import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import utils
from dm_control.utils import rewards
from einops import rearrange, reduce, repeat
from agent.modules.attention import Block, CausalSelfAttention
from agent.mdp import MaskedDPMultimodal


class MDP_MM_GoalAgent:
    def __init__(
        self,
        name,
        obs_shape,
        action_shape,
        device,
        lr,
        batch_size,
        use_tb,
        finetune,
        transformer_cfg,
        path=None,
    ):
        self.action_dim = action_shape[0]
        self.lr = lr
        self.device = device
        self.use_tb = use_tb

        # init from snapshot
        payload = None
        if path is not None:
            print("loading existing model...")
            payload = torch.load(path)
            self.config = payload["cfg"]
        else:
            self.config = transformer_cfg
        self.mdp = MaskedDPMultimodal(obs_shape[0], action_shape[0], self.config).to(device)
        print("number of parameters: %e", sum(p.numel() for p in self.mdp.parameters()))
        if path is not None:
            self.mdp.load_state_dict(payload["model"])
            
        # Check if this is a multimodal architecture
        self.is_multimodal = self._check_multimodal()
        self.is_early_fusion = hasattr(self.mdp, 'early_fusion_blocks')
        mode_str = 'Early Fusion' if self.is_early_fusion else ('Late Fusion' if self.is_multimodal else 'Legacy')
        print(f"Model type: {mode_str}")

        self.finetune = finetune
        self._freeze_layers()
        self.train()
        
    def _check_multimodal(self):
        """Check if the model has multimodal architecture components."""
        has_state_encoder = hasattr(self.mdp, 'state_encoder_blocks')
        has_action_encoder = hasattr(self.mdp, 'action_encoder_blocks')
        has_early_fusion = hasattr(self.mdp, 'early_fusion_blocks')
        has_fusion = hasattr(self.mdp, 'fusion_blocks')
        return ((has_state_encoder and has_action_encoder) or has_early_fusion) and has_fusion

    def _freeze_layers(self):
        frozen_layers = [
            self.mdp.state_embed,
            self.mdp.action_embed,
            self.mdp.decoder_pos_embed,
            self.mdp.decoder_state_embed,
            self.mdp.decoder_action_embed,
            self.mdp.mask_token,
        ]

        if self.finetune == "decoder":
            if self.is_early_fusion:
                frozen_layers += [
                    self.mdp.early_fusion_blocks, 
                    self.mdp.state_encoder_norm,
                    self.mdp.action_encoder_norm,
                    self.mdp.fusion_blocks,
                    self.mdp.fusion_norm,
                ]
                if self.mdp.fusion_type_embed is not None:
                    frozen_layers.append(self.mdp.fusion_type_embed)
            elif self.is_multimodal:
                frozen_layers += [
                    self.mdp.state_encoder_blocks, 
                    self.mdp.state_encoder_norm,
                    self.mdp.action_encoder_blocks, 
                    self.mdp.action_encoder_norm,
                    self.mdp.state_proj,
                    self.mdp.action_proj,
                    self.mdp.fusion_blocks,
                    self.mdp.fusion_norm,
                ]
                if self.mdp.fusion_type_embed is not None:
                    frozen_layers.append(self.mdp.fusion_type_embed)
            else:
                # Legacy: freeze encoder blocks
                if hasattr(self.mdp, 'encoder_blocks'):
                    frozen_layers.append(self.mdp.encoder_blocks)

        if self.finetune == "linear":
            frozen_layers += [
                self.mdp.decoder_blocks,
                self.mdp.decoder_state_embed,
                self.mdp.decoder_action_embed,
            ]

        for m in frozen_layers:
            if isinstance(m, nn.Module):
                for param in m.parameters():
                    param.requires_grad = False
            elif isinstance(m, nn.Parameter):
                m.requires_grad = False
        # optimizers
        print(
            "number of parameters to be tuned: %e",
            sum(
                p.numel()
                for p in filter(lambda p: p.requires_grad, self.mdp.parameters())
            ),
        )
        self.opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.mdp.parameters()), lr=self.lr
        )

    def train(self, training=True):
        self.training = training
        self.mdp.train(training)
        
    def _act_multimodal(self, obs, goal, T):
        """Goal-conditioned action generation for multimodal architecture."""
        batch_size = obs.shape[0]

        if 2 * (T + 1) > self.mdp.decoder_pos_embed.shape[1]:
            enc_pos_embed = utils.interpolate_pos_embed(self.mdp.pos_embed, 2 * (T + 1))
            decoder_pos_embed = utils.interpolate_pos_embed(
                self.mdp.decoder_pos_embed, 2 * (T + 1)
            )
            attn_mask = torch.ones(2 * (T + 1), 2 * (T + 1))[None, None, ...].to(
                self.device
            )
        else:
            enc_pos_embed = self.mdp.pos_embed              # [1, max_len, enc_n_embd]
            decoder_pos_embed = self.mdp.decoder_pos_embed  # [1, max_len, n_embd]
            attn_mask = self.mdp.attn_mask

        # ENCODER PHASE
        s_emb = self.mdp.state_embed(obs) + enc_pos_embed[:, 0:1]       # [B, 1, enc_n_embd]
        g_emb = self.mdp.state_embed(goal) + enc_pos_embed[:, 2*T:2*T+1]
        s_enc_input = torch.cat([s_emb, g_emb], dim=1)                  # [B, 2, enc_n_embd]

        # Placeholder de acciones en enc_n_embd (enc_mask_token si existe, mask_token si no)
        _enc_mtok = getattr(self.mdp, 'enc_mask_token', self.mdp.mask_token)
        a_emb = _enc_mtok.repeat(batch_size, 2, 1)                      # [B, 2, enc_n_embd]
        a_emb[:, 0] = a_emb[:, 0] + enc_pos_embed[:, 1]
        a_emb[:, 1] = a_emb[:, 1] + enc_pos_embed[:, 2*T+1]

        if self.is_early_fusion:
            # EARLY FUSION
            s_encoded = s_enc_input
            a_encoded = a_emb
            for blk in self.mdp.early_fusion_blocks:
                s_encoded, a_encoded = blk(
                    x_s=s_encoded, x_a=a_encoded,
                    mask_s=attn_mask, mask_a=attn_mask,
                    pad_mask_s=None, pad_mask_a=None
                )
        else:
            # LATE FUSION (Baseline)
            # State encoder: process [s0, s_goal]
            s_encoded = s_enc_input
            for blk in self.mdp.state_encoder_blocks:
                s_encoded = blk(s_encoded, attn_mask)
            # Action encoder: process zeros (no ground-truth actions)
            a_encoded = a_emb
            for blk in self.mdp.action_encoder_blocks:
                a_encoded = blk(a_encoded, attn_mask)
        
        s_encoded = self.mdp.state_encoder_norm(s_encoded)  # [B, 2, D]
        a_encoded = self.mdp.action_encoder_norm(a_encoded)  # [B, 2, D]

        s_encoded = self.mdp.state_adapter(s_encoded)
        a_encoded = self.mdp.action_adapter(a_encoded)

        s_encoded = self.mdp.state_proj(s_encoded)
        a_encoded = self.mdp.action_proj(a_encoded)

        # Fusion: combine state and action information
        # Create ids_keep for fusion (interleaved: [0=s0, 1=a0, 2=s_goal, 3=a_goal])
        ids_keep = torch.arange(4, device=self.device).unsqueeze(0).expand(batch_size, -1)
        
        if self.mdp.fusion_type == 'cross':
            # Cross-attention fusion
            x_s = s_encoded
            x_a = a_encoded
            
            if self.mdp.fusion_type_embed is not None:
                type_ids_s = torch.zeros(batch_size, 2, dtype=torch.long, device=self.device)
                type_ids_a = torch.ones(batch_size, 2, dtype=torch.long, device=self.device)
                x_s = x_s + self.mdp.fusion_type_embed(type_ids_s)
                x_a = x_a + self.mdp.fusion_type_embed(type_ids_a)
            
            for blk in self.mdp.fusion_blocks:
                x_s, x_a = blk(x_s, x_a, mask_s=None, mask_a=None)
            
            x_fused = self.mdp._combine_kept_tokens(x_s, x_a, ids_keep)
            x_fused = self.mdp.fusion_norm(x_fused)
        else:
            # Self-attention fusion
            x_fused = self.mdp._combine_kept_tokens(s_encoded, a_encoded, ids_keep)
            
            if self.mdp.fusion_type_embed is not None:
                type_ids = (ids_keep % 2).long()
                x_fused = x_fused + self.mdp.fusion_type_embed(type_ids)
            
            for blk in self.mdp.fusion_blocks:
                x_fused = blk(x_fused, attn_mask)
            x_fused = self.mdp.fusion_norm(x_fused)
        
        # DECODER PHASE
        # Prepare decoder inputs: states are [s0, mask, ..., mask, s_goal]
        if T > 1:
            mask_states = self.mdp.mask_token.repeat(batch_size, T - 1, 1)
            obs_dec = torch.cat([
                x_fused[:, 0].unsqueeze(1),  # s0 (fused)
                mask_states,                   # intermediate masked states
                x_fused[:, 2].unsqueeze(1)     # s_goal (fused)
            ], dim=1)  # [B, T+1, D]
        else:
            obs_dec = torch.cat([
                x_fused[:, 0].unsqueeze(1),
                x_fused[:, 2].unsqueeze(1)
            ], dim=1)  # [B, 2, D]
        
        # Actions are all masked (need to be predicted)
        mask_actions = self.mdp.mask_token.repeat(batch_size, T + 1, 1)

        obs_dec = self.mdp.decoder_state_embed(obs_dec)
        mask_actions = self.mdp.decoder_action_embed(mask_actions)

        x_dec = torch.stack([obs_dec, mask_actions], dim=2).reshape(
            batch_size, 2 * (T + 1), self.config.n_embd
        )
        x_dec += decoder_pos_embed[:, :2 * (T + 1)]
        
        # Apply decoder blocks
        for blk in self.mdp.decoder_blocks:
            x_dec = blk(x_dec, attn_mask)
        
        # Extract actions (odd positions, exclude last)
        actions = self.mdp.action_head(x_dec[:, 1::2])[:, :-1]  # [B, T, action_dim]
        
        return actions.cpu().numpy()

    def _act_legacy(self, obs, goal, T):
        """Legacy goal-conditioned action generation (original MaskDP)."""
        batch_size = obs.shape[0]
        
        if 2 * (T + 1) > self.mdp.decoder_pos_embed.shape[1]:
            pos_embed = utils.interpolate_pos_embed(
                self.mdp.decoder_pos_embed, 2 * (T + 1)
            )
            decoder_pos_embed = pos_embed
            attn_mask = torch.ones(2 * (T + 1), 2 * (T + 1))[None, None, ...].to(
                self.device
            )
        else:
            pos_embed = self.mdp.decoder_pos_embed
            decoder_pos_embed = self.mdp.decoder_pos_embed
            attn_mask = self.mdp.attn_mask

        s_emb = self.mdp.state_embed(obs) + pos_embed[:, 0]
        g_emb = self.mdp.state_embed(goal) + pos_embed[:, 2 * T]
        
        # encoder
        x = torch.cat([s_emb, g_emb], dim=1)
        for blk in self.mdp.state_encoder_blocks:
            x = blk(x, attn_mask)
        x = self.mdp.state_encoder_norm(x)

        if T > 1:
            mask_states = self.mdp.mask_token.repeat(batch_size, T - 1, 1)
            obs = torch.cat([x[:, 0].unsqueeze(1), mask_states], dim=1)
            obs = torch.cat([obs, x[:, -1].unsqueeze(1)], dim=1)
        else:
            obs = x

        mask_actions = self.mdp.mask_token.repeat(batch_size, T + 1, 1)
        obs = self.mdp.decoder_state_embed(obs)
        mask_actions = self.mdp.decoder_action_embed(mask_actions)

        x = (
            torch.stack([obs, mask_actions], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 2 * (T + 1), self.config.n_embd)
        )
        x += decoder_pos_embed[:, : 2 * (T + 1)]
        
        # apply Transformer blocks
        for blk in self.mdp.decoder_blocks:
            x = blk(x, attn_mask)
        actions = self.mdp.action_head(x[:, 1::2])[:, :-1]
        return actions.cpu().numpy()
    
    def _multi_goal_act_multimodal(self, obs, goal, time_budgets):
        """Multi-goal action generation for multimodal architecture."""
        batch_size = obs.shape[0]
        T = time_budgets[-1]
        num_goals = goal.shape[1]

        if 2 * (T + 1) > self.mdp.decoder_pos_embed.shape[1]:
            enc_pos_embed = utils.interpolate_pos_embed(self.mdp.pos_embed, 2 * (T + 1))
            decoder_pos_embed = utils.interpolate_pos_embed(
                self.mdp.decoder_pos_embed, 2 * (T + 1)
            )
            attn_mask = torch.ones(2 * (T + 1), 2 * (T + 1))[None, None, ...].to(self.device)
        else:
            enc_pos_embed = self.mdp.pos_embed
            decoder_pos_embed = self.mdp.decoder_pos_embed
            attn_mask = self.mdp.attn_mask


        # ENCODER PHASE
        s_emb = self.mdp.state_embed(obs) + enc_pos_embed[:, 0:1]  # [B, 1, D]
        g_emb = self.mdp.state_embed(goal) + enc_pos_embed[:, time_budgets * 2]  # [B, num_goals, D]
        s_enc_input = torch.cat([s_emb, g_emb], dim=1)  # [B, num_goals+1, D]

        _enc_mtok = getattr(self.mdp, 'enc_mask_token', self.mdp.mask_token)
        a_emb = _enc_mtok.repeat(batch_size, num_goals + 1, 1)  # [B, num_goals+1, D]
        a_emb[:, 0] = a_emb[:, 0] + enc_pos_embed[:, 1]  # position for a0 
        a_emb[:, 1] = a_emb[:, 1] + enc_pos_embed[:, 2*T+1]  # position for a_goal

        if self.is_early_fusion:
            # EARLY FUSION
            s_encoded = s_enc_input
            a_encoded = a_emb
            for blk in self.mdp.early_fusion_blocks:
                s_encoded, a_encoded = blk(
                    x_s=s_encoded, x_a=a_encoded,
                    mask_s=attn_mask, mask_a=attn_mask,
                    pad_mask_s=None, pad_mask_a=None
                )
        else:
            #LATE FUSION
            # State encoder: process [s0, g1, g2, ..., gN]
            s_encoded = s_enc_input
            for blk in self.mdp.state_encoder_blocks:
                s_encoded = blk(s_encoded, attn_mask)
            # Action encoder: process zeros (no ground-truth actions)
            a_encoded = a_emb
            for blk in self.mdp.action_encoder_blocks:
                a_encoded = blk(a_encoded, attn_mask)

        s_encoded = self.mdp.state_encoder_norm(s_encoded)
        a_encoded = self.mdp.action_encoder_norm(a_encoded)

        s_encoded = self.mdp.state_adapter(s_encoded)
        a_encoded = self.mdp.action_adapter(a_encoded)

        s_encoded = self.mdp.state_proj(s_encoded)
        a_encoded = self.mdp.action_proj(a_encoded)

        total_tokens = (num_goals + 1) * 2
        ids_keep = torch.arange(total_tokens, device=self.device).unsqueeze(0).expand(batch_size, -1)
        
        if self.mdp.fusion_type == 'cross':
            # Cross-attention fusion
            x_s = s_encoded
            x_a = a_encoded
            
            if self.mdp.fusion_type_embed is not None:
                type_ids_s = torch.zeros(batch_size, num_goals + 1, dtype=torch.long, device=self.device)
                type_ids_a = torch.ones(batch_size, num_goals + 1, dtype=torch.long, device=self.device)
                x_s = x_s + self.mdp.fusion_type_embed(type_ids_s)
                x_a = x_a + self.mdp.fusion_type_embed(type_ids_a)
            
            for blk in self.mdp.fusion_blocks:
                x_s, x_a = blk(x_s, x_a, mask_s=None, mask_a=None)
            
            x_fused = self.mdp._combine_kept_tokens(x_s, x_a, ids_keep)
            x_fused = self.mdp.fusion_norm(x_fused)
        else:
            # Self-attention fusion
            x_fused = self.mdp._combine_kept_tokens(s_encoded, a_encoded, ids_keep)
            
            if self.mdp.fusion_type_embed is not None:
                type_ids = (ids_keep % 2).long()
                x_fused = x_fused + self.mdp.fusion_type_embed(type_ids)
            
            for blk in self.mdp.fusion_blocks:
                x_fused = blk(x_fused, attn_mask)
            x_fused = self.mdp.fusion_norm(x_fused)

        state_indices = torch.arange(0, total_tokens, 2, device=self.device)
        x_states = x_fused[:, state_indices]  # [B, num_goals+1, D]
        
        # DECODER PHASE
        # Prepare decoder inputs: place s0 and goals at their time positions
        if T > 1:
            obs_dec = self.mdp.mask_token.repeat(batch_size, T + 1, 1)
            obs_dec[:, 0] = x_states[:, 0]  # s0
            obs_dec[:, time_budgets] = x_states[:, 1:]  # goals at their time budgets
        else:
            obs_dec = x_states
        
        # Actions are all masked (need to be predicted)
        mask_actions = self.mdp.mask_token.repeat(batch_size, T + 1, 1)
        
        obs_dec = self.mdp.decoder_state_embed(obs_dec)
        mask_actions = self.mdp.decoder_action_embed(mask_actions)
        
        x_dec = torch.stack([obs_dec, mask_actions], dim=2).reshape(
            batch_size, 2 * (T + 1), self.config.n_embd
        )
        x_dec += decoder_pos_embed[:, :2 * (T + 1)]
        
        # Apply decoder blocks
        for blk in self.mdp.decoder_blocks:
            x_dec = blk(x_dec, attn_mask)
        
        # Extract actions (odd positions, exclude last)
        actions = self.mdp.action_head(x_dec[:, 1::2])[:, :-1]  # [B, T, action_dim]
        
        return actions.cpu().numpy()

    def _multi_goal_act_legacy(self, obs, goal, time_budgets):
        """Multi-goal action generation for legacy architecture (original MaskDP)."""
        batch_size = obs.shape[0]
        T = time_budgets[-1]
        
        if 2 * (T + 1) > self.mdp.decoder_pos_embed.shape[1]:
            pos_embed = utils.interpolate_pos_embed(
                self.mdp.decoder_pos_embed, 2 * (T + 1)
            )
            decoder_pos_embed = pos_embed
            attn_mask = torch.ones(2 * (T + 1), 2 * (T + 1))[None, None, ...].to(
                self.device
            )
        else:
            pos_embed = self.mdp.decoder_pos_embed
            decoder_pos_embed = self.mdp.decoder_pos_embed
            attn_mask = self.mdp.attn_mask

        s_emb = self.mdp.state_embed(obs) + pos_embed[:, 0]
        g_emb = self.mdp.state_embed(goal) + pos_embed[:, time_budgets * 2]
        
        # encoder
        x = torch.cat([s_emb, g_emb], dim=1)
        for blk in self.mdp.state_encoder_blocks:
            x = blk(x, attn_mask)
        x = self.mdp.state_encoder_norm(x)

        if T > 1:
            obs = self.mdp.mask_token.repeat(batch_size, T + 1, 1)
            obs[:, 0] = x[:, 0]
            obs[:, time_budgets] = x[:, 1:]
        else:
            obs = x

        mask_actions = self.mdp.mask_token.repeat(batch_size, T + 1, 1)
        obs = self.mdp.decoder_state_embed(obs)
        mask_actions = self.mdp.decoder_action_embed(mask_actions)

        x = (
            torch.stack([obs, mask_actions], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 2 * (T + 1), self.config.n_embd)
        )
        x += decoder_pos_embed[:, : 2 * (T + 1)]
        
        # apply Transformer blocks
        for blk in self.mdp.decoder_blocks:
            x = blk(x, attn_mask)
        actions = self.mdp.action_head(x[:, 1::2])[:, :-1]
        return actions.cpu().numpy()

    def multi_goal_act(self, obs, goal, time_budgets):
        """Multi-goal action generation interface (for multi-goal evaluation)."""
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        goal = torch.as_tensor(goal, device=self.device).unsqueeze(0)
        
        assert goal.shape[1] == len(time_budgets)
        
        if self.is_multimodal:
            actions = self._multi_goal_act_multimodal(obs, goal, time_budgets)
        else:
            actions = self._multi_goal_act_legacy(obs, goal, time_budgets)
        
        return actions[0]

    def act(self, obs, goal, T):
        """Main action generation interface."""
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        goal = torch.as_tensor(goal, device=self.device).unsqueeze(0)
        
        if self.is_multimodal:
            actions = self._act_multimodal(obs, goal, T)
        else:
            actions = self._act_legacy(obs, goal, T)
        
        return actions[0]
