import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from collections import OrderedDict

import utils
from dm_control.utils import rewards
from einops import rearrange, reduce, repeat
from agent.modules.attention import Block, CausalSelfAttention


class MaskedDPMultimodal(nn.Module):
    def __init__(self, obs_dim, action_dim, config):
        super().__init__()
        # Padding sentinel (used for ragged state/action sequences within a batch)
        # Prefer a value that will never appear in real embedded tokens.
        self.pad_value = float(getattr(config, "pad_value", 1e9))
        # MAE encoder specifics
        self.n_embd = config.n_embd
        self.max_len = config.traj_length * 2
        # self.mask_ratio = config.mask_ratio
        self.pe = config.pe
        self.norm = config.norm
        print("norm", self.norm)
        self.state_embed = nn.Linear(obs_dim, self.n_embd)
        self.action_embed = nn.Linear(action_dim, self.n_embd)
        
        # Separate encoders for state and action
        self.state_encoder_blocks = nn.ModuleList(
            [Block(config) for _ in range(config.n_enc_layer)]
        )
        self.action_encoder_blocks = nn.ModuleList(
            [Block(config) for _ in range(config.n_enc_layer)]
        )
        
        # Normalization for encoders
        self.state_encoder_norm = nn.LayerNorm(self.n_embd)
        self.action_encoder_norm = nn.LayerNorm(self.n_embd)

        # --------------------------------------------------------------------------
        # Fusion encoder (cross-modal interaction after separate encoders, before decoder)
        self.n_fuse_layer = int(getattr(config, "n_fuse_layer", 0))

        # Toggles modality/type embeddings (state vs action)
        self.use_fusion_type_embed = bool(getattr(config, "use_fusion_type_embed", False))

        # Categoric embedding with state and actions classes (Like A/B on BERT)
        self.fusion_type_embed = nn.Embedding(2, self.n_embd) if self.use_fusion_type_embed else None

        #Fusion Block
        self.fusion_blocks = nn.ModuleList(
            [Block(config) for _ in range(self.n_fuse_layer)]
        )
        self.fusion_norm = nn.LayerNorm(self.n_embd) if self.n_fuse_layer > 0 else nn.Identity()

        # --------------------------------------------------------------------------
        # Mask tokens for decoder
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.n_embd))
        
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_state_embed = nn.Linear(self.n_embd, self.n_embd)
        self.decoder_action_embed = nn.Linear(self.n_embd, self.n_embd)

        self.decoder_blocks = nn.ModuleList(
            [Block(config) for _ in range(config.n_dec_layer)]
        )

        self.action_head = nn.Sequential(
            nn.LayerNorm(self.n_embd),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_embd, action_dim),
            nn.Tanh(),
        )  # decoder to patch
        self.state_head = nn.Sequential(
            nn.LayerNorm(self.n_embd),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_embd, obs_dim),
        )
        # --------------------------------------------------------------------------
        self.initialize_weights()

    def initialize_weights(self):
        # Positional embeddings SHOULD NOT be separated, this actually breaks the
        # original trayectory order
        pos_embed = utils.get_1d_sincos_pos_embed_from_grid(self.n_embd, self.max_len)
        pe = torch.from_numpy(pos_embed).float().unsqueeze(0) / 2.0
        
        self.register_buffer("pos_embed", pe)
        
        # For decoder we use full pos_embed
        self.register_buffer("decoder_pos_embed", pe)
        
        self.register_buffer(
            "attn_mask", torch.ones(self.max_len, self.max_len)[None, None, ...]
        )
        
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # Init mask tokens
        torch.nn.init.normal_(self.mask_token, std=0.02)
        if self.fusion_type_embed is not None:
            torch.nn.init.normal_(self.fusion_type_embed.weight, std=0.02)
        
        self.apply(self._init_weights)
        
        # Initialize FUSION with zeros to stabilize freezing training
        for blk in self.fusion_blocks:
            # Zero-init Attention Output Projection
            nn.init.zeros_(blk.attn.proj.weight)
            nn.init.zeros_(blk.attn.proj.bias)

            # Zero-init MLP Output Projection
            nn.init.zeros_(blk.mlp[2].weight)
            nn.init.zeros_(blk.mlp[2].bias)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence

        Applies the same masking pattern for states and actions
        to maintain temporal consistency (concatenated sequence)
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        # noise independent between modalities
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        # Same pattern for s & a
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def build_blockdiag_pad_attn_mask(self, lengths: torch.Tensor, L: int) -> torch.Tensor:
        """
        lengths: [B] long, number of valid tokens (no-padding) per batch element
        return: [B, 1, L, L] float mask with blocks:
        - valid <-> valid
        - padding <-> padding
        - valid <-> padding blocked
        - padding <-> valid blocked
        """
        # [L]
        idx = torch.arange(L, device=lengths.device)
        # [B, L]  True when a token is valid
        valid = idx.unsqueeze(0) < lengths.unsqueeze(1)

        # [B, L, L]
        vv = valid.unsqueeze(2) & valid.unsqueeze(1)
        pp = (~valid).unsqueeze(2) & (~valid).unsqueeze(1)

        mask2d = (vv | pp)  # block diagonal

        # [B, 1, L, L] float
        return mask2d.unsqueeze(1).to(dtype=torch.float32)

    # Auxiliary function to concatenate unmasked states and actions
    def _combine_kept_tokens(self, s_tokens: torch.Tensor, a_tokens: torch.Tensor, ids_keep: torch.Tensor) -> torch.Tensor:
        """Reconstruct the kept token sequence in `ids_keep` order by interleaving state/action streams.

        Parity convention (from interleaving): even indices are states, odd indices are actions.

        Args:
            s_tokens: [B, Ls, D] encoded state tokens (padded along Ls)
            a_tokens: [B, La, D] encoded action tokens (padded along La)
            ids_keep: [B, len_keep] original indices of kept tokens in the interleaved sequence

        Returns:
            x_keep: [B, len_keep, D] kept tokens in ids_keep order
        """
        B, len_keep = ids_keep.shape
        D = s_tokens.size(-1)

        is_state = (ids_keep % 2) == 0  # [B, len_keep]
        is_action = ~is_state

        # Per-slot index into the (state/action) token lists
        s_idx = torch.cumsum(is_state.to(torch.long), dim=1) - 1  # [B, len_keep]
        a_idx = torch.cumsum(is_action.to(torch.long), dim=1) - 1  # [B, len_keep]
        s_idx = s_idx.clamp(min=0)
        a_idx = a_idx.clamp(min=0)

        # Gather candidates. Unused candidates are ignored by torch.where.
        if s_tokens.size(1) > 0:
            s_slots = s_tokens.gather(1, s_idx.unsqueeze(-1).expand(-1, -1, D))
        else:
            s_slots = s_tokens.new_zeros(B, len_keep, D)

        if a_tokens.size(1) > 0:
            a_slots = a_tokens.gather(1, a_idx.unsqueeze(-1).expand(-1, -1, D))
        else:
            a_slots = a_tokens.new_zeros(B, len_keep, D)

        x_keep = torch.where(is_state.unsqueeze(-1), s_slots, a_slots)  # [B, len_keep, D]
        return x_keep

    def forward_fusion(self, s_encoded: torch.Tensor, a_encoded: torch.Tensor, ids_keep: torch.Tensor) -> torch.Tensor:
        """Optional fusion encoder over kept tokens (after separate state/action encoders).

        If `n_fuse_layer == 0` and `use_fusion_type_embed == False`, this is effectively an identity mapping
        (it just reconstructs the kept sequence in ids_keep order).
        """
        x = self._combine_kept_tokens(s_encoded, a_encoded, ids_keep)  # [B, len_keep, D]

        if self.fusion_type_embed is not None:
            type_ids = (ids_keep % 2).long()  # 0=state (even), 1=action (odd)
            x = x + self.fusion_type_embed(type_ids)

        if len(self.fusion_blocks) > 0:
            B, L, _ = x.shape
            fuse_mask = torch.ones(B, 1, L, L, device=x.device, dtype=torch.float32)
            for blk in self.fusion_blocks:
                x = blk(x, fuse_mask)
            x = self.fusion_norm(x)

        return x

    def forward_encoder(self, states, actions, mask_ratio):
        batch_size, T, obs_dim = states.size()
        
        # Embeddings
        s_emb = self.state_embed(states)
        a_emb = self.action_embed(actions)
        
        # Base:
        """
        x = torch.stack([s_emb, a_emb], dim=1).permute(0, 2, 1, 3).reshape(batch_size, 2 * T, self.n_embd)
        x = x + self.pos_embed
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        """
        # Separating the positional embeddings this ways loses the interaction between 
        # states and actions. In a nutshell, the interleaving is lost, this should be 
        # fixed with the fusion encoder, so we leave for the moment.

        # Separate positional embeddings and consider even and odd pos
        # States are even: 0, 2, 4, 6, ... (idex s0, s1, s2, s3, ...)
        # Actions are odd: 1, 3, 5, 7, ... (index a0, a1, a2, a3, ...)
        s_emb = s_emb + self.pos_embed[:, 0:2*T:2, :]
        a_emb = a_emb + self.pos_embed[:, 1:2*T:2, :]

        # Interleave states and actions: [s0, a0, s1, a1, s2, a2, ...]
        x = torch.stack([s_emb, a_emb], dim=2).reshape(batch_size, 2 * T, self.n_embd)
        
        # Apply masking on the full interleaved sequence
        x_masked, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)

        # Now separate the kept tokens into states and actions
        # ids_keep[b, j] tells us the original index in [0, 2T-1]
        # Even indices are states, odd indices are actions

        # Determine which kept tokens are states vs actions based on their ORIGINAL indices
        is_state = (ids_keep % 2) == 0  # [N, len_keep]
        is_action = ~is_state  # [N, len_keep]
        
        # Extract state and action tokens
        # We need to maintain batch processing, so we'll use masking
        s_masked = []
        a_masked = []
        
        for b in range(batch_size):
            s_masked.append(x_masked[b, is_state[b]])  # [num_states_kept, D]
            a_masked.append(x_masked[b, is_action[b]])  # [num_actions_kept, D]
        

        # Pad to same length within batch using a sentinel value. (removing python for loops)
        s_masked = pad_sequence(s_masked, batch_first=True, padding_value=self.pad_value)  # [B, Ls, D]
        a_masked = pad_sequence(a_masked, batch_first=True, padding_value=self.pad_value)  # [B, La, D]

        # 1D padding masks (True where padding)
        s_pad_mask_1d = (s_masked == self.pad_value).all(dim=-1)
        a_pad_mask_1d = (a_masked == self.pad_value).all(dim=-1)

        # Valid-only attention masks
        # [B, L] -> [B, L, L] via outer-product of validity, then add head dim -> [B,1,L,L]
        s_valid = ~s_pad_mask_1d
        a_valid = ~a_pad_mask_1d
        s_attn_mask = (s_valid.unsqueeze(2) & s_valid.unsqueeze(1)).unsqueeze(1).to(dtype=torch.float32)
        a_attn_mask = (a_valid.unsqueeze(2) & a_valid.unsqueeze(1)).unsqueeze(1).to(dtype=torch.float32)
        
        # Process with separate encoders
        s_encoded = s_masked
        for blk in self.state_encoder_blocks:
            s_encoded = blk(s_encoded, s_attn_mask)
        s_encoded = self.state_encoder_norm(s_encoded)
        
        a_encoded = a_masked
        for blk in self.action_encoder_blocks:
            a_encoded = blk(a_encoded, a_attn_mask)
        a_encoded = self.action_encoder_norm(a_encoded)
        
        # Fuse and return kept tokens for the decoder
        x_fused = self.forward_fusion(s_encoded, a_encoded, ids_keep)
        
        # Return also ids_keep to track state/action positions
        return x_fused, mask, ids_restore, ids_keep

    def forward_decoder(self, x_fused: torch.Tensor, ids_restore: torch.Tensor):
        """MAE-style decoder.
        Args:
            x_fused: [B, len_keep, D] kept tokens in ids_keep / ids_shuffle-kept order (post-fusion)
            ids_restore: [B, total_len] indices to restore original interleaved order
        Returns:
            s_pred: [B, T, obs_dim]
            a_pred: [B, T, action_dim]
        """

        batch_size = x_fused.shape[0]
        total_len = ids_restore.shape[1]
        len_keep = x_fused.shape[1]

        # Append mask tokens to reach total_len
        n_mask = total_len - len_keep
        if n_mask > 0:
            mask_tokens = self.mask_token.expand(batch_size, n_mask, self.n_embd)
        else:
            mask_tokens = x_fused.new_empty(batch_size, 0, self.n_embd)

        x_full = torch.cat([x_fused, mask_tokens], dim=1)  # [B, total_len, D]
        
        # Unshuffle using ids_restore to get back to interleaved [s0,a0,s1,a1,...]
        x = torch.gather(
            x_full, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_full.shape[2])
        )
        
        # Project to decoder embedding space
        s = self.decoder_state_embed(x[:, ::2])
        a = self.decoder_action_embed(x[:, 1::2])
        
        # Interleave actions and states for the decoder
        x = torch.stack([s, a], dim=2).reshape(batch_size, total_len, self.n_embd)
        
        # Add positional embeddings
        x = x + self.decoder_pos_embed[:, :x.shape[1], :]
        
        # Apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, self.attn_mask)
        
        # Split and predict
        s_pred = self.state_head(x[:, ::2])
        a_pred = self.action_head(x[:, 1::2])
        
        return s_pred, a_pred

    def forward_loss(self, target_s, target_a, pred_s, pred_a, mask):
        batch_size, T, _ = target_s.size()
        
        # State normalization
        if self.norm == "l2":
            target_s = target_s / torch.norm(target_s, dim=-1, keepdim=True)
        elif self.norm == "mae":
            mean = target_s.mean(dim=-1, keepdim=True)
            var = target_s.var(dim=-1, keepdim=True)
            target_s = (target_s - mean) / (var + 1.0e-6) ** 0.5

        # MSE per dimention
        loss_s = (pred_s - target_s) ** 2
        loss_a = (pred_a - target_a) ** 2

        # mean MSE per token 
        loss_s_t = loss_s.mean(dim=-1)
        loss_a_t = loss_a.mean(dim=-1)

        # Intercalate [s0,a0,s1,a1,...] -> shape [B, 2T]
        loss_tokens = torch.stack([loss_s_t, loss_a_t], dim=-1)  # [B, T, 2]
        loss_tokens = loss_tokens.reshape(batch_size, 2 * T)     # [B, 2T]
            
        # Only ONE masked_loss for both modalities (same idea than the original)
        masked_loss = (loss_tokens * mask).sum() / mask.sum()

        loss_s = loss_s.mean()
        loss_a = loss_a.mean()
        
        return masked_loss, loss_s, loss_a


class MaskedDPMultimodalAgent:
    def __init__(
        self,
        name,
        obs_shape,
        action_shape,
        device,
        lr,
        batch_size,
        use_tb,
        mask_ratio,
        transformer_cfg,
        freeze_schedule=None,
    ):
        self.action_dim = action_shape[0]
        self.lr = lr
        self.device = device
        self.use_tb = use_tb
        self.config = transformer_cfg
        
        # Schedule {'module_nane': [start_step, end_step]}
        self.freeze_schedule = freeze_schedule if freeze_schedule is not None else {}

        # models
        self.model = MaskedDPMultimodal(obs_shape[0], action_shape[0], transformer_cfg).to(device)
        self.mask_ratio = mask_ratio
        # optimizers
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        print(
            "number of parameters: %e", sum(p.numel() for p in self.model.parameters())
        )

        self.train()
        
    def set_module_requires_grad(self, module, requires_grad):
        for param in module.parameters():
            param.requires_grad = requires_grad
            
    def _rebuild_optimizer(self):
        """Rebuild optimizer with only trainable parameters"""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        print(f"Rebuilding optimizer with {len(trainable_params)} trainable parameter groups")
        self.opt = torch.optim.Adam(trainable_params, lr=self.lr)
            
    def check_freeze_schedule(self, step):
        """Applies freezing to a block if step is inside an interval [start, end)"""
        if step is None:
            return
        
        needs_optimizer_rebuild = False

        for module_name, (start_step, end_step) in self.freeze_schedule.items():
            module = getattr(self.model, module_name, None)
            if module is None:
                continue
            
            should_train = not (start_step <= step < end_step)
            current_status = next(module.parameters()).requires_grad
            
            if should_train != current_status:
                status = "TRAINABLE" if should_train else "FROZEN"
                print(f"[{step}] Module '{module_name}' -> {status} (interval: [{start_step}, {end_step}))")
                self.set_module_requires_grad(module, should_train)
                needs_optimizer_rebuild = True
                
        if needs_optimizer_rebuild:
            self._rebuild_optimizer()

    def train(self, training=True):
        self.training = training
        self.model.train(training)

    def update_mdp(self, states, actions, step=None):
        self.check_freeze_schedule(step)
        metrics = dict()
        mask_ratio = np.random.choice(self.mask_ratio)
        
        # Encoder (dual + optional fusion)
        x_fused, mask, ids_restore, ids_keep = \
            self.model.forward_encoder(states, actions, mask_ratio)
        
        # Decoder
        pred_s, pred_a = self.model.forward_decoder(
            x_fused, ids_restore
        )
        
        # Loss
        mask_loss, state_loss, action_loss = self.model.forward_loss(
            states, actions, pred_s, pred_a, mask
        )
        
        if self.config.loss == "masked":
            loss = mask_loss
        elif self.config.loss == "total":
            loss = state_loss + action_loss
        else:
            raise NotImplementedError

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        if self.use_tb:
            metrics["mask_loss"] = mask_loss.item()
            metrics["state_loss"] = state_loss.item()
            metrics["action_loss"] = action_loss.item()

        return metrics

    def eval_validation(self, val_iter, step=None):
        metrics = dict()
        batch = next(val_iter)
        obs, action, _, _, _, _ = utils.to_torch(batch, self.device)
        
        mask_ratio = np.random.choice(self.mask_ratio)
        x_fused, mask, ids_restore, ids_keep = \
            self.model.forward_encoder(obs, action, mask_ratio)
        
        pred_s, pred_a = self.model.forward_decoder(
            x_fused, ids_restore
        )
        
        mask_loss, state_loss, action_loss = self.model.forward_loss(
            obs, action, pred_s, pred_a, mask
        )

        if self.use_tb:
            metrics["val_mask_loss"] = mask_loss.item()
            metrics["val_state_loss"] = state_loss.item()
            metrics["val_action_loss"] = action_loss.item()

        return metrics

    def update(self, replay_iter, step=None):
        metrics = dict()

        batch = next(replay_iter)
        obs, action, _, _, _, _ = utils.to_torch(batch, self.device)

        # update critic
        metrics.update(self.update_mdp(obs, action, step=step))

        return metrics
