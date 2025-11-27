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


class MaskedDPMultimodal(nn.Module):
    def __init__(self, obs_dim, action_dim, config):
        super().__init__()
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
        
        # Mask tokens for encoders
        self.state_mask_token = nn.Parameter(torch.zeros(1, 1, self.n_embd))
        self.action_mask_token = nn.Parameter(torch.zeros(1, 1, self.n_embd))
        
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
        torch.nn.init.normal_(self.state_mask_token, std=0.02)
        torch.nn.init.normal_(self.action_mask_token, std=0.02)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, s_emb, a_emb, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence

        Applies the same masking pattern for states and actions
        to maintain temporal consistency
        """
        N, L, D = s_emb.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        # Use the same noise for both modalities (if time t is masked, s_t AND a_t are too)
        noise = torch.rand(N, L, device=s_emb.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        # Same pattern for s & a
        s_masked = torch.gather(s_emb, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        a_masked = torch.gather(a_emb, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=s_emb.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return s_masked, a_masked, mask, ids_restore

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
        
        # Apply masking SHOULD NOT be separately
        s_masked, a_masked, mask, ids_restore = self.random_masking(
            s_emb, a_emb, mask_ratio
        )
        
        # Process with separate encoders
        s_encoded = s_masked
        for blk in self.state_encoder_blocks:
            s_encoded = blk(s_encoded, self.attn_mask)
        s_encoded = self.state_encoder_norm(s_encoded)
        
        a_encoded = a_masked
        for blk in self.action_encoder_blocks:
            a_encoded = blk(a_encoded, self.attn_mask)
        a_encoded = self.action_encoder_norm(a_encoded)
        
        return s_encoded, a_encoded, mask, ids_restore

    def forward_decoder(self, s_encoded, a_encoded, ids_restore):
        # Append mask tokens (same number of state and actions)
        s_mask_tokens = self.state_mask_token.repeat(
            s_encoded.shape[0], ids_restore.shape[1] - s_encoded.shape[1], 1
        )
        a_mask_tokens = self.action_mask_token.repeat(
            a_encoded.shape[0], ids_restore.shape[1] - a_encoded.shape[1], 1
        )
        
        # Unshuffle states and actions
        s_full = torch.cat([s_encoded, s_mask_tokens], dim=1)
        s_unshuffled = torch.gather(
            s_full, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, s_full.shape[2])
        )
        
        a_full = torch.cat([a_encoded, a_mask_tokens], dim=1)
        a_unshuffled = torch.gather(
            a_full, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, a_full.shape[2])
        )
        
        # Project to decoder
        s = self.decoder_state_embed(s_unshuffled)
        a = self.decoder_action_embed(a_unshuffled)
        
        # Interleave actions and states for the decoder
        x = torch.stack([s, a], dim=2).reshape(s.shape[0], s.shape[1] * 2, s.shape[2])
        
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

        loss_s = (pred_s - target_s) ** 2
        loss_a = (pred_a - target_a) ** 2
        
        # Separate mask losses
        masked_loss_s = (loss_s.mean(dim=-1) * mask).sum() / mask.sum()
        masked_loss_a = (loss_a.mean(dim=-1) * mask).sum() / mask.sum()
        
        masked_loss = masked_loss_s + masked_loss_a
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
    ):
        self.action_dim = action_shape[0]
        self.lr = lr
        self.device = device
        self.use_tb = use_tb
        self.config = transformer_cfg

        # models
        self.model = MaskedDPMultimodal(obs_shape[0], action_shape[0], transformer_cfg).to(device)
        self.mask_ratio = mask_ratio
        # optimizers
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        print(
            "number of parameters: %e", sum(p.numel() for p in self.model.parameters())
        )

        self.train()

    def train(self, training=True):
        self.training = training
        self.model.train(training)

    def update_mdp(self, states, actions):
        metrics = dict()
        mask_ratio = np.random.choice(self.mask_ratio)
        
        # Separate encoders
        s_enc, a_enc, mask, ids_restore = \
            self.model.forward_encoder(states, actions, mask_ratio)
        
        # Decoder
        pred_s, pred_a = self.model.forward_decoder(
            s_enc, a_enc, ids_restore
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
        s_enc, a_enc, mask, ids_restore = \
            self.model.forward_encoder(obs, action, mask_ratio)
        
        pred_s, pred_a = self.model.forward_decoder(
            s_enc, a_enc, ids_restore
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
        metrics.update(self.update_mdp(obs, action))

        return metrics
