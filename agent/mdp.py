import math
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
from agent.modules.pixel_encoder import PixelEncoder
from agent.modules.attention import Block, CausalSelfAttention, CoAttentionBlock, ParallelCoAttentionBlock, AdapterMLP


class MaskedDPMultimodal(nn.Module):

    def _embed_states(self, states: torch.Tensor) -> torch.Tensor:
        if self.pixel_encoder is not None:
            return self.pixel_encoder(states)
        return self.state_embed(states)

    def __init__(self, obs_dim, action_dim, config, train_mode='joint'):
        super().__init__()
        # Pretrain configuration
        self.train_mode = train_mode  # 'joint', 'state_only', 'action_only'
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Padding sentinel (used for ragged state/action sequences within a batch)
        # Prefer a value that will never appear in real embedded tokens.
        self.pad_value = float(getattr(config, "pad_value", 1e9))
        # MAE encoder specifics
        self.n_embd = config.n_embd
        self.max_len = config.traj_length * 2
        # Temporal jitter configuration
        raw_traj_lengths = getattr(config, "traj_lengths", None)
        self.jitter_strategy = str(getattr(config, "jitter_strategy", "mix50"))

        if raw_traj_lengths is not None:
            self.traj_lengths = [int(t) for t in raw_traj_lengths]
            assert len(self.traj_lengths) >= 1, "traj_lengths must have at least one element"
            assert all(t >= 1 for t in self.traj_lengths), "all traj_lengths must be >= 1"
            assert all(t <= config.traj_length for t in self.traj_lengths), \
                f"all traj_lengths must be <= traj_length ({config.traj_length}). Got: {self.traj_lengths}"
            assert self.jitter_strategy in ('mix50', 'uniform', 'uniform_log'), \
                f"jitter_strategy must be 'mix50', 'uniform', or 'uniform_log'. Got: '{self.jitter_strategy}'"
            print(f"Temporal Jitter ENABLED: strategy='{self.jitter_strategy}', T candidates={self.traj_lengths}")
        else:
            self.traj_lengths = None
            print(f"Temporal Jitter DISABLED: fixed T={config.traj_length}")
        # self.mask_ratio = config.mask_ratio
        self.pe = config.pe
        self.norm = config.norm
        print("norm", self.norm)
        self.use_pixel_obs = getattr(config, "use_pixel_obs", False)
        if self.use_pixel_obs:
            pixel_obs_shape = tuple(config.pixel_obs_shape)  # (64, 64, 3)
            self.pixel_encoder = PixelEncoder(pixel_obs_shape, self.n_embd)
            self.state_embed = nn.Identity()
            # Freeze CNN — FIXED representations for CNN placeholder
            trainable = sum(p.numel() for p in self.pixel_encoder.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.pixel_encoder.parameters())
            print(f"[PixelEncoder] Trainable params: {trainable}/{total} (should be ~131k/11M)")
        else:
            self.pixel_encoder = None
            self.state_embed = nn.Linear(obs_dim, self.n_embd)
        self.action_embed = nn.Linear(action_dim, self.n_embd)
        
        # Modality droupout
        self.modality_dropout = bool(getattr(config, "modality_dropout", True))
        self.modality_dropout_prob = float(getattr(config, "modality_dropout_prob", 0.25))
        self.p_drop_action = float(getattr(config, "action_dropout_prob", 1.0))
        self.p_drop_state = float(getattr(config, "state_dropout_prob", 0.0))
        self.min_keep_states = int(getattr(config, "min_keep_states",  1))
        self.min_keep_actions = int(getattr(config, "min_keep_actions", 0))
        if self.modality_dropout:
            print(f"Modality Dropout ENABLED (Global Prob={self.modality_dropout_prob})")
            print(f"Rel. Weights -> Action: {self.p_drop_action}, State: {self.p_drop_state}")
            print(f"Min. Tokens -> Actions: {self.min_keep_actions}, States: {self.min_keep_states}")

        # Feature Flag for Ablation
        self.use_early_fusion = bool(getattr(config, "use_early_fusion", False))

        if self.use_early_fusion:
            # EARLY FUSION: Parallel Blocks
            self.early_fusion_blocks = nn.ModuleList(
                [ParallelCoAttentionBlock(config) for _ in range(config.n_enc_layer)]
            )
        else:
            # LATE FUSION (Baseline): Separate encoders for state and action
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
        # Optional MLP adapter — sits between encoder norms and fusion neck.
        # When use_adapter_mlp=False both adapters are nn.Identity() (zero overhead).
        self.use_adapter_mlp = bool(getattr(config, "use_adapter_mlp", False))
        if self.use_adapter_mlp:
            _ratio     = int(getattr(config, "adapter_mlp_ratio",    2))
            _layers    = int(getattr(config, "adapter_mlp_layers",   2))
            _norm      = bool(getattr(config, "adapter_mlp_norm",    True))
            _residual  = bool(getattr(config, "adapter_mlp_residual", True))
            print(f"AdapterMLP ENABLED — ratio={_ratio}, layers={_layers}, norm={_norm}, residual={_residual}")
            self.state_adapter  = AdapterMLP(self.n_embd, _ratio, _layers, _norm, _residual)
            self.action_adapter = AdapterMLP(self.n_embd, _ratio, _layers, _norm, _residual)
        else:
            self.state_adapter  = nn.Identity()
            self.action_adapter = nn.Identity()

        # --------------------------------------------------------------------------
        # Fusion encoder (cross-modal interaction after separate encoders, before decoder)
        self.n_fuse_layer = int(getattr(config, "n_fuse_layer", 0))

        # Toggles modality/type embeddings (state vs action)
        self.use_fusion_type_embed = bool(getattr(config, "use_fusion_type_embed", False))

        # Categoric embedding with state and actions classes (Like A/B on BERT)
        self.fusion_type_embed = nn.Embedding(2, self.n_embd) if self.use_fusion_type_embed else None

        #Fusion Block
        self.fusion_type = getattr(config, "fusion_type", "self")
        # 'cross' for co-attention, 'self' for self-attention
        if self.fusion_type == 'cross':
            self.fusion_blocks = nn.ModuleList(
                [CoAttentionBlock(config) for _ in range(self.n_fuse_layer)]
            )
        else:
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
        self.state_pred_dim = self.n_embd if self.use_pixel_obs else obs_dim
        self.state_head = nn.Sequential(
            nn.LayerNorm(self.n_embd),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_embd, self.state_pred_dim),
        )
        # --------------------------------------------------------------------------
        self.initialize_weights()
        
        if self.train_mode != 'joint':
            # Unimodal case, force fusion_type='none' and n_fuse_layer=0
            assert self.fusion_type == 'none' or self.n_fuse_layer == 0, \
                f"Error: Modo '{self.train_mode}' requiere fusion_type='none'. Recibido: {self.fusion_type}"

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
        # The initialization procedures depends on the fusion type
        for blk in self.fusion_blocks:
            if isinstance(blk, CoAttentionBlock):
                # Init Stream S
                nn.init.zeros_(blk.cross_attn_s.proj.weight)
                nn.init.zeros_(blk.cross_attn_s.proj.bias)
                nn.init.zeros_(blk.mlp_s[2].weight)
                nn.init.zeros_(blk.mlp_s[2].bias)

                # Init Stream A
                nn.init.zeros_(blk.cross_attn_a.proj.weight)
                nn.init.zeros_(blk.cross_attn_a.proj.bias)
                nn.init.zeros_(blk.mlp_a[2].weight)
                nn.init.zeros_(blk.mlp_a[2].bias)
            else:
                # Zero-init Attention Output Projection
                nn.init.zeros_(blk.attn.proj.weight)
                nn.init.zeros_(blk.attn.proj.bias)

                # Zero-init MLP Output Projection
                nn.init.zeros_(blk.mlp[2].weight)
                nn.init.zeros_(blk.mlp[2].bias)

        # Estable Early Fusion with zero init
        # Same strategy as the fusion block
        if getattr(self, "use_early_fusion", False):
            for blk in self.early_fusion_blocks:
                # Init Stream S
                nn.init.zeros_(blk.cross_attn_s.proj.weight)
                nn.init.zeros_(blk.cross_attn_s.proj.bias)
                nn.init.zeros_(blk.mlp_s[2].weight)
                nn.init.zeros_(blk.mlp_s[2].bias)

                # Init Stream A
                nn.init.zeros_(blk.cross_attn_a.proj.weight)
                nn.init.zeros_(blk.cross_attn_a.proj.bias)
                nn.init.zeros_(blk.mlp_a[2].weight)
                nn.init.zeros_(blk.mlp_a[2].bias)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio, noise=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence

        Applies the same masking pattern for states and actions
        to maintain temporal consistency (concatenated sequence)
        
        If noise is provided, use it to bias the sorting order.
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        # noise independent between modalities
        if noise is None:
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

    def forward_fusion(
        self, 
        s_encoded: torch.Tensor, 
        a_encoded: torch.Tensor, 
        ids_keep: torch.Tensor,
        s_pad_mask=None, 
        a_pad_mask=None,
    ) -> torch.Tensor:
        """Optional fusion encoder over kept tokens (after separate state/action encoders).

        If `n_fuse_layer == 0` and `use_fusion_type_embed == False`, this is effectively an identity mapping
        (it just reconstructs the kept sequence in ids_keep order).
        
        Multimodal fusion is done via either self-attention or co-attention blocks.
        """
 
        # --- CROSS ATTENTION ---       
        if self.fusion_type == 'cross':
            x_s = s_encoded
            x_a = a_encoded

            if self.fusion_type_embed is not None:
                B, L_s, _ = x_s.shape
                B, L_a, _ = x_a.shape

                # 0 states, 1 actions
                type_ids_s = torch.zeros(B, L_s, dtype=torch.long, device=x_s.device)
                type_ids_a = torch.ones(B, L_a, dtype=torch.long, device=x_a.device)
                
                x_s = x_s + self.fusion_type_embed(type_ids_s)
                x_a = x_a + self.fusion_type_embed(type_ids_a)

            for blk in self.fusion_blocks:
                x_s, x_a = blk(x_s, x_a, mask_s=s_pad_mask, mask_a=a_pad_mask)

            x = self._combine_kept_tokens(x_s, x_a, ids_keep)
            x = self.fusion_norm(x)

            return x

        # --- SELF ATTENTION ---
        else:
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

    def _sample_jitter_T(self) -> int:
        """Sample an effective trajectory length from the discrete traj_lengths list.

        Strategies:
            mix50       : 50% returns traj_lengths[-1] (full T), 50% uniform from list.
            uniform     : uniform sampling from list — all lengths equally likely.
            uniform_log : weighted by w_i = 1/T_i, normalized — strongly favors short T.

        The last element of traj_lengths is treated as the full-T reference (used by mix50).
        Always call this only when traj_lengths is not None and model is in training mode.
        """
        if self.jitter_strategy == 'mix50':
            if np.random.rand() < 0.5:
                return self.traj_lengths[-1]
            return int(np.random.choice(self.traj_lengths))

        if self.jitter_strategy == 'uniform':
            return int(np.random.choice(self.traj_lengths))

        if self.jitter_strategy == 'uniform_log':
            weights = 1.0 / np.array(self.traj_lengths, dtype=float)
            weights /= weights.sum()
            return int(np.random.choice(self.traj_lengths, p=weights))

        return self.traj_lengths[-1]  # fallback

    def forward_encoder(self, states, actions, mask_ratio):
        """MAE-style encoder with separate state and action streams."""

        # Input sanitization according to training mode
        if self.train_mode == 'state_only':
            actions = torch.full_like(actions, self.pad_value)
        elif self.train_mode == 'action_only':
            states = torch.full_like(states, self.pad_value)

        batch_size, T = states.shape[0], states.shape[1]
        
        # Embeddings
        s_emb = self._embed_states(states)
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
        noise = torch.rand(batch_size, 2 * T, device=states.device)
        
        # Modality Dropout (only during training)
        if self.training and self.modality_dropout and (np.random.rand() < self.modality_dropout_prob):
            total_prob = self.p_drop_action + self.p_drop_state
            p_action = self.p_drop_action / total_prob if total_prob > 0 else 0.5
            # Biased masking: masking high noisy modality tokens
            drop_actions = np.random.rand() < p_action
            start_idx = 1 if drop_actions else 0
            noise[:, start_idx::2] += 100.0
            
        # Modality shields (min keep tokens)
        if self.min_keep_states > 0:
            k_s = min(self.min_keep_states, T)
            _, top_s_idx = torch.topk(torch.rand(batch_size, T, device=states.device), k_s, dim=1)
            top_s_idx_interleaved = top_s_idx * 2
            noise.scatter_(1, top_s_idx_interleaved, -100.0)
            
        if self.min_keep_actions > 0:
            k_a = min(self.min_keep_actions, T)
            _, top_a_idx = torch.topk(torch.rand(batch_size, T, device=states.device), k_a, dim=1)
            top_a_idx_interleaved = top_a_idx * 2 + 1
            noise.scatter_(1, top_a_idx_interleaved, -100.0)
        
        # Apply masking on the full interleaved sequence
        x_masked, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio, noise=noise)

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
        
        # Process with separate encoders (Early vs Late Fusion)
        s_encoded = s_masked
        a_encoded = a_masked

        if self.use_early_fusion:
            #EARLY FUSION
            for blk in self.early_fusion_blocks:
                s_encoded, a_encoded = blk(
                    x_s=s_encoded, x_a=a_encoded,
                    mask_s=s_attn_mask, mask_a=a_attn_mask,
                    pad_mask_s=s_pad_mask_1d, pad_mask_a=a_pad_mask_1d
                )
        else:
            for blk in self.state_encoder_blocks:
                s_encoded = blk(s_encoded, s_attn_mask)
            for blk in self.action_encoder_blocks:
                a_encoded = blk(a_encoded, a_attn_mask)

        s_encoded = self.state_encoder_norm(s_encoded)
        a_encoded = self.action_encoder_norm(a_encoded)

        # Optional adapter — identity when use_adapter_mlp=False
        s_encoded = self.state_adapter(s_encoded)
        a_encoded = self.action_adapter(a_encoded)
        
        # Fuse and return kept tokens for the decoder
        x_fused = self.forward_fusion(
            s_encoded, 
            a_encoded, 
            ids_keep, 
            s_pad_mask=s_pad_mask_1d, 
            a_pad_mask=a_pad_mask_1d
        )
        
        # Return also ids_keep to track state/action positions
        return x_fused, mask, ids_restore, ids_keep

    def forward_decoder(self, x_fused: torch.Tensor, ids_restore: torch.Tensor):
        """MAE-style decoder with support for unimodal training.
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
        
        if self.train_mode == 'state_only':
            s = self.decoder_state_embed(x[:, ::2])
            x_dec = torch.zeros(batch_size, total_len, self.n_embd, device=x.device)
            x_dec[:, ::2] = s
            x_dec = x_dec + self.decoder_pos_embed[:, :total_len, :]

            for blk in self.decoder_blocks:
                x_dec = blk(x_dec, self.attn_mask)

            # dummy for actions
            s_pred = self.state_head(x_dec[:, ::2])
            a_pred = torch.zeros(batch_size, total_len // 2, self.action_dim, device=x.device)
        
        elif self.train_mode == 'action_only':
            a = self.decoder_action_embed(x[:, 1::2])
            x_dec = torch.zeros(batch_size, total_len, self.n_embd, device=x.device)
            x_dec[:, 1::2] = a
            x_dec = x_dec + self.decoder_pos_embed[:, :total_len, :]

            for blk in self.decoder_blocks:
                x_dec = blk(x_dec, self.attn_mask)

            # dummy for states
            a_pred = self.action_head(x_dec[:, 1::2])
            s_pred = torch.zeros(batch_size, total_len // 2, self.state_pred_dim, device=x.device)
        
        else:  # 'joint'
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

        # MSE per dimension
        loss_s = (pred_s - target_s) ** 2
        loss_a = (pred_a - target_a) ** 2
        
        if self.train_mode == 'state_only':
            loss_a = torch.zeros_like(loss_a)
        elif self.train_mode == 'action_only':
            loss_s = torch.zeros_like(loss_s)

        # Mean MSE per token 
        loss_s_t = loss_s.mean(dim=-1)
        loss_a_t = loss_a.mean(dim=-1)

        # Intercalate [s0,a0,s1,a1,...] -> shape [B, 2T]
        loss_tokens = torch.stack([loss_s_t, loss_a_t], dim=-1)  # [B, T, 2]
        loss_tokens = loss_tokens.reshape(batch_size, 2 * T)     # [B, 2T]

        # Only ONE masked_loss for both modalities (same idea than the original)
        masked_loss = (loss_tokens * mask).sum() / mask.sum()

        # Per-modality average losses
        state_loss = loss_s.mean()
        action_loss = loss_a.mean()

        if self.train_mode == 'state_only':
            state_loss = loss_s.mean()
            action_loss = torch.tensor(0.0, device=target_a.device)
        elif self.train_mode == 'action_only':
            state_loss = torch.tensor(0.0, device=target_s.device)
            action_loss = loss_a.mean()
        else:
            state_loss = loss_s.mean()
            action_loss = loss_a.mean()
        
        return masked_loss, state_loss, action_loss


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
        train_mode='joint',
        warmup_steps=0,
        finetune_lr=None,
    ):
        self.action_dim = action_shape[0]
        self.lr = lr
        self.finetune_lr = finetune_lr if finetune_lr is not None else lr
        self.device = device
        self.use_tb = use_tb
        self.config = transformer_cfg
        
        # Schedule {'module_nane': [start_step, end_step]}
        self.freeze_schedule = freeze_schedule if freeze_schedule is not None else {}

        # Warmup logic
        if warmup_steps > 0:
            self.warmup_steps = math.ceil(warmup_steps / 1000.0) * 1000
            print(f"Warmup ENABLED: Requested {warmup_steps} -> Adjusted to {self.warmup_steps} steps")
        else:
            self.warmup_steps = 0
        self.warmup_start_step = None

        # models
        self.model = MaskedDPMultimodal(
            obs_shape[0], 
            action_shape[0], 
            transformer_cfg,
            train_mode=train_mode
        ).to(device)
        self.mask_ratio = mask_ratio
        # optimizers
        # Tag the initial group with target_lr so that warmup logic works correctly
        # even when no freeze_schedule is active (full-model training with warmup).
        # When a freeze_schedule IS active, _rebuild_optimizer / _add_encoder_param_groups
        # will replace this at step 0 before any gradient is taken.
        self.opt = torch.optim.Adam(
            [{'params': list(self.model.parameters()), 'lr': lr, 'target_lr': lr, 'name': 'all_params'}]
        )
        print(
            "number of parameters: %e", sum(p.numel() for p in self.model.parameters())
        )

        self.train()
        
    def set_module_requires_grad(self, module, requires_grad):
        for param in module.parameters():
            param.requires_grad = requires_grad

    def _get_encoder_param_ids(self):
        # Used to distinguish encoder params from fusion/decoder params when building groups.
        ids = set()
        for module_name in self.freeze_schedule:
            module = getattr(self.model, module_name, None)
            if module is not None:
                for p in module.parameters():
                    ids.add(p.data_ptr())
        return ids

    def _rebuild_optimizer(self):
        """Rebuild optimizer with only trainable parameters"""
        encoder_ids = self._get_encoder_param_ids()
        encoder_params, other_params = [], []

        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            if p.data_ptr() in encoder_ids:
                encoder_params.append(p)
            else:
                other_params.append(p)

        param_groups = []
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': self.lr,
                'target_lr': self.lr,
                'name': 'fusion_decoder',
            })
        if encoder_params:
            param_groups.append({
                'params': encoder_params,
                'lr': self.finetune_lr,
                'target_lr': self.finetune_lr,
                'name': 'encoders',
            })

        # Fallback: nothing in either bucket (shouldn't happen, but be safe)
        if not param_groups:
            param_groups = [{'params': [], 'lr': self.lr, 'target_lr': self.lr, 'name': 'empty'}]

        print(
            f"[Optimizer] Full rebuild — "
            f"{len(other_params)} fusion/decoder params @ lr={self.lr}, "
            f"{len(encoder_params)} encoder params @ lr={self.finetune_lr}"
        )
        self.opt = torch.optim.Adam(param_groups)

    def _add_encoder_param_groups(self, newly_unfrozen_names):
        # Collect all param data_ptrs already tracked by the optimizer
        existing_ids = set()
        for grp in self.opt.param_groups:
            for p in grp['params']:
                existing_ids.add(p.data_ptr())

        new_params = []
        for name in newly_unfrozen_names:
            module = getattr(self.model, name, None)
            if module is None:
                continue
            for p in module.parameters():
                if p.requires_grad and p.data_ptr() not in existing_ids:
                    new_params.append(p)

        if not new_params:
            print(
                f"[Optimizer] _add_encoder_param_groups: no new params to add "
                f"(already tracked or empty). Skipping."
            )
            return

        self.opt.add_param_group({
            'params': new_params,
            'lr': self.finetune_lr,
            'target_lr': self.finetune_lr,
            'name': 'encoders',
        })
        print(
            f"[Optimizer] add_param_group — {len(new_params)} encoder params "
            f"@ lr={self.finetune_lr}. Fusion/decoder Adam state PRESERVED."
        )

    # ------------------------------------------------------------------

    def check_freeze_schedule(self, step):
        """Applies freezing to a block if step is inside an interval [start, end)"""
        if step is None:
            return

        newly_unfrozen = []   # modules transitioning frozen → trainable
        needs_full_rebuild = False  # any module transitioning trainable → frozen

        for module_name, (start_step, end_step) in self.freeze_schedule.items():
            module = getattr(self.model, module_name, None)
            if module is None:
                print(f"[Warning] freeze_schedule: module '{module_name}' not found in model.")
                continue

            params = list(module.parameters())
            if not params:
                continue

            should_train = not (start_step <= step < end_step)
            current_status = params[0].requires_grad

            if should_train != current_status:
                status_str = "TRAINABLE" if should_train else "FROZEN"
                print(
                    f"[{step}] Module '{module_name}' -> {status_str} "
                    f"(schedule: [{start_step}, {end_step}))"
                )
                self.set_module_requires_grad(module, should_train)

                if should_train:
                    newly_unfrozen.append(module_name)
                else:
                    # Freezing requires full rebuild to drop these params from Adam
                    needs_full_rebuild = True

        if not newly_unfrozen and not needs_full_rebuild:
            return  # No state change — nothing to do

        if needs_full_rebuild:
            # Freeze event (or mixed freeze+unfreeze): full rebuild, momentum lost.
            # Mixed events are extremely unlikely in normal use (FZ+FT pattern only unfreezes).
            self._rebuild_optimizer()
        else:
            # Pure unfreeze event: momentum-preserving add_param_group path.
            self._add_encoder_param_groups(newly_unfrozen)

    def train(self, training=True):
        self.training = training
        self.model.train(training)

    def update_mdp(self, states, actions, step=None):
        self.check_freeze_schedule(step)
        
        # Warmup Logic
        # Each param_group carries a 'target_lr' key. Warmup scales each group
        # relative to its own target, so fusion/decoder (lr=1e-4) and encoders
        # (lr=5e-5) both warm up proportionally — not to a single shared value.
        # Warmup only fires once (anchored to warmup_start_step), so encoder
        # params added via add_param_group after the warmup window has elapsed
        # will correctly start at their full finetune_lr with no warmup applied.
        if step is not None and self.warmup_steps > 0:
            if self.warmup_start_step is None:
                self.warmup_start_step = step
            
            steps_since_start = step - self.warmup_start_step
            
            if steps_since_start < self.warmup_steps:
                # Linear warmup: scale each group by the same factor, but relative
                # to that group's own target_lr (not a single global self.lr).
                warmup_factor = (steps_since_start + 1) / float(self.warmup_steps)
                for param_group in self.opt.param_groups:
                    target = param_group.get('target_lr', self.lr)
                    param_group['lr'] = target * warmup_factor
            
            elif steps_since_start == self.warmup_steps:
                # Snap each group to its exact target_lr at the end of warmup
                for param_group in self.opt.param_groups:
                    param_group['lr'] = param_group.get('target_lr', self.lr)

        metrics = dict()
        mask_ratio = np.random.choice(self.mask_ratio)

        # Temporal jitter: sample T_eff from the discrete traj_lengths list and truncate
        # the batch before the forward pass. T is constant within a batch to avoid
        # variable-length collation issues. T is already fully dynamic throughout
        # forward_encoder and forward_decoder (all shapes derived from states.size(1)),
        # so no other changes are needed.
        if self.model.traj_lengths is not None and self.training:
            T_eff = self.model._sample_jitter_T()
            states  = states[:, :T_eff]
            actions = actions[:, :T_eff]

        # Encoder (dual + optional fusion)
        x_fused, mask, ids_restore, ids_keep = \
            self.model.forward_encoder(states, actions, mask_ratio)
        
        # Decoder
        pred_s, pred_a = self.model.forward_decoder(
            x_fused, ids_restore
        )
        
        # Loss
        with torch.no_grad():
            target_s = self.model._embed_states(states)  # (B, T, n_embd)

        # DEBUG — remover después de verificar
        if step % 5000 == 0:
            norms = torch.norm(target_s, dim=-1)  # debería ser ~1.0 por L2 norm
            sim = torch.nn.functional.cosine_similarity(
                target_s[:, 0].unsqueeze(1),   # primer frame
                target_s[:, 1:],               # resto de frames
                dim=-1
            ).mean()
            print(f"[DEBUG step={step}] embedding norm: {norms.mean():.4f} ± {norms.std():.4f} | inter-frame cosine sim: {sim:.4f}")
        
        mask_loss, state_loss, action_loss = self.model.forward_loss(
            target_s, actions, pred_s, pred_a, mask
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
