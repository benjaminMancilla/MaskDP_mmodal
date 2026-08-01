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
from agent.modules.attention import Block, CausalSelfAttention, CoAttentionBlock
from agent.modules.pixel_encoder import PixelEncoder
from agent.modules.load_pretrained_encoder import load_drqbc_convnet, load_procgen_impala
from agent.modules.pixel_recon_decoder import PixelReconDecoder


class MaskedDPMultimodal(nn.Module):

    def _embed_states(self, states: torch.Tensor) -> torch.Tensor:
        if self.pixel_encoder is not None:
            return self.pixel_encoder(states)
        return self.state_embed(states)

    def _make_state_target(self, states: torch.Tensor) -> torch.Tensor:
        if self.state_target == "pixel":
            return states.float() / 255.0
        return self._embed_states(states)

    def _predict_state(self, dec_tokens: torch.Tensor) -> torch.Tensor:
        if self.state_target == "pixel":
            # [B, T, D] -> [B, T, 1, D] -> PixelReconDecoder -> [B, T, H, W, C]
            return self.pixel_recon_decoder(dec_tokens.unsqueeze(2))
        return self.state_head(dec_tokens)

    def _zero_state_pred(self, batch_size: int, T: int, device) -> torch.Tensor:
        if self.state_target == "pixel":
            d = self.pixel_recon_decoder
            return torch.zeros(batch_size, T, d.H, d.W, d.C, device=device)
        return torch.zeros(batch_size, T, self.state_pred_dim, device=device)

    def __init__(self, obs_dim, action_dim, config):
        super().__init__()
        # Pretrain configuration
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Discrete action support (classification instead of regression).
        self.discrete_actions = bool(getattr(config, "discrete_actions", False))
        self.num_actions = int(getattr(config, "num_actions", 0))
        # Lambda to rebalance MSE(state) vs CE(action)
        self.action_loss_weight = float(getattr(config, "action_loss_weight", 1.0))
        # Label smoothing for the action CE loss ONLY
        self.label_smoothing = float(getattr(config, "label_smoothing", 0.0))
        # dropout on the CNN->transformer state features,
        self.input_dropout = nn.Dropout(float(getattr(config, "input_dropout", 0.0)))
        if self.discrete_actions:
            assert self.num_actions > 0, "num_actions must be > 0 with discrete_actions=True"
            print(f"Discrete actions ENABLED: num_actions={self.num_actions}, action_loss_weight={self.action_loss_weight}")
        else:
            print(f"Discrete actions DISABLED (continuous action_dim={action_dim})")

        # Padding sentinel (used for ragged state/action sequences within a batch)
        # Prefer a value that will never appear in real embedded tokens.
        self.pad_value = float(getattr(config, "pad_value", 1e9))
        # MAE encoder specifics
        # Hidden for decoder and fusion neck
        self.n_embd = config.n_embd
        # Hidden dim for unimodal encoders
        self.enc_n_embd = int(getattr(config, 'enc_n_embd', config.n_embd))
        # Projection layer for encoder -> neck n_emb difference
        self._has_enc_proj = (self.enc_n_embd != self.n_embd)
        if self._has_enc_proj:
            print(f"R2 mode: enc_n_embd={self.enc_n_embd}, n_embd(fusion/dec)={self.n_embd}")
            assert self.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
            assert self.enc_n_embd % config.n_head == 0, "enc_n_embd must be divisible by n_head"

        # Hidden dim for decoders
        self.dec_n_embd = int(getattr(config, 'dec_n_embd', config.n_embd))
        # Projection layer for decoder -> neck n_emb difference
        self._has_dec_proj = (self.dec_n_embd != self.n_embd)
        if self._has_dec_proj:
            print(f"Dec reduction: dec_n_embd={self.dec_n_embd}, n_embd(fusion)={self.n_embd}")
            assert self.dec_n_embd % config.n_head == 0, \
                f"dec_n_embd ({self.dec_n_embd}) must be divisible by n_head ({config.n_head})"

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
            pixel_obs_shape = tuple(config.pixel_obs_shape)
            self.pixel_obs_shape = pixel_obs_shape
            pixel_encoder_type = str(getattr(config, "pixel_encoder_type", "drqv2"))

            encoder_trainable = bool(getattr(config, "encoder_trainable", False))
            encoder_init = str(getattr(config, "encoder_init", "pretrained"))
            if encoder_init not in ("pretrained", "random"):
                raise ValueError(
                    f"encoder_init must be 'pretrained' or 'random', got '{encoder_init}'"
                )
            print(f"[PixelEncoder] encoder_trainable={encoder_trainable}, encoder_init='{encoder_init}'")

            self.pixel_encoder = PixelEncoder(pixel_obs_shape, self.enc_n_embd, encoder_type=pixel_encoder_type)

            pretrained_path = getattr(config, "pretrained_encoder_path", None)
            if encoder_init == "pretrained" and pretrained_path is not None:
                # Dispatch by encoder_type: each pretrained checkpoint format
                # (DrQ-v2 convnet-only vs. Procgen IMPALA convnet+projection).
                # freeze=not encoder_trainable: frozen baseline OR trainable fine-tune.
                if pixel_encoder_type == "drqv2":
                    load_drqbc_convnet(self.pixel_encoder, pretrained_path, freeze=not encoder_trainable)
                elif pixel_encoder_type == "procgen_impala":
                    load_procgen_impala(self.pixel_encoder, pretrained_path, freeze=not encoder_trainable)
                else:
                    raise ValueError(
                        f"No pretrained-weights loader registered for "
                        f"pixel_encoder_type='{pixel_encoder_type}'. "
                        f"Either add one in load_pretrained_encoder.py and dispatch "
                        f"it here, or omit 'pretrained_encoder_path' to train "
                        f"'{pixel_encoder_type}' from scratch."
                    )
            elif encoder_init == "random":
                # Skip pretrained loading; apply freeze if encoder_trainable=False.
                if not encoder_trainable:
                    for p in self.pixel_encoder.parameters():
                        p.requires_grad = False
                print(
                    f"  [PixelEncoder] encoder_init=random → random init "
                    f"({'frozen' if not encoder_trainable else 'trainable'})."
                )
            else:
                raise ValueError(
                    f"pretrained_path is missing with 'pretrained' encoder_init"
                )
                

            self.state_embed = nn.Identity()
            trainable = sum(p.numel() for p in self.pixel_encoder.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.pixel_encoder.parameters())
            print(f"[PixelEncoder] Trainable params: {trainable}/{total}")
        else:
            self.pixel_encoder = None
            self.state_embed = nn.Linear(obs_dim, self.enc_n_embd)

        if self.discrete_actions:
            self.action_embed = nn.Embedding(self.num_actions, self.enc_n_embd)
        else:
            self.action_embed = nn.Linear(action_dim, self.enc_n_embd)
        
        # Modality droupout
        self.modality_dropout = bool(getattr(config, "modality_dropout", True))
        self.modality_dropout_prob = float(getattr(config, "modality_dropout_prob", 0.25))
        self.p_drop_action = float(getattr(config, "action_dropout_prob", 1.0))
        self.p_drop_state = float(getattr(config, "state_dropout_prob", 0.0))
        self.min_keep_states = int(getattr(config, "min_keep_states",  1))
        self.min_keep_actions = int(getattr(config, "min_keep_actions", 0))
        # Floor on real (non-padding) context tokens
        self.min_keep_context = int(getattr(config, "min_keep_context", 1))
        self.use_decoder_valid_mask = bool(getattr(config, "use_decoder_valid_mask", False))
        if self.modality_dropout:
            print(f"Modality Dropout ENABLED (Global Prob={self.modality_dropout_prob})")
            print(f"Rel. Weights -> Action: {self.p_drop_action}, State: {self.p_drop_state}")
            print(f"Min. Tokens -> Actions: {self.min_keep_actions}, States: {self.min_keep_states}")

        # Separate encoders for state and action (Late Fusion Baseline)
        if self._has_enc_proj:
            from omegaconf import OmegaConf
            _enc_cfg_dict = OmegaConf.to_container(config, resolve=True)
            _enc_cfg_dict['n_embd'] = self.enc_n_embd
            enc_config = OmegaConf.create(_enc_cfg_dict)
        else:
            enc_config = config

        self.state_encoder_blocks = nn.ModuleList(
            [Block(enc_config) for _ in range(config.n_enc_layer)]
        )
        self.action_encoder_blocks = nn.ModuleList(
            [Block(enc_config) for _ in range(config.n_enc_layer)]
        )
        
        # Normalization for encoders
        self.state_encoder_norm = nn.LayerNorm(self.enc_n_embd)
        self.action_encoder_norm = nn.LayerNorm(self.enc_n_embd)

        # --------------------------------------------------------------------------
        # Projection enc_n_embd -> n_embd (indetity if no reduction)
        # Sits between encoder norms and the optional adapter / fusion neck.
        if self._has_enc_proj:
            self.state_proj  = nn.Linear(self.enc_n_embd, self.n_embd)
            self.action_proj = nn.Linear(self.enc_n_embd, self.n_embd)
        else:
            self.state_proj  = nn.Identity()
            self.action_proj = nn.Identity()

        if self._has_enc_proj:
            self.enc_mask_token = nn.Parameter(torch.zeros(1, 1, self.enc_n_embd))


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
        self.decoder_state_embed  = nn.Linear(self.n_embd, self.dec_n_embd)
        self.decoder_action_embed = nn.Linear(self.n_embd, self.dec_n_embd)

        if self._has_dec_proj:
            from omegaconf import OmegaConf
            _dec_cfg_dict = OmegaConf.to_container(config, resolve=True)
            _dec_cfg_dict['n_embd'] = self.dec_n_embd
            dec_config = OmegaConf.create(_dec_cfg_dict)
        else:
            dec_config = config

        self.decoder_blocks = nn.ModuleList(
            [Block(dec_config) for _ in range(config.n_dec_layer)]
        )

        if self.discrete_actions:
            self.action_head = nn.Sequential(
                nn.LayerNorm(self.dec_n_embd),
                nn.ReLU(inplace=True),
                nn.Linear(self.dec_n_embd, self.num_actions),
            )  # decoder to logits, softmax is applied inside cross_entropy, not here
        else:
            self.action_head = nn.Sequential(
                nn.LayerNorm(self.dec_n_embd),
                nn.ReLU(inplace=True),
                nn.Linear(self.dec_n_embd, action_dim),
                nn.Tanh(),
            )  # decoder to patch
        self.state_pred_dim = self.enc_n_embd if self.use_pixel_obs else obs_dim
        self.state_head = nn.Sequential(
            nn.LayerNorm(self.dec_n_embd),
            nn.ReLU(inplace=True),
            nn.Linear(self.dec_n_embd, self.state_pred_dim),
        )

        # --------------------------------------------------------------------------
        # state_target: what the state/reconstruction loss targets.
        #   'embedding' (default): predict the (frozen) encoder embedding
        #   'pixel': predict raw pixels (MAE-style). Target is the external
        # Routes through PixelReconDecoder instead of state_head when 'pixel'.
        self.state_target = str(getattr(config, "state_target", "embedding"))
        if self.state_target not in ("embedding", "pixel"):
            raise ValueError(
                f"state_target must be 'embedding' or 'pixel', got '{self.state_target}'"
            )
        if self.state_target == "pixel":
            assert self.use_pixel_obs, (
                "state_target='pixel' requires use_pixel_obs=True — the pixel-recon "
                "target is the raw frame, which only exists when observations are pixels."
            )
            if self.use_pixel_obs and not bool(getattr(config, "encoder_trainable", False)):
                print(
                    "[state_target=pixel] Warning: encoder_trainable=False — pixel-recon "
                    "with a frozen encoder is a valid ablation, but the main point of "
                    "this target (frozen->trainable + MAE) needs encoder_trainable=True."
                )
            self.pixel_recon_decoder = PixelReconDecoder(
                d_model=self.dec_n_embd,
                tokens_per_frame=1,
                frame_hw=(self.pixel_obs_shape[0], self.pixel_obs_shape[1]),
                channels=self.pixel_obs_shape[2],
            )
            print(
                f"[state_target=pixel] PixelReconDecoder: frame_hw="
                f"{self.pixel_obs_shape[:2]}, channels={self.pixel_obs_shape[2]}, "
                f"d_model={self.dec_n_embd}"
            )
        else:
            self.pixel_recon_decoder = None
            if self.use_pixel_obs and bool(getattr(config, "encoder_trainable", False)):
                print(
                    "[state_target=embedding] Warning: encoder_trainable=True with "
                    "state_target='embedding' is the naive unfreeze that the brief "
                    "warns breaks (moving-target/collapse). Did you mean state_target='pixel'?"
                )
        # --------------------------------------------------------------------------
        self.initialize_weights()
        

    def initialize_weights(self):
        # Positional embeddings SHOULD NOT be separated, this actually breaks the
        # original trayectory order
        enc_pe = utils.get_1d_sincos_pos_embed_from_grid(self.enc_n_embd, self.max_len)
        enc_pe = torch.from_numpy(enc_pe).float().unsqueeze(0) / 2.0
        
        self.register_buffer("pos_embed", enc_pe)
        
        # For decoder we use full pos_embed
        dec_pe = utils.get_1d_sincos_pos_embed_from_grid(self.dec_n_embd, self.max_len)
        dec_pe = torch.from_numpy(dec_pe).float().unsqueeze(0) / 2.0
        self.register_buffer("decoder_pos_embed", dec_pe)
        
        self.register_buffer(
            "attn_mask", torch.ones(self.max_len, self.max_len)[None, None, ...]
        )
        
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # Init mask tokens
        torch.nn.init.normal_(self.mask_token, std=0.02)
        if hasattr(self, 'enc_mask_token'):
            torch.nn.init.normal_(self.enc_mask_token, std=0.02)
        if self.fusion_type_embed is not None:
            torch.nn.init.normal_(self.fusion_type_embed.weight, std=0.02)
        
        self.apply(self._init_weights)
        
        # Dinamic last linear layer search
        def zero_init_last_linear(module):
            for m in reversed(list(module)):
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                    break

        # Initialize FUSION with zeros to stabilize freezing training
        for blk in self.fusion_blocks:
            if isinstance(blk, CoAttentionBlock):
                # Init Stream S
                nn.init.zeros_(blk.cross_attn_s.proj.weight)
                nn.init.zeros_(blk.cross_attn_s.proj.bias)
                zero_init_last_linear(blk.mlp_s)

                # Init Stream A
                nn.init.zeros_(blk.cross_attn_a.proj.weight)
                nn.init.zeros_(blk.cross_attn_a.proj.bias)
                zero_init_last_linear(blk.mlp_a)  
                              
                
            else:
                nn.init.zeros_(blk.attn.proj.weight)
                nn.init.zeros_(blk.attn.proj.bias)
                zero_init_last_linear(blk.mlp)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio, noise=None, len_keep=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence

        Applies the same masking pattern for states and actions
        to maintain temporal consistency (concatenated sequence)
        
        If noise is provided, use it to bias the sorting order.
        If len_keep is provided, use it instead of int(L * (1 - mask_ratio))
        """
        N, L, D = x.shape  # batch, length, dim
        if len_keep is None:
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

    def _combine_kept_flags(self, s_flags: torch.Tensor, a_flags: torch.Tensor, ids_keep: torch.Tensor) -> torch.Tensor:
        combined = self._combine_kept_tokens(
            s_flags.unsqueeze(-1).float(), a_flags.unsqueeze(-1).float(), ids_keep
        )                                              # [B, len_keep, 1]
        return combined.squeeze(-1) > 0.5               # [B, len_keep] bool

    def forward_fusion(
        self,
        s_encoded: torch.Tensor,
        a_encoded: torch.Tensor,
        ids_keep: torch.Tensor,
        s_pad_mask=None,
        a_pad_mask=None,
        tar_layer: int = None,          # ATTATTR: index into self.fusion_blocks to intervene on
        tmp_att_s: torch.Tensor = None, # ATTATTR: override (alpha*A) for cross_attn_s at tar_layer
        tmp_att_a: torch.Tensor = None, # ATTATTR: override for cross_attn_a at tar_layer
        capture_att: bool = False,      # ATTATTR: True on the baseline pass (no override) to read real A
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

            att_s_out, att_a_out = None, None
            for layer_index, blk in enumerate(self.fusion_blocks):
                if tar_layer is not None and layer_index == tar_layer:
                    need_att = capture_att or (tmp_att_s is not None) or (tmp_att_a is not None)
                    out = blk(x_s, x_a, mask_s=s_pad_mask, mask_a=a_pad_mask,
                              tmp_att_s=tmp_att_s, tmp_att_a=tmp_att_a, return_att=need_att)
                    if need_att:
                        x_s, x_a, att_s_out, att_a_out = out
                    else:
                        x_s, x_a = out
                else:
                    x_s, x_a = blk(x_s, x_a, mask_s=s_pad_mask, mask_a=a_pad_mask)

            x = self._combine_kept_tokens(x_s, x_a, ids_keep)
            x = self.fusion_norm(x)

            if capture_att or tmp_att_s is not None or tmp_att_a is not None:
                return x, att_s_out, att_a_out
            return x

        # --- SELF ATTENTION ---
        else:
            x = self._combine_kept_tokens(s_encoded, a_encoded, ids_keep)  # [B, len_keep, D]

            if self.fusion_type_embed is not None:
                type_ids = (ids_keep % 2).long()  # 0=state (even), 1=action (odd)
                x = x + self.fusion_type_embed(type_ids)

            if len(self.fusion_blocks) > 0:
                B, L, _ = x.shape
                if self.use_decoder_valid_mask and s_pad_mask is not None and a_pad_mask is not None:
                    token_pad = self._combine_kept_flags(s_pad_mask, a_pad_mask, ids_keep)  # [B, L] True=pad
                    token_valid = ~token_pad
                    vv = token_valid.unsqueeze(2) & token_valid.unsqueeze(1)
                    pp = token_pad.unsqueeze(2) & token_pad.unsqueeze(1)
                    fuse_mask = (vv | pp).unsqueeze(1).to(dtype=torch.float32)
                else:
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

    def forward_encoder(self, states, actions, mask_ratio, valid_mask=None):
        """MAE-style encoder with separate state and action streams."""
        batch_size, T = states.shape[0], states.shape[1]
        
        # Embeddings
        s_emb = self._embed_states(states)
        # Optional input dropout on CNN/state features entering the transformer
        s_emb = self.input_dropout(s_emb)
        if self.discrete_actions:
            a_idx = actions.long()
            if a_idx.dim() == 3 and a_idx.size(-1) == 1:
                a_idx = a_idx.squeeze(-1)        # [B, T, 1] -> [B, T]
            a_emb = self.action_embed(a_idx)     # [B, T] -> [B, T, enc_n_embd]
        else:
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
        x = torch.stack([s_emb, a_emb], dim=2).reshape(batch_size, 2 * T, self.enc_n_embd)
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

        # Partition the interleaved sequence into 3 bands per row so that
        # argsort produces exactly [kept-real] < [pad-filler] < [removed-real]:
        #   band A (kept-real)    : real tokens that survive as context
        #   band B (pad-filler)   : padding, fills the dense width up to len_keep
        #   band C (removed-real) : real tokens to reconstruct (the loss targets)
        if valid_mask is not None:
            v = valid_mask.to(device=states.device, dtype=torch.bool)
            if v.dim() == 3 and v.size(-1) == 1:
                v = v.squeeze(-1)                          # [B, T, 1] -> [B, T]
            valid_il = v.repeat_interleave(2, dim=1)       # [B, 2T], s_t/a_t share validity

            n_real = valid_il.sum(dim=1)                                          # [B]
            keep_real = torch.floor((1.0 - mask_ratio) * n_real.float()).long()   # [B]
            keep_real = keep_real.clamp(min=self.min_keep_context)                # >=1 real context token
            keep_real = torch.minimum(keep_real, n_real)                          # never more than exist

            # Rank of each REAL token within its row by noise (ascending). Padding
            # gets +inf so it never occupies a "real" rank.
            u_real = noise.masked_fill(~valid_il, float('inf'))
            real_rank = u_real.argsort(dim=1).argsort(dim=1)                      # [B, 2T]

            is_kept_real    = valid_il & (real_rank <  keep_real.unsqueeze(1))    # band A
            is_removed_real = valid_il & (real_rank >= keep_real.unsqueeze(1))    # band C
            is_pad = ~valid_il                                                     # band B

            noise = noise + 1.0e4 * is_pad.float() + 2.0e4 * is_removed_real.float()
            len_keep = int(keep_real.max().item())
        else:
            len_keep = None

        # Apply masking on the full interleaved sequence
        x_masked, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio, noise=noise, len_keep=len_keep)

        if valid_mask is not None:
            kept_valid = torch.gather(valid_il, 1, ids_keep)   # [B, len_keep]
        else:
            kept_valid = None

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
        s_valid_kept = []
        a_valid_kept = []

        for b in range(batch_size):
            s_masked.append(x_masked[b, is_state[b]])  # [num_states_kept, D]
            a_masked.append(x_masked[b, is_action[b]])  # [num_actions_kept, D]
            if kept_valid is not None:
                s_valid_kept.append(kept_valid[b, is_state[b]])
                a_valid_kept.append(kept_valid[b, is_action[b]])
        

        # Pad to same length within batch using a sentinel value. (removing python for loops)
        s_masked = pad_sequence(s_masked, batch_first=True, padding_value=self.pad_value)  # [B, Ls, D]
        a_masked = pad_sequence(a_masked, batch_first=True, padding_value=self.pad_value)  # [B, La, D]

        # 1D padding masks (True where padding)
        s_pad_mask_1d = (s_masked == self.pad_value).all(dim=-1)
        a_pad_mask_1d = (a_masked == self.pad_value).all(dim=-1)

        if kept_valid is not None:
            s_vk = pad_sequence(s_valid_kept, batch_first=True, padding_value=False)
            a_vk = pad_sequence(a_valid_kept, batch_first=True, padding_value=False)
            s_pad_mask_1d = s_pad_mask_1d | (~s_vk)
            a_pad_mask_1d = a_pad_mask_1d | (~a_vk)

        # Valid-only attention masks
        # [B, L] -> [B, L, L] via outer-product of validity, then add head dim -> [B,1,L,L]
        s_valid = ~s_pad_mask_1d
        a_valid = ~a_pad_mask_1d
        s_attn_mask = (s_valid.unsqueeze(2) & s_valid.unsqueeze(1)).unsqueeze(1).to(dtype=torch.float32)
        a_attn_mask = (a_valid.unsqueeze(2) & a_valid.unsqueeze(1)).unsqueeze(1).to(dtype=torch.float32)
        
        # Process with separate encoders
        s_encoded = s_masked
        a_encoded = a_masked

        for blk in self.state_encoder_blocks:
            s_encoded = blk(s_encoded, s_attn_mask)
        for blk in self.action_encoder_blocks:
            a_encoded = blk(a_encoded, a_attn_mask)

        s_encoded = self.state_encoder_norm(s_encoded)   # [B, Ls, enc_n_embd]
        a_encoded = self.action_encoder_norm(a_encoded)   # [B, La, enc_n_embd]

        # Project enc_n_embd -> n_embd
        s_encoded = self.state_proj(s_encoded)            # [B, Ls, n_embd]
        a_encoded = self.action_proj(a_encoded)            # [B, La, n_embd]

        
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

    def forward_decoder(self, x_fused: torch.Tensor, ids_restore: torch.Tensor, valid_il: torch.Tensor = None):
        """MAE-style decoder with support for unimodal training.
        Args:
            x_fused: [B, len_keep, D] kept tokens in ids_keep / ids_shuffle-kept order (post-fusion)
            ids_restore: [B, total_len] indices to restore original interleaved order
            valid_il: optional [B, total_len] bool, True = real token, False = padding
        Returns:
            s_pred: [B, T, state_pred_dim] if state_target=='embedding', or
                    [B, T, H, W, C]        if state_target=='pixel'
            a_pred: [B, T, action_dim] (continuous) or [B, T, num_actions] logits (discrete)
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

        use_valid_mask = self.use_decoder_valid_mask and (valid_il is not None)
        if use_valid_mask:
            v = valid_il.to(device=x.device, dtype=torch.bool)
            x = torch.where(
                v.unsqueeze(-1),
                x,
                self.mask_token.expand(batch_size, total_len, self.n_embd),
            )
            # Block-diagonal: real tokens only attend to real tokens, pad tokens
            # only attend to (inert) pad tokens.
            vv = v.unsqueeze(2) & v.unsqueeze(1)
            pp = (~v).unsqueeze(2) & (~v).unsqueeze(1)
            dec_attn_mask = (vv | pp).unsqueeze(1).to(dtype=torch.float32)   # [B, 1, 2T, 2T]
        else:
            dec_attn_mask = self.attn_mask
        
        # Project to decoder embedding space
        s = self.decoder_state_embed(x[:, ::2])
        a = self.decoder_action_embed(x[:, 1::2])
        
        # Interleave actions and states for the decoder
        x = torch.stack([s, a], dim=2).reshape(batch_size, total_len, self.dec_n_embd)
        
        # Add positional embeddings
        x = x + self.decoder_pos_embed[:, :x.shape[1], :]
        
        # Apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, dec_attn_mask)
        
        # Split and predict
        s_pred = self._predict_state(x[:, ::2])
        a_pred = self.action_head(x[:, 1::2])
        
        return s_pred, a_pred

    def forward_loss(self, target_s, target_a, pred_s, pred_a, mask, valid_mask=None, label_smoothing=0.0):
        # label_smoothing: applied ONLY to the action Cross-Entropy
        # Pixel-recon target/pred arrive as [B, T, H, W, C] (state_target='pixel').
        # Flatten to [B, T, H*W*C] so everything below (norm, MSE, masking) is
        # identical regardless of whether the state target is an embedding or a frame
        if target_s.dim() == 5:
            b5, t5, h5, w5, c5 = target_s.shape
            target_s = target_s.reshape(b5, t5, h5 * w5 * c5)
            pred_s = pred_s.reshape(b5, t5, h5 * w5 * c5)

        batch_size, T, _ = target_s.size()

        with torch.no_grad():
            target_norm = torch.norm(target_s, dim=-1).mean().item()
            pred_norm = torch.norm(pred_s, dim=-1).mean().item()
        
        # State normalization
        if self.norm == "l2":
            # clamp avoids 0/0 -> NaN when target_s is an exact-zero embedding
            target_s = target_s / torch.norm(target_s, dim=-1, keepdim=True).clamp(min=1.0e-6)
        elif self.norm == "mae":
            mean = target_s.mean(dim=-1, keepdim=True)
            var = target_s.var(dim=-1, keepdim=True)
            target_s = (target_s - mean) / (var + 1.0e-6) ** 0.5

        # --- State: MSE per dimension -> mean per token ---
        loss_s = (pred_s - target_s) ** 2
        loss_s_t = loss_s.mean(dim=-1)                    # [B, T]

        # --- Action: MSE per token (continuous) or cross-entropy per token (discrete) ---
        if self.discrete_actions:
            A = pred_a.shape[-1]                          # num_actions
            a_tgt = target_a.long()
            if a_tgt.dim() == 3 and a_tgt.size(-1) == 1:
                a_tgt = a_tgt.squeeze(-1)                  # [B, T, 1] -> [B, T]
            loss_a_t = F.cross_entropy(
                pred_a.reshape(batch_size * T, A),
                a_tgt.reshape(batch_size * T),
                reduction='none',
                label_smoothing=label_smoothing,
            ).reshape(batch_size, T)                       # [B, T] CE per token
        else:
            loss_a = (pred_a - target_a) ** 2
            loss_a_t = loss_a.mean(dim=-1)                 # [B, T]


        if valid_mask is not None:
            v_t = valid_mask.to(device=loss_s_t.device, dtype=loss_s_t.dtype)
            if v_t.dim() == 3 and v_t.size(-1) == 1:
                v_t = v_t.squeeze(-1)                  # [B, T, 1] -> [B, T]
            loss_s_t = loss_s_t * v_t
            loss_a_t = loss_a_t * v_t
        else:
            v_t = torch.ones_like(loss_s_t)

        # Intercalate [s0,a0,s1,a1,...] -> shape [B, 2T]
        # action_loss_weight is applied only for total/masked loss
        loss_tokens = torch.stack(
            [loss_s_t, loss_a_t * self.action_loss_weight], dim=-1
        )  # [B, T, 2]
        loss_tokens = loss_tokens.reshape(batch_size, 2 * T)     # [B, 2T]

        v_interleaved = v_t.repeat_interleave(2, dim=1)               # [B, 2T]
        combined_weight = mask * v_interleaved
        masked_loss = (loss_tokens * combined_weight).sum() / combined_weight.sum().clamp(min=1.0e-8)

        denom = v_t.sum().clamp(min=1.0e-8)
        state_loss = loss_s_t.sum() / denom
        action_loss = loss_a_t.sum() / denom

        # state_loss (above) mixes context tokens with masked/removed
        # tokens (the actual prediction task), restrict to masked+valid only, so
        # this metric reflects reconstruction quality specifically.
        state_removed = mask[:, 0::2]                              # [B, T], 1 = masked-for-recon
        sel_s = state_removed * v_t                                # exclude padding too
        state_loss_masked = (loss_s_t * sel_s).sum() / sel_s.sum().clamp(min=1.0e-8)

        # action_acc only on the masked tokens (not including padding)
        # action_acc_all_valid on masked + unmasked (not including padding)
        action_acc = None
        action_acc_all_valid = None
        if self.discrete_actions:
            with torch.no_grad():
                pred_idx = pred_a.argmax(dim=-1)              # [B, T]
                correct = (pred_idx == a_tgt)                 # [B, T]
                action_removed = mask[:, 1::2].bool()
                valid_bool = v_t.bool()
                sel = action_removed & valid_bool
                action_acc = correct[sel].float().mean() if sel.any() else pred_a.new_tensor(float('nan'))
                action_acc_all_valid = correct[valid_bool].float().mean() if valid_bool.any() else pred_a.new_tensor(float('nan'))

        return masked_loss, state_loss, action_loss, action_acc, action_acc_all_valid, state_loss_masked


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
        finetune_lr=None,
    ):
        self.action_dim = action_shape[0]
        self.lr = lr
        self.finetune_lr = finetune_lr if finetune_lr is not None else lr
        self.device = device
        self.use_tb = use_tb
        self.config = transformer_cfg
        # Decoupled weight decay (AdamW). Applied only to the "decay" param
        # group built by _classify_params (2D matrices; CNN always excluded).
        self.weight_decay = float(getattr(self.config, "weight_decay", 0.1))
        
        # Schedule {'module_nane': [start_step, end_step]}
        self.freeze_schedule = freeze_schedule if freeze_schedule is not None else {}


        # models
        self.model = MaskedDPMultimodal(
            obs_shape[0], 
            action_shape[0], 
            transformer_cfg,
        ).to(device)
        self.mask_ratio = mask_ratio
        # optimizers
        if self.model.pixel_encoder is not None:
            encoder_param_ids = {p.data_ptr() for p in self.model.pixel_encoder.parameters()}
        else:
            encoder_param_ids = set()

        param_groups, (od_len, ond_len, ed_len, end_len) = self._build_param_groups(encoder_param_ids)

        if end_len > 0:
            print(f"[Optimizer] pixel_encoder: {end_len} trainable param tensors "
                  f"@ lr={self.finetune_lr}, weight_decay=0.0 (separate group from the rest @ lr={self.lr})")

        print(f"[Optimizer] AdamW groups — "
              f"all_params: {od_len} decay / {ond_len} no_decay "
              f"(weight_decay={self.weight_decay}) | "
              f"pixel_encoder: {ed_len} decay / {end_len} no_decay")

        self.opt = torch.optim.AdamW(param_groups)
        print(
            "number of parameters: %e", sum(p.numel() for p in self.model.parameters())
        )

        self.train()
        
    def set_module_requires_grad(self, module, requires_grad):
        for param in module.parameters():
            param.requires_grad = requires_grad

    def _classify_params(self):
        decay_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
        no_decay_modules = (nn.LayerNorm, nn.Embedding)

        pixel_encoder_ids = set()
        if self.model.pixel_encoder is not None:
            pixel_encoder_ids = {p.data_ptr() for p in self.model.pixel_encoder.parameters()}

        decay_ids, no_decay_ids = set(), set()
        for _, module in self.model.named_modules():
            for pn, p in module.named_parameters(recurse=False):
                ptr = p.data_ptr()
                if ptr in pixel_encoder_ids:
                    no_decay_ids.add(ptr)
                elif pn.endswith("bias"):
                    no_decay_ids.add(ptr)
                elif isinstance(module, no_decay_modules):
                    no_decay_ids.add(ptr)
                elif isinstance(module, decay_modules) and pn.endswith("weight"):
                    decay_ids.add(ptr)
                else:
                    # Standalone nn.Parameter not owned by Linear/Conv/Norm/Embedding
                    # (e.g. mask_token, enc_mask_token) -> embedding-like, no decay.
                    no_decay_ids.add(ptr)

        return decay_ids, no_decay_ids
    
    def _build_param_groups(self, encoder_ids):
        """Método auxiliar para construir los grupos del optimizador sin duplicar código."""
        decay_ids, no_decay_ids = self._classify_params()

        encoder_decay, encoder_no_decay = [], []
        other_decay, other_no_decay = [], []
        
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            ptr = p.data_ptr()
            is_encoder = ptr in encoder_ids
            is_decay = ptr in decay_ids
            
            if is_encoder:
                (encoder_decay if is_decay else encoder_no_decay).append(p)
            else:
                (other_decay if is_decay else other_no_decay).append(p)

        assert not encoder_decay, (
            "[Optimizer] Invariant violated: pixel_encoder (CNN) params must never "
            "land in a weight-decay group. Check _classify_params()."
        )

        param_groups = []
        if other_decay:
            param_groups.append({'params': other_decay, 'lr': self.lr,
                                'weight_decay': self.weight_decay, 'name': 'fusion_decoder_decay'})
        if other_no_decay:
            param_groups.append({'params': other_no_decay, 'lr': self.lr,
                                'weight_decay': 0.0, 'name': 'fusion_decoder_no_decay'})
        if encoder_decay:
            param_groups.append({'params': encoder_decay, 'lr': self.finetune_lr,
                                'weight_decay': self.weight_decay, 'name': 'encoders_decay'})
        if encoder_no_decay:
            param_groups.append({'params': encoder_no_decay, 'lr': self.finetune_lr,
                                'weight_decay': 0.0, 'name': 'encoders_no_decay'})

        if not param_groups:
            param_groups = [{'params': [], 'lr': self.lr,
                            'weight_decay': self.weight_decay, 'name': 'empty'}]

        stats = (len(other_decay), len(other_no_decay), len(encoder_decay), len(encoder_no_decay))
        return param_groups, stats

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
        """Rebuild optimizer with only trainable parameters (decay/no_decay x encoder/other)."""
        encoder_ids = self._get_encoder_param_ids()
        param_groups, (od_len, ond_len, ed_len, end_len) = self._build_param_groups(encoder_ids)

        print(
            f"[Optimizer] Full rebuild — "
            f"fusion/decoder: {od_len} decay / {ond_len} no_decay @ lr={self.lr}, "
            f"encoders: {ed_len} decay / {end_len} no_decay @ lr={self.finetune_lr}"
        )
        self.opt = torch.optim.AdamW(param_groups)

    def _add_encoder_param_groups(self, newly_unfrozen_names):
        # Collect all param data_ptrs already tracked by the optimizer
        existing_ids = set()
        for grp in self.opt.param_groups:
            for p in grp['params']:
                existing_ids.add(p.data_ptr())

        decay_ids, no_decay_ids = self._classify_params()

        new_decay, new_no_decay = [], []
        for name in newly_unfrozen_names:
            module = getattr(self.model, name, None)
            if module is None:
                continue
            for p in module.parameters():
                if p.requires_grad and p.data_ptr() not in existing_ids:
                    (new_decay if p.data_ptr() in decay_ids else new_no_decay).append(p)

        if not new_decay and not new_no_decay:
            print(
                f"[Optimizer] _add_encoder_param_groups: no new params to add "
                f"(already tracked or empty). Skipping."
            )
            return

        if new_decay:
            self.opt.add_param_group({
                'params': new_decay,
                'lr': self.finetune_lr,
                'weight_decay': self.weight_decay,
                'name': 'encoders_decay',
            })
        if new_no_decay:
            self.opt.add_param_group({
                'params': new_no_decay,
                'lr': self.finetune_lr,
                'weight_decay': 0.0,
                'name': 'encoders_no_decay',
            })
        print(
            f"[Optimizer] add_param_group — {len(new_decay)} decay + {len(new_no_decay)} no_decay "
            f"encoder params @ lr={self.finetune_lr}. Fusion/decoder Adam state PRESERVED."
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

    def update_mdp(self, states, actions, step=None, valid_mask=None):
        """
        valid_mask: optional [B,T] or [B,T,1] tensor, 1=real timestep, 0=padding
        (e.g. from OfflineReplayBuffer when an episode is shorter than
        traj_length). Default None means "all real" - exact original behavior.
        """
        self.check_freeze_schedule(step)
        

        metrics = dict()
        mask_ratio = np.random.choice(self.mask_ratio)

        # Temporal jitter: sample T_eff from the discrete traj_lengths list and truncate
        # the batch before the forward pass. T is constant within a batch to avoid
        # variable-length collation issues. T is already fully dynamic throughout
        # forward_encoder and forward_decoder (all shapes derived from states.size(1)),
        # so no other changes are needed.
        T_eff = None
        if self.model.traj_lengths is not None and self.training:
            T_eff = self.model._sample_jitter_T()
            states  = states[:, :T_eff]
            actions = actions[:, :T_eff]
            if valid_mask is not None:
                valid_mask = valid_mask[:, :T_eff]

        # Encoder (dual + optional fusion)
        x_fused, mask, ids_restore, ids_keep = \
            self.model.forward_encoder(states, actions, mask_ratio, valid_mask=valid_mask)

        valid_il = None
        if valid_mask is not None:
            v = valid_mask.to(device=states.device, dtype=torch.bool)
            if v.dim() == 3 and v.size(-1) == 1:
                v = v.squeeze(-1)                          # [B, T, 1] -> [B, T]
            valid_il = v.repeat_interleave(2, dim=1)        # [B, 2T]

        # Decoder
        pred_s, pred_a = self.model.forward_decoder(
            x_fused, ids_restore, valid_il=valid_il
        )
        
        # Loss
        with torch.no_grad():
            target_s = self.model._make_state_target(states)
        
        mask_loss, state_loss, action_loss, action_acc, action_acc_all_valid, state_loss_masked = self.model.forward_loss(
            target_s, actions, pred_s, pred_a, mask, valid_mask=valid_mask,
            label_smoothing=self.model.label_smoothing,
        )
        
        if self.config.loss == "masked":
            loss = mask_loss
        elif self.config.loss == "total":
            # lamba for balancing total loss
            loss = state_loss + self.model.action_loss_weight * action_loss
        else:
            raise NotImplementedError

        # Gradient-norm probe (diagnostic, DELETE later)
        log_now = self.use_tb and step is not None and step % 1000 == 0
        grad_norm_state_trunk = None
        grad_norm_action_trunk = None
        grad_ratio_state_over_action = None
        if log_now and state_loss.requires_grad and action_loss.requires_grad:
            probe_params = [p for m in (self.model.fusion_blocks, self.model.decoder_blocks)
                            for p in m.parameters() if p.requires_grad]
            if probe_params:
                g_s = torch.autograd.grad(state_loss, probe_params, retain_graph=True, allow_unused=True)
                g_a = torch.autograd.grad(self.model.action_loss_weight * action_loss, probe_params,
                                          retain_graph=True, allow_unused=True)
                ns = torch.sqrt(sum((g**2).sum() for g in g_s if g is not None))
                na = torch.sqrt(sum((g**2).sum() for g in g_a if g is not None))
                grad_norm_state_trunk = ns.item()
                grad_norm_action_trunk = na.item()
                grad_ratio_state_over_action = (ns / na.clamp(min=1e-12)).item()

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        if self.use_tb:
            if T_eff is not None:
                metrics["jitter_T_eff"] = T_eff
            metrics["mask_loss"] = mask_loss.item()
            metrics["state_loss"] = state_loss.item()
            metrics["action_loss"] = action_loss.item()
            if not torch.isnan(state_loss_masked):
                metrics["state_loss_masked"] = state_loss_masked.item()
            if action_acc is not None and not torch.isnan(action_acc):
                metrics["action_acc"] = action_acc.item()
            if action_acc_all_valid is not None and not torch.isnan(action_acc_all_valid):
                metrics["action_acc_all_valid"] = action_acc_all_valid.item()

            # CE/accuracy per masking ratio
            mr_key = f"{mask_ratio:.2f}"
            if not np.isnan(action_loss.item()):
                metrics[f"by_mr/action_loss_mr{mr_key}"] = action_loss.item()
            if action_acc is not None and not torch.isnan(action_acc):
                metrics[f"by_mr/action_acc_mr{mr_key}"] = action_acc.item()

            if grad_ratio_state_over_action is not None:
                metrics["grad_norm_state_trunk"] = grad_norm_state_trunk
                metrics["grad_norm_action_trunk"] = grad_norm_action_trunk
                metrics["grad_ratio_state_over_action"] = grad_ratio_state_over_action

        return metrics

    def eval_validation(self, val_iter, step=None):
        metrics = dict()
        batch = next(val_iter)
        obs, action, _, _, _, valid_mask = utils.to_torch(batch, self.device)
        
        mask_ratio = np.random.choice(self.mask_ratio)
        x_fused, mask, ids_restore, ids_keep = \
            self.model.forward_encoder(obs, action, mask_ratio, valid_mask=valid_mask)

        valid_il = None
        if valid_mask is not None:
            v = valid_mask.to(device=obs.device, dtype=torch.bool)
            if v.dim() == 3 and v.size(-1) == 1:
                v = v.squeeze(-1)                          # [B, T, 1] -> [B, T]
            valid_il = v.repeat_interleave(2, dim=1)        # [B, 2T]

        pred_s, pred_a = self.model.forward_decoder(
            x_fused, ids_restore, valid_il=valid_il
        )

        with torch.no_grad():
            target_s = self.model._make_state_target(obs)

        mask_loss, state_loss, action_loss, action_acc, action_acc_all_valid, state_loss_masked = self.model.forward_loss(
            target_s, action, pred_s, pred_a, mask, valid_mask=valid_mask,
            label_smoothing=0.0,
        )

        if self.use_tb:
            metrics["val_mask_loss"] = mask_loss.item()
            metrics["val_state_loss"] = state_loss.item()
            metrics["val_action_loss"] = action_loss.item()
            if not torch.isnan(state_loss_masked):
                metrics["val_state_loss_masked"] = state_loss_masked.item()
            if action_acc is not None and not torch.isnan(action_acc):
                metrics["val_action_acc"] = action_acc.item()
            if action_acc_all_valid is not None and not torch.isnan(action_acc_all_valid):
                metrics["val_action_acc_all_valid"] = action_acc_all_valid.item()

        return metrics

    def update(self, replay_iter, step=None):
        metrics = dict()

        batch = next(replay_iter)
        obs, action, _, _, _, valid_mask = utils.to_torch(batch, self.device)

        # update critic
        metrics.update(self.update_mdp(obs, action, step=step, valid_mask=valid_mask))

        return metrics