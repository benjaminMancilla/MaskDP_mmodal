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
from agent.modules.pixel_encoder import PixelEncoder
from agent.modules.load_pretrained_encoder import load_drqbc_convnet, load_procgen_impala


class MaskedDP(nn.Module):
    def __init__(self, obs_dim, action_dim, config):
        super().__init__()
        self.config = config 
        # MAE encoder specifics
        self.n_embd = config.n_embd
        self.max_len = config.traj_length * 2
        # self.mask_ratio = config.mask_ratio
        self.pe = config.pe
        self.norm = config.norm

        # -- Temporal Jitter --------------------------
        raw_traj_lengths = getattr(config, "traj_lengths", None)
        self.jitter_strategy = str(getattr(config, "jitter_strategy", "mix50"))
        if raw_traj_lengths is not None:
            self.traj_lengths = [int(t) for t in raw_traj_lengths]
            assert all(t <= config.traj_length for t in self.traj_lengths)
            print(f"Temporal Jitter ENABLED: strategy='{self.jitter_strategy}', T candidates={self.traj_lengths}")
        else:
            self.traj_lengths = None
            print(f"Temporal Jitter DISABLED: fixed T={config.traj_length}")

        # -- Modality Dropout --------------------------
        self.modality_dropout      = bool(getattr(config, "modality_dropout", False))
        self.modality_dropout_prob = float(getattr(config, "modality_dropout_prob", 0.25))
        self.p_drop_action         = float(getattr(config, "action_dropout_prob", 1.0))
        self.p_drop_state          = float(getattr(config, "state_dropout_prob", 0.0))
        self.min_keep_states       = int(getattr(config, "min_keep_states", 1))
        self.min_keep_actions      = int(getattr(config, "min_keep_actions", 0))
        if self.modality_dropout:
            print(f"Modality Dropout ENABLED (prob={self.modality_dropout_prob}, "
                f"75A/25S ratio, min_states={self.min_keep_states}, min_actions={self.min_keep_actions})")

        print("norm", self.norm)

        # -- Discrete actions (classification) vs continuous (regression) --
        self.discrete_actions   = bool(getattr(config, "discrete_actions", False))
        self.num_actions        = int(getattr(config, "num_actions", 0))
        self.action_loss_weight = float(getattr(config, "action_loss_weight", 1.0))
        self.label_smoothing    = float(getattr(config, "label_smoothing", 0.0))
        if self.discrete_actions:
            assert self.num_actions > 0, "discrete_actions=True requires num_actions > 0"
            print(f"Discrete actions ENABLED: num_actions={self.num_actions}, "
                  f"action_loss_weight={self.action_loss_weight}, label_smoothing={self.label_smoothing}")

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

            self.pixel_encoder = PixelEncoder(
                pixel_obs_shape, self.n_embd, encoder_type=pixel_encoder_type
            )

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
            print(f"[PixelEncoder] type='{pixel_encoder_type}' | "
                  f"trainable params: {trainable}/{total}")
        else:
            self.pixel_encoder = None
            self.state_embed = nn.Linear(obs_dim, self.n_embd)

        if self.discrete_actions:
            self.action_embed = nn.Embedding(self.num_actions, self.n_embd)
        else:
            self.action_embed = nn.Linear(action_dim, self.n_embd)
        self.encoder_blocks = nn.ModuleList(
            [Block(config) for _ in range(config.n_enc_layer)]
        )
        self.encoder_norm = nn.LayerNorm(self.n_embd)
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_state_embed = nn.Linear(self.n_embd, self.n_embd)
        self.decoder_action_embed = nn.Linear(self.n_embd, self.n_embd)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.n_embd))

        self.decoder_blocks = nn.ModuleList(
            [Block(config) for _ in range(config.n_dec_layer)]
        )

        if self.discrete_actions:
            self.action_head = nn.Sequential(
                nn.LayerNorm(self.n_embd),
                nn.ReLU(inplace=True),
                nn.Linear(self.n_embd, self.num_actions),
            )  # logits over discrete actions (no Tanh)
        else:
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

    def _embed_states(self, states: torch.Tensor) -> torch.Tensor:
        if self.pixel_encoder is not None:
            return self.pixel_encoder(states)
        return self.state_embed(states)

    def initialize_weights(self):
        pos_embed = utils.get_1d_sincos_pos_embed_from_grid(self.n_embd, self.max_len)
        pe = torch.from_numpy(pos_embed).float().unsqueeze(0) / 2.0
        self.register_buffer("pos_embed", pe)
        self.register_buffer("decoder_pos_embed", pe)
        self.register_buffer(
            "attn_mask", torch.ones(self.max_len, self.max_len)[None, None, ...]
        )
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        # initialize nn.Linear and nn.LayerNorm
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

    def random_masking(self, x, mask_ratio, noise):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        if noise is None:
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def _sample_jitter_length(self):
        """Sample a trajectory length T for temporal jitter."""
        if self.traj_lengths is None:
            return None
        T_max = self.traj_lengths[-1]
        if self.jitter_strategy == 'mix50':
            if np.random.rand() < 0.5:
                return T_max
            candidates = self.traj_lengths[:-1] if len(self.traj_lengths) > 1 else self.traj_lengths
            return int(np.random.choice(candidates))
        elif self.jitter_strategy == 'uniform':
            return int(np.random.choice(self.traj_lengths))
        elif self.jitter_strategy == 'uniform_log':
            weights = np.array([1.0 / t for t in self.traj_lengths], dtype=float)
            weights /= weights.sum()
            return int(np.random.choice(self.traj_lengths, p=weights))
        return T_max


    def forward_encoder(self, states, actions, mask_ratio, valid_mask=None):
        batch_size, T_full = states.shape[0], states.shape[1]
        obs_dim = states.shape[2] if states.ndim == 3 else None

        # Temporal Jitter: recortar secuencia a T_jitter
        T_jitter = self._sample_jitter_length()
        if T_jitter is not None and T_jitter < T_full:
            # Offset aleatorio para no siempre tomar desde el inicio
            max_offset = T_full - T_jitter
            offset = np.random.randint(0, max_offset + 1)
            states  = states[:, offset:offset + T_jitter, :]
            actions = actions[:, offset:offset + T_jitter, :]
            if valid_mask is not None:
                valid_mask = valid_mask[:, offset:offset + T_jitter]
            T = T_jitter
        else:
            offset = 0
            T = T_full

        # Per-token validity for padding (short episodes): interleave [v0,v0,v1,v1,...].
        # None -> no padding (V-D4RL / dm_control), behaves exactly as before.
        valid_tok = None
        if valid_mask is not None:
            v = valid_mask
            if v.dim() == 3:
                v = v.squeeze(-1)                                  # [B, T]
            v = v.to(dtype=states.dtype)
            valid_tok = v.repeat_interleave(2, dim=1)              # [B, 2T]


        batch_size, T = states.shape[0], states.shape[1]
        s_emb = self._embed_states(states)
        if self.discrete_actions:
            a_idx = actions.long()
            if a_idx.dim() == 3 and a_idx.size(-1) == 1:
                a_idx = a_idx.squeeze(-1)          # [B, T, 1] -> [B, T]
            a_emb = self.action_embed(a_idx)       # [B, T, n_embd]
        else:
            a_emb = self.action_embed(actions)

        x = (
            torch.stack([s_emb, a_emb], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, 2 * T, self.n_embd)
        )
        x = x + self.pos_embed[:, :2 * T, :]

        # Modality Dropout
        noise = torch.rand(batch_size, 2 * T, device=states.device)

        if self.training and self.modality_dropout and (np.random.rand() < self.modality_dropout_prob):
            total_prob = self.p_drop_action + self.p_drop_state
            p_action = self.p_drop_action / total_prob if total_prob > 0 else 0.5
            drop_actions = np.random.rand() < p_action
            start_idx = 1 if drop_actions else 0
            noise[:, start_idx::2] += 100.0

        # Modality shields (min keep tokens)
        if self.min_keep_states > 0:
            k_s = min(self.min_keep_states, T)
            _, top_s_idx = torch.topk(torch.rand(batch_size, T, device=states.device), k_s, dim=1)
            noise.scatter_(1, top_s_idx * 2, -100.0)

        if self.min_keep_actions > 0:
            k_a = min(self.min_keep_actions, T)
            _, top_a_idx = torch.topk(torch.rand(batch_size, T, device=states.device), k_a, dim=1)
            noise.scatter_(1, top_a_idx * 2 + 1, -100.0)


        # Padding bias: push padded tokens to the end of the ranking so random_masking
        # removes them first (applied AFTER the shields so it dominates them). Guarantees
        # at least min(len_keep, #valid) valid kept tokens -> no fully-masked attention row.
        if valid_tok is not None:
            noise = noise + (1.0 - valid_tok) * 1.0e4

        x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio, noise=noise)

        # Encoder attention mask: block attention TO any padded token that survived into
        # the kept set (only happens when the episode is shorter than the kept length).
        if valid_tok is not None:
            kept_valid = torch.gather(valid_tok, 1, ids_keep)      # [B, len_keep]
            enc_attn = kept_valid[:, None, None, :]                # block padded key columns
        else:
            enc_attn = self.attn_mask

        # apply Transformer blocks
        for blk in self.encoder_blocks:
            x = blk(x, enc_attn)
        x = self.encoder_norm(x)
        return x, mask, ids_restore, T, offset, valid_tok

    def forward_decoder(self, x, ids_restore, valid_tok=None):
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1
        )
        x_ = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        s = self.decoder_state_embed(x[:, ::2])
        a = self.decoder_action_embed(x[:, 1::2])

        x = torch.stack([s, a], dim=1).permute(0, 2, 1, 3).reshape_as(x)

        # add pos embed
        x = x + self.decoder_pos_embed[:, :x.shape[1], :]

        # Decoder attention mask: block padded key columns over the full 2T sequence.
        dec_attn = self.attn_mask if valid_tok is None else valid_tok[:, None, None, :]

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, dec_attn)

        # predictor projection
        s = self.state_head(x[:, ::2])
        a = self.action_head(x[:, 1::2])

        return s, a

    def forward_loss(self, target_s, target_a, pred_s, pred_a, mask, label_smoothing=0.0, valid_mask=None):
        # label_smoothing: applied ONLY to the action Cross-Entropy (discrete actions).
        # valid_mask [B, T] (1=real, 0=pad): padded tokens are excluded from every term.
        batch_size, T, _ = target_s.size()
        # State target normalization
        if self.norm == "l2":
            # clamp avoids 0/0 -> NaN when target_s is an exact-zero embedding
            target_s = target_s / torch.norm(target_s, dim=-1, keepdim=True).clamp(min=1.0e-6)
        elif self.norm == "mae":
            mean = target_s.mean(dim=-1, keepdim=True)
            var = target_s.var(dim=-1, keepdim=True)
            target_s = (target_s - mean) / (var + 1.0e-6) ** 0.5

        # --- State: MSE per dim -> per token ---
        loss_s = (pred_s - target_s) ** 2
        loss_s_t = loss_s.mean(dim=-1)                     # [B, T]

        # --- Action: MSE per token (continuous) or cross-entropy per token (discrete) ---
        if self.discrete_actions:
            A = pred_a.shape[-1]                           # num_actions
            a_tgt = target_a.long()
            if a_tgt.dim() == 3 and a_tgt.size(-1) == 1:
                a_tgt = a_tgt.squeeze(-1)                  # [B, T, 1] -> [B, T]
            loss_a_t = F.cross_entropy(
                pred_a.reshape(batch_size * T, A),
                a_tgt.reshape(batch_size * T),
                reduction='none',
                label_smoothing=label_smoothing,
            ).reshape(batch_size, T)                        # [B, T] CE per token
        else:
            loss_a_t = ((pred_a - target_a) ** 2).mean(dim=-1)   # [B, T]

        # Per-timestep validity (1=real, 0=pad); ones when there is no padding.
        if valid_mask is not None:
            v_t = valid_mask.to(device=loss_s_t.device, dtype=loss_s_t.dtype)
            if v_t.dim() == 3 and v_t.size(-1) == 1:
                v_t = v_t.squeeze(-1)                      # [B, T]
            loss_s_t = loss_s_t * v_t
            loss_a_t = loss_a_t * v_t
        else:
            v_t = torch.ones_like(loss_s_t)

        # Interleave [s0,a0,s1,a1,...] -> [B, 2T]; action term scaled by action_loss_weight.
        loss_tokens = torch.stack(
            [loss_s_t, loss_a_t * self.action_loss_weight], dim=-1
        ).reshape(batch_size, 2 * T)
        v_interleaved = v_t.repeat_interleave(2, dim=1)            # [B, 2T]
        combined_weight = mask * v_interleaved
        masked_loss = (loss_tokens * combined_weight).sum() / combined_weight.sum().clamp(min=1.0e-8)

        denom = v_t.sum().clamp(min=1.0e-8)
        state_loss = loss_s_t.sum() / denom
        action_loss = loss_a_t.sum() / denom

        # Discrete action accuracy on masked-for-recon, valid action tokens only.
        action_acc = None
        if self.discrete_actions:
            with torch.no_grad():
                pred_idx = pred_a.argmax(dim=-1)           # [B, T]
                correct = (pred_idx == a_tgt)
                action_removed = mask[:, 1::2].bool()      # 1 = action token masked
                sel = action_removed & v_t.bool()
                action_acc = (correct[sel].float().mean()
                              if sel.any() else pred_a.new_tensor(float('nan')))

        return masked_loss, state_loss, action_loss, action_acc


class MaskedDPAgent:
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
        self.model = MaskedDP(obs_shape[0], action_shape[0], transformer_cfg).to(device)
        self.mask_ratio = mask_ratio
        # optimizers
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.Adam(trainable_params, lr=lr)
        print(
            "number of parameters: %e", sum(p.numel() for p in self.model.parameters())
        )

        self.train()

    def train(self, training=True):
        self.training = training
        self.model.train(training)

    def update_mdp(self, states, actions, valid_mask=None, step=0):
        metrics = dict()
        mask_ratio = np.random.choice(self.mask_ratio)
        # Freeze pixel encoder after convergence (used in trainble CNN)
        freeze_after = getattr(self.config, "freeze_encoder_after", 10000)
        if step >= freeze_after and self.model.pixel_encoder is not None:
            if any(p.requires_grad for p in self.model.pixel_encoder.parameters()):
                for p in self.model.pixel_encoder.parameters():
                    p.requires_grad = False
                print(f"[step {step}] Pixel encoder frozen.")
        latent, mask, ids_restore, T, offset, valid_tok = self.model.forward_encoder(
            states, actions, mask_ratio, valid_mask=valid_mask
        )

        with torch.no_grad():
            if self.model.use_pixel_obs:
                target_s = self.model._embed_states(states[:, offset:offset + T])
            else:
                target_s = states[:, offset:offset + T]
        target_a = actions[:, offset:offset + T, :]

        pred_s, pred_a = self.model.forward_decoder(
            latent, ids_restore, valid_tok
        )  # [N, L, p*p*3]
        loss_valid = valid_tok[:, 0::2] if valid_tok is not None else None
        mask_loss, state_loss, action_loss, action_acc = self.model.forward_loss(
            target_s, target_a, pred_s, pred_a, mask,
            label_smoothing=self.model.label_smoothing,
            valid_mask=loss_valid,
        )
        if self.config.loss == "masked":
            loss = mask_loss
        elif self.config.loss == "total":
            loss = state_loss + self.model.action_loss_weight * action_loss
        else:
            raise NotImplementedError

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        if self.use_tb:
            metrics["mask_loss"] = mask_loss.item()
            metrics["state_loss"] = state_loss.item()
            metrics["action_loss"] = action_loss.item()
            if action_acc is not None:
                metrics["action_acc"] = action_acc.item()

        return metrics

    def eval_validation(self, val_iter, step=None):
        metrics = dict()
        batch = next(val_iter)
        obs, action, _, _, _, valid_mask = utils.to_torch(batch, self.device)
        mask_ratio = np.random.choice(self.mask_ratio)
        latent, mask, ids_restore, T, offset, valid_tok = self.model.forward_encoder(
            obs, action, mask_ratio, valid_mask=valid_mask
        )
        with torch.no_grad():
            if self.model.use_pixel_obs:
                target_s = self.model._embed_states(obs[:, offset:offset + T])
            else:
                target_s = obs[:, offset:offset + T]
        target_a = action[:, offset:offset + T, :]
        pred_s, pred_a = self.model.forward_decoder(
            latent, ids_restore, valid_tok
        )  # [N, L, p*p*3]
        loss_valid = valid_tok[:, 0::2] if valid_tok is not None else None
        mask_loss, state_loss, action_loss, action_acc = self.model.forward_loss(
            target_s, target_a, pred_s, pred_a, mask,
            label_smoothing=0.0,
            valid_mask=loss_valid,
        )

        if self.use_tb:
            metrics["val_mask_loss"] = mask_loss.item()
            metrics["val_state_loss"] = state_loss.item()
            metrics["val_action_loss"] = action_loss.item()
            if action_acc is not None:
                metrics["val_action_acc"] = action_acc.item()

        return metrics

    def update(self, replay_iter, step=None):
        metrics = dict()

        batch = next(replay_iter)
        obs, action, _, _, _, valid_mask = utils.to_torch(batch, self.device)

        # update critic
        metrics.update(self.update_mdp(obs, action, valid_mask, step))

        return metrics
