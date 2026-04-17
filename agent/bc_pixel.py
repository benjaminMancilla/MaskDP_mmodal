# agent/bc_pixel.py
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent.mdp import MaskedDPMultimodal
import utils


class BCHead(nn.Module):
    """Trainable MLP head on top of frozen MaskDP encoder+fusion."""
    def __init__(self, n_embd: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(n_embd),
            nn.Linear(n_embd, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BCPixelAgent:
    """
    Behavioral Cloning agent built on top of pretrained MaskDP multimodal encoder.

    Forward pass:
        obs_seq    (B, T, H, W, C) uint8    * state stream
        action_seq (B, T, action_dim)       * action stream (dataset during FT, own actions during eval)
            - pixel_encoder                 -> (B, T, n_embd)
            - action_embed                  -> (B, T, n_embd)
            + positional embeddings
            - state_encoder_blocks + norm + adapter
            - action_encoder_blocks + norm + adapter
            - fusion_type_embed
            - fusion_blocks (cross or self)
            - fusion_norm
            - x_s[:, -1, :]                 -> (B, n_embd)  last state token
            - BCHead                        -> (B, action_dim)

    With T=1 this degenerates to single-frame BC.

    During online eval:
        - obs buffer grows from 1 to T, then slides (no padding)
        - action buffer starts with zeros at t=0, then uses agent's own actions
        - env delivers (C, H, W) channels-first → converted to (H, W, C)
    """

    def __init__(
        self,
        obs_shape,          # (H, W, C) single frame
        action_shape,
        device,
        lr,
        context_length,     # T during finetuning
        hidden_dim,
        use_tb,
        transformer_cfg,
        freeze_encoder=True,
        path=None,          # pretrained MaskDP snapshot
    ):
        self.device = device
        self.use_tb = use_tb
        self.context_length = context_length
        self.obs_shape = obs_shape      # (H, W, C)
        self.action_dim = action_shape[0]
        self.freeze_encoder = freeze_encoder

        # Build MaskDP model
        if path is not None:
            print(f"[BCPixelAgent] Loading pretrained weights: {path}")
            payload = torch.load(path, map_location=device)
            cfg = payload["cfg"]
        else:
            cfg = transformer_cfg

        self.mdp = MaskedDPMultimodal(obs_shape, self.action_dim, cfg).to(device)
        self.config = cfg

        if path is not None:
            self.mdp.load_state_dict(payload["model"])
            print(f"[BCPixelAgent] Pretrained weights loaded.")

        if freeze_encoder:
            for param in self.mdp.parameters():
                param.requires_grad = False
            print(f"[BCPixelAgent] Encoder frozen.")

        # Trainable BC head
        self.bc_head = BCHead(cfg.n_embd, self.action_dim, hidden_dim).to(device)

        # Optimizer
        params = list(self.bc_head.parameters())
        if not freeze_encoder:
            params += list(self.mdp.parameters())
        self.opt = torch.optim.Adam(params, lr=lr)

        trainable = sum(p.numel() for p in params)
        total = sum(p.numel() for p in self.mdp.parameters()) + \
                sum(p.numel() for p in self.bc_head.parameters())
        print(f"[BCPixelAgent] Trainable: {trainable:,} / {total:,}")

        self.train()

        # Online eval buffers
        self._obs_buffer    = None
        self._action_buffer = None

    def train(self, training=True):
        self.training = training
        self.mdp.train(False)           # encoder always in eval mode
        self.bc_head.train(training)

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def _encode(
        self,
        obs_seq: torch.Tensor,      # (B, T, H, W, C) uint8
        action_seq: torch.Tensor,   # (B, T, action_dim) float32
    ) -> torch.Tensor:
        """Returns (B, n_embd) — last state token after full encoder+fusion."""

        B, T = obs_seq.shape[0], obs_seq.shape[1]

        # --- Positional embeddings (interpolate if T > max_len/2) ---
        max_pos = self.mdp.pos_embed.shape[1]
        if 2 * T > max_pos:
            pos_embed = utils.interpolate_pos_embed(self.mdp.pos_embed, 2 * T)
        else:
            pos_embed = self.mdp.pos_embed

        attn_mask = torch.ones(B, 1, T, T, device=self.device)

        # --- State stream ---
        # pixel_encoder: (B, T, H, W, C) → (B, T, n_embd)
        s_emb = self.mdp._embed_states(obs_seq)
        s_emb = s_emb + pos_embed[:, 0:2*T:2, :]   # even positions

        for blk in self.mdp.state_encoder_blocks:
            s_emb = blk(s_emb, attn_mask)
        s_emb = self.mdp.state_encoder_norm(s_emb)
        s_emb = self.mdp.state_adapter(s_emb)       # identity if not used

        # --- Action stream ---
        # action_embed: (B, T, action_dim) → (B, T, n_embd)
        a_emb = self.mdp.action_embed(action_seq)
        a_emb = a_emb + pos_embed[:, 1:2*T:2, :]   # odd positions

        for blk in self.mdp.action_encoder_blocks:
            a_emb = blk(a_emb, attn_mask)
        a_emb = self.mdp.action_encoder_norm(a_emb)
        a_emb = self.mdp.action_adapter(a_emb)

        # --- Fusion (Opción B — directo sin forward_fusion) ---
        x_s, x_a = s_emb, a_emb

        if self.mdp.fusion_type_embed is not None:
            type_ids_s = torch.zeros(B, T, dtype=torch.long, device=self.device)
            type_ids_a = torch.ones(B,  T, dtype=torch.long, device=self.device)
            x_s = x_s + self.mdp.fusion_type_embed(type_ids_s)
            x_a = x_a + self.mdp.fusion_type_embed(type_ids_a)

        if self.mdp.fusion_type == 'cross':
            for blk in self.mdp.fusion_blocks:
                x_s, x_a = blk(x_s, x_a, mask_s=None, mask_a=None)
        else:
            # self-attention: interleave and apply blocks
            x_interleaved = torch.stack([x_s, x_a], dim=2).reshape(B, 2*T, self.config.n_embd)
            fuse_mask = torch.ones(B, 1, 2*T, 2*T, device=self.device)
            for blk in self.mdp.fusion_blocks:
                x_interleaved = blk(x_interleaved, fuse_mask)
            # extract state tokens (even positions)
            x_s = x_interleaved[:, 0::2, :]

        if self.mdp.fusion_blocks:
            x_s = self.mdp.fusion_norm(x_s)

        # Last state token → (B, n_embd)
        return x_s[:, -1, :]

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def update(self, replay_iter, step):
        metrics = dict()

        batch = next(replay_iter)
        obs, action, *_ = utils.to_torch(batch, self.device)
        # obs:    (B, T, H, W, C) uint8
        # action: (B, T, action_dim) float32

        # Action context: shift by 1 — at t we give actions a_0..a_{t-1}
        # a_0 has no previous action → use zeros
        action_context = torch.zeros_like(action)
        action_context[:, 1:] = action[:, :-1]     # (B, T, action_dim)

        with torch.set_grad_enabled(not self.freeze_encoder):
            features = self._encode(obs, action_context)    # (B, n_embd)

        pred_action = self.bc_head(features)                # (B, action_dim)
        target_action = action[:, -1, :]                    # (B, action_dim)

        loss = F.mse_loss(pred_action, target_action)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        if self.use_tb:
            metrics["bc_loss"] = loss.item()

        return metrics

    # ------------------------------------------------------------------
    # Online eval
    # ------------------------------------------------------------------

    def act(self, obs_chw: np.ndarray, step: int = 0) -> np.ndarray:
        """
        obs_chw: (C, H, W) uint8 — format from dm_control FrameStackWrapper
        returns: (action_dim,) float32
        """
        # channels-first → channels-last
        obs_hwc = obs_chw.transpose(1, 2, 0)  # (H, W, C)

        if self._obs_buffer is None:
            # First step of episode — initialize buffers
            self._obs_buffer    = deque([obs_hwc], maxlen=self.context_length)
            self._action_buffer = deque(
                [np.zeros(self.action_dim, dtype=np.float32)],
                maxlen=self.context_length
            )
        else:
            self._obs_buffer.append(obs_hwc)
            # action buffer already has the last action appended by act() below

        T = len(self._obs_buffer)

        # Build tensors: (1, T, ...)
        obs_seq = torch.as_tensor(
            np.stack(list(self._obs_buffer), axis=0)[np.newaxis],
            device=self.device
        )  # (1, T, H, W, C)

        action_seq = torch.as_tensor(
            np.stack(list(self._action_buffer), axis=0)[np.newaxis],
            device=self.device,
            dtype=torch.float32
        )  # (1, T, action_dim)

        with torch.no_grad():
            features = self._encode(obs_seq, action_seq)    # (1, n_embd)
            action = self.bc_head(features)                 # (1, action_dim)

        action_np = action.cpu().numpy()[0]

        # Append this action so next step has it as context
        self._action_buffer.append(action_np)

        return action_np

    def reset_obs_buffer(self):
        """Call at the start of each eval episode."""
        self._obs_buffer    = None
        self._action_buffer = None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path):
        torch.save({
            "bc_head": self.bc_head.state_dict(),
            "mdp": self.mdp.state_dict() if not self.freeze_encoder else None,
        }, path)

    def load(self, path):
        payload = torch.load(path, map_location=self.device)
        self.bc_head.load_state_dict(payload["bc_head"])
        if not self.freeze_encoder and payload["mdp"] is not None:
            self.mdp.load_state_dict(payload["mdp"])