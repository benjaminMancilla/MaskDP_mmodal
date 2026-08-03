import numpy as np
import torch
import torch.nn as nn
from collections import deque

from agent.mdp import MaskedDP


class BCTEvalAgent:
    """
    Closed-loop, per-timestep BC eval agent for the single-stream MaskedDP model,
    matching gen_dgrl's `eval_DT_agent` with model_type='naive' (BCT): act every
    step from the current sliding window, no returns/rtg, no open-loop horizon.

    Protocol per step t:
      1. Append the new observation s_t to the obs window (maxlen=K).
      2. The action buffer (maxlen=K-1) does NOT yet contain a_t -> a_t stays masked
         by construction.
      3. _forward() runs the single-stream encoder + decoder over the window.
      4. Read the prediction at the current-action position: pred_a[n_s - 1].
      5. Decode to a concrete action (sample at temperature, or argmax), execute,
         append to the action buffer, repeat.

    Because the window holds s_0..s_{n_s-1} and a_0..a_{n_s-2}, the real tokens
    occupy the contiguous prefix [s0,a0,s1,a1,...,s_{n_s-1}] of the interleaved
    2T sequence. The masked tokens (a_{n_s-1} onward) are simply appended, so
    ids_restore is the identity -- no scatter needed.
    """

    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        K=None,                 # sliding window length; default = transformer_cfg.traj_length
        temperature=1.0,
        sample=True,            # True: Categorical(logits/temperature).sample(); False: argmax
        path=None,
        transformer_cfg=None,
        **kwargs,
    ):
        self.device = device
        self.temperature = temperature
        self.sample = sample

        if path is not None:
            print(f"[BCTEvalAgent] Loading snapshot: {path}")
            payload = torch.load(path, map_location=device)
            self.config = payload["cfg"]
        else:
            assert transformer_cfg is not None, "Must provide either `path` or `transformer_cfg`."
            self.config = transformer_cfg

        assert bool(getattr(self.config, "discrete_actions", False)), (
            "BCTEvalAgent only supports discrete-actions"
        )
        self.num_actions = int(self.config.num_actions)

        self.K = int(K) if K is not None else int(self.config.traj_length)
        assert 1 <= self.K <= int(self.config.traj_length), (
            f"K={self.K} must satisfy 1 <= K <= traj_length={self.config.traj_length}."
        )

        self.mdp = MaskedDP(obs_shape[0], action_shape[0], self.config).to(device)

        if path is not None:
            self.mdp.load_state_dict(payload["model"])
            print("[BCTEvalAgent] Weights loaded.")

        for param in self.mdp.parameters():
            param.requires_grad = False
        self.mdp.eval()

        n_params = sum(p.numel() for p in self.mdp.parameters())
        print(f"[BCTEvalAgent] Parameters: {n_params:,} (all frozen)")
        print(f"[BCTEvalAgent] K={self.K} | num_actions={self.num_actions} | "
              f"temperature={self.temperature} | sample={self.sample}")

        self._obs_buffer = None
        self._action_buffer = None

    def reset(self):
        self._obs_buffer = deque(maxlen=self.K)
        self._action_buffer = deque(maxlen=self.K - 1)

    def eval(self):
        self.mdp.eval()

    def _forward(self):
        """Single-stream encoder + decoder over the current window. Returns (K, num_actions)."""
        T = self.K
        D = self.mdp.n_embd
        n_s = len(self._obs_buffer)
        n_a = len(self._action_buffer)
        len_keep = n_s + n_a                       # real tokens = contiguous prefix

        # --- embed real tokens ---
        obs_np = np.stack(list(self._obs_buffer))                    # (n_s, H, W, C)
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        s_emb = self.mdp._embed_states(obs_t)                        # (1, n_s, D)

        x_keep = s_emb.new_empty(1, len_keep, D)
        x_keep[:, 0::2] = s_emb                                      # states at even prefix positions
        if n_a > 0:
            act_np = np.stack(list(self._action_buffer))             # (n_a,) ints
            act_t = torch.as_tensor(act_np, dtype=torch.long, device=self.device).unsqueeze(0)
            a_emb = self.mdp.action_embed(act_t)                     # (1, n_a, D)
            x_keep[:, 1::2] = a_emb                                  # actions at odd prefix positions

        # positional embedding (prefix 0..len_keep-1) + encoder (all real, no padding)
        x_keep = x_keep + self.mdp.pos_embed[:, :len_keep, :]
        for blk in self.mdp.encoder_blocks:
            x_keep = blk(x_keep, self.mdp.attn_mask)
        x_keep = self.mdp.encoder_norm(x_keep)

        # decoder: identity restore (kept prefix + appended mask tokens = natural order)
        ids_restore = torch.arange(2 * T, device=self.device).unsqueeze(0)
        _, pred_a = self.mdp.forward_decoder(x_keep, ids_restore, valid_tok=None)   # (1, K, num_actions)
        return pred_a[0]                                             # (K, num_actions)

    def _forward_for_attr(self, tar_layer, tmp_att=None, capture_att=False):
        """
        Same pipeline as _forward(), but threads the attention override hooks through
        the encoder_blocks at tar_layer.
        """
        T = self.K
        D = self.mdp.n_embd
        n_s = len(self._obs_buffer)
        n_a = len(self._action_buffer)
        len_keep = n_s + n_a

        obs_np = np.stack(list(self._obs_buffer))                    # (n_s, H, W, C)
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        s_emb = self.mdp._embed_states(obs_t)                        # (1, n_s, D)

        x_keep = s_emb.new_empty(1, len_keep, D)
        x_keep[:, 0::2] = s_emb
        if n_a > 0:
            act_np = np.stack(list(self._action_buffer))             # (n_a,) ints
            act_t = torch.as_tensor(act_np, dtype=torch.long, device=self.device).unsqueeze(0)
            a_emb = self.mdp.action_embed(act_t)                     # (1, n_a, D)
            x_keep[:, 1::2] = a_emb

        x_keep = x_keep + self.mdp.pos_embed[:, :len_keep, :]

        need_att = capture_att or (tmp_att is not None)
        att = None
        for layer_index, blk in enumerate(self.mdp.encoder_blocks):
            if layer_index == tar_layer:
                out = blk(x_keep, self.mdp.attn_mask, tmp_att=tmp_att, return_att=need_att)
                x_keep, att = out if need_att else (out, None)
            else:
                x_keep = blk(x_keep, self.mdp.attn_mask)
        x_keep = self.mdp.encoder_norm(x_keep)

        ids_restore = torch.arange(2 * T, device=self.device).unsqueeze(0)
        _, pred_a = self.mdp.forward_decoder(x_keep, ids_restore, valid_tok=None)   # (1, K, num_actions)
        return pred_a[0], att                                        # (K, num_actions), ...

    def _embed_state_stream(self, obs_list, n_s, override_index, override_batch, B):
        """
        Embeds the current obs-buffer contents into (B, n_s, D).

        Read-only over `self._obs_buffer`. If `override_batch` is None, embeds
        the buffer as-is with B=1. Otherwise, `override_batch` replaces the frame
        at `override_index`; the remaining frames are embedded once and
        broadcasted to all B rows.
        """
        D = self.mdp.n_embd

        if override_batch is None:
            obs_np = np.stack(obs_list, axis=0)                                # (n_s, H, W, C)
            obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.mdp._embed_states(obs_t)                               # (1, n_s, D)

        override_t = torch.as_tensor(override_batch, dtype=torch.float32, device=self.device)  # (B, H, W, C)
        override_embed = self.mdp._embed_states(override_t)                    # (B, D)

        s_emb = override_embed.new_empty(B, n_s, D)
        s_emb[:, override_index, :] = override_embed

        intact_positions = [i for i in range(n_s) if i != override_index]
        if intact_positions:
            intact_np = np.stack([obs_list[i] for i in intact_positions], axis=0)  # (n_s-1, H, W, C)
            intact_t = torch.as_tensor(intact_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            intact_embed = self.mdp._embed_states(intact_t)                    # (1, n_s-1, D)
            s_emb[:, intact_positions, :] = intact_embed.expand(B, -1, -1)

        return s_emb

    def _forward_batched(self, override_index=None, override_batch=None, B=1):
        """
        Batched variant of `_forward()`. Read-only over the internal buffers.
        With `override_batch=None` it reproduces the standard `_forward()` (B=1).
        Otherwise, `override_batch` independently replaces the frame at
        `override_index` per batch row.

        Returns (B, K, num_actions) raw logits (no temperature applied).
        """
        T = self.K
        D = self.mdp.n_embd
        n_s = len(self._obs_buffer)
        n_a = len(self._action_buffer)
        len_keep = n_s + n_a

        obs_list = list(self._obs_buffer)
        s_emb = self._embed_state_stream(obs_list, n_s, override_index, override_batch, B)  # (B, n_s, D)

        x_keep = s_emb.new_empty(B, len_keep, D)
        x_keep[:, 0::2] = s_emb
        if n_a > 0:
            act_np = np.stack(list(self._action_buffer))             # (n_a,) ints
            act_t = torch.as_tensor(act_np, dtype=torch.long, device=self.device).unsqueeze(0).expand(B, -1)
            a_emb = self.mdp.action_embed(act_t)                     # (B, n_a, D)
            x_keep[:, 1::2] = a_emb

        x_keep = x_keep + self.mdp.pos_embed[:, :len_keep, :]
        for blk in self.mdp.encoder_blocks:
            x_keep = blk(x_keep, self.mdp.attn_mask)                 # broadcasts against B, no expand needed
        x_keep = self.mdp.encoder_norm(x_keep)

        ids_restore = torch.arange(2 * T, device=self.device).unsqueeze(0).expand(B, -1)
        _, pred_a = self.mdp.forward_decoder(x_keep, ids_restore, valid_tok=None)   # (B, K, num_actions)
        return pred_a

    def logits_for(self, obs_override=None, override_index=-1):
        """
        Returns raw logits (num_actions,) for the current buffer contents,
        optionally replacing ONE frame in the observation buffer.

        obs_override:   (H, W, C) or (N, H, W, C) -- perturbed frame(s), in [0, 255].
        override_index: Position in the obs buffer to replace.
                        -1 for the current frame (spatial saliency) or
                        j < n_s-1 for past frames (temporal saliency).
        """
        n_s = len(self._obs_buffer)
        assert n_s > 0, "logits_for() requires a non-empty obs buffer (call act() at least once after reset())."

        idx = override_index if override_index >= 0 else n_s + override_index
        assert 0 <= idx < n_s, f"override_index={override_index} out of range for n_s={n_s}."

        if obs_override is None:
            with torch.no_grad():
                pred_a = self._forward_batched(override_index=None, override_batch=None, B=1)
            return pred_a[0, n_s - 1]                                          # (num_actions,)

        override_arr = np.asarray(obs_override)
        single = override_arr.ndim == 3
        if single:
            override_arr = override_arr[None]                                 # (1, H, W, C)
        B = override_arr.shape[0]

        with torch.no_grad():
            pred_a = self._forward_batched(override_index=idx, override_batch=override_arr, B=B)  # (B, K, num_actions)

        logits = pred_a[:, n_s - 1]                                            # (B, num_actions)
        return logits[0] if single else logits

    def act(self, obs):
        obs_frame = obs[0] if obs.ndim == 4 else obs   # (H, W, C)
        self._obs_buffer.append(obs_frame)

        with torch.no_grad():
            pred_a = self._forward()              # (K, num_actions) logits

        n_s = len(self._obs_buffer)
        logits_t = pred_a[n_s - 1]                 # current-action position, always masked

        if self.sample:
            dist = torch.distributions.Categorical(logits=logits_t / self.temperature)
            action_int = int(dist.sample().item())
        else:
            action_int = int(torch.argmax(logits_t).item())

        self._action_buffer.append(action_int)
        return np.array([action_int], dtype=np.int64)
