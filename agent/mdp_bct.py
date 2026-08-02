import numpy as np
import torch
import torch.nn as nn
from collections import deque

from agent.mdp import MaskedDPMultimodal


class BCTEvalAgentMultimodal:
    """
    Closed-loop, per-timestep BC eval agent for the multimodal MaskedDPMultimodal model,
    matching the evaluation of gen_dgrl's `eval_DT_agent` with model_type='naive'
    (BCT): act every single step from the current sliding window, no returns/rtg, no
    open-loop planning horizon.

    Protocol per step t:
      1. Append the new observation s_t to the obs window (maxlen=K).
      2. The action buffer (maxlen=K-1) does NOT yet contain a_t -> a_t stays masked
         by construction
      3. _forward() runs streams + forward_fusion + forward_decoder over the window.
      4. Read the prediction at the current-action position: pred_a[n_s - 1].
      5. Decode to a concrete action (sample at temperature, or argmax), execute,
         append to the action buffer, repeat.
    """

    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        K=None,                 # sliding window length; default = transformer_cfg.traj_length
        temperature=1.0,
        sample=True,            # True: Categorical(logits/temperature).sample() (matches Meta's eval_DT_agent default)
                                 # False: argmax
        path=None,
        transformer_cfg=None,
        **kwargs,
    ):
        self.device = device
        self.temperature = temperature
        self.sample = sample

        if path is not None:
            print(f"[BCTEvalAgentMultimodal] Loading snapshot: {path}")
            payload = torch.load(path, map_location=device)
            self.config = payload["cfg"]
        else:
            assert transformer_cfg is not None, "Must provide either `path` or `transformer_cfg`."
            self.config = transformer_cfg

        assert bool(getattr(self.config, "discrete_actions", False)), (
            "BCTEvalAgentMultimodal onlu supports discrete-actions"
        )
        self.num_actions = int(self.config.num_actions)

        self.K = int(K) if K is not None else int(self.config.traj_length)
        assert 1 <= self.K <= int(self.config.traj_length), (
            f"K={self.K} must satisfy 1 <= K <= traj_length={self.config.traj_length}."
        )

        self.mdp = MaskedDPMultimodal(
            obs_shape[0], action_shape[0], self.config
        ).to(device)

        if path is not None:
            self.mdp.load_state_dict(payload["model"])
            print("[BCTEvalAgentMultimodal] Weights loaded.")

        for param in self.mdp.parameters():
            param.requires_grad = False
        self.mdp.eval()

        n_params = sum(p.numel() for p in self.mdp.parameters())
        print(f"[BCTEvalAgentMultimodal] Parameters: {n_params:,} (all frozen)")
        print(f"[BCTEvalAgentMultimodal] K={self.K} | num_actions={self.num_actions} | "
              f"temperature={self.temperature} | sample={self.sample}")

        self._obs_buffer = None
        self._action_buffer = None

    def reset(self):
        self._obs_buffer = deque(maxlen=self.K)
        self._action_buffer = deque(maxlen=self.K - 1)

    def eval(self):
        self.mdp.eval()

    def _embed_and_encode(self, enc_target=None, enc_tmp_att=None, enc_capture_att=False):
        """
        Shared plumbing for `_forward()`/`_forward_for_attr()`/`_forward_for_attr_encoder()`:
        embeds the obs/action buffers, runs `state_encoder_blocks`/`action_encoder_blocks`,
        builds `ids_keep`. Read-only over the buffers.

        `enc_target`: None (both encoder loops run unhooked) or `(stream, layer_index)`
        with `stream in ('s', 'a')` -- that stream's loop becomes `enumerate` + intervene
        at `layer_index` (same pattern as `forward_fusion`'s `tar_layer`); the other
        stream's loop is untouched. `enc_tmp_att`/`enc_capture_att`: override / capture-only
        at that layer.

        Returns (x_s, x_a, ids_keep, a_pad_mask, real_positions, enc_att). `enc_att` is
        None unless `enc_target` is set and `enc_tmp_att`/`enc_capture_att` requested it.
        """
        enc_D = self.mdp.enc_n_embd
        D = self.mdp.n_embd
        n_s = len(self._obs_buffer)
        n_a = len(self._action_buffer)

        pos_s = torch.arange(n_s, device=self.device) * 2
        pos_a = torch.arange(n_a, device=self.device) * 2 + 1
        pos_embed = self.mdp.pos_embed

        obs_np = np.stack(list(self._obs_buffer))                    # (n_s, H, W, C)
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        s_real = self.mdp._embed_states(obs_t)                       # (1, n_s, enc_D)
        s_real = s_real + pos_embed[:, pos_s, :]

        if n_a > 0:
            act_np = np.stack(list(self._action_buffer))             # (n_a,) ints
            act_t = torch.as_tensor(act_np, dtype=torch.long, device=self.device).unsqueeze(0)
            a_real = self.mdp.action_embed(act_t)                     # (1, n_a, enc_D)
            a_real = a_real + pos_embed[:, pos_a, :]
            a_pad_mask = torch.zeros(1, n_a, dtype=torch.bool, device=self.device)
        else:
            a_real = s_real.new_empty(1, 0, enc_D)
            a_pad_mask = torch.ones(1, 0, dtype=torch.bool, device=self.device)

        enc_stream, enc_layer = enc_target if enc_target is not None else (None, None)
        enc_att = None
        enc_need_att = enc_capture_att or (enc_tmp_att is not None)

        s_attn = torch.ones(1, 1, n_s, n_s, device=self.device)
        x_s = s_real
        if enc_stream == 's':
            for layer_index, blk in enumerate(self.mdp.state_encoder_blocks):
                if layer_index == enc_layer:
                    out = blk(x_s, s_attn, tmp_att=enc_tmp_att, return_att=enc_need_att)
                    x_s, enc_att = out if enc_need_att else (out, None)
                else:
                    x_s = blk(x_s, s_attn)
        else:
            for blk in self.mdp.state_encoder_blocks:
                x_s = blk(x_s, s_attn)
        x_s = self.mdp.state_encoder_norm(x_s)
        x_s = self.mdp.state_proj(x_s)

        if n_a > 0:
            a_attn = torch.ones(1, 1, n_a, n_a, device=self.device)
            x_a = a_real
            if enc_stream == 'a':
                for layer_index, blk in enumerate(self.mdp.action_encoder_blocks):
                    if layer_index == enc_layer:
                        out = blk(x_a, a_attn, tmp_att=enc_tmp_att, return_att=enc_need_att)
                        x_a, enc_att = out if enc_need_att else (out, None)
                    else:
                        x_a = blk(x_a, a_attn)
            else:
                for blk in self.mdp.action_encoder_blocks:
                    x_a = blk(x_a, a_attn)
            x_a = self.mdp.action_encoder_norm(x_a)
            x_a = self.mdp.action_proj(x_a)
        else:
            x_a = x_s.new_empty(1, 0, D)

        real_positions = torch.cat([pos_s, pos_a], dim=0)
        _, sort_idx = torch.sort(real_positions)
        ids_keep = real_positions[sort_idx].unsqueeze(0)

        return x_s, x_a, ids_keep, a_pad_mask, real_positions, enc_att

    def _build_ids_restore(self, real_positions, ids_keep):
        total_len = 2 * self.K
        mask = torch.ones(total_len, dtype=torch.bool, device=self.device)
        mask[real_positions] = False
        masked_positions = torch.arange(total_len, device=self.device)[mask]
        ids_shuffle = torch.cat([ids_keep[0], masked_positions], dim=0)
        return torch.argsort(ids_shuffle).unsqueeze(0)

    def _forward(self):
        """
        Same streams -> forward_fusion -> forward_decoder pipeline as
        mdp_return.py::_forward, with T_total fixed to self.K (not T_cond+T_pred
        there is no separate warmup/planning phase here, every step is the same).
        """
        x_s, x_a, ids_keep, a_pad_mask, real_positions, _ = self._embed_and_encode()

        x_fused = self.mdp.forward_fusion(
            x_s, x_a, ids_keep, s_pad_mask=None, a_pad_mask=a_pad_mask,
        )
        ids_restore = self._build_ids_restore(real_positions, ids_keep)

        # valid_il=None: closed-loop has no concept of padding. Every position
        # after the current one is legitimate future not yet observed.
        _, pred_a = self.mdp.forward_decoder(x_fused, ids_restore, valid_il=None)   # (1, K, num_actions) logits
        return pred_a[0]                                             # (K, num_actions)

    def _forward_for_attr(self, tar_layer, tmp_att_s=None, tmp_att_a=None, capture_att=False):
        """
        Same pipeline as `_forward()`, but threads the fusion-layer attention override
        hooks through `forward_fusion`.

        Returns (pred_a, att_s, att_a) with pred_a of shape (K, num_actions).
        """
        x_s, x_a, ids_keep, a_pad_mask, real_positions, _ = self._embed_and_encode()

        need_att = capture_att or (tmp_att_s is not None) or (tmp_att_a is not None)
        out = self.mdp.forward_fusion(
            x_s, x_a, ids_keep, s_pad_mask=None, a_pad_mask=a_pad_mask,
            tar_layer=tar_layer, tmp_att_s=tmp_att_s, tmp_att_a=tmp_att_a, capture_att=capture_att,
        )
        if need_att:
            x_fused, att_s, att_a = out
        else:
            x_fused, att_s, att_a = out, None, None
        ids_restore = self._build_ids_restore(real_positions, ids_keep)

        _, pred_a = self.mdp.forward_decoder(x_fused, ids_restore, valid_il=None)   # (1, K, num_actions)
        return pred_a[0], att_s, att_a                                              # (K, num_actions), ...

    def _forward_for_attr_encoder(self, stream, tar_layer, tmp_att=None, capture_att=False):
        """
        Same pipeline as `_forward()`, but threads an attention override hook through
        `state_encoder_blocks`/`action_encoder_blocks` instead of fusion. Fusion runs
        with its real, unhooked attention.

        Returns (pred_a, enc_att) with pred_a of shape (K, num_actions).
        """
        assert stream in ('s', 'a'), f"stream must be 's' or 'a', got {stream!r}"
        if stream == 'a':
            assert len(self._action_buffer) > 0, (
                "no action tokens in the buffer yet -- action-encoder attribution "
                "needs at least one action (not the very first step of an episode)."
            )

        x_s, x_a, ids_keep, a_pad_mask, real_positions, enc_att = self._embed_and_encode(
            enc_target=(stream, tar_layer), enc_tmp_att=tmp_att, enc_capture_att=capture_att,
        )

        x_fused = self.mdp.forward_fusion(
            x_s, x_a, ids_keep, s_pad_mask=None, a_pad_mask=a_pad_mask,
        )
        ids_restore = self._build_ids_restore(real_positions, ids_keep)

        _, pred_a = self.mdp.forward_decoder(x_fused, ids_restore, valid_il=None)   # (1, K, num_actions)
        return pred_a[0], enc_att                                                   # (K, num_actions), ...

    def _embed_state_stream(self, obs_list, n_s, override_index, override_batch, B):
        """
        Embeds the current obs-buffer contents into (B, n_s, enc_D).

        Read-only over `self._obs_buffer`. If `override_batch` is None, embeds 
        the buffer as-is with B=1. Otherwise, `override_batch` replaces the frame 
        at `override_index`; the remaining frames are embedded ONCE and broadcasted 
        to all B rows (avoiding redundant computation per perturbation).
        """
        enc_D = self.mdp.enc_n_embd

        if override_batch is None:
            obs_np = np.stack(obs_list, axis=0)                                # (n_s, H, W, C)
            obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.mdp._embed_states(obs_t)                               # (1, n_s, enc_D)

        override_t = torch.as_tensor(override_batch, dtype=torch.float32, device=self.device)  # (B, H, W, C)
        override_embed = self.mdp._embed_states(override_t)                    # (B, enc_D)

        s_real = override_embed.new_empty(B, n_s, enc_D)
        s_real[:, override_index, :] = override_embed

        intact_positions = [i for i in range(n_s) if i != override_index]
        if intact_positions:
            intact_np = np.stack([obs_list[i] for i in intact_positions], axis=0)  # (n_s-1, H, W, C)
            intact_t = torch.as_tensor(intact_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            intact_embed = self.mdp._embed_states(intact_t)                    # (1, n_s-1, enc_D)
            s_real[:, intact_positions, :] = intact_embed.expand(B, -1, -1)

        return s_real

    def _forward_batched(self, override_index=None, override_batch=None, B=1):
        """
        Batched variant of `_forward()`. Read-only over the internal buffers.
        With `override_batch=None` it reproduces the standard `_forward()` (B=1). 
        Otherwise, `override_batch` independently replaces the frame at 
        `override_index` per batch row.

        Returns (B, K, num_actions) raw logits (no temperature applied).
        """
        T = self.K
        enc_D = self.mdp.enc_n_embd
        D = self.mdp.n_embd
        n_s = len(self._obs_buffer)
        n_a = len(self._action_buffer)

        pos_s = torch.arange(n_s, device=self.device) * 2
        pos_a = torch.arange(n_a, device=self.device) * 2 + 1
        pos_embed = self.mdp.pos_embed

        obs_list = list(self._obs_buffer)
        s_real = self._embed_state_stream(obs_list, n_s, override_index, override_batch, B)  # (B, n_s, enc_D)
        s_real = s_real + pos_embed[:, pos_s, :]

        if n_a > 0:
            act_np = np.stack(list(self._action_buffer))             # (n_a,) ints
            act_t = torch.as_tensor(act_np, dtype=torch.long, device=self.device).unsqueeze(0).expand(B, -1)
            a_real = self.mdp.action_embed(act_t)                     # (B, n_a, enc_D)
            a_real = a_real + pos_embed[:, pos_a, :]
            a_pad_mask = torch.zeros(B, n_a, dtype=torch.bool, device=self.device)
        else:
            a_real = s_real.new_empty(B, 0, enc_D)
            a_pad_mask = torch.ones(B, 0, dtype=torch.bool, device=self.device)

        # Broadcasts against (B, n_head, n_s, n_s) attention scores 
        # inside CausalSelfAttention -- no need to expand to B.
        s_attn = torch.ones(1, 1, n_s, n_s, device=self.device)
        x_s = s_real
        for blk in self.mdp.state_encoder_blocks:
            x_s = blk(x_s, s_attn)
        x_s = self.mdp.state_encoder_norm(x_s)
        x_s = self.mdp.state_proj(x_s)

        if n_a > 0:
            a_attn = torch.ones(1, 1, n_a, n_a, device=self.device)
            x_a = a_real
            for blk in self.mdp.action_encoder_blocks:
                x_a = blk(x_a, a_attn)
            x_a = self.mdp.action_encoder_norm(x_a)
            x_a = self.mdp.action_proj(x_a)
        else:
            x_a = x_s.new_empty(B, 0, D)

        # real_positions/ids_keep/ids_restore are identical for every batch row.
        # Computed once, then expanded to B for gather ops in forward_fusion/decoder.
        real_positions = torch.cat([pos_s, pos_a], dim=0)
        _, sort_idx = torch.sort(real_positions)
        ids_keep = real_positions[sort_idx].unsqueeze(0).expand(B, -1)

        x_fused = self.mdp.forward_fusion(
            x_s, x_a, ids_keep, s_pad_mask=None, a_pad_mask=a_pad_mask,
        )

        total_len = 2 * T
        mask = torch.ones(total_len, dtype=torch.bool, device=self.device)
        mask[real_positions] = False
        masked_positions = torch.arange(total_len, device=self.device)[mask]
        ids_shuffle = torch.cat([ids_keep[0], masked_positions], dim=0)
        ids_restore = torch.argsort(ids_shuffle).unsqueeze(0).expand(B, -1)

        # valid_il=None: closed-loop prediction expects legitimate future steps
        _, pred_a = self.mdp.forward_decoder(x_fused, ids_restore, valid_il=None)   # (B, K, num_actions)
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
