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

    def _forward(self):
        """
        Same streams -> forward_fusion -> forward_decoder pipeline as
        mdp_return.py::_forward, with T_total fixed to self.K (not T_cond+T_pred
        there is no separate warmup/planning phase here, every step is the same).
        """
        T = self.K
        enc_D = self.mdp.enc_n_embd
        D = self.mdp.n_embd
        n_s = len(self._obs_buffer)
        n_a = len(self._action_buffer)

        pos_s = torch.arange(n_s, device=self.device) * 2
        pos_a = torch.arange(n_a, device=self.device) * 2 + 1
        pos_embed = self.mdp.pos_embed

        obs_np = np.stack(list(self._obs_buffer))                    # (n_s, H, W, C
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

        s_attn = torch.ones(1, 1, n_s, n_s, device=self.device)
        x_s = s_real
        for blk in self.mdp.state_encoder_blocks:
            x_s = blk(x_s, s_attn)
        x_s = self.mdp.state_encoder_norm(x_s)
        x_s = self.mdp.state_proj(x_s)
        x_s = self.mdp.state_adapter(x_s)

        if n_a > 0:
            a_attn = torch.ones(1, 1, n_a, n_a, device=self.device)
            x_a = a_real
            for blk in self.mdp.action_encoder_blocks:
                x_a = blk(x_a, a_attn)
            x_a = self.mdp.action_encoder_norm(x_a)
            x_a = self.mdp.action_proj(x_a)
            x_a = self.mdp.action_adapter(x_a)
        else:
            x_a = x_s.new_empty(1, 0, D)

        real_positions = torch.cat([pos_s, pos_a], dim=0)
        _, sort_idx = torch.sort(real_positions)
        ids_keep = real_positions[sort_idx].unsqueeze(0)

        x_fused = self.mdp.forward_fusion(
            x_s, x_a, ids_keep, s_pad_mask=None, a_pad_mask=a_pad_mask,
        )

        total_len = 2 * T
        mask = torch.ones(total_len, dtype=torch.bool, device=self.device)
        mask[real_positions] = False
        masked_positions = torch.arange(total_len, device=self.device)[mask]
        ids_shuffle = torch.cat([ids_keep[0], masked_positions], dim=0)
        ids_restore = torch.argsort(ids_shuffle).unsqueeze(0)

        # valid_il=None: closed-loop has no concept of padding. Every position
        # after the current one is legitimate future not yet observed.
        _, pred_a = self.mdp.forward_decoder(x_fused, ids_restore, valid_il=None)   # (1, K, num_actions) logits
        return pred_a[0]                                             # (K, num_actions)

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
