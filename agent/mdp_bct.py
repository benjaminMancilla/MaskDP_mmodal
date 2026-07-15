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
