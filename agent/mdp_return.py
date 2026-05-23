import numpy as np
from collections import deque

import torch
import torch.nn as nn

from agent.mdp import MaskedDPMultimodal
import utils


class MaskingEvalAgentMultimodal:
    """
    Eval agent for the hierarchical MaskedDPMultimodal model.

    Protocol identical to MaskingEvalAgent (unimodal):
        Warmup  (T_cond steps): context grows 1 -> T_cond, predict 1 action per step.
        Eval    (rest of episode): context fixed at T_cond, predict T_pred actions per
                cycle, execute replan_freq of them before replanning.
    """

    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        T_cond,
        T_pred,
        replan_freq,
        path=None,
        transformer_cfg=None,
        **kwargs,
    ):
        self.device = device
        self.action_dim = action_shape[0]
        self.T_cond = T_cond
        self.T_pred = T_pred
        self.T_total = T_cond + T_pred
        self.replan_freq = replan_freq

        assert T_pred % replan_freq == 0, \
            f"replan_freq ({replan_freq}) must divide T_pred ({T_pred})"

        if path is not None:
            print(f"[MaskingEvalAgentMultimodal] Loading snapshot: {path}")
            payload = torch.load(path, map_location=device)
            self.config = payload["cfg"]
        else:
            assert transformer_cfg is not None
            self.config = transformer_cfg

        self.mdp = MaskedDPMultimodal(
            obs_shape[0], action_shape[0], self.config
        ).to(device)

        if path is not None:
            self.mdp.load_state_dict(payload["model"])
            print(f"[MaskingEvalAgentMultimodal] Weights loaded.")

        for param in self.mdp.parameters():
            param.requires_grad = False
        self.mdp.eval()

        assert self.T_total == self.config.traj_length, (
            f"T_cond + T_pred = {self.T_total} must equal "
            f"traj_length = {self.config.traj_length}"
        )

        n_params = sum(p.numel() for p in self.mdp.parameters())
        print(f"[MaskingEvalAgentMultimodal] Parameters: {n_params:,} (all frozen)")
        print(f"[MaskingEvalAgentMultimodal] T_cond={T_cond} | T_pred={T_pred} | "
              f"replan_freq={replan_freq} | T_total={self.T_total}")

        self._obs_buffer    = None
        self._action_buffer = None
        self._warmup_done   = False
        self._action_queue  = []

    def reset(self):
        self._obs_buffer    = deque(maxlen=self.T_cond)
        self._action_buffer = deque(maxlen=self.T_cond)
        self._warmup_done   = False
        self._action_queue  = []

    @property
    def in_warmup(self):
        return not self._warmup_done

    def _forward(self):
        T     = self.T_total
        D     = self.config.n_embd
        enc_D = self.mdp.enc_n_embd
        n_s   = len(self._obs_buffer)
        n_a   = len(self._action_buffer)

        # Pos embeddings
        pos_s = torch.arange(n_s, device=self.device) * 2
        pos_a = torch.arange(n_a, device=self.device) * 2 + 1
        pos_embed = self.mdp.pos_embed                     # (1, max_len, enc_D)

        # Embed state tokens
        obs_np = np.stack(list(self._obs_buffer))
        obs_t  = torch.as_tensor(
            obs_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0)                                      # (1, n_s, ...)
        s_real = self.mdp._embed_states(obs_t)             # (1, n_s, enc_D)
        s_real = s_real + pos_embed[:, pos_s, :]

        # Embed action tokens
        if n_a > 0:
            act_np = np.stack(list(self._action_buffer))
            act_t  = torch.as_tensor(
                act_np, dtype=torch.float32, device=self.device
            ).unsqueeze(0)                                  # (1, n_a, action_dim)
            a_real = self.mdp.action_embed(act_t)          # (1, n_a, enc_D)
            a_real = a_real + pos_embed[:, pos_a, :]
            a_pad_mask = torch.zeros(1, n_a, dtype=torch.bool, device=self.device)
        else:
            a_real     = s_real.new_empty(1, 0, enc_D)
            a_pad_mask = torch.ones(1, 0, dtype=torch.bool, device=self.device)

        # State encoder
        s_attn = torch.ones(1, 1, n_s, n_s, device=self.device)
        x_s = s_real
        for blk in self.mdp.state_encoder_blocks:
            x_s = blk(x_s, s_attn)
        x_s = self.mdp.state_encoder_norm(x_s)
        x_s = self.mdp.state_proj(x_s)                    # (1, n_s, D)
        x_s = self.mdp.state_adapter(x_s)

        # Action encoder
        if n_a > 0:
            a_attn = torch.ones(1, 1, n_a, n_a, device=self.device)
            x_a = a_real
            for blk in self.mdp.action_encoder_blocks:
                x_a = blk(x_a, a_attn)
            x_a = self.mdp.action_encoder_norm(x_a)
            x_a = self.mdp.action_proj(x_a)               # (1, n_a, D)
            x_a = self.mdp.action_adapter(x_a)
        else:
            x_a = x_s.new_empty(1, 0, D)

        # ids_keep: real token positions in ascending interleaved order
        real_positions = torch.cat([pos_s, pos_a], dim=0)  # (n_s + n_a,)
        _, sort_idx    = torch.sort(real_positions)
        ids_keep = real_positions[sort_idx].unsqueeze(0)   # (1, n_real)

        # Fusion
        x_fused = self.mdp.forward_fusion(
            x_s, x_a, ids_keep,
            s_pad_mask=None,
            a_pad_mask=a_pad_mask,
        )                                                   # (1, n_real, D)

        # Build ids_restore for forward_decoder
        total_len = 2 * T
        masked_positions = torch.tensor(
            [p for p in range(total_len)
             if p not in set(real_positions.tolist())],
            device=self.device, dtype=torch.long
        )
        ids_shuffle = torch.cat([ids_keep[0], masked_positions], dim=0)  # (2*T,)
        ids_restore = torch.argsort(ids_shuffle).unsqueeze(0)            # (1, 2*T)

        # Decoder
        _, pred_a = self.mdp.forward_decoder(x_fused, ids_restore)  # (1, T, action_dim)

        return pred_a[0]                                             # (T, action_dim)

    def act(self, obs):
        if obs.ndim == 3:
            obs = obs.transpose(1, 2, 0)
        self._obs_buffer.append(obs)

        if len(self._action_queue) > 0:
            action = self._action_queue.pop(0)
            self._action_buffer.append(action)
            return action

        with torch.no_grad():
            pred_a = self._forward()                               # (T_total, action_dim)

            if self.in_warmup:
                ctx_len = len(self._obs_buffer)
                action  = pred_a[ctx_len - 1].cpu().numpy()
                if ctx_len >= self.T_cond:
                    self._warmup_done = True
            else:
                actions = pred_a[
                    self.T_cond : self.T_cond + self.replan_freq
                ].cpu().numpy()
                action = actions[0]
                self._action_queue = list(actions[1:])

        self._action_buffer.append(action)
        return action