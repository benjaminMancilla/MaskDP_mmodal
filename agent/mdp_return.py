import numpy as np
from collections import deque

import torch
import torch.nn as nn

from agent.mdp import MaskedDP
import utils


class MaskingEvalAgent:
    """
    Behavioral Cloning (Online RL).

    Protocol:
    Warmup  (T_cond steps): context grows 1 -> T_cond, predict 1 action per step.
    Eval    (rest of episode): context fixed at T_cond, predict T_pred actions per cycle,
            execute replan_freq of them before replanning. Return is measured.
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
            print(f"[MaskingEvalAgent] Loading snapshot: {path}")
            payload = torch.load(path, map_location=device)
            self.config = payload["cfg"]
        else:
            assert transformer_cfg is not None
            self.config = transformer_cfg

        self.mdp = MaskedDP(obs_shape[0], action_shape[0], self.config).to(device)

        if path is not None:
            self.mdp.load_state_dict(payload["model"])
            print(f"[MaskingEvalAgent] Weights loaded.")

        # Fully frozen
        for param in self.mdp.parameters():
            param.requires_grad = False
        self.mdp.eval()

        #assert self.T_total == self.config.traj_length, (
        #    f"T_cond + T_pred = {self.T_total} must equal "
        #    f"traj_length = {self.config.traj_length}"
        #)

        n_params = sum(p.numel() for p in self.mdp.parameters())
        print(f"[MaskingEvalAgent] Parameters: {n_params:,} (all frozen)")
        print(f"[MaskingEvalAgent] T_cond={T_cond} | T_pred={T_pred} | "
              f"replan_freq={replan_freq} | T_total={self.T_total}")

        # Online buffers (populated during eval)
        self._obs_buffer    = None
        self._action_buffer = None
        self._warmup_done   = False
        self._action_queue  = []    # pending predicted actions from last cycle


    # Episode management
    def reset(self):
        self._obs_buffer    = deque(maxlen=self.T_cond)
        self._action_buffer = deque(maxlen=self.T_cond)
        self._warmup_done   = False
        self._action_queue  = []

    @property
    def in_warmup(self):
        return not self._warmup_done


    def _forward(self):
        T      = self.T_total
        D      = self.config.n_embd
        n_s    = len(self._obs_buffer)
        n_a    = len(self._action_buffer)

        # Embed visible tokens 
        obs_np = np.stack(list(self._obs_buffer))          # (n_s, ...) — H,W,C or obs_dim
        obs_t  = torch.as_tensor(
            obs_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0)                                      # (1, n_s, ...)
        s_real = self.mdp._embed_states(obs_t)             # (1, n_s, D)

        if n_a > 0:
            act_np = np.stack(list(self._action_buffer))   # (n_a, action_dim)
            act_t  = torch.as_tensor(
                act_np, dtype=torch.float32, device=self.device
            ).unsqueeze(0)                                  # (1, n_a, action_dim)
            a_real = self.mdp.action_embed(act_t)          # (1, n_a, D)
        else:
            a_real = s_real.new_empty(1, 0, D)

        # Original interleaved positions of each real token
        pos_s = torch.arange(n_s, device=self.device) * 2          # [0, 2, ..., 2*(n_s-1)]
        pos_a = torch.arange(n_a, device=self.device) * 2 + 1      # [1, 3, ..., 2*n_a-1]

        # Add positional embeddings
        pos_embed = self.mdp.pos_embed                             # (1, max_len, D)
        s_real = s_real + pos_embed[:, pos_s, :]
        a_real = a_real + pos_embed[:, pos_a, :]

        # Interleave real tokens ordered by original position
        real_positions = torch.cat([pos_s, pos_a], dim=0)          # (n_s + n_a,)
        real_tokens    = torch.cat([s_real, a_real], dim=1)        # (1, n_s + n_a, D)

        sorted_positions, sort_idx = torch.sort(real_positions)    # ascending
        x_keep = real_tokens[:, sort_idx, :]                       # (1, n_real, D)
        n_real = x_keep.shape[1]                                   # n_s + n_a

        # Encoder sees only real tokens
        attn_mask_full = self.mdp.attn_mask                        # (1, 1, max_len, max_len)
        x = x_keep
        for blk in self.mdp.encoder_blocks:
            x = blk(x, attn_mask_full)
        x = self.mdp.encoder_norm(x)                               # (1, n_real, D)

        # Reconstruct full 2*T sequence; mask_token at unfilled positions
        full = self.mdp.mask_token.expand(1, 2 * T, D).clone()    # (1, 2*T, D)
        full[:, sorted_positions, :] = x

        # Decoder
        s_dec = self.mdp.decoder_state_embed(full[:, 0::2, :])    # (1, T, D)
        a_dec = self.mdp.decoder_action_embed(full[:, 1::2, :])   # (1, T, D)

        x_dec = torch.stack([s_dec, a_dec], dim=2).reshape(1, 2 * T, D)
        x_dec = x_dec + self.mdp.decoder_pos_embed[:, :2 * T, :]

        for blk in self.mdp.decoder_blocks:
            x_dec = blk(x_dec, attn_mask_full)

        pred_a = self.mdp.action_head(x_dec[:, 1::2, :])          # (1, T, action_dim)
        return pred_a[0]                                            # (T, action_dim)


    # Online action selection
    def act(self, obs):
        if obs.ndim == 3:
            obs = obs.transpose(1, 2, 0)
        self._obs_buffer.append(obs)

        # If queued actions exist from a previous prediction batch, use the next one
        if len(self._action_queue) > 0:
            action = self._action_queue.pop(0)
            self._action_buffer.append(action)
            return action

        # Need to predict
        with torch.no_grad():
            pred_a = self._forward()                               # (T_total, action_dim)

            if self.in_warmup:
                # Predict 1 action — at the position of the current (last) state
                ctx_len = len(self._obs_buffer)
                action  = pred_a[ctx_len - 1].cpu().numpy()

                if ctx_len >= self.T_cond:
                    self._warmup_done = True

            else:
                # Predict replan_freq actions from the first masked position (T_cond)
                actions = pred_a[
                    self.T_cond : self.T_cond + self.replan_freq
                ].cpu().numpy()                                    # (replan_freq, action_dim)

                action = actions[0]
                self._action_queue = list(actions[1:])             # queue the rest

        self._action_buffer.append(action)
        return action