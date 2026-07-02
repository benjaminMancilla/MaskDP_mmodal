import numpy as np
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from agent.mdp import MaskedDPMultimodal
import utils


def _stack_frames(frames: np.ndarray, indices: np.ndarray, k: int) -> np.ndarray:
    stacked = []
    for j in range(k):
        shifted = np.clip(indices - (k - 1 - j), 0, len(frames) - 1)
        stacked.append(frames[shifted])   # (T, H, W, 3)
    return np.concatenate(stacked, axis=-1)   # (T, H, W, k*3)


def _pad_zeros(arr: np.ndarray, n_pad: int) -> np.ndarray:
    pad_shape = (n_pad,) + arr.shape[1:]
    return np.concatenate([arr, np.zeros(pad_shape, dtype=arr.dtype)], axis=0)


class ReconstructionEvalAgent:
    """
    Eval agent that measures reconstruction quality on offline val sequences.

    Supports three modalities:
        "actions" : Continuous actions (discrete_actions=False, e.g. V-D4RL):
                        MSE between predicted and ground-truth actions at masked positions.
                    Discrete actions (discrete_actions=True, e.g. Procgen):
                        Top-1 accuracy between argmax(logits) and ground-truth action id
                        at masked positions.
        "states"  : MSE between predicted and L2-normalised CNN features at masked positions.
                    Target = pixel_encoder(s) / ||pixel_encoder(s)||_2, identical to
                    the normalisation used in forward_loss during training.
                    (Unaffected by discrete_actions — states are never discrete.)
        "both"    : computes both metrics in a single forward pass per episode.
                    Logs to separate W&B projects. total_recon_loss = action + state loss
                    ONLY when actions are continuous (both are losses, lower=better).
                    When actions are discrete, action metric is an accuracy (higher=better)
                    and state metric is an MSE (lower=better).

    Masking schemes
    ---------------
    "second_half"
        Mask the last T//2 timesteps completely (both states and actions).
        First T//2 timesteps are always visible as context.
        In the 2T interleaved sequence: keep positions 0..T-1, mask T..2T-1.

    "random"
        Draw mask_ratio ~ Uniform(mask_ratio list).
        Apply random masking over the full 2T interleaved sequence with a single
        noise tensor — identical mechanism to training (forward_encoder).

    "temporal_split"
        Flexible generalisation of second_half with a configurable split ratio.
        Keep the first `split_ratio` fraction of timesteps visible (context);
        mask the remaining (1 - split_ratio) fraction.
        split_ratio must be in (0, 1]

    "last_action"
        Same split_ratio knob as temporal_split but ALSO reveals the state at
        position `cut` and masks/scores ONLY the action at position `cut`.

    Observation types
    -----------------
    obs_type="states"  : reads episode["observation"] (proprioceptive, float32).
    obs_type="pixels"  : reads episode["pixel_observation"] for the 9-channel custom
                         dataset (individual 3-ch frames) and builds frame_stack-deep
                         stacks on the fly; falls back to episode["observation"] for
                         VD4RL pixel episodes that were renamed by the loader.
    """

    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_shape: Tuple[int],
        device: torch.device,
        T: int,
        masking_scheme: str,
        mask_ratio: List[float],
        modality: str = "actions",
        path: Optional[str] = None,
        transformer_cfg=None,
        split_ratio: float = 0.5,
        obs_type: str = "pixels",
        frame_stack: int = 1,
        has_dummy_transition: bool = True,
        padding_mode: str = "off",
        **kwargs,
    ):
        self.device = device
        self.T = T
        self.action_dim = action_shape[0]
        self.masking_scheme = masking_scheme
        self.mask_ratio = list(mask_ratio)
        self.modality = modality
        self.split_ratio = split_ratio
        self.obs_type = obs_type
        self.frame_stack = frame_stack
        self.has_dummy_transition = bool(has_dummy_transition)
        self.padding_mode = padding_mode
        self._debug_done = False          # first-episode input/embedding debug print guard
        self._debug_printed_pred = False  # first-episode prediction-shape debug print guard
        self._debug_printed_padding = False  # first-padded-episode debug print guard

        assert masking_scheme in ("second_half", "random", "temporal_split", "last_action"), (
            f"Unknown masking_scheme '{masking_scheme}'. "
            "Supported: 'second_half', 'random', 'temporal_split', 'last_action'."
        )
        assert modality in ("actions", "states", "both"), (
            f"Unknown modality '{modality}'. "
            "Supported: 'actions', 'states', 'both'."
        )
        assert 0.0 < split_ratio <= 1.0, (
            f"split_ratio must be in (0, 1]. Got {split_ratio}."
        )
        assert obs_type in ("pixels", "states"), (
            f"Unknown obs_type '{obs_type}'. Supported: 'pixels', 'states'."
        )
        assert padding_mode in ("off", "on", "force"), (
            f"Unknown padding_mode '{padding_mode}'. Supported: 'off', 'on', 'force'."
        )

        if path is not None:
            print(f"[ReconstructionEvalAgent] Loading snapshot: {path}")
            payload = torch.load(path, map_location=device)
            self.config = payload["cfg"]
        else:
            assert transformer_cfg is not None, (
                "Either path or transformer_cfg must be provided."
            )
            self.config = transformer_cfg

        self.mdp = MaskedDPMultimodal(
            obs_shape[0], action_shape[0], self.config
        ).to(device)

        assert not self.mdp.use_early_fusion, (
            "Early fusion is a legacy option that is not supported"
        )

        if path is not None:
            self.mdp.load_state_dict(payload["model"])
            print("[ReconstructionEvalAgent] Weights loaded.")

        for param in self.mdp.parameters():
            param.requires_grad = False
        self.mdp.eval()

        self.discrete_actions = bool(self.mdp.discrete_actions)
        if self.discrete_actions and self.modality in ("actions", "both"):
            print(
                f"[ReconstructionEvalAgent] discrete_actions=True detected -> "
                f"action metric will be TOP-1 ACCURACY (num_actions={self.mdp.num_actions}), "
                f"not MSE."
            )

        n_params = sum(p.numel() for p in self.mdp.parameters())
        scheme_detail = (
            f"split_ratio={split_ratio}"
            if masking_scheme in ("temporal_split", "last_action") else ""
        )
        print(
            f"[ReconstructionEvalAgent] "
            f"Parameters: {n_params:,} (all frozen) | "
            f"T={T} | scheme={masking_scheme}"
            + (f"({scheme_detail})" if scheme_detail else "")
            + f" | modality={modality}"
            + f" | obs_type={obs_type} frame_stack={frame_stack}"
            + f" | has_dummy_transition={self.has_dummy_transition}"
            + f" | discrete_actions={self.discrete_actions}"
            + f" | padding_mode={self.padding_mode}"
        )


    # Masking
    def _apply_masking(self, T: int, valid_t: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:

        total_len = 2 * T

        if valid_t is None:
            valid_t = np.ones(T, dtype=bool)
        n_real = int(valid_t.sum())

        if self.masking_scheme == "second_half":
            half = n_real // 2
            keep_t  = np.zeros(T, dtype=bool); keep_t[:half] = True
            score_t = np.zeros(T, dtype=bool); score_t[half:n_real] = True
            ids_keep = np.where(np.repeat(keep_t, 2))[0]
            mask_interleaved = np.repeat(score_t, 2).astype(np.float32)

        elif self.masking_scheme == "random":
            ratio = float(np.random.choice(self.mask_ratio))
            valid_il = np.repeat(valid_t, 2)
            real_idx = np.where(valid_il)[0]
            n_real_tok = len(real_idx)
            len_keep = max(1, int(n_real_tok * (1.0 - ratio)))
            
            # Single noise vector over real tokens pool
            noise = np.random.rand(n_real_tok)
            order = np.argsort(noise)                    # ascending: small = keep
            ids_keep = np.sort(real_idx[order[:len_keep]])
            mask_interleaved = np.zeros(total_len, dtype=np.float32)
            mask_interleaved[real_idx[order[len_keep:]]] = 1.0

        elif self.masking_scheme == "temporal_split":
            cut = int(n_real * self.split_ratio)
            keep_t  = np.zeros(T, dtype=bool); keep_t[:cut] = True
            score_t = np.zeros(T, dtype=bool); score_t[cut:n_real] = True
            ids_keep = np.where(np.repeat(keep_t, 2))[0]
            mask_interleaved = np.repeat(score_t, 2).astype(np.float32)

        elif self.masking_scheme == "last_action":
            mask_interleaved = np.zeros(total_len, dtype=np.float32)
            if n_real < 1:
                ids_keep = np.array([], dtype=np.int64)  # degenerate: nothing real
            else:
                cut = int(n_real * self.split_ratio)
                cut = min(cut, n_real - 1)  # need >=1 real timestep left to predict
                ids_keep = np.arange(2 * cut + 1)
                mask_interleaved[2 * cut + 1] = 1.0

        else:
            raise ValueError(f"Unknown masking_scheme '{self.masking_scheme}'.")

        return ids_keep, mask_interleaved


    # Forward pass
    def _forward(self, states_np: np.ndarray, actions_np: np.ndarray, valid_t: Optional[np.ndarray] = None):
        T = self.T
        device = self.device
        enc_D = self.mdp.enc_n_embd
        D = self.mdp.n_embd

        states_t = torch.as_tensor(
            states_np, dtype=torch.float32, device=device
        ).unsqueeze(0)   # (1, T, H, W, C)

        if self.mdp.discrete_actions:
            a_idx = torch.as_tensor(
                actions_np, dtype=torch.long, device=device
            ).unsqueeze(0)                                    # (1, T) or (1, T, 1)
            if a_idx.dim() == 3 and a_idx.size(-1) == 1:
                a_idx = a_idx.squeeze(-1)                       # -> (1, T)

            a_min, a_max = int(a_idx.min().item()), int(a_idx.max().item())
            assert 0 <= a_min and a_max < self.mdp.num_actions, (
                f"[ReconstructionEvalAgent] Discrete action ids out of range: "
                f"min={a_min} max={a_max} but num_actions={self.mdp.num_actions}. "
                f"Check that the val .npz files store 0-indexed integer actions "
                f"matching agent.transformer_cfg.num_actions."
            )
            actions_t = a_idx  # kept for the debug print below
        else:
            actions_t = torch.as_tensor(
                actions_np, dtype=torch.float32, device=device
            ).unsqueeze(0)   # (1, T, action_dim)

        if not self._debug_done:
            print(
                f"[ReconstructionEvalAgent][debug] states_np: shape={states_np.shape} "
                f"dtype={states_np.dtype}"
            )
            print(
                f"[ReconstructionEvalAgent][debug] actions_np: shape={actions_np.shape} "
                f"dtype={actions_np.dtype} min={np.min(actions_np)} max={np.max(actions_np)} "
                f"| embedded as: {'nn.Embedding (long indices)' if self.mdp.discrete_actions else 'nn.Linear (float vector)'}"
            )

        # embed states and actions
        pos_embed = self.mdp.pos_embed

        s_emb_raw = self.mdp._embed_states(states_t)

        # State target: L2-normalised CNN features (same as forward_loss normalisation)
        if self.obs_type == "pixels":
            state_target_t = s_emb_raw / torch.norm(s_emb_raw, dim=-1, keepdim=True)
            state_target = state_target_t[0].cpu().numpy()  # (T, enc_D)
        else:
            # Proprioceptive objective: L2-normalize raw states to match training loss
            states_t = torch.tensor(states_np, dtype=torch.float32, device=self.device)
            state_target_t = states_t / (torch.norm(states_t, dim=-1, keepdim=True) + 1e-6)
            state_target = state_target_t.cpu().numpy()  # (T, obs_dim)

        a_emb = self.mdp.action_embed(actions_t)   # (1, T, enc_D)

        if not self._debug_done:
            print(
                f"[ReconstructionEvalAgent][debug] s_emb_raw: {tuple(s_emb_raw.shape)} | "
                f"a_emb: {tuple(a_emb.shape)}"
            )
            self._debug_done = True

        # pos embeddings
        s_emb = s_emb_raw + pos_embed[:, 0:2*T:2, :]   # (1, T, enc_D)
        a_emb = a_emb     + pos_embed[:, 1:2*T:2, :]   # (1, T, enc_D)

        # masking
        ids_keep_np, mask_interleaved = self._apply_masking(T, valid_t=valid_t)
        is_state_np = (ids_keep_np % 2) == 0     # (len_keep,)

        if not self._debug_printed_padding and valid_t is not None and not valid_t.all():
            n_real = int(valid_t.sum())
            print(
                f"[ReconstructionEvalAgent][debug] Padded window: "
                f"real_timesteps={n_real}/{T} | context_tokens={len(ids_keep_np)} | "
                f"scored_tokens={int(mask_interleaved.sum())} "
                f"(padding is excluded from both, per padding_mode='{self.padding_mode}')"
            )
            self._debug_printed_padding = True

        # timestep indices
        state_timesteps  = (ids_keep_np[ is_state_np] // 2).tolist()
        action_timesteps = (ids_keep_np[~is_state_np] // 2).tolist()

        n_s = len(state_timesteps)
        n_a = len(action_timesteps)

        if n_s > 0:
            s_idx_t = torch.tensor(state_timesteps, device=device, dtype=torch.long)
            x_s = s_emb[:, s_idx_t, :]   # (1, n_s, enc_D)

        if n_a > 0:
            a_idx_t = torch.tensor(action_timesteps, device=device, dtype=torch.long)
            x_a = a_emb[:, a_idx_t, :]   # (1, n_a, enc_D)

        # Separate encoder blocks
        if n_s > 0:
            s_attn = torch.ones(1, 1, n_s, n_s, device=device)
            for blk in self.mdp.state_encoder_blocks:
                x_s = blk(x_s, s_attn)
            x_s = self.mdp.state_encoder_norm(x_s)   # (1, n_s, enc_D)
            x_s = self.mdp.state_proj(x_s)            # (1, n_s, D)
            x_s = self.mdp.state_adapter(x_s)         # (1, n_s, D)
        else:
            x_s = torch.zeros(1, 0, D, device=device)

        if n_a > 0:
            a_attn = torch.ones(1, 1, n_a, n_a, device=device)
            for blk in self.mdp.action_encoder_blocks:
                x_a = blk(x_a, a_attn)
            x_a = self.mdp.action_encoder_norm(x_a)   # (1, n_a, enc_D)
            x_a = self.mdp.action_proj(x_a)            # (1, n_a, D)
            x_a = self.mdp.action_adapter(x_a)         # (1, n_a, D)
        else:
            x_a = torch.zeros(1, 0, D, device=device)

        # padding masks for fusion
        s_pad_mask = (
            torch.ones(1, 0, dtype=torch.bool, device=device)
            if n_s == 0 else None
        )
        a_pad_mask = (
            torch.ones(1, 0, dtype=torch.bool, device=device)
            if n_a == 0 else None
        )

        ids_keep_t = torch.tensor(
            ids_keep_np, device=device, dtype=torch.long
        ).unsqueeze(0)   # (1, len_keep)

        # Fusion
        x_fused = self.mdp.forward_fusion(
            x_s, x_a, ids_keep_t,
            s_pad_mask=s_pad_mask,
            a_pad_mask=a_pad_mask,
        )   # (1, len_keep, D)

        # Decoder
        total_len = 2 * T
        kept_set = set(ids_keep_np.tolist())
        masked_positions_t = torch.tensor(
            [p for p in range(total_len) if p not in kept_set],
            device=device, dtype=torch.long,
        )
        ids_shuffle = torch.cat(
            [ids_keep_t[0], masked_positions_t], dim=0
        )                                                      # (2T,)
        ids_restore = torch.argsort(ids_shuffle).unsqueeze(0)  # (1, 2T)

        pred_s_t, pred_a_t = self.mdp.forward_decoder(x_fused, ids_restore)
        # pred_s_t: (1, T, enc_D)
        # pred_a_t: (1, T, action_dim) continuous, or (1, T, num_actions) logits if discrete

        pred_s = pred_s_t[0].cpu().numpy()   # (T, enc_D)
        pred_a = pred_a_t[0].cpu().numpy()   # (T, action_dim) or (T, num_actions)

        if not self._debug_printed_pred:
            print(
                f"[ReconstructionEvalAgent][debug] pred_s: {pred_s.shape} | "
                f"pred_a: {pred_a.shape}"
                + (f" (logits over {self.mdp.num_actions} actions)" if self.mdp.discrete_actions else " (continuous)")
            )
            self._debug_printed_pred = True

        state_mask  = mask_interleaved[0::2].astype(bool)
        action_mask = mask_interleaved[1::2].astype(bool)

        return pred_s, pred_a, state_target, state_mask, action_mask, ids_keep_np, mask_interleaved


    # Metric helpers
    def _compute_episode_mse(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple:
        masked_indices = np.where(mask)[0]   # indices of masked timesteps
        if len(masked_indices) == 0:
            return None, None, None

        pred_masked   = pred[masked_indices]    # (n_m, D)
        target_masked = target[masked_indices]  # (n_m, D)

        # Per-token MSE: mean over feature dimensions -> scalar per timestep
        mse_per_token = ((pred_masked - target_masked) ** 2).mean(axis=-1)  # (n_m,)
        return float(mse_per_token.mean()), mse_per_token, masked_indices

    def _compute_episode_action_accuracy(
        self,
        pred_logits: np.ndarray,
        target: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple:
        masked_indices = np.where(mask)[0]
        if len(masked_indices) == 0:
            return None, None, None

        target_arr = np.asarray(target)
        if target_arr.ndim == 2 and target_arr.shape[-1] == 1:
            target_arr = target_arr[:, 0]

        pred_masked   = pred_logits[masked_indices]                    # (n_m, num_actions)
        target_masked = target_arr[masked_indices].astype(np.int64)    # (n_m,)

        pred_idx = pred_masked.argmax(axis=-1).astype(np.int64)        # (n_m,)
        correct  = (pred_idx == target_masked).astype(np.float64)      # (n_m,) 1.0/0.0 per token

        return float(correct.mean()), correct, masked_indices

    def _aggregate_metrics(
        self,
        episode_losses: List[float],
        pos_mse_sum: np.ndarray,
        pos_mse_count: np.ndarray,
        prefix: str,
    ) -> Dict:

        if not episode_losses:
            # Degenerate: all episodes had no masked tokens
            return {
                f"{prefix}/mean":        float("nan"),
                f"{prefix}/std":         float("nan"),
                f"{prefix}/by_position": [0.0] * self.T,
            }

        safe_count = np.where(pos_mse_count > 0, pos_mse_count, 1)
        by_position = np.where(
            pos_mse_count > 0,
            pos_mse_sum / safe_count,
            0.0,
        )

        return {
            f"{prefix}/mean":        float(np.mean(episode_losses)),
            f"{prefix}/std":         float(np.std(episode_losses)),
            f"{prefix}/by_position": by_position.tolist(),
        }


    # Single-modality evaluation
    def _evaluate_modality(
        self,
        val_episodes: List[Dict],
        num_eval_episodes: int,
        modality: str,
    ) -> Dict:
        assert modality in ("actions", "states")

        use_accuracy = (modality == "actions") and self.discrete_actions
        prefix = (
            "action_recon_acc" if use_accuracy else
            "action_recon_loss" if modality == "actions" else
            "state_recon_loss"
        )
        episode_losses = []
        pos_mse_sum   = np.zeros(self.T, dtype=np.float64)
        pos_mse_count = np.zeros(self.T, dtype=np.int64)

        for ep_idx in range(num_eval_episodes):
            states_np, actions_np, valid_t = self._sample_window(val_episodes)

            with torch.no_grad():
                pred_s, pred_a, state_target, state_mask, action_mask, _, _ = \
                    self._forward(states_np, actions_np, valid_t)

            if modality == "actions":
                pred, target, mask = pred_a, actions_np, action_mask
            else:
                pred, target, mask = pred_s, state_target, state_mask

            if use_accuracy:
                ep_loss, per_token, masked_indices = \
                    self._compute_episode_action_accuracy(pred, target, mask)
            else:
                ep_loss, per_token, masked_indices = \
                    self._compute_episode_mse(pred, target, mask)

            if ep_loss is None:
                # Degenerate: no tokens masked.
                continue

            episode_losses.append(ep_loss)

            # Accumulate per-position
            for local_i, pos in enumerate(masked_indices):
                pos_mse_sum[pos]   += per_token[local_i]
                pos_mse_count[pos] += 1

        return self._aggregate_metrics(episode_losses, pos_mse_sum, pos_mse_count, prefix)


    # Both-modality evaluation (single forward pass per episode)
    def _evaluate_both(
        self,
        val_episodes: List[Dict],
        num_eval_episodes: int,
    ) -> Dict:

        a_losses, s_losses = [], []
        a_pos_sum   = np.zeros(self.T, dtype=np.float64)
        a_pos_count = np.zeros(self.T, dtype=np.int64)
        s_pos_sum   = np.zeros(self.T, dtype=np.float64)
        s_pos_count = np.zeros(self.T, dtype=np.int64)

        a_prefix = "action_recon_acc" if self.discrete_actions else "action_recon_loss"

        for ep_idx in range(num_eval_episodes):
            states_np, actions_np, valid_t = self._sample_window(val_episodes)

            with torch.no_grad():
                pred_s, pred_a, state_target, state_mask, action_mask, _, _ = \
                    self._forward(states_np, actions_np, valid_t)

            # Actions
            if self.discrete_actions:
                ep_a, val_a, idx_a = \
                    self._compute_episode_action_accuracy(pred_a, actions_np, action_mask)
            else:
                ep_a, val_a, idx_a = \
                    self._compute_episode_mse(pred_a, actions_np, action_mask)
            if ep_a is not None:
                a_losses.append(ep_a)
                for local_i, pos in enumerate(idx_a):
                    a_pos_sum[pos]   += val_a[local_i]
                    a_pos_count[pos] += 1

            # States
            ep_s, mse_s, idx_s = self._compute_episode_mse(pred_s, state_target, state_mask)
            if ep_s is not None:
                s_losses.append(ep_s)
                for local_i, pos in enumerate(idx_s):
                    s_pos_sum[pos]   += mse_s[local_i]
                    s_pos_count[pos] += 1

        metrics_a = self._aggregate_metrics(a_losses, a_pos_sum, a_pos_count, a_prefix)
        metrics_s = self._aggregate_metrics(s_losses, s_pos_sum, s_pos_count, "state_recon_loss")

        if self.discrete_actions:
            total = float("nan")
        else:
            mean_a = metrics_a["action_recon_loss/mean"]
            mean_s = metrics_s["state_recon_loss/mean"]
            total  = (
                mean_a + mean_s
                if not (np.isnan(mean_a) or np.isnan(mean_s))
                else float("nan")
            )

        return {
            "actions":          metrics_a,
            "states":           metrics_s,
            "total_recon_loss": total,
        }

    
    # Episode helpers
    def _get_frames(self, episode: Dict) -> np.ndarray:
        if self.obs_type == "pixels":
            if "pixel_observation" in episode:
                return episode["pixel_observation"]   # (N[+1], H, W, 3)
            return episode["observation"]              # (N[+1], H, W, 3)
        return episode["observation"]                  # (N[+1], obs_dim)

    def _episode_real_length(self, episode: Dict) -> int:
        n_frames = self._get_frames(episode).shape[0]
        return n_frames - 1 if self.has_dummy_transition else n_frames

    def _filter_episodes_for_padding_mode(self, val_episodes: List[Dict]) -> List[Dict]:
        long_eps  = [ep for ep in val_episodes if self._episode_real_length(ep) >= self.T]
        short_eps = [ep for ep in val_episodes if self._episode_real_length(ep) <  self.T]

        print(
            f"[ReconstructionEvalAgent] Episode length check (T={self.T}): "
            f"{len(long_eps)}/{len(val_episodes)} episodes have real_length >= T "
            f"(usable as-is), {len(short_eps)}/{len(val_episodes)} are shorter "
            f"(would need padding) | padding_mode='{self.padding_mode}'"
        )

        if self.padding_mode == "off":
            usable = long_eps
            if not usable:
                raise AssertionError(
                    f"padding_mode='off' and ALL {len(val_episodes)} val episodes are "
                    f"shorter than T={self.T}. Use padding_mode='on' or 'force', reduce "
                    f"T, or point val_replay_dir at a split with longer episodes."
                )
        elif self.padding_mode == "on":
            usable = val_episodes
        else:  # "force"
            usable = short_eps
            if not usable:
                raise AssertionError(
                    f"padding_mode='force' but ALL {len(val_episodes)} val episodes "
                    f"have real_length >= T={self.T} — there is nothing to pad, so "
                    f"there is nothing to force-evaluate. Use padding_mode='on' or "
                    f"'off' instead."
                )

        print(
            f"[ReconstructionEvalAgent] padding_mode='{self.padding_mode}' -> "
            f"{len(usable)}/{len(val_episodes)} episodes usable for this eval."
        )
        return usable

    # Episode sampling
    def _sample_window(self, val_episodes: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a random window of T timesteps from a val episode.
        """
        episode = random.choice(val_episodes)
        frames = self._get_frames(episode)

        n_frames = frames.shape[0]
        ep_len = n_frames - 1 if self.has_dummy_transition else n_frames

        if ep_len >= self.T:
            if self.has_dummy_transition:
                # V-D4RL / dm_control convention: +1 dummy frame at the start.
                # action[t] is the action taken FROM state obs[t] -> action index
                # leads the state index by 1.
                idx = np.random.randint(0, ep_len - self.T + 1) + 1
                state_start = idx - 1
            else:
                # Procgen convention: no dummy frame, action[t] aligned with obs[t].
                idx = np.random.randint(0, ep_len - self.T + 1)
                state_start = idx

            if self.obs_type == "pixels" and self.frame_stack > 1:
                abs_indices = np.arange(state_start, state_start + self.T)
                states = _stack_frames(frames, abs_indices, self.frame_stack)  # (T,H,W,k*3)
            else:
                states = frames[state_start : state_start + self.T]   # (T,H,W,3) or (T,obs_dim)

            actions = episode["action"][idx : idx + self.T]    # (T,) or (T, action_dim)
            valid_t = np.ones(self.T, dtype=bool)
            return states, actions, valid_t

        # Padding needed case
        assert self.padding_mode in ("on", "force"), (
            f"Internal error: got a short episode (real_length={ep_len} < T={self.T}) "
            f"with padding_mode='{self.padding_mode}'. evaluate() is supposed to filter "
            f"these out via _filter_episodes_for_padding_mode before padding_mode='off' "
            f"ever reaches _sample_window — this should not be reachable."
        )
        assert self.frame_stack == 1, (
            f"Padding a short episode with frame_stack={self.frame_stack} > 1 is not "
            f"supported."
        )

        L = ep_len
        pad = self.T - L
        if self.has_dummy_transition:
            state_idx0, action_idx0 = 0, 1
        else:
            state_idx0, action_idx0 = 0, 0

        obs_real    = frames[state_idx0 : state_idx0 + L]
        action_real = episode["action"][action_idx0 : action_idx0 + L]

        states  = _pad_zeros(obs_real, pad)
        actions = _pad_zeros(action_real, pad)

        valid_t = np.zeros(self.T, dtype=bool)
        valid_t[:L] = True

        return states, actions, valid_t


    # Evaluation
    def evaluate(self, val_episodes: List[Dict], num_eval_episodes: int) -> Dict:
        """
        Run reconstruction evaluation on val episodes.

        One episode = one random window of T timesteps from a val file.
        Seed must be set outside this function (via utils.set_seed_everywhere)
        to guarantee reproducibility across model snapshots.

        Returns
        -------
        modality="actions" (continuous):
            {"actions": {"action_recon_loss/mean": ..., "action_recon_loss/std": ...,
                         "action_recon_loss/by_position": [...]}}
        modality="actions" (discrete_actions=True in the snapshot's config):
            {"actions": {"action_recon_acc/mean": ..., "action_recon_acc/std": ...,
                         "action_recon_acc/by_position": [...]}}   # accuracy, higher=better
        modality="states":
            {"states": {"state_recon_loss/mean": ..., ...}}
        modality="both":
            {"actions": {...}, "states": {...}, "total_recon_loss": float}
            total_recon_loss is NaN when actions are discrete (accuracy + MSE don't sum).
        """
        usable_episodes = self._filter_episodes_for_padding_mode(val_episodes)

        if self.modality == "actions":
            return {"actions": self._evaluate_modality(usable_episodes, num_eval_episodes, "actions")}
        elif self.modality == "states":
            return {"states": self._evaluate_modality(usable_episodes, num_eval_episodes, "states")}
        elif self.modality == "both":
            return self._evaluate_both(usable_episodes, num_eval_episodes)