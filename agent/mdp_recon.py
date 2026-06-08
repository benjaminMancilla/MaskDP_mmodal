import numpy as np
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from agent.mdp import MaskedDPMultimodal
import utils


class ReconstructionEvalAgent:
    """
    Eval agent that measures reconstruction loss on offline val sequences.

    Supports three modalities:
        "actions" : MSE between predicted and ground-truth actions at masked positions.
        "states"  : MSE between predicted and L2-normalised CNN features at masked positions.
                    Target = pixel_encoder(s) / ||pixel_encoder(s)||_2, identical to
                    the normalisation used in forward_loss during training.
        "both"    : computes both losses in a single forward pass per episode.
                    Logs to separate W&B projects; total_recon_loss = action + state loss.

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
        **kwargs,
    ):
        self.device = device
        self.T = T
        self.action_dim = action_shape[0]
        self.masking_scheme = masking_scheme
        self.mask_ratio = list(mask_ratio)
        self.modality = modality
        self.split_ratio = split_ratio

        assert masking_scheme in ("second_half", "random", "temporal_split"), (
            f"Unknown masking_scheme '{masking_scheme}'. "
            "Supported: 'second_half', 'random', 'temporal_split'."
        )
        assert modality in ("actions", "states", "both"), (
            f"Unknown modality '{modality}'. "
            "Supported: 'actions', 'states', 'both'."
        )
        assert 0.0 < split_ratio <= 1.0, (
            f"split_ratio must be in (0, 1]. Got {split_ratio}."
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

        n_params = sum(p.numel() for p in self.mdp.parameters())
        scheme_detail = (
            f"split_ratio={split_ratio}" if masking_scheme == "temporal_split" else ""
        )
        print(
            f"[ReconstructionEvalAgent] "
            f"Parameters: {n_params:,} (all frozen) | "
            f"T={T} | scheme={masking_scheme}"
            + (f"({scheme_detail})" if scheme_detail else "")
            + f" | modality={modality}"
        )


    # Masking
    def _apply_masking(self, T: int) -> Tuple[np.ndarray, np.ndarray]:

        total_len = 2 * T

        if self.masking_scheme == "second_half":
            ids_keep = np.arange(T, dtype=np.int64)
            mask_interleaved = np.zeros(total_len, dtype=np.float32)
            mask_interleaved[T:] = 1.0

        elif self.masking_scheme == "random":
            ratio = float(np.random.choice(self.mask_ratio))
            len_keep = max(1, int(total_len * (1.0 - ratio)))
            # Single noise vector over 2T — same mechanism as forward_encoder
            noise = np.random.rand(total_len)
            ids_shuffle = np.argsort(noise)          # ascending: small = keep
            ids_keep = np.sort(ids_shuffle[:len_keep])  # sort to restore order
            mask_interleaved = np.ones(total_len, dtype=np.float32)
            mask_interleaved[ids_keep] = 0.0

        elif self.masking_scheme == "temporal_split":
            cut = int(T * self.split_ratio)
            ids_keep = np.arange(2 * cut, dtype=np.int64)
            mask_interleaved = np.zeros(total_len, dtype=np.float32)
            mask_interleaved[2 * cut:] = 1.0

        else:
            raise ValueError(f"Unknown masking_scheme '{self.masking_scheme}'.")

        return ids_keep, mask_interleaved


    # Forward pass
    def _forward(self, states_np: np.ndarray, actions_np: np.ndarray):
        T = self.T
        device = self.device
        enc_D = self.mdp.enc_n_embd
        D = self.mdp.n_embd

        states_t = torch.as_tensor(
            states_np, dtype=torch.float32, device=device
        ).unsqueeze(0)   # (1, T, H, W, C)

        actions_t = torch.as_tensor(
            actions_np, dtype=torch.float32, device=device
        ).unsqueeze(0)   # (1, T, action_dim)

        # embed states and actions
        pos_embed = self.mdp.pos_embed

        s_emb_raw = self.mdp._embed_states(states_t)

        # State target: L2-normalised CNN features (same as forward_loss normalisation)
        state_target_t = s_emb_raw / torch.norm(s_emb_raw, dim=-1, keepdim=True)
        state_target = state_target_t[0].cpu().numpy()  # (T, enc_D)

        a_emb = self.mdp.action_embed(actions_t)   # (1, T, enc_D)

        # pos embeddings
        s_emb = s_emb_raw + pos_embed[:, 0:2*T:2, :]   # (1, T, enc_D)
        a_emb = a_emb     + pos_embed[:, 1:2*T:2, :]   # (1, T, enc_D)

        # masking
        ids_keep_np, mask_interleaved = self._apply_masking(T)
        is_state_np = (ids_keep_np % 2) == 0     # (len_keep,)

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
        # pred_s_t: (1, T, enc_D) | pred_a_t: (1, T, action_dim)

        pred_s = pred_s_t[0].cpu().numpy()   # (T, enc_D)
        pred_a = pred_a_t[0].cpu().numpy()   # (T, action_dim)

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

        prefix        = "action_recon_loss" if modality == "actions" else "state_recon_loss"
        episode_losses = []
        pos_mse_sum   = np.zeros(self.T, dtype=np.float64)
        pos_mse_count = np.zeros(self.T, dtype=np.int64)

        for ep_idx in range(num_eval_episodes):
            states_np, actions_np = self._sample_window(val_episodes)

            with torch.no_grad():
                pred_s, pred_a, state_target, state_mask, action_mask, _, _ = \
                    self._forward(states_np, actions_np)

            if modality == "actions":
                pred, target, mask = pred_a, actions_np, action_mask
            else:
                pred, target, mask = pred_s, state_target, state_mask

            ep_loss, mse_per_token, masked_indices = self._compute_episode_mse(pred, target, mask)
            if ep_loss is None:
                # Degenerate: no tokens masked.
                continue

            episode_losses.append(ep_loss)

            # Accumulate per-position
            for local_i, pos in enumerate(masked_indices):
                pos_mse_sum[pos]   += mse_per_token[local_i]
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

        for ep_idx in range(num_eval_episodes):
            states_np, actions_np = self._sample_window(val_episodes)

            with torch.no_grad():
                pred_s, pred_a, state_target, state_mask, action_mask, _, _ = \
                    self._forward(states_np, actions_np)

            # Actions
            ep_a, mse_a, idx_a = self._compute_episode_mse(pred_a, actions_np, action_mask)
            if ep_a is not None:
                a_losses.append(ep_a)
                for local_i, pos in enumerate(idx_a):
                    a_pos_sum[pos]   += mse_a[local_i]
                    a_pos_count[pos] += 1

            # States
            ep_s, mse_s, idx_s = self._compute_episode_mse(pred_s, state_target, state_mask)
            if ep_s is not None:
                s_losses.append(ep_s)
                for local_i, pos in enumerate(idx_s):
                    s_pos_sum[pos]   += mse_s[local_i]
                    s_pos_count[pos] += 1

        metrics_a = self._aggregate_metrics(a_losses, a_pos_sum, a_pos_count, "action_recon_loss")
        metrics_s = self._aggregate_metrics(s_losses, s_pos_sum, s_pos_count, "state_recon_loss")

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


    # Episode sampling
    def _sample_window(self, val_episodes: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample a random window of T timesteps from a val episode.
        """
        episode = random.choice(val_episodes)
        # episode["observation"] has shape (n_steps + 1, H, W, C)
        # episode_len = n_observations - 1 (subtract dummy first transition)
        ep_len = episode["observation"].shape[0] - 1

        assert ep_len >= self.T, (
            f"Episode length {ep_len} < T={self.T}. "
            "Use a smaller T or longer episodes."
        )

        idx = np.random.randint(0, ep_len - self.T + 1) + 1
        states  = episode["observation"][idx - 1 : idx - 1 + self.T]   # (T, H, W, C)
        actions = episode["action"][idx : idx + self.T]                  # (T, action_dim)

        return states, actions


    # Evaluation
    def evaluate(self, val_episodes: List[Dict], num_eval_episodes: int) -> Dict:
        """
        Run reconstruction evaluation on val episodes.

        One episode = one random window of T timesteps from a val file.
        Seed must be set outside this function (via utils.set_seed_everywhere)
        to guarantee reproducibility across model snapshots.

        Returns
        -------
        modality="actions":
            {"actions": {"action_recon_loss/mean": ..., "action_recon_loss/std": ...,
                         "action_recon_loss/by_position": [...]}}
        modality="states":
            {"states": {"state_recon_loss/mean": ..., ...}}
        modality="both":
            {"actions": {...}, "states": {...}, "total_recon_loss": float}
        """
        if self.modality == "actions":
            return {"actions": self._evaluate_modality(val_episodes, num_eval_episodes, "actions")}
        elif self.modality == "states":
            return {"states": self._evaluate_modality(val_episodes, num_eval_episodes, "states")}
        elif self.modality == "both":
            return self._evaluate_both(val_episodes, num_eval_episodes)