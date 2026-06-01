import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
import torch
import wandb
import omegaconf

import utils
from agent.mdp_action_recon import ActionReconstructionEvalAgentUnimodal

torch.backends.cudnn.benchmark = True


def load_val_episodes(replay_dir: str, val_split_ratio: float) -> List[Dict]:
    eps_fns = sorted(Path(replay_dir).rglob("*.npz"))
    n_total = len(eps_fns)
    assert n_total > 0, f"No .npz files found in {replay_dir}"

    n_train  = int(n_total * (1.0 - val_split_ratio))
    val_fns  = eps_fns[n_train:]
    n_val    = len(val_fns)

    print(
        f"[Val loader] {n_val}/{n_total} files "
        f"(train={n_train}, val={n_val}, val_split_ratio={val_split_ratio})"
    )
    assert n_val > 0, (
        f"val split is empty. Check val_split_ratio={val_split_ratio} "
        f"with {n_total} total files."
    )

    episodes = []
    for fn in val_fns:
        with fn.open("rb") as f:
            raw = np.load(f)
            ep  = {k: raw[k] for k in raw.keys()}
        # V-D4RL stores pixel obs under "image"; normalise to "observation"
        if "image" in ep and "observation" not in ep:
            ep["observation"] = ep.pop("image")
        episodes.append(ep)

    return episodes


def _log_metrics_to_wandb(metrics: Dict, global_step: int, prefix: str = "eval") -> None:
    log_data = {"eval/snapshot_step": global_step}

    for k, v in metrics.items():
        if k == "action_recon_loss/by_position":
            # One scalar per position: eval/action_mse_t000 … eval/action_mse_tXXX
            for t, mse in enumerate(v):
                log_data[f"{prefix}/action_mse_t{t:03d}"] = float(mse)
        else:
            log_data[f"{prefix}/{k}"] = v

    wandb.log(log_data, step=global_step)


@hydra.main(config_path=".", config_name="eval_action_recon")
def main(cfg):
    work_dir = Path.cwd()
    print(f"workspace: {work_dir}")

    device = torch.device(cfg.device)

    val_episodes = load_val_episodes(cfg.val_replay_dir, cfg.val_split_ratio)

    first_ep     = val_episodes[0]
    obs_shape    = first_ep["observation"].shape[1:]   # (H, W, C)
    action_dim   = first_ep["action"].shape[1]
    action_shape = (action_dim,)
    print(f"obs_shape={obs_shape}, action_shape={action_shape}")

    snapshots = cfg.get("eval_snapshots", None)
    if snapshots is None or len(snapshots) == 0:
        assert cfg.agent.path is not None, (
            "Provide eval_snapshots=[...] or agent.path=... in the config."
        )
        snapshots = [cfg.agent.path]

    first_snap = str(snapshots[0])
    agent = ActionReconstructionEvalAgentUnimodal(
        obs_shape=obs_shape,
        action_shape=action_shape,
        device=device,
        T=cfg.agent.T,
        masking_scheme=cfg.agent.masking_scheme,
        mask_ratio=list(cfg.agent.mask_ratio),
        path=first_snap,
    )

    exp_name     = str(cfg.exp_name)
    wandb_config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb_kwargs = dict(
        project  = cfg.project,
        name     = exp_name,
        config   = wandb_config,
        settings = wandb.Settings(start_method="thread", _disable_stats=True),
        mode     = "online" if cfg.use_wandb else "offline",
        notes    = cfg.notes,
    )
    if cfg.get("wandb_run_id", None):
        wandb_kwargs["id"]     = str(cfg.wandb_run_id)
        wandb_kwargs["resume"] = "allow"

    wandb.init(**wandb_kwargs)

    # Eval loop
    for snap_path in snapshots:
        snap_path = str(snap_path)
        print(f"\n[Eval] Loading snapshot: {snap_path}")

        payload = torch.load(snap_path, map_location=device)
        agent.mdp.load_state_dict(payload["model"])
        agent.mdp.eval()

        try:
            global_step = int(Path(snap_path).stem.split("_")[-1])
        except Exception:
            global_step = 0

        # Re-seed before every snapshot: guarantees all models see identical
        # episode windows and (for random scheme) identical masking patterns.
        utils.set_seed_everywhere(cfg.seed)

        print(
            f"[Eval] step={global_step} | "
            f"T={cfg.agent.T} | scheme={cfg.agent.masking_scheme} | "
            f"episodes={cfg.num_eval_episodes} | seed={cfg.seed}"
        )

        metrics = agent.evaluate(val_episodes, cfg.num_eval_episodes)

        if cfg.use_wandb:
            _log_metrics_to_wandb(metrics, global_step)

        by_pos = metrics["action_recon_loss/by_position"]
        masked_mses = [v for v in by_pos if v > 0.0]
        print(
            f"  action_recon_loss | "
            f"mean={metrics['action_recon_loss/mean']:.6f}  "
            f"std={metrics['action_recon_loss/std']:.6f}  "
            f"(masked positions: min={min(masked_mses):.6f}  "
            f"max={max(masked_mses):.6f})"
            if masked_mses else
            f"  action_recon_loss | "
            f"mean={metrics['action_recon_loss/mean']:.6f}  "
            f"std={metrics['action_recon_loss/std']:.6f}"
        )

    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()