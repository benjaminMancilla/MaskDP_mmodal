import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import numpy as np
import torch
import wandb
import omegaconf

import utils
from agent.mdp_recon import ReconstructionEvalAgentUnimodal

torch.backends.cudnn.benchmark = True

# W&B projects are hardcoded per modality
WANDB_PROJECTS = {
    "actions": "maskdp-mm-vd4rl-action-recon",
    "states":  "maskdp-mm-vd4rl-state-recon",
}


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


def _infer_obs_shape(episodes: List[Dict], obs_type: str, frame_stack: int) -> Tuple:
    first = episodes[0]
    if obs_type == "pixels":
        if "pixel_observation" in first:
            H, W, C = first["pixel_observation"].shape[1:]   # C=3 (single frame)
        else:
            H, W, C = first["observation"].shape[1:]          # V-D4RL: already pixels
        return (H, W, C * frame_stack)
    else:
        return first["observation"].shape[1:]   # (obs_dim,) for proprioceptive


def _log_modality_to_wandb(
    run: wandb.sdk.wandb_run.Run,
    metrics: Dict,
    prefix: str,
    global_step: int,
) -> None:
    """
    Log a single modality's metrics to the given wandb run.

    Scalar metrics are logged directly. by_position is logged as one scalar
    per timestep so curves can be overlaid across models in W&B.
    """
    log_data = {"eval/snapshot_step": global_step}

    for k, v in metrics.items():
        if k.endswith("/by_position"):
            short = "action_mse" if "action" in k else "state_mse"
            for t, mse in enumerate(v):
                log_data[f"{prefix}/{short}_t{t:03d}"] = float(mse)
        else:
            log_data[f"{prefix}/{k}"] = v

    run.log(log_data, step=global_step)


def _print_modality(label: str, metrics: Dict) -> None:
    mean = metrics.get(f"{label}/mean", float("nan"))
    std  = metrics.get(f"{label}/std",  float("nan"))
    by_pos = metrics.get(f"{label}/by_position", [])
    masked_mses = [v for v in by_pos if v > 0.0]
    print(
        f"  {label} | "
        f"mean={mean:.6f}  std={std:.6f}  "
        f"(masked: min={min(masked_mses):.6f}  max={max(masked_mses):.6f})"
        if masked_mses else
        f"  {label} | mean={mean:.6f}  std={std:.6f}"
    )


def _init_wandb_run(project: str, exp_name: str, cfg) -> wandb.sdk.wandb_run.Run:
    wandb_config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    kwargs = dict(
        project  = project,
        name     = exp_name,
        config   = wandb_config,
        settings = wandb.Settings(start_method="thread", _disable_stats=True),
        mode     = "online" if cfg.use_wandb else "offline",
        notes    = cfg.notes,
    )
    if cfg.get("wandb_run_id", None):
        kwargs["id"]     = str(cfg.wandb_run_id)
        kwargs["resume"] = "allow"
    return wandb.init(**kwargs)


@hydra.main(config_path=".", config_name="eval_recon")
def main(cfg):
    work_dir = Path.cwd()
    print(f"workspace: {work_dir}")

    device      = torch.device(cfg.device)
    modality    = cfg.agent.modality
    obs_type    = str(cfg.agent.obs_type)
    frame_stack = int(cfg.agent.frame_stack)

    val_episodes = load_val_episodes(cfg.val_replay_dir, cfg.val_split_ratio)

    obs_shape    = _infer_obs_shape(val_episodes, obs_type, frame_stack)
    action_dim   = val_episodes[0]["action"].shape[1]
    action_shape = (action_dim,)
    print(f"obs_type={obs_type}  frame_stack={frame_stack}  "
          f"obs_shape={obs_shape}  action_shape={action_shape}")

    snapshots = cfg.get("eval_snapshots", None)
    if snapshots is None or len(snapshots) == 0:
        assert cfg.agent.path is not None, (
            "Provide eval_snapshots=[...] or agent.path=... in the config."
        )
        snapshots = [cfg.agent.path]

    first_snap = str(snapshots[0])
    agent = ReconstructionEvalAgentUnimodal(
        obs_shape=obs_shape,
        action_shape=action_shape,
        device=device,
        T=cfg.agent.T,
        masking_scheme=cfg.agent.masking_scheme,
        mask_ratio=list(cfg.agent.mask_ratio),
        split_ratio=cfg.agent.split_ratio,
        modality=modality,
        obs_type=obs_type,
        frame_stack=frame_stack,
        path=first_snap,
    )

    exp_name = str(cfg.exp_name)

    # Eval loop
    # Accumulate all metrics before touching W&B.
    all_results: List[tuple] = []   # [(global_step, metrics), ...]

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

        scheme_detail = (
            f"split_ratio={cfg.agent.split_ratio} | "
            if cfg.agent.masking_scheme == "temporal_split"
            else ""
        )
        print(
            f"[Eval] step={global_step} | "
            f"T={cfg.agent.T} | scheme={cfg.agent.masking_scheme} | "
            + scheme_detail
            + f"modality={modality} | episodes={cfg.num_eval_episodes} | seed={cfg.seed}"
        )

        metrics = agent.evaluate(val_episodes, cfg.num_eval_episodes)
        all_results.append((global_step, metrics))

        # Print immediately so progress is visible in the cluster log
        if modality == "actions":
            _print_modality("action_recon_loss", metrics["actions"])
        elif modality == "states":
            _print_modality("state_recon_loss", metrics["states"])
        elif modality == "both":
            _print_modality("action_recon_loss", metrics["actions"])
            _print_modality("state_recon_loss",  metrics["states"])
            print(f"  total_recon_loss = {metrics['total_recon_loss']:.6f}")

    # W&B logging
    if not cfg.use_wandb:
        return

    if modality in ("actions", "both"):
        run = _init_wandb_run(WANDB_PROJECTS["actions"], exp_name, cfg)
        for global_step, metrics in all_results:
            _log_modality_to_wandb(run, metrics["actions"], "eval", global_step)
            if modality == "both":
                run.log({"eval/total_recon_loss": metrics["total_recon_loss"]},
                        step=global_step)
        run.finish()

    if modality in ("states", "both"):
        run = _init_wandb_run(WANDB_PROJECTS["states"], exp_name, cfg)
        for global_step, metrics in all_results:
            _log_modality_to_wandb(run, metrics["states"], "eval", global_step)
            if modality == "both":
                run.log({"eval/total_recon_loss": metrics["total_recon_loss"]},
                        step=global_step)
        run.finish()


if __name__ == "__main__":
    main()