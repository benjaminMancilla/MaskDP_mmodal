import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path

import hydra
import numpy as np
import torch

import dmc
import utils
import wandb
import omegaconf

from agent.mdp_return import MaskingEvalAgent

torch.backends.cudnn.benchmark = True


def get_domain(task):
    if task.startswith("point_mass_maze"):
        return "point_mass_maze"
    return task.split("_", 1)[0]


def eval_masking(
    global_step,
    agent,
    env,
    num_eval_episodes,
    discount,
    T_cond,
):
    total_returns_full      = []
    total_returns_postwarmup = []
    total_lengths           = []

    for episode in range(num_eval_episodes):
        agent.reset()
        time_step = env.reset()
        episode_return_full       = 0.0
        episode_return_postwarmup = 0.0
        episode_length            = 0
        # discount_full             = 1.0
        # discount_post             = 1.0

        while not time_step.last():
            obs = time_step.observation  # (C, H, W) pixels or (obs_dim,) states

            with torch.no_grad():
                action = agent.act(obs)

            time_step = env.step(action)

            episode_return_full += time_step.reward # * discount_full
            # discount_full       *= discount
            episode_length      += 1

            # Post-warmup return
            if episode_length > T_cond:
                episode_return_postwarmup += time_step.reward # * discount_post
                # discount_post             *= discount

        total_returns_full.append(episode_return_full)
        total_returns_postwarmup.append(episode_return_postwarmup)
        total_lengths.append(episode_length)

    normalized = [r / 10.0 for r in total_returns_full]

    print(f"raw_return={total_returns_full[-1]:.1f}, normalized={normalized[-1]:.3f}")

    return {
        "episode_return":             float(np.mean(total_returns_full)),
        "episode_return_postwarmup":  float(np.mean(total_returns_postwarmup)),
        "episode_return_normalized":  float(np.mean(normalized)),
        "episode_return_std":         float(np.std(normalized)),
        "episode_length":             float(np.mean(total_lengths)),
    }


@hydra.main(config_path=".", config_name="eval_return")
def main(cfg):
    work_dir = Path.cwd()
    print(f"workspace: {work_dir}")

    utils.set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)

    # Environment
    obs_type   = cfg.get("obs_type", "pixels")
    pixel_size = cfg.get("pixel_size", 64)
    env = dmc.make(cfg.task, seed=cfg.seed, obs_type=obs_type,
               pixel_size=pixel_size, action_repeat=cfg.get("action_repeat", 2))

    # Agent (built once, weights are swapped per snapshot inside the loop)
    agent = hydra.utils.instantiate(
        cfg.agent,
        obs_shape=env.observation_spec().shape,
        action_shape=env.action_spec().shape,
    )

    cfg.agent.obs_shape = list(env.observation_spec().shape)
    cfg.agent.action_shape = list(env.action_spec().shape)

    # WandB
    exp_name = str(cfg.exp_name)
    wandb_config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb_kwargs = {
        "project": cfg.project,
        "name": exp_name,
        "config": wandb_config,
        "settings": wandb.Settings(start_method="thread", _disable_stats=True),
        "mode": "online" if cfg.use_wandb else "offline",
        "notes": cfg.notes,
    }
    if cfg.get("wandb_run_id", None):
        wandb_kwargs["id"]     = str(cfg.wandb_run_id)
        wandb_kwargs["resume"] = "allow"

    wandb.init(**wandb_kwargs)

    # Snapshot list (single-run-of-wandb, multi-snapshot loop)
    snapshots = cfg.get("eval_snapshots", None)
    if snapshots is None or len(snapshots) == 0:
        # Backwards-compat: single snapshot via agent.path
        assert cfg.agent.path is not None, "Either eval_snapshots or agent.path must be set."
        snapshots = [cfg.agent.path]

    for snap_path in snapshots:
        print(f"\n[Eval] Loading snapshot: {snap_path}")
        payload = torch.load(snap_path, map_location=device)
        agent.mdp.load_state_dict(payload["model"])
        agent.mdp.eval()

        try:
            global_step = int(Path(snap_path).stem.split("_")[-1])
        except Exception:
            global_step = 0

        print(f"[Eval] step={global_step} | "
              f"T_cond={cfg.agent.T_cond} | T_pred={cfg.agent.T_pred} | "
              f"replan_freq={cfg.agent.replan_freq} | "
              f"episodes={cfg.num_eval_episodes}")

        metrics = eval_masking(
            global_step,
            agent,
            env,
            cfg.num_eval_episodes,
            cfg.discount,
            cfg.agent.T_cond,
        )

        # Log to wandb with explicit step so the curve plots against snapshot step.
        if cfg.use_wandb:
            wandb_data = {f"eval/{k}": v for k, v in metrics.items()}
            wandb_data["eval/snapshot_step"] = global_step
            wandb.log(wandb_data, step=global_step)

        print(f"  return={metrics['episode_return']:.2f} "
              f"(norm={metrics['episode_return_normalized']:.3f}) "
              f"post_warmup={metrics['episode_return_postwarmup']:.2f} "
              f"len={metrics['episode_length']:.1f}")

    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()