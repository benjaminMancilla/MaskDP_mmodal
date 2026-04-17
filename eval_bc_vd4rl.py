# eval_bc_vd4rl.py
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
from logger import Logger
from replay_buffer import make_replay_loader
import wandb
import omegaconf

from agent.bc_pixel import BCPixelAgent

torch.backends.cudnn.benchmark = True


def get_domain(task):
    if task.startswith("point_mass_maze"):
        return "point_mass_maze"
    return task.split("_", 1)[0]


def eval_bc(
    global_step,
    agent,
    env,
    num_eval_episodes,
    device,
    discount,
):
    """
    Runs num_eval_episodes in dm_control live.
    Returns dict with mean/std of episode return, normalized to [0, 100].
    """
    total_returns = []
    total_lengths = []

    for episode in range(num_eval_episodes):
        agent.reset_obs_buffer()
        time_step = env.reset()
        episode_return = 0.0
        episode_length = 0
        discount_acc = 1.0

        while not time_step.last():
            obs = time_step.observation  # (C, H, W) uint8
            with torch.no_grad():
                action = agent.act(obs, step=global_step)
            time_step = env.step(action)
            episode_return += time_step.reward * discount_acc
            discount_acc   *= discount
            episode_length += 1

        total_returns.append(episode_return)
        total_lengths.append(episode_length)

    # Normalize to [0, 100] as V-D4RL paper (max return ~1000)
    normalized_returns = [r / 10.0 for r in total_returns]

    return {
        "eval/episode_return":            np.mean(total_returns),
        "eval/episode_return_normalized": np.mean(normalized_returns),
        "eval/episode_return_std":        np.std(normalized_returns),
        "eval/episode_length":            np.mean(total_lengths),
    }


@hydra.main(config_path=".", config_name="eval_bc_vd4rl")
def main(cfg):
    work_dir = Path.cwd()
    print(f"workspace: {work_dir}")

    utils.set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)

    # Environment
    obs_type   = cfg.get("obs_type", "pixels")
    pixel_size = cfg.get("pixel_size", 64)
    env = dmc.make(cfg.task, seed=cfg.seed, obs_type=obs_type, pixel_size=pixel_size)

    obs_shape    = (pixel_size, pixel_size, 3)   # HWC — formato del replay buffer
    action_shape = env.action_spec().shape

    # Agent
    pretrain_path = cfg.get("pretrain_path", None)
    agent = BCPixelAgent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        device=device,
        lr=cfg.agent.lr,
        context_length=cfg.agent.context_length,
        hidden_dim=cfg.agent.hidden_dim,
        use_tb=cfg.use_tb,
        transformer_cfg=None,           # se lee del snapshot
        freeze_encoder=cfg.agent.freeze_encoder,
        path=pretrain_path,
    )

    # Logger + WandB
    exp_name = str(cfg.exp_name)
    wandb_config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(
        project=cfg.project,
        name=exp_name,
        config=wandb_config,
        settings=wandb.Settings(start_method="thread", _disable_stats=True),
        mode="online" if cfg.use_wandb else "offline",
        notes=cfg.notes,
    )
    logger = Logger(work_dir, use_tb=cfg.use_tb, use_wandb=cfg.use_wandb)

    # BC finetuning data loader — split "bc"
    domain = get_domain(cfg.task)
    bc_dir = Path(cfg.replay_buffer_dir)
    print(f"BC data dir: {bc_dir}")

    bc_loader = make_replay_loader(
        env,
        bc_dir,
        cfg.replay_buffer_size,
        cfg.batch_size,
        cfg.replay_buffer_num_workers,
        cfg.discount,
        domain,
        cfg.agent.context_length,
        relabel=False,
        file_split="bc",
        train_ratio=cfg.get("train_ratio", 0.8),
        eval_ratio=cfg.get("eval_ratio", 0.1),
        bc_ratio=cfg.get("bc_ratio", 0.1),
    )
    bc_iter = iter(bc_loader)

    # Snapshot dir for BC head weights
    snapshot_dir = work_dir / Path(cfg.snapshot_dir) / domain / str(cfg.seed)
    snapshot_dir.mkdir(exist_ok=True, parents=True)

    # Training + eval loop
    train_until_step = utils.Until(cfg.num_grad_steps)
    eval_every_step  = utils.Every(cfg.eval_every_steps)
    log_every_step   = utils.Every(cfg.log_every_steps)

    global_step = 0
    timer = utils.Timer()

    while train_until_step(global_step):

        # BC finetuning step
        metrics = agent.update(bc_iter, global_step)
        logger.log_metrics(metrics, global_step, ty="train")

        # Periodic eval in live env
        if eval_every_step(global_step):
            print(f"[{global_step}] Running BC eval ({cfg.num_eval_episodes} episodes)...")
            agent.train(training=False)
            eval_metrics = eval_bc(
                global_step,
                agent,
                env,
                cfg.num_eval_episodes,
                device,
                cfg.discount,
            )
            agent.train(training=True)

            for key, value in eval_metrics.items():
                logger.log_metrics({key: value}, global_step, ty="eval")
            logger.dump(global_step, ty="eval")

            print(f"  return={eval_metrics['eval/episode_return']:.1f} "
                  f"(normalized={eval_metrics['eval/episode_return_normalized']:.1f})")

        if log_every_step(global_step):
            elapsed_time, total_time = timer.reset()
            with logger.log_and_dump_ctx(global_step, ty="train") as log:
                log("fps", cfg.log_every_steps / elapsed_time)
                log("total_time", total_time)
                log("step", global_step)

        # Save BC head snapshot
        if global_step in cfg.snapshots:
            snapshot = snapshot_dir / f"bc_snapshot_{global_step}.pt"
            agent.save(snapshot)
            print(f"Saved BC snapshot: {snapshot}")

        global_step += 1

    # Final eval
    print("Final evaluation...")
    agent.train(training=False)
    final_metrics = eval_bc(
        global_step, agent, env, cfg.num_eval_episodes, device, cfg.discount
    )
    for key, value in final_metrics.items():
        logger.log_metrics({key: value}, global_step, ty="eval")
    logger.dump(global_step, ty="eval")
    print(f"Final return: {final_metrics['eval/episode_return']:.1f} "
          f"(normalized={final_metrics['eval/episode_return_normalized']:.1f})")


if __name__ == "__main__":
    main()