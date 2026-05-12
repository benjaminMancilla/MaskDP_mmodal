import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs

import dmc
import utils
from logger import Logger
from replay_buffer import make_replay_loader
from video import VideoRecorder
import wandb
import omegaconf
import agent.mdp_goal as mdp_goal_module

torch.backends.cudnn.benchmark = True


def get_dir(cfg):
    resume_dir = Path(cfg.resume_dir)
    snapshot = resume_dir / str(cfg.seed) / f"snapshot_{cfg.resume_step}.pt"
    print("loading from", snapshot)
    return snapshot


def get_domain(task):
    if task.startswith("point_mass_maze"):
        return "point_mass_maze"
    return task.split("_", 1)[0]


def get_data_seed(seed, num_data_seeds):
    return (seed - 1) % num_data_seeds + 1

def create_goal_agent_from_snapshot(pretrain_agent, device, cfg):
    """Creates a goal-conditioned agent loading the pretrain agent weights."""
    goal_agent = mdp_goal_module.MDPGoalAgent(
        name="mdp_goal",
        obs_shape=cfg.agent.obs_shape,
        action_shape=cfg.agent.action_shape,
        device=device,
        lr=cfg.agent.lr,
        batch_size=cfg.agent.batch_size,
        use_tb=cfg.use_tb,
        finetune='decoder',
        transformer_cfg=pretrain_agent.config,
        path=None,
    )
    goal_agent.mdp.load_state_dict(pretrain_agent.model.state_dict())
    for param in goal_agent.mdp.parameters():
        param.requires_grad = False
    goal_agent.train(training=False)
    return goal_agent


def eval_goal_reaching(global_step, goal_agent, env, logger,
                       goal_iter, device, num_eval_episodes,
                       video_recorder, replan=False):
    """Evaluate goal-reaching performance."""
    step, episode, total_dist2goal = 0, 0, []
    eval_until_episode = utils.Until(num_eval_episodes)
    batch = next(goal_iter)
    start_obs, start_physics, goal_obs, goal_physics, timestep = utils.to_torch(
        batch, device
    )
    while eval_until_episode(episode):
        time_step = env.reset()
        with env.physics.reset_context():
            env.physics.set_state(start_physics[episode].cpu())
        dist2goal = 1e6
        video_recorder.init(env, enabled=False)

        if not replan:
            with torch.no_grad(), utils.eval_mode(goal_agent):
                actions = goal_agent.act(
                    start_obs[episode].unsqueeze(0),
                    goal_obs[episode].unsqueeze(0),
                    timestep[episode],
                )
            for a in actions:
                time_step = env.step(a)
                step += 1
                dist = np.linalg.norm(
                    time_step.observation - goal_obs[episode].cpu().numpy()
                )
                dist2goal = min(dist2goal, dist)
        else:
            obs = start_obs[episode]
            for t in range(timestep[episode]):
                with torch.no_grad(), utils.eval_mode(goal_agent):
                    action = goal_agent.act(
                        obs.unsqueeze(0),
                        goal_obs[episode].unsqueeze(0),
                        timestep[episode] - t,
                    )[0, ...]
                time_step = env.step(action)
                obs = np.asarray(time_step.observation)
                obs = torch.as_tensor(obs, device=device)
                dist = np.linalg.norm(
                    time_step.observation - goal_obs[episode].cpu().numpy()
                )
                dist2goal = min(dist2goal, dist)
                step += 1

        episode += 1
        total_dist2goal.append(dist2goal)

    return {
        "distance2goal": np.mean(total_dist2goal),
        "std": np.std(total_dist2goal),
        "episode_length": step / episode,
    }


@hydra.main(config_path=".", config_name="pretrain_full")
def main(cfg):
    work_dir = Path.cwd()
    print(f"workspace: {work_dir}")

    utils.set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)

    # create envs
    obs_type   = cfg.get("obs_type", "states")
    pixel_size = cfg.get("pixel_size", 84)
    env = dmc.make(cfg.task, seed=cfg.seed, obs_type=obs_type, pixel_size=pixel_size)

    # create agent
    agent = hydra.utils.instantiate(
        cfg.agent,
        obs_shape=env.observation_spec().shape,
        action_shape=env.action_spec().shape,
    )

    if cfg.resume is True:
        resume_dir = get_dir(cfg)
        payload = torch.load(resume_dir)
        agent.model.load_state_dict(payload["model"])

    domain = get_domain(cfg.task)
    if cfg.get("resume", False):
        snapshot_dir = Path(cfg.snapshot_dir) / domain / str(cfg.seed)
    else:
        snapshot_dir = work_dir / Path(cfg.snapshot_dir) / domain / str(cfg.seed)
    snapshot_dir.mkdir(exist_ok=True, parents=True)

    # create logger
    cfg.agent.obs_shape = env.observation_spec().shape
    cfg.agent.action_shape = env.action_spec().shape
    exp_name = str(cfg.exp_name) if cfg.get("exp_name", None) else "_".join([cfg.agent.name, domain, str(cfg.seed)])
    wandb_config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb_kwargs = {
        "project": cfg.project,
        "name": exp_name,
        "config": wandb_config,
        "settings": wandb.Settings(start_method="thread", _disable_stats=True),
        "mode": "online" if cfg.use_wandb else "offline",
        "notes": cfg.notes,
    }
    if cfg.get("wandb_id", None):
        wandb_kwargs["id"] = str(cfg.wandb_id)
        wandb_kwargs["resume"] = "allow"
    wandb.init(**wandb_kwargs)
    logger = Logger(work_dir, use_tb=cfg.use_tb, use_wandb=cfg.use_wandb)

    if cfg.get("use_raw_replay_dir", False):
        replay_train_dir = Path(cfg.replay_buffer_dir)
    else:
        replay_train_dir = Path(cfg.replay_buffer_dir) / domain
    print(f"replay dir: {replay_train_dir}")
    train_loader = make_replay_loader(
        env,
        replay_train_dir,
        cfg.replay_buffer_size,
        cfg.batch_size,
        cfg.replay_buffer_num_workers,
        cfg.discount,
        domain,
        cfg.agent.transformer_cfg.traj_length,
        relabel=False,
        file_split=cfg.get("train_file_split", "all"),
        train_ratio=cfg.get("train_ratio", 0.8),
        eval_ratio=cfg.get("eval_ratio", 0.1),
        bc_ratio=cfg.get("bc_ratio", 0.1),
    )
    train_iter = iter(train_loader)

    # Goal evaluation loader
    goal_iter = None
    video_recorder = VideoRecorder(work_dir if cfg.save_video else None)
    if hasattr(cfg, 'goal_buffer_dir') and cfg.goal_buffer_dir is not None:
        goal_dir = Path(cfg.goal_buffer_dir) / cfg.task
        if goal_dir.exists():
            print(f"goal evaluation dir: {goal_dir}")
            goal_loader = make_replay_loader(
                env,
                goal_dir,
                cfg.goal_buffer_size,
                cfg.num_goal_eval_episodes,
                cfg.goal_buffer_num_workers,
                cfg.discount,
                domain=domain,
                traj_length=1,
                mode="goal",
                cfg=cfg.agent.transformer_cfg,
                relabel=False,
            )
            goal_iter = iter(goal_loader)

    # create video recorders

    timer = utils.Timer()

    global_step = cfg.resume_step

    train_until_step = utils.Until(cfg.num_grad_steps)
    eval_every_step = utils.Every(cfg.eval_every_steps)
    log_every_step = utils.Every(cfg.log_every_steps)

    while train_until_step(global_step):
        # try to evaluate
        metrics = agent.update(train_iter, global_step)
        logger.log_metrics(metrics, global_step, ty="train")
        if log_every_step(global_step):
            elapsed_time, total_time = timer.reset()
            with logger.log_and_dump_ctx(global_step, ty="train") as log:
                log("fps", cfg.log_every_steps / elapsed_time)
                log("total_time", total_time)
                log("step", global_step)

        if goal_iter is not None and eval_every_step(global_step):
            print(f"[{global_step}] Running goal-reaching evaluation...")
            goal_agent = create_goal_agent_from_snapshot(agent, device, cfg)

            with torch.no_grad():
                metrics_openloop = eval_goal_reaching(
                    global_step, goal_agent, env, logger,
                    goal_iter, device, cfg.num_goal_eval_episodes,
                    video_recorder, replan=False,
                )
            for key, value in metrics_openloop.items():
                logger.log_metrics({f"goal_openloop/{key}": value}, global_step, ty="eval")

            with torch.no_grad():
                metrics_replan = eval_goal_reaching(
                    global_step, goal_agent, env, logger,
                    goal_iter, device, cfg.num_goal_eval_episodes,
                    video_recorder, replan=True,
                )
            for key, value in metrics_replan.items():
                logger.log_metrics({f"goal_replan/{key}": value}, global_step, ty="eval")

            logger.dump(global_step, ty="eval")
            del goal_agent
            torch.cuda.empty_cache()
            agent.train(training=True)

        if global_step in cfg.snapshots:
            snapshot = snapshot_dir / f"snapshot_{global_step}.pt"
            payload = {
                "model": agent.model.state_dict(),
                "cfg": cfg.agent.transformer_cfg,
            }
            with snapshot.open("wb") as f:
                torch.save(payload, f)

        global_step += 1


if __name__ == "__main__":
    main()
