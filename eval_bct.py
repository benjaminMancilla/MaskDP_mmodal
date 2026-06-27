import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
from pathlib import Path

import hydra
import numpy as np
import torch
import procgen
import omegaconf

import utils
import wandb

torch.backends.cudnn.benchmark = True

class VecExtractDictObs:
    """
    Copy paste without any extra dependencies (newer version of gym) of: 
    `baselines.common.vec_env.VecExtractDictObs` (openai/baselines).
    """
 
    def __init__(self, venv, key):
        self.venv = venv
        self.key = key
        self.observation_space = venv.observation_space.spaces[key]
        self.action_space = venv.action_space
        self.num_envs = venv.num_envs
 
    def reset(self):
        obs = self.venv.reset()
        return obs[self.key]
 
    def step(self, action):
        obs, rew, done, info = self.venv.step(action)
        return obs[self.key], rew, done, info
 
    def close(self):
        return self.venv.close()


PROCGEN = {
    "bigfish":    {"easy": (1, 40),    "hard": (0, 40)},
    "bossfight":  {"easy": (0.5, 13),  "hard": (0.5, 13)},
    "caveflyer":  {"easy": (3.5, 12),  "hard": (2, 13.4)},
    "chaser":     {"easy": (0.5, 13),  "hard": (0.5, 14.2)},
    "climber":    {"easy": (2, 12.6),  "hard": (1, 12.6)},
    "coinrun":    {"easy": (5, 10),    "hard": (5, 10)},
    "dodgeball":  {"easy": (1.5, 19),  "hard": (1.5, 19)},
    "fruitbot":   {"easy": (-1.5, 32.4), "hard": (-0.5, 27.2)},
    "heist":      {"easy": (3.5, 10),  "hard": (2, 10)},
    "jumper":     {"easy": (1, 10),    "hard": (1, 10)},
    "leaper":     {"easy": (1.5, 10),  "hard": (1.5, 10)},
    "maze":       {"easy": (5, 10),    "hard": (4, 10)},
    "miner":      {"easy": (1.5, 13),  "hard": (1.5, 20)},
    "ninja":      {"easy": (3.5, 10),  "hard": (2, 10)},
    "plunder":    {"easy": (4.5, 30),  "hard": (3, 30)},
    "starpilot":  {"easy": (2.5, 64),  "hard": (1.5, 35)},
}


def normalize_return(raw_return, env_name, distribution_mode):
    r_min, r_max = PROCGEN[env_name][distribution_mode]
    return (raw_return - r_min) / (r_max - r_min)


def eval_bct_split(
    agent,
    device,
    env_name="coinrun",
    num_levels=0,
    start_level=0,
    distribution_mode="easy",
    num_episodes=100,
):
    env = procgen.ProcgenEnv(
        num_envs=1,
        env_name=env_name,
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode=distribution_mode,
    )
    env = VecExtractDictObs(env, "rgb")

    agent.eval()
    eval_episode_rewards = []

    for _ in range(num_episodes):
        agent.reset()                 # clear internal deques
        obs = env.reset()             # (1, H, W, C)
        done = False
        episode_reward = 0.0

        while not done:
            action = agent.act(obs)               # np.ndarray shape (1,), int64
            obs, reward, done, infos = env.step(action)
            episode_reward += reward[0]

        eval_episode_rewards.append(episode_reward)

    env.close()
    mean_raw = float(np.mean(eval_episode_rewards))
    std_raw = float(np.std(eval_episode_rewards))
    return mean_raw, std_raw, eval_episode_rewards


@hydra.main(config_path=".", config_name="eval_bct")
def main(cfg):
    work_dir = Path.cwd()
    print(f"workspace: {work_dir}")

    utils.set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)

    env_name = cfg.env_name
    distribution_mode = cfg.distribution_mode
    assert env_name in PROCGEN, f"{env_name} no está en el dict PROCGEN de normalización."
    assert distribution_mode in PROCGEN[env_name], (
        f"distribution_mode='{distribution_mode}' no definido para env_name='{env_name}' en PROCGEN."
    )
    
    pixel_obs_shape = tuple(cfg.agent.transformer_cfg.pixel_obs_shape) \
        if cfg.agent.get("transformer_cfg") is not None else (64, 64, 3)
    obs_shape = pixel_obs_shape
    action_shape = (1,)

    agent = hydra.utils.instantiate(
        cfg.agent,
        obs_shape=obs_shape,
        action_shape=action_shape,
    )

    cfg.agent.obs_shape = list(obs_shape)
    cfg.agent.action_shape = list(action_shape)

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
        wandb_kwargs["id"] = str(cfg.wandb_run_id)
        wandb_kwargs["resume"] = "allow"
    wandb.init(**wandb_kwargs)

    snapshots = cfg.get("eval_snapshots", None)
    if snapshots is None or len(snapshots) == 0:
        assert cfg.agent.path is not None, "Either eval_snapshots or agent.path must be set."
        snapshots = [cfg.agent.path]

    splits = cfg.splits

    for snap_path in snapshots:
        print(f"\n[Eval BCT] Loading snapshot: {snap_path}")
        payload = torch.load(snap_path, map_location=device)
        agent.mdp.load_state_dict(payload["model"])
        agent.mdp.eval()

        try:
            global_step = int(Path(snap_path).stem.split("_")[-1])
        except Exception:
            global_step = 0

        print(f"[Eval BCT] step={global_step} | env={env_name} | mode={distribution_mode} | "
              f"K={agent.K} | temperature={agent.temperature} | sample={agent.sample} | "
              f"episodes/split={cfg.num_episodes}")

        results = {}
        for split_name, split_cfg in splits.items():
            mean_raw, std_raw, raw_rewards = eval_bct_split(
                agent,
                device,
                env_name=env_name,
                num_levels=split_cfg.num_levels,
                start_level=split_cfg.start_level,
                distribution_mode=distribution_mode,
                num_episodes=cfg.num_episodes,
            )
            norm_rewards = [normalize_return(r, env_name, distribution_mode) for r in raw_rewards]
            mean_norm = float(np.mean(norm_rewards))
            std_norm = float(np.std(norm_rewards))

            results[split_name] = {
                "raw_return": mean_raw,
                "raw_return_std": std_raw,
                "normalized_return": mean_norm,
                "normalized_return_std": std_norm,
            }
            print(f"  [{split_name:>5s}] start_level={split_cfg.start_level:>4d} "
                  f"num_levels={split_cfg.num_levels:>4d} | raw={mean_raw:.2f}±{std_raw:.2f} "
                  f"| norm={mean_norm:.3f}±{std_norm:.3f}")

        if cfg.use_wandb:
            wandb_data = {}
            for split_name, r in results.items():
                for k, v in r.items():
                    wandb_data[f"eval_bct/{split_name}/{k}"] = v
            wandb_data["eval_bct/snapshot_step"] = global_step
            wandb.log(wandb_data, step=global_step)

        out_dir = Path(snap_path).parent
        csv_path = out_dir / "evaluate_bct.csv"
        with open(csv_path, "w") as f:
            f.write("final_test_ret,final_train_ret,final_val_ret,"
                    "final_test_ret_norm,final_train_ret_norm,final_val_ret_norm\n")
            f.write(
                f"{results['test']['raw_return']},{results['train']['raw_return']},"
                f"{results['val']['raw_return']},{results['test']['normalized_return']},"
                f"{results['train']['normalized_return']},{results['val']['normalized_return']}\n"
            )
        print(f"  -> {csv_path}")

    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
