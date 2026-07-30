import pickle
from collections import deque
from pathlib import Path

import numpy as np
import procgen


class VecExtractDictObs:

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


def _unwrap_procgen(env):
    """
    Unwraps the environment to find the underlying object that exposes
    `get_state()` and `set_state()`. Fails loudly if neither method is found
    in the known wrapper chain.
    """
    candidates = [env.venv, getattr(env.venv, "env", None), getattr(env.venv, "gym3_env", None)]
    for c in candidates:
        if c is not None and hasattr(c, "get_state") and hasattr(c, "set_state"):
            return c
    raise RuntimeError(
        "Could not find get_state()/set_state() in the wrapper chain. "
        "Checked candidates: env.venv, env.venv.env, env.venv.gym3_env. "
        "Check your procgen version and adjust _unwrap_procgen()."
    )


def get_env_state(env):
    raw = _unwrap_procgen(env)
    return raw.get_state()[0]  # num_envs=1 -> list[bytes] of length 1


def set_env_state(env, state_bytes):
    raw = _unwrap_procgen(env)
    raw.set_state([state_bytes])


def record_reference_trajectory(agent, env_name, level, distribution_mode):
    """
    Performs a real rollout of `agent` on a single fixed level and saves it 
    in a reusable format. It is architecture-agnostic, working with any agent 
    that exposes `.reset()`, `.act(obs)`, and `.sample`.

    Records the exact environment state and buffer states at each step so
    the exact context can be reproduced later for evaluation.

    Buffer capture order: `action_buffer` is captured BEFORE calling `act()`
    (pre-decision state), and `obs_buffer` is captured AFTER (to include
    the newly observed frame).

    Requires `agent.sample=False` (argmax) for exploitation, not exploration.
    """
    assert not agent.sample, "record_reference_trajectory requires agent.sample=False (argmax)."

    env = procgen.ProcgenEnv(
        num_envs=1, env_name=env_name, num_levels=1,
        start_level=level, distribution_mode=distribution_mode,
    )
    env = VecExtractDictObs(env, "rgb")

    agent.reset()
    obs = env.reset()
    done = False
    t = 0
    steps = []

    while not done:
        env_state = get_env_state(env)
        action_buffer_snapshot = list(agent._action_buffer)          # pre-decision
        action = agent.act(obs)                                      # decide + append
        obs_buffer_snapshot = [f.copy() for f in agent._obs_buffer]  # post-append

        steps.append({
            "t": t,
            "env_state": env_state,
            "obs_buffer": obs_buffer_snapshot,
            "action_buffer": action_buffer_snapshot,
            "traj_action": int(action[0]),
        })

        obs, reward, done, infos = env.step(action)
        t += 1

    env.close()
    return steps


def quantile_indices(episode_len, num_states):
    if episode_len <= num_states:
        return list(range(episode_len))
    fracs = np.linspace(0.05, 0.95, num_states)
    idx = np.unique(np.round(fracs * (episode_len - 1)).astype(int))
    return idx.tolist()


def select_quantile_states(steps, num_states):
    idx = quantile_indices(len(steps), num_states)
    return [steps[i] for i in idx]


def save_trajectory_states(out_root, name, arch, seed, checkpoint_step, trajectories):
    """`trajectories`: {level: [state, ...]}. Returns the written path."""
    states_path = Path(out_root) / f"states_{name}.pkl"
    with open(states_path, "wb") as f:
        pickle.dump({
            "levels": trajectories,
            "arch": arch,
            "seed": seed,
            "checkpoint_step": checkpoint_step,
        }, f)
    return states_path


def load_trajectory_states(out_root, name):
    path = Path(out_root) / f"states_{name}.pkl"
    assert path.exists(), f"Missing {path}. Record a trajectory for '{name}' first."
    with open(path, "rb") as f:
        return pickle.load(f)["levels"]


def load_buffers(agent, obs_buffer_snapshot, action_buffer_snapshot):
    """
    Loads a saved snapshot back into the agent's buffers for teacher forcing. 
    The agent does not need to be the one that originally recorded it.
    """
    assert len(obs_buffer_snapshot) <= agent.K, (
        f"obs_buffer_snapshot has {len(obs_buffer_snapshot)} frames but "
        f"agent.K={agent.K}. The agent replaying this snapshot must share "
        f"the K it was recorded with."
    )
    agent._obs_buffer = deque(obs_buffer_snapshot, maxlen=agent.K)
    agent._action_buffer = deque(action_buffer_snapshot, maxlen=agent.K - 1)
