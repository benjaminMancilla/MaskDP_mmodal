import datetime
import io
import random
import traceback
import copy
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
from utils import get_norm


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open("wb") as f:
            f.write(bs.read())


def load_episode(fn, domain, obs):
    with fn.open("rb") as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}

        if "image" in episode and "observation" not in episode:
            episode["observation"] = episode.pop("image")

        for key in ("reward", "discount"):
            if key in episode and episode[key].ndim == 1:
                episode[key] = episode[key][:, np.newaxis].astype(np.float32)

        if "physics" not in episode:
            n = episode["observation"].shape[0]
            episode["physics"] = np.zeros((n, 18), dtype=np.float64)

        return episode


def relable_episode(env, episode):
    rewards = []
    reward_spec = env.reward_spec()
    states = episode["physics"]
    for i in range(states.shape[0]):
        with env.physics.reset_context():
            env.physics.set_state(states[i])
        reward = env.task.get_reward(env.physics)
        reward = np.full(reward_spec.shape, reward, reward_spec.dtype)
        rewards.append(reward)
    episode["reward"] = np.array(rewards, dtype=reward_spec.dtype)
    return episode


class OfflineReplayBuffer(IterableDataset):
    def __init__(
        self,
        env,
        replay_dir,
        max_size,
        num_workers,
        discount,
        domain,
        traj_length,
        mode,
        cfg,
        relabel,
        obs,
        file_split="all",
        train_ratio=0.8,
        eval_ratio=0.1,
        bc_ratio=0.1,
        frame_stack: int = 1,
    ):
        self._env = env
        self._replay_dir = replay_dir
        self._domain = domain
        self._mode = mode
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._discount = discount
        self._loaded = False
        self._traj_length = traj_length
        self._cfg = cfg
        self._relabel = relabel
        self._obs = obs
        self._frame_stack = frame_stack

        assert abs(train_ratio + eval_ratio + bc_ratio - 1.0) < 1e-6, \
            f"train_ratio + eval_ratio + bc_ratio debe ser 1.0, got {train_ratio + eval_ratio + bc_ratio}"

        self._file_split = file_split
        self._train_ratio = train_ratio
        self._eval_ratio = eval_ratio
        self._bc_ratio = bc_ratio

    def _load(self, relable=True):
        if relable:
            print("Labeling data...")
        else:
            print("loading reward free data...")
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(
            self._replay_dir.rglob("*.npz")
        )  # get all episodes recursively

        if self._file_split != "all":
            n_total = len(eps_fns)
            n_train = int(n_total * self._train_ratio)
            n_eval  = int(n_total * self._eval_ratio)
            n_bc    = int(n_total * self._bc_ratio)

            if self._file_split == "train":
                eps_fns = eps_fns[:n_train]
            elif self._file_split == "eval":
                eps_fns = eps_fns[n_train:n_train + n_eval]
            elif self._file_split == "bc":
                eps_fns = eps_fns[n_train + n_eval:n_train + n_eval + n_bc]

            print(f"[split={self._file_split}] {len(eps_fns)}/{n_total} archivos "
                f"(train={n_train}, eval={n_eval}, bc={n_bc})")

        for eps_idx, eps_fn in enumerate(eps_fns):
            if self._size > self._max_size:
                print("over size", self._max_size)
                break
            if eps_idx % self._num_workers != worker_id:
                continue
            episode = load_episode(eps_fn, self._domain, self._obs)
            if relable:
                episode = self._relable_reward(episode)
            self._episode_fns.append(eps_fn)
            self._episodes[eps_fn] = episode
            self._size += episode_len(episode)

    def _stack_pixel_frames(self, frames, indices, k):
        """
        Build a frame-stacked observation for each requested token.

        Args:
            frames:  (N_ep, H, W, 3) uint8 — full episode pixel observations.
            indices: 1-D int array of absolute episode indices (0-indexed).
            k:       number of frames to stack (oldest → newest).
        """
        # j=0 oldest frame, j=k-1 newest frame
        idxs = np.stack(
            [np.clip(indices - (k - 1 - j), 0, None) for j in range(k)],
            axis=1,
        )  # (n, k)
        stacked = frames[idxs]  # (n, k, H, W, 3) — fancy indexing always copies
        # Concatenate frames along the channel axis: oldest RGB first
        return np.concatenate([stacked[:, j] for j in range(k)], axis=-1)  # (n, H, W, k*3)

    def _sample_episode(self):
        if not self._loaded:
            self._load(self._relabel)
            self._loaded = True
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _relable_reward(self, episode):
        return relable_episode(self._env, episode)

    def _sample(self):
        episode = self._sample_episode()
        L = episode_len(episode)

        if L >= self._traj_length:
            # add +1 for the first dummy transition
            idx = np.random.randint(0, L - self._traj_length + 1) + 1

            if self._obs == "pixels":
                frames = episode.get("pixel_observation", episode["observation"])
                if self._frame_stack > 1:
                    obs_abs      = np.arange(idx - 1, idx - 1 + self._traj_length)
                    next_obs_abs = np.arange(idx,     idx     + self._traj_length)
                    obs      = self._stack_pixel_frames(frames, obs_abs,      self._frame_stack)
                    next_obs = self._stack_pixel_frames(frames, next_obs_abs, self._frame_stack)
                else:
                    obs      = frames[idx - 1 : idx - 1 + self._traj_length]
                    next_obs = frames[idx     : idx     + self._traj_length]
            else:
                obs      = episode["observation"][idx - 1 : idx - 1 + self._traj_length]
                next_obs = episode["observation"][idx     : idx     + self._traj_length]

            action   = episode["action"][idx : idx + self._traj_length]
            reward   = episode["reward"][idx : idx + self._traj_length]
            discount = episode["discount"][idx : idx + self._traj_length] * self._discount
            mask     = np.ones((self._traj_length, 1), dtype=np.float32)
            return (obs, action, reward, discount, next_obs, mask)

        # frame_stack > 1 not supported, V-D4RL always is L >= self._traj_length
        if self._frame_stack > 1:
            raise NotImplementedError(
                f"Short Episode (L={L} < traj_length={self._traj_length}) with "
                f"frame_stack={self._frame_stack} > 1"
            )

        # Apply padding of 0 + mask for the missing timesteps
        idx = 1
        pad = self._traj_length - L

        frames = episode.get("pixel_observation", episode["observation"]) if self._obs == "pixels" \
            else episode["observation"]

        obs_real      = frames[idx - 1 : idx - 1 + L]
        next_obs_real = frames[idx     : idx     + L]
        action_real   = episode["action"][idx : idx + L]
        reward_real   = episode["reward"][idx : idx + L]
        discount_real = episode["discount"][idx : idx + L] * self._discount

        def _pad_zeros(arr, n_pad):
            pad_shape = (n_pad,) + arr.shape[1:]
            return np.concatenate([arr, np.zeros(pad_shape, dtype=arr.dtype)], axis=0)

        obs      = _pad_zeros(obs_real, pad)
        next_obs = _pad_zeros(next_obs_real, pad)
        action   = _pad_zeros(action_real, pad)
        reward   = _pad_zeros(reward_real, pad)
        discount = _pad_zeros(discount_real, pad)

        mask = np.zeros((self._traj_length, 1), dtype=np.float32)
        mask[:L] = 1.0

        return (obs, action, reward, discount, next_obs, mask)


    def _sample_goal(self):
        episode = self._sample_episode()
        ep_len = episode_len(episode)
        max_start = max(1, ep_len - 30)
        # add +1 for the first dummy transition
        start_idx = np.random.randint(0, max_start)
        length    = np.random.randint(15, min(20, ep_len - start_idx))
        goal_idx  = start_idx + length - 1

        start_physics = episode["physics"][start_idx]
        goal_physics  = episode["physics"][goal_idx]
        # goal_obs_prop: always proprioceptive "observation", used for the L2 metric
        goal_obs_prop = episode["observation"][goal_idx]
        timestep = length - 1

        if self._obs == "pixels":
            frames = episode.get("pixel_observation", episode["observation"])
            if self._frame_stack > 1:
                start_obs = self._stack_pixel_frames(
                    frames, np.array([start_idx]), self._frame_stack
                )[0]  # (H, W, 3*k)
                goal_obs = self._stack_pixel_frames(
                    frames, np.array([goal_idx]), self._frame_stack
                )[0]  # (H, W, 3*k)
            else:
                start_obs = frames[start_idx]
                goal_obs  = frames[goal_idx]
        else:
            start_obs = episode["observation"][start_idx]
            goal_obs  = goal_obs_prop  # same array in state mode

        return (start_obs, start_physics, goal_obs, goal_obs_prop, goal_physics, timestep)

    def _sample_multiple_goal(self):
        episode = self._sample_episode()
        ep_len = episode_len(episode)
        time_budget  = np.array([12, 24, 36, 48, 60])
        max_start    = max(1, ep_len - time_budget[-1] - 2)
        # add +1 for the first dummy transition
        start_idx    = np.random.randint(0, max_start)
        goal_indices = start_idx + time_budget  # absolute episode indices, shape (5,)

        start_physics = episode["physics"][start_idx]
        goal_physics  = episode["physics"][goal_indices]
        # goal_prop: always proprioceptive, for L2 metric
        goal_prop = episode["observation"][goal_indices]

        if self._obs == "pixels":
            frames = episode.get("pixel_observation", episode["observation"])
            if self._frame_stack > 1:
                start_obs = self._stack_pixel_frames(
                    frames, np.array([start_idx]), self._frame_stack
                )[0]  # (H, W, 3*k)
                goal = self._stack_pixel_frames(
                    frames, goal_indices, self._frame_stack
                )  # (5, H, W, 3*k)
            else:
                start_obs = frames[start_idx]
                goal      = frames[goal_indices]
        else:
            start_obs = episode["observation"][start_idx]
            goal      = goal_prop  # same array in state mode

        return (start_obs, start_physics, goal, goal_prop, goal_physics, time_budget)

    def _sample_context(self):
        episode = self._sample_episode()
        context_length = self._cfg.context_length
        forecast_length = self._cfg.forecast_length
        # add +1 for the first dummy transition
        start_idx = np.random.randint(100, 850)
        obs = episode["observation"][
            start_idx - 1 : start_idx + context_length
        ]  # last state is the initial obs
        action = episode["action"][start_idx : start_idx + context_length]
        reward = episode["reward"][
            start_idx + context_length : start_idx + context_length + forecast_length
        ]
        physics = episode["physics"][start_idx - 1 : start_idx + context_length]
        remaining = episode["action"][
            start_idx + context_length : start_idx + context_length + forecast_length
        ]
        return (obs, action, physics, reward, remaining)

    def __iter__(self):
        while True:
            if self._mode is None:
                yield self._sample()
            elif self._mode == "goal":
                yield self._sample_goal()
            elif self._mode == "multi_goal":
                yield self._sample_multiple_goal()
            elif self._mode == "prompt":
                yield self._sample_context()


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(
    env,
    replay_dir,
    max_size,
    batch_size,
    num_workers,
    discount,
    domain,
    traj_length=1,
    mode=None,
    cfg=None,
    multi_task=False,
    relabel=True,
    obs="states",
    file_split="all",
    train_ratio=0.8,
    eval_ratio=0.1,
    bc_ratio=0.1,
    frame_stack: int = 1,
):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = OfflineReplayBuffer(
        env,
        replay_dir,
        max_size_per_worker,
        num_workers,
        discount,
        domain,
        traj_length,
        mode,
        cfg,
        relabel,
        obs,
        file_split=file_split,
        train_ratio=train_ratio,
        eval_ratio=eval_ratio,
        bc_ratio=bc_ratio,
        frame_stack=frame_stack,
    )

    loader = torch.utils.data.DataLoader(
        iterable,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )
    return loader
