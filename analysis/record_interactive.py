"""
Play Procgen with the keyboard and save marked states (env state + K-frame /
(K-1)-action history), in the same format `trajectory_recorder.py` uses for
policy-recorded trajectories.

Controls:
  F5             mark the current state, saved once the next action is known
  arrow keys     move
  ESCAPE         quit

Also auto-saves the state right before a win (tagged "source": "auto_win").
Deaths and timeouts save nothing unless you pressed F5.

Requires a display and procgen/gym3 installed.
"""
import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
from gym3 import Interactive, unwrap
from procgen import ProcgenGym3Env

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.trajectory_recorder import save_trajectory_states


class RecordingInteractive(Interactive):

    def __init__(self, env, k, level, out_dir, name, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.k = k
        self.level = level
        self.out_dir = Path(out_dir)
        self.name = name

        self._obs_hist = deque(maxlen=k)
        self._action_hist = deque(maxlen=k - 1)
        self._pending_snapshot = None
        self.snapshots = []

        _, obs0, _ = self._env.observe()
        self._obs_hist.append(obs0["rgb"][0].copy())

    def _snapshot_now(self):
        return {
            "t": self._steps,
            "env_state": unwrap(self._env).get_state()[0],
            "obs_buffer": list(self._obs_hist),
            "action_buffer": list(self._action_hist),
        }

    def _mark_snapshot(self):
        self._pending_snapshot = self._snapshot_now()
        print(f"[record] snapshot marked at t={self._steps}, finalizing on next action...")

    def _finalize(self, snapshot, action_int, source):
        snapshot["traj_action"] = action_int
        snapshot["source"] = source
        self.snapshots.append(snapshot)
        path = self._persist()
        print(f"[record] snapshot finalized (source={source}, traj_action={action_int}), "
              f"total={len(self.snapshots)}, written to {path}")

    def _persist(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = save_trajectory_states(
            self.out_dir, self.name, arch="human", seed=None,
            checkpoint_step=None, trajectories={self.level: self.snapshots},
        )
        return path

    def _update(self, dt, keys_clicked, keys_pressed):
        if "F5" in keys_clicked:
            self._mark_snapshot()
        super()._update(dt, keys_clicked, keys_pressed)

    def _act(self, ac):
        action_int = int(np.asarray(ac).reshape(-1)[0])

        # Taken BEFORE stepping: if this action ends the episode with a win,
        # env_state is only recoverable now -- procgen auto-resets internally,
        # so get_state() right after the step returns the NEXT episode's state,
        # not the one this action was taken from.
        prestep_snapshot = self._snapshot_now()

        if self._pending_snapshot is not None:
            self._finalize(self._pending_snapshot, action_int, source="manual")
            self._pending_snapshot = None

        first = super()._act(ac)
        won = first and bool(self._last_info.get("prev_level_complete", 0))
        if won:
            self._finalize(prestep_snapshot, action_int, source="auto_win")

        obs_frame = self._last_ob["rgb"][0].copy()
        if first:
            print(f"[record] episode boundary -- history reset (won={bool(won)}, "
                  f"prev_level_complete={self._last_info.get('prev_level_complete')}, "
                  f"reward={self._last_rew})")
            self._obs_hist.clear()
            self._action_hist.clear()
            self._obs_hist.append(obs_frame)
        else:
            self._action_hist.append(action_int)
            self._obs_hist.append(obs_frame)

        return first


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, required=True,
                         help="Context window length. MUST match agent.K of the "
                              "model(s) that will later load these snapshots -- "
                              "load_buffers() asserts len(obs_buffer) <= agent.K.")
    parser.add_argument("--env-name", default="coinrun")
    parser.add_argument("--level-seed", type=int, required=True,
                         help="Pins start_level + num_levels=1 (plan invariant: "
                              "iterate explicit levels, never leave this open).")
    parser.add_argument("--distribution-mode", default="easy",
                         help="Procgen's own default is 'hard'; this repo's "
                              "invariant is 'easy' (matches the frozen encoder's "
                              "training distribution). Overriding this is a red flag.")
    parser.add_argument("--out-dir", required=True,
                         help="Directory for states_{name}.pkl, same layout "
                              "trajectory_recorder.save_trajectory_states uses.")
    parser.add_argument("--name", required=True,
                         help="Tag for the output file, e.g. 'human_l250'.")
    args = parser.parse_args()

    if args.distribution_mode != "easy":
        print(f"[record] WARNING: --distribution-mode={args.distribution_mode!r}, "
              f"not 'easy'. Snapshots would be off-distribution for the frozen "
              f"encoder unless that's intentional.")

    env = ProcgenGym3Env(
        num=1,
        env_name=args.env_name,
        start_level=args.level_seed,
        num_levels=1,
        distribution_mode=args.distribution_mode,
    )

    ia = RecordingInteractive(
        env, k=args.k, level=args.level_seed, out_dir=args.out_dir, name=args.name,
        ob_key="rgb", width=64 * 12, height=64 * 12,
    )

    print("[record] F5 = mark snapshot (finalized on next action, written to "
          "disk immediately -- nothing lost if the window is closed or crashes). "
          "ESCAPE to quit.")

    ia.run()
    path = ia._persist()
    print(f"[record] done. {len(ia.snapshots)} snapshot(s) at {path}")


if __name__ == "__main__":
    main()
