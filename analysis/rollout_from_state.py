import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import procgen
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.mdp_bct import BCTEvalAgentMultimodal
from analysis.trajectory_recorder import (
    VecExtractDictObs,
    load_buffers,
    load_trajectory_states,
    set_env_state,
)


def play_branch(agent, env_name, distribution_mode, state, horizon):
    env = procgen.ProcgenEnv(
        num_envs=1, env_name=env_name, num_levels=1,
        start_level=0, distribution_mode=distribution_mode,
    )
    env = VecExtractDictObs(env, "rgb")
    env.reset()
    set_env_state(env, state["env_state"])

    load_buffers(agent, state["obs_buffer"], state["action_buffer"])
    with torch.no_grad():
        # Read the decision for the restored context via logits_for() (no buffer
        # mutation), then apply it manually -- mirrors what act() does internally.
        # Every step after this is a normal closed-loop agent.act() call.
        action0 = int(torch.argmax(agent.logits_for()).item())
    agent._action_buffer.append(action0)

    obs, reward, done, infos = env.step(np.array([action0]))
    frames = [state["obs_buffer"][-1], obs[0].copy()]
    actions = [action0]
    rewards = [float(reward[0])]
    dones = [bool(done[0])]

    while not dones[-1] and len(actions) < horizon:
        action = agent.act(obs)
        obs, reward, done, infos = env.step(action)
        frames.append(obs[0].copy())
        actions.append(int(action[0]))
        rewards.append(float(reward[0]))
        dones.append(bool(done[0]))

    won = dones[-1] and bool(infos[0].get("prev_level_complete", 0))
    env.close()

    return {
        "frames": np.stack(frames),
        "actions": np.array(actions),
        "rewards": np.array(rewards),
        "dones": np.array(dones),
        "total_reward": float(np.sum(rewards)),
        "length": len(actions),
        "won": won,
        "truncated": not dones[-1],
    }


def load_cases(args):
    if args.qualitative_pkl:
        with open(args.qualitative_pkl, "rb") as f:
            states = pickle.load(f)
        assert states, f"{args.qualitative_pkl} is empty -- no disagreement states to roll out from."
        cases = [(f"q{i:04d}_t{s['t']}", s) for i, s in enumerate(states)]
    else:
        assert args.name, "--name is required together with --states-dir."
        levels = load_trajectory_states(args.states_dir, args.name)
        cases = [(f"L{level}_t{s['t']}", s) for level, snaps in levels.items() for s in snaps]
        assert cases, f"No snapshots found for name={args.name} in {args.states_dir}."

    if args.limit:
        cases = cases[: args.limit]
    return cases


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--qualitative-pkl", default=None)
    source.add_argument("--states-dir", default=None)
    parser.add_argument("--name", default=None, help="Required with --states-dir.")

    parser.add_argument("--model-a-name", required=True)
    parser.add_argument("--model-a-path", required=True)
    parser.add_argument("--model-b-name", required=True)
    parser.add_argument("--model-b-path", required=True)

    parser.add_argument("--env-name", default="coinrun")
    parser.add_argument("--distribution-mode", default="easy")
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    assert args.distribution_mode == "easy", (
        f"distribution_mode='{args.distribution_mode}' -- 'easy' is required, "
        "it's the distribution the shared frozen encoder was trained on."
    )

    cases = load_cases(args)
    device = torch.device(args.device)
    obs_shape, action_shape = (64, 64, 3), (1,)

    agent_a = BCTEvalAgentMultimodal(
        obs_shape=obs_shape, action_shape=action_shape, device=device,
        temperature=1.0, sample=False, path=args.model_a_path,
    )
    agent_b = BCTEvalAgentMultimodal(
        obs_shape=obs_shape, action_shape=action_shape, device=device,
        temperature=1.0, sample=False, path=args.model_b_path,
    )
    assert agent_a.K == agent_b.K, (
        f"K mismatch: {args.model_a_name}={agent_a.K} vs {args.model_b_name}={agent_b.K}. "
        "Rolling out from a shared context requires the same K."
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_first_disagree = 0
    for case_id, state in cases:
        branch_a = play_branch(agent_a, args.env_name, args.distribution_mode, state, args.horizon)
        branch_b = play_branch(agent_b, args.env_name, args.distribution_mode, state, args.horizon)

        first_disagree = branch_a["actions"][0] != branch_b["actions"][0]
        n_first_disagree += int(first_disagree)

        out_path = out_dir / f"rollout_{case_id}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump({
                "case_id": case_id,
                "t": state["t"],
                "source": state.get("source"),
                args.model_a_name: branch_a,
                args.model_b_name: branch_b,
            }, f)

        print(f"[rollout_from_state] {case_id}: first action {args.model_a_name}={branch_a['actions'][0]} "
              f"{args.model_b_name}={branch_b['actions'][0]} ({'DIFFER' if first_disagree else 'same'}) | "
              f"{args.model_a_name} won={branch_a['won']} len={branch_a['length']} | "
              f"{args.model_b_name} won={branch_b['won']} len={branch_b['length']}")

    print(f"\n[rollout_from_state] {len(cases)} case(s) -> {out_dir}. "
          f"First-action disagreement: {n_first_disagree}/{len(cases)}")


if __name__ == "__main__":
    main()
