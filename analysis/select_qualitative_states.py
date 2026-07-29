import argparse
import pickle
from pathlib import Path

import numpy as np


def load_cell(sarfa_dir, traj_source, eval_model):
    path = Path(sarfa_dir) / f"traj_{traj_source}__eval_{eval_model}.npz"
    return dict(np.load(path, allow_pickle=False))


def load_states(sarfa_dir, traj_source):
    path = Path(sarfa_dir) / f"states_{traj_source}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def disagreement_mask(sarfa_dir, traj_source, model_a, model_b):
    """
    Compares the predicted actions (argmax of logits) of two models on the same trajectory states. 
    Returns a boolean array indicating where the models disagree (True where their predictions differ).
    """
    cell_a = load_cell(sarfa_dir, traj_source, model_a)
    cell_b = load_cell(sarfa_dir, traj_source, model_b)

    assert np.array_equal(cell_a["levels"], cell_b["levels"]), (
        f"'levels' does not match between cells ({traj_source}, {model_a}) and "
        f"({traj_source}, {model_b}) -- they should come from the SAME eval_sarfa.py run."
    )
    assert np.array_equal(cell_a["timesteps"], cell_b["timesteps"]), (
        f"'timesteps' does not match between cells ({traj_source}, {model_a}) and "
        f"({traj_source}, {model_b})."
    )

    return cell_a["a_hat"] != cell_b["a_hat"], cell_a["levels"], cell_a["timesteps"]


def select_disagreement_states(sarfa_dir, traj_source, model_a, model_b):
    """
    Returns a list of raw state snapshots where `model_a` and `model_b` disagree. 
    These snapshots contain metadata and can be restored using the agent's state or buffer loading methods.

    Matches states by (level, t) instead of relying on iteration order, ensuring 
    robustness across different files and runs.
    """
    mask, levels, timesteps = disagreement_mask(sarfa_dir, traj_source, model_a, model_b)
    states_by_level = load_states(sarfa_dir, traj_source)["levels"]

    lookup = {}
    for level, states in states_by_level.items():
        for s in states:
            lookup[(int(level), int(s["t"]))] = s

    selected = []
    for level, t, disagree in zip(levels, timesteps, mask):
        if not disagree:
            continue
        key = (int(level), int(t))
        assert key in lookup, (
            f"Raw snapshot not found for (level={level}, t={t}) in "
            f"states_{traj_source}.pkl -- ensure it comes from the SAME "
            f"eval_sarfa.py run that produced traj_{traj_source}__eval_{model_a}.npz."
        )
        selected.append(lookup[key])

    print(f"[select_qualitative_states] traj={traj_source} {model_a} vs {model_b}: "
          f"{len(selected)}/{len(mask)} disagreeing states.")
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sarfa_dir", help="'sarfa/' directory produced by eval_sarfa.py")
    parser.add_argument("traj_source", help="name of the model that generated the reference trajectory")
    parser.add_argument("model_a", help="name of a model evaluated on that trajectory (e.g. hier_*)")
    parser.add_argument("model_b", help="name of the other model evaluated on that trajectory (e.g. uni_*)")
    args = parser.parse_args()

    selected = select_disagreement_states(args.sarfa_dir, args.traj_source, args.model_a, args.model_b)

    out_path = Path(args.sarfa_dir) / f"qualitative_{args.traj_source}_{args.model_a}_vs_{args.model_b}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(selected, f)
    print(f"-> {out_path}")
