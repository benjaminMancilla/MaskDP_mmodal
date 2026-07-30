import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.select_qualitative_states import load_cell

TOPK_FRACS = (0.05, 0.10, 0.20)


def discover_models(output_dir):
    """Extracts {name: {arch, seed, checkpoint_step}} from the saved states_*.pkl files."""
    models = {}
    for p in sorted(Path(output_dir).glob("states_*.pkl")):
        name = p.stem[len("states_"):]
        with open(p, "rb") as f:
            meta = pickle.load(f)
        models[name] = {
            "arch": meta["arch"],
            "seed": meta["seed"],
            "checkpoint_step": meta["checkpoint_step"],
        }
    return models


def _weighted_centroid(m):
    m = np.clip(m, 0, None)
    total = m.sum()
    h, w = m.shape
    if total <= 0:
        return np.array([(h - 1) / 2, (w - 1) / 2])
    rows, cols = np.indices(m.shape)
    return np.array([(rows * m).sum() / total, (cols * m).sum() / total])


def map_entropy(m):
    """Shannon entropy of a raw map treated as an unnormalized distribution."""
    m = np.clip(m, 0, None)
    total = m.sum()
    if total <= 0:
        return 0.0
    p = (m / total).flatten()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def per_state_metrics(map_a, map_b):
    flat_a, flat_b = map_a.flatten(), map_b.flatten()
    metrics = {"spearman": float(spearmanr(flat_a, flat_b).correlation)}
    for k in TOPK_FRACS:
        n = max(1, round(k * flat_a.size))
        top_a = set(np.argsort(flat_a)[-n:])
        top_b = set(np.argsort(flat_b)[-n:])
        union = top_a | top_b
        metrics[f"iou_top{int(k * 100)}"] = len(top_a & top_b) / len(union) if union else float("nan")
    metrics["centroid_dist"] = float(np.linalg.norm(_weighted_centroid(map_a) - _weighted_centroid(map_b)))
    return metrics


def paired_indices(cell_a, cell_b):
    """Index pairs (i_a, i_b) matched by (level, t) -- not by row order."""
    key_b = {(int(l), int(t)): i for i, (l, t) in enumerate(zip(cell_b["levels"], cell_b["timesteps"]))}
    pairs = []
    for i, (l, t) in enumerate(zip(cell_a["levels"], cell_a["timesteps"])):
        key = (int(l), int(t))
        assert key in key_b, (
            f"state (level={l}, t={t}) missing in the other cell -- both must "
            f"come from record runs against the same `levels`."
        )
        pairs.append((i, key_b[key]))
    return pairs


def compare_on_trajectory(output_dir, traj_name, model_a, model_b, map_key):
    """
    Metrics for model_a vs model_b, both evaluated on traj_name's states.
    Aggregated per level first, then averaged over levels. This prevents any 
    single level from dominating the metrics if episode lengths are unbalanced.
    """
    cell_a = load_cell(output_dir, traj_name, model_a)
    cell_b = load_cell(output_dir, traj_name, model_b)
    pairs = paired_indices(cell_a, cell_b)

    per_level = defaultdict(list)
    a_hat_match_per_level = defaultdict(list)
    for ia, ib in pairs:
        level = int(cell_a["levels"][ia])
        per_level[level].append(per_state_metrics(cell_a[map_key][ia], cell_b[map_key][ib]))
        a_hat_match_per_level[level].append(int(cell_a["a_hat"][ia]) == int(cell_b["a_hat"][ib]))

    keys = next(iter(per_level.values()))[0].keys()
    level_means = {k: [] for k in keys}
    a_hat_rate_per_level = []
    for level, ms in per_level.items():
        for k in keys:
            level_means[k].append(np.nanmean([m[k] for m in ms]))
        a_hat_rate_per_level.append(np.mean(a_hat_match_per_level[level]))

    result = {k: float(np.nanmean(vals)) for k, vals in level_means.items()}
    result["a_hat_match_rate"] = float(np.mean(a_hat_rate_per_level))
    return result


def symmetric_distance(output_dir, model_a, model_b, map_key):
    """
    Computes the unbiased symmetric distance between two models:
    d(A, B) = 1/2 * [ d(S_A|traj_A, S_B|traj_A) + d(S_A|traj_B, S_B|traj_B) ]
    
    Averaging the distances on A's trajectory and B's trajectory makes the 
    comparison unbiased, as evaluating on just one favors the on-distribution model.
    """
    on_a = compare_on_trajectory(output_dir, model_a, model_a, model_b, map_key)
    on_b = compare_on_trajectory(output_dir, model_b, model_a, model_b, map_key)
    return {k: float(np.mean([on_a[k], on_b[k]])) for k in on_a}


def model_entropy_summary(output_dir, model_name, map_key):
    """Mean entropy of a model's own saliency maps on its own (on-distribution) trajectory."""
    cell = load_cell(output_dir, model_name, model_name)
    return float(np.mean([map_entropy(m) for m in cell[map_key]]))


def main():
    parser = argparse.ArgumentParser(
        description="Comparison metrics over the M x M saliency matrix. "
                    "Pairs are split into cross-architecture (the signal) and "
                    "intra-architecture (the noise floor). The main claim holds if "
                    "the first is systematically above the second."
    )
    parser.add_argument("output_dir")
    parser.add_argument("--map-key", default="answer_maps", choices=["answer_maps", "dP_maps"])
    args = parser.parse_args()

    models = discover_models(args.output_dir)
    assert len(models) >= 2, f"Found {len(models)} models in {args.output_dir}, need >= 2."
    names = sorted(models.keys())
    print(f"Models found: {names}")

    cross_arch, intra_arch = [], []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = symmetric_distance(args.output_dir, a, b, args.map_key)
            arch_pair = "cross" if models[a]["arch"] != models[b]["arch"] else "intra"
            record = {"model_a": a, "model_b": b, "arch_a": models[a]["arch"], "arch_b": models[b]["arch"], **d}
            (cross_arch if arch_pair == "cross" else intra_arch).append(record)
            print(f"{a} vs {b} [{arch_pair}]: " + ", ".join(f"{k}={v:.4f}" for k, v in d.items()))

    print("\n=== Entropy per model (own trajectory, on-distribution) ===")
    entropies = {name: model_entropy_summary(args.output_dir, name, args.map_key) for name in names}
    for name in names:
        print(f"  {name} (arch={models[name]['arch']}): entropy={entropies[name]:.4f}")

    print("\n=== Summary: cross-architecture (signal) vs intra-architecture (noise floor) ===")
    for label, records in [("cross-architecture", cross_arch), ("intra-architecture", intra_arch)]:
        print(f"\n{label}: n={len(records)}")
        if not records:
            continue
        for key in records[0]:
            if key in ("model_a", "model_b", "arch_a", "arch_b"):
                continue
            vals = [r[key] for r in records]
            print(f"  {key}: mean={np.mean(vals):.4f} std={np.std(vals):.4f}")

    out_path = Path(args.output_dir) / "compare_saliency_results.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"cross_arch": cross_arch, "intra_arch": intra_arch, "entropies": entropies}, f)
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
