"""
Renders the ATTATTR figures from an npz produced by analysis/tests/run_attn_attr.py.

The npz holds attribution matrices but no pixels, so the recorded states SARFA
already uses (states_<name>.pkl) are reloaded here for the frames.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from analysis.attn_attr import token_timesteps
from analysis.render.attn_attr_figures import (
    find_state,
    head_sum,
    load_attr_npz,
    parse_node,
    render_attribution_matrix,
    render_attribution_tree,
    render_filmstrip,
    render_matrix_detail,
    save_figure,
    token_masses,
    upper_triangle_mass,
)
from analysis.render_saliency import find_frame
from analysis.select_qualitative_states import load_cell
from analysis.trajectory_recorder import load_trajectory_states


def check_invariants(data, state, states_by_level, level, ts_state, ts_action):
    """
    The timestep mapping holds only while the eval buffers stay interleaved and
    unmasked. Fail loudly rather than draw a plausible but wrong time axis.
    """
    n_s = len(state["obs_buffer"])
    n_a = len(state["action_buffer"])
    assert n_a == n_s - 1, f"expected n_a == n_s - 1, got n_s={n_s}, n_a={n_a}"

    attr_s = data["attr_s"]
    assert attr_s.shape[1] == n_s, f"attr_s rows {attr_s.shape[1]} != n_s {n_s}"
    assert attr_s.shape[2] == n_a, f"attr_s cols {attr_s.shape[2]} != n_a {n_a}"
    assert data["attr_a"].shape[1:] == (n_a, n_s), (
        f"attr_a shape {data['attr_a'].shape} inconsistent with (n_a, n_s)=({n_a}, {n_s})"
    )
    assert len(ts_state) == n_s and len(ts_action) == n_a
    assert int(ts_state[-1]) == int(state["t"]), (
        f"last state token maps to t={ts_state[-1]} but the state is at t={state['t']}"
    )
    # same "newest frame is the last state token" convention SARFA's renderer uses
    assert np.array_equal(state["obs_buffer"][-1],
                          find_frame(states_by_level, level, state["t"]))


def sarfa_overlays(sarfa_dir, traj_name, eval_name, level, ts_state, mass_state, top_k):
    """
    SARFA maps only exist for timesteps recorded as states, so a token gets an
    overlay only if its timestep is in the cell. Returns {state_token_index: map}
    for the top_k highest-attribution tokens that have one.
    """
    cell = load_cell(sarfa_dir, traj_name, eval_name)
    rows = {(int(cell["levels"][i]), int(cell["timesteps"][i])): i
            for i in range(len(cell["levels"]))}

    available = [i for i in range(len(ts_state)) if (level, int(ts_state[i])) in rows]
    available.sort(key=lambda i: mass_state[i], reverse=True)
    chosen = available[:top_k]
    print(f"[render_attn_attr] SARFA overlays: {len(chosen)} of {len(ts_state)} state "
          f"tokens had a recorded map ({len(available)} available)")
    return {i: cell["answer_maps"][rows[(level, int(ts_state[i]))]] for i in chosen}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("attr_npz", help="output of analysis/tests/run_attn_attr.py")
    parser.add_argument("states_dir")
    parser.add_argument("name", help="traj_source, i.e. states_<name>.pkl")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--figures", default="all",
                        choices=["all", "matrix", "detail", "tree", "filmstrip"])
    parser.add_argument("--normalize", default="max", choices=["max", "sum", "none"])
    parser.add_argument("--merge-runs", action="store_true",
                        help="collapse consecutive same-action nodes in the tree")
    parser.add_argument("--quiet-frac", type=float, default=0.05,
                        help="filmstrip accordion: column mass below this fraction of "
                             "the peak counts as quiet")
    parser.add_argument("--max-arcs", type=int, default=40)
    parser.add_argument("--window", type=int, default=None,
                        help="filmstrip: keep only the last N timesteps. A crop, so "
                             "it is reported in the caption; the matrix stays whole")
    parser.add_argument("--thumb-zoom", type=float, default=0.42)
    parser.add_argument("--sarfa-eval", default=None,
                        help="eval_model of the SARFA cell to overlay on tree nodes; "
                             "the cell is read from states_dir")
    parser.add_argument("--topk-sarfa", type=int, default=3)
    args = parser.parse_args()

    data = load_attr_npz(args.attr_npz)
    assert data["t"] is not None, (
        "this npz has no 't': it came from run_attn_attr.py's fresh-recording "
        "fallback, with no states_<name>.pkl to pair with. Re-run it with "
        "--states-dir/--name for a real timestep axis."
    )
    level = int(data["level"])
    t_target = int(data["t"])
    states_by_level = load_trajectory_states(args.states_dir, args.name)
    state = find_state(states_by_level, level, t_target)

    frames = state["obs_buffer"]
    actions = state["action_buffer"]
    n_s, n_a = len(frames), len(actions)

    ts_state, ts_action = token_timesteps(t_target, n_s, n_a)
    check_invariants(data, state, states_by_level, level, ts_state, ts_action)

    a_s = head_sum(data["attr_s"])
    a_a = head_sum(data["attr_a"])
    a_enc_s = head_sum(data["attr_enc_s"]) if "attr_enc_s" in data else None
    a_enc_a = head_sum(data["attr_enc_a"]) if "attr_enc_a" in data else None

    for name, mat, query, key in (("a_s", a_s, "s", "a"), ("a_a", a_a, "a", "s"),
                                  ("a_enc_s", a_enc_s, "s", "s"),
                                  ("a_enc_a", a_enc_a, "a", "a")):
        if mat is not None:
            print(f"[render_attn_attr] {name}: |attribution| on later tokens = "
                  f"{upper_triangle_mass(mat, query, key):.4f} of the total "
                  f"(0 means causally masked)")

    out_dir = Path(args.out_dir or Path(args.states_dir) / "renders")
    out_dir = out_dir / "attn_attr" / f"traj_{args.name}" / f"L{level}"
    stem = f"t{int(t_target):04d}"
    subtitle = (f"level {level}, t={int(t_target)}, action={int(data['action_idx'])}, "
                f"tar_layer={int(data['tar_layer'])}, m={int(data['m'])}, "
                f"tau={float(data['tau']):.4g}")
    wanted = {args.figures} if args.figures != "all" else {
        "matrix", "detail", "tree", "filmstrip"}
    written = []

    if "matrix" in wanted:
        fig = render_attribution_matrix(
            a_s, a_a, ts_state, ts_action, a_enc_s, a_enc_a,
            normalize_mode=args.normalize, subtitle=subtitle,
        )
        written += save_figure(fig, out_dir, f"{stem}_matrix")

    tree_state_idx = sorted({i for label in data["V"]
                             for stream, i in [parse_node(label)]
                             if stream == "s"})
    tree_action_idx = sorted({i for label in data["V"]
                              for stream, i in [parse_node(label)]
                              if stream == "a"})

    if "detail" in wanted and tree_state_idx and tree_action_idx:
        fig = render_matrix_detail(
            a_s, ts_state, ts_action, tree_state_idx, tree_action_idx,
            row_frames=frames, col_actions=actions,
            normalize_mode=args.normalize,
            title=f"fusion a_s, crop on the tree's tokens -- {subtitle}",
        )
        written += save_figure(fig, out_dir, f"{stem}_matrix_detail")

    mass_state, _ = token_masses(a_s, a_a, a_enc_s, a_enc_a)

    if "tree" in wanted:
        overlays = None
        if args.sarfa_eval:
            overlays = sarfa_overlays(args.states_dir, args.name, args.sarfa_eval,
                                      level, ts_state, mass_state, args.topk_sarfa)
        mats = {"a_s": a_s, "a_a": a_a, "a_enc_s": a_enc_s, "a_enc_a": a_enc_a}
        # run_attn_attr.py only prints the tree diagnostics; its diag_* keys are
        # the attribution range, so the orphan count is recomputed from V
        n_candidates = n_s + n_a
        tree_diag = {"n_candidate_tokens": n_candidates,
                     "n_orphans": n_candidates - (len(data["V"]) - 1)}
        fig = render_attribution_tree(
            data["V"], data["E"], mats, ts_state, ts_action, frames, actions,
            t_target=t_target, tree_diag=tree_diag,
            thumb_zoom=args.thumb_zoom, merge_runs=args.merge_runs,
            sarfa_by_state_idx=overlays,
            title=f"Attribution tree -- {subtitle}",
        )
        written += save_figure(fig, out_dir, f"{stem}_tree")

    if "filmstrip" in wanted:
        fig = render_filmstrip(
            a_s, a_a, ts_state, ts_action, frames, actions, a_enc_s, a_enc_a,
            quiet_frac=args.quiet_frac, max_arcs=args.max_arcs,
            thumb_zoom=args.thumb_zoom, window=args.window,
            title=f"Attribution filmstrip -- {subtitle}",
        )
        written += save_figure(fig, out_dir, f"{stem}_filmstrip")

    for path in written:
        print(f"[render_attn_attr] -> {path}")


if __name__ == "__main__":
    main()
