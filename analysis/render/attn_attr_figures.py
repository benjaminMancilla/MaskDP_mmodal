"""
Figures for the ATTATTR attribution maps and trees (`analysis/attn_attr.py`):
the 2x2 attribution matrix, the filmstrip of arcs, and the attribution tree.

States render as raw game pixels, actions as glyphs from the action id.
Attribution is signed, so the colormaps are diverging and centered at zero.
"""
import re
import sys
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from analysis.render.procgen_actions import action_glyph

CMAP = "RdBu_r"
COLOR_FUSION = "#d95f02"
COLOR_ENC_S = "#1f6fb4"
COLOR_ENC_A = "#1b7837"
COLOR_TARGET = "#c0392b"

_NODE_RE = re.compile(r"^([sa])_(\d+)$")


# --------------------------------------------------------------------------
# loading / shared numerics
# --------------------------------------------------------------------------

def _scalar(value):
    array = np.asarray(value)
    return array.item() if array.ndim == 0 else value


def load_attr_npz(path):
    """Reads an npz from run_attn_attr.py. allow_pickle: V/E are object arrays."""
    raw = np.load(path, allow_pickle=True)
    data = {key: _scalar(raw[key]) for key in raw.files}
    data["V"] = [str(v) for v in raw["V"]]
    data["E"] = [tuple(str(x) for x in edge) for edge in raw["E"]]
    return data


def find_state(states_by_level, level, t):
    """The whole state dict; `render_saliency.find_frame` returns only obs_buffer[-1]."""
    for state in states_by_level.get(level, []):
        if state["t"] == t:
            return state
    raise KeyError(f"(level={level}, t={t}) not in the recorded states")


def head_sum(attr):
    """[n_head, T, T'] -> [T, T']."""
    return np.asarray(attr).sum(axis=0)


def normalize(mat, mode):
    """
    Attribution scale varies per run, so an un-normalized difference between two
    models measures scale rather than behaviour.
    """
    if mat is None or mode in (None, "none"):
        return mat
    mat = np.asarray(mat, dtype=np.float64)
    if mat.size == 0:  # n_a == 0 on the first step of an episode
        return mat
    if mode == "max":
        peak = np.abs(mat).max()
        return mat / peak if peak > 0 else mat
    if mode == "sum":
        total = np.abs(mat).sum()
        return mat / total if total > 0 else mat
    raise ValueError(f"unknown normalization mode: {mode!r}")


def upper_triangle_mass(mat, query_stream, key_stream):
    """
    Fraction of |attribution| the query spends on keys that are later in time.
    Zero means the path is causally masked.

    The cut is not the same in every block. Tokens interleave as
    s_0 a_0 s_1 a_1..., so s_i sits at position 2i and a_j at 2j+1: for the
    state -> action block a_j is already in the future at j == i, while every
    other block only looks forward at j > i.
    """
    if mat is None:
        return float("nan")
    mat = np.abs(np.asarray(mat))
    rows, cols = np.indices(mat.shape)
    forward = cols >= rows if (query_stream, key_stream) == ("s", "a") else cols > rows
    total = mat.sum()
    return float(mat[forward].sum() / total) if total > 0 else 0.0


def parse_node(label):
    """'s_3' -> ('s', 3); 'TARGET' -> ('TARGET', None)."""
    if label == "TARGET":
        return "TARGET", None
    match = _NODE_RE.match(label)
    assert match is not None, (
        f"node label {label!r} is not the default '{{stream}}_{{idx}}' format; the "
        "figures cannot map a custom label_fn back to a token index"
    )
    return match.group(1), int(match.group(2))


def node_depths(V, E):
    """
    Depth by BFS from V[0], the TopNode. Not readable off the edge kind: the
    fusion phase chains, so a 'real' edge can sit at any depth. 'terminal' edges
    are skipped -- TARGET is wired to every node and carries no score.
    """
    adjacency = {}
    for src, dst, kind in E:
        if kind != "terminal":
            adjacency.setdefault(src, []).append(dst)

    root = V[0]
    depths = {root: 0}
    queue = deque([root])
    while queue:
        node = queue.popleft()
        for child in adjacency.get(node, []):
            if child not in depths:
                depths[child] = depths[node] + 1
                queue.append(child)
    return depths


def edge_weight(src, dst, kind, mats):
    """Looks up the weight E does not carry. `mats` keys: a_s, a_a, a_enc_s, a_enc_a."""
    stream_src, idx_src = parse_node(src)
    _, idx_dst = parse_node(dst)
    if kind == "real":
        mat = mats["a_s"] if stream_src == "s" else mats["a_a"]
    elif kind == "real_encoder":
        mat = mats["a_enc_s"] if stream_src == "s" else mats["a_enc_a"]
    else:
        return None
    assert mat is not None, (
        f"edge {src}->{dst} has kind {kind!r} but its matrix was not supplied"
    )
    return float(mat[idx_src, idx_dst])


def save_figure(fig, out_dir, stem):
    """Writes <stem>.pdf and <stem>.png."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("pdf", "png"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def _ticks(timesteps, max_ticks=12):
    n = len(timesteps)
    if n == 0:
        return [], []
    stride = max(1, int(np.ceil(n / max_ticks)))
    positions = list(range(0, n, stride))
    return positions, [str(int(timesteps[p])) for p in positions]


def _apply_ticks(ax, axis, positions, labels, fontsize=7, rotation=0):
    """set_xticks(positions, labels) is matplotlib >= 3.5; the pinned env is 3.4.2."""
    if axis == "x":
        ax.set_xticks(list(positions))
        ax.set_xticklabels(labels, fontsize=fontsize, rotation=rotation)
    else:
        ax.set_yticks(list(positions))
        ax.set_yticklabels(labels, fontsize=fontsize, rotation=rotation)


def crop_center(frame, crop_px):
    """
    Centered square crop of a game frame. It can hide part of the scene, so the
    figures that use it say so in the caption.
    """
    frame = np.asarray(frame)
    height, width = frame.shape[:2]
    if crop_px is None or crop_px >= min(height, width):
        return frame
    top, left = (height - crop_px) // 2, (width - crop_px) // 2
    return frame[top:top + crop_px, left:left + crop_px]


def _thumb(frame, crop_px, zoom):
    """Crop plus the zoom that keeps the drawn box the size it had uncropped."""
    full = np.asarray(frame)
    cropped = crop_center(full, crop_px)
    return cropped, zoom * (full.shape[0] / cropped.shape[0])


def _glyph_fontsize(thumb_zoom, frame_px):
    """Keeps an action box roughly the size of a state thumbnail."""
    return max(6.0, 0.55 * frame_px * thumb_zoom)


def _symmetric_limit(mat):
    mat = np.asarray(mat)
    if mat.size == 0:
        return 1.0
    peak = np.abs(mat).max()
    return float(peak) if peak > 0 else 1.0


# --------------------------------------------------------------------------
# figure 1 -- attribution matrix (2x2 block)
# --------------------------------------------------------------------------

def render_attribution_matrix(a_s, a_a, ts_state, ts_action,
                              a_enc_s=None, a_enc_a=None,
                              normalize_mode="max", title=None, subtitle=None):
    """
    Diagonal blocks are the pre-fusion encoder streams, off-diagonal the fusion
    cross-attention. One colorbar per block: they come from different layers and
    the encoder gradient is larger.
    """
    blocks = [
        [("state encoder  (s -> s)", a_enc_s, ts_state, ts_state, COLOR_ENC_S),
         ("fusion  a_s  (state -> action)", a_s, ts_state, ts_action, COLOR_FUSION)],
        [("fusion  a_a  (action -> state)", a_a, ts_action, ts_state, COLOR_FUSION),
         ("action encoder  (a -> a)", a_enc_a, ts_action, ts_action, COLOR_ENC_A)],
    ]
    n_s, n_a = len(ts_state), max(len(ts_action), 1)

    fig, axes = plt.subplots(
        2, 2, figsize=(12, 11),
        gridspec_kw={"width_ratios": [n_s, n_a], "height_ratios": [n_s, n_a]},
    )

    for row in range(2):
        for col in range(2):
            ax = axes[row][col]
            name, mat, row_ts, col_ts, accent = blocks[row][col]
            if mat is None or np.asarray(mat).size == 0:
                reason = "not computed" if mat is None else "empty (no action tokens)"
                ax.set_facecolor("#eeeeee")
                ax.text(0.5, 0.5, f"{name}\n{reason}", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="#666666")
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            mat = normalize(mat, normalize_mode)
            limit = _symmetric_limit(mat)
            image = ax.imshow(mat, cmap=CMAP, vmin=-limit, vmax=limit,
                              interpolation="nearest", aspect="auto")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

            ax.set_title(name, fontsize=10, color=accent)
            xt, xl = _ticks(col_ts)
            yt, yl = _ticks(row_ts)
            _apply_ticks(ax, "x", xt, xl, rotation=90)
            _apply_ticks(ax, "y", yt, yl)
            ax.set_xlabel("key timestep", fontsize=8)
            ax.set_ylabel("query timestep", fontsize=8)

    header = title or "Attribution matrix"
    note = f"normalization: {normalize_mode} (per block)"
    if subtitle:
        note = f"{subtitle}    |    {note}"
    fig.suptitle(header, fontsize=13)
    fig.text(0.5, 0.005, note, ha="center", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    return fig


def render_matrix_detail(mat, row_ts, col_ts, row_idx, col_idx,
                         row_frames=None, col_actions=None, crop_px=None,
                         normalize_mode="max", title=None):
    """
    Crop of one block, with frames as row headers and action glyphs as column
    headers. Pass the token indices the tree already selected.

    The colorbar gets its own gridspec column: attaching it to the heatmap axes
    steals width from it and leaves the glyph row wider, so the two stop lining up.
    """
    row_idx, col_idx = list(row_idx), list(col_idx)
    assert row_idx and col_idx, "detail crop needs at least one row and one column"

    block = normalize(np.asarray(mat)[np.ix_(row_idx, col_idx)], normalize_mode)
    limit = _symmetric_limit(block)
    n_rows, n_cols = block.shape

    fig = plt.figure(figsize=(1.1 * n_cols + 3.4, 1.1 * n_rows + 1.8))
    grid = fig.add_gridspec(2, 3, width_ratios=[1, n_cols, 0.35],
                            height_ratios=[1, n_rows], wspace=0.10, hspace=0.04)

    ax_cols = fig.add_subplot(grid[0, 1])
    ax_cols.set_xlim(-0.5, n_cols - 0.5)
    ax_cols.set_ylim(0, 1)
    ax_cols.axis("off")
    if col_actions is not None:
        for position, index in enumerate(col_idx):
            action_id = int(col_actions[index])
            ax_cols.text(position, 0.55, action_glyph(action_id),
                         ha="center", va="center", fontsize=15)
            ax_cols.text(position, 0.15, str(action_id),
                         ha="center", va="center", fontsize=6, color="#777777")

    ax_rows = fig.add_subplot(grid[1, 0])
    ax_rows.axis("off")
    if row_frames is not None:
        tiles = [crop_center(row_frames[i], crop_px) for i in row_idx]
        ax_rows.imshow(np.concatenate(tiles, axis=0), aspect="auto",
                       interpolation="nearest")
        height = tiles[0].shape[0]
        for position in range(1, len(tiles)):  # otherwise the frames read as one blur
            ax_rows.axhline(position * height, color="white", linewidth=1.5)

    ax_heat = fig.add_subplot(grid[1, 1])
    image = ax_heat.imshow(block, cmap=CMAP, vmin=-limit, vmax=limit,
                           interpolation="nearest", aspect="auto")
    _apply_ticks(ax_heat, "x", range(n_cols),
                 [str(int(col_ts[i])) for i in col_idx], rotation=90)
    _apply_ticks(ax_heat, "y", range(n_rows), [str(int(row_ts[i])) for i in row_idx])
    ax_heat.yaxis.tick_right()
    ax_heat.set_xlabel("key timestep", fontsize=8)
    fig.colorbar(image, cax=fig.add_subplot(grid[1, 2]))

    fig.suptitle(title or "Attribution detail", fontsize=12)
    if crop_px:
        fig.text(0.5, 0.005, f"frames cropped to a centered {crop_px}x{crop_px}",
                 ha="center", fontsize=8, color="#444444")
    return fig


def render_difference_matrix(blocks_a, blocks_b, ts_state, ts_action,
                             label_a="model A", label_b="model B",
                             normalize_mode="max", title=None):
    """
    Two models on the same fixed trajectory plus their signed difference;
    `blocks_*` are (a_s, a_a) pairs. Normalized per model first, otherwise the
    difference mostly reports whose gradients are larger.
    """
    assert normalize_mode not in (None, "none"), (
        "an un-normalized difference measures attribution scale, not behaviour; "
        "pass normalize_mode='max' or 'sum'"
    )
    rows = [
        ("fusion  a_s  (state -> action)", blocks_a[0], blocks_b[0], ts_state, ts_action),
        ("fusion  a_a  (action -> state)", blocks_a[1], blocks_b[1], ts_action, ts_state),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for row, (name, mat_a, mat_b, row_ts, col_ts) in enumerate(rows):
        mat_a = normalize(mat_a, normalize_mode)
        mat_b = normalize(mat_b, normalize_mode)
        difference = mat_a - mat_b
        shared = max(_symmetric_limit(mat_a), _symmetric_limit(mat_b))

        panels = [(label_a, mat_a, shared), (label_b, mat_b, shared),
                  (f"{label_a} - {label_b}", difference, _symmetric_limit(difference))]
        for col, (panel_name, mat, limit) in enumerate(panels):
            ax = axes[row][col]
            image = ax.imshow(mat, cmap=CMAP, vmin=-limit, vmax=limit,
                              interpolation="nearest", aspect="auto")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{name}\n{panel_name}", fontsize=9)
            xt, xl = _ticks(col_ts)
            yt, yl = _ticks(row_ts)
            _apply_ticks(ax, "x", xt, xl, fontsize=6, rotation=90)
            _apply_ticks(ax, "y", yt, yl, fontsize=6)

    fig.suptitle(title or "Attribution difference (same fixed trajectory)", fontsize=13)
    fig.text(0.5, 0.005, f"normalization: {normalize_mode}, applied per model before "
             "subtracting", ha="center", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    return fig


# --------------------------------------------------------------------------
# figure 2 -- attribution tree
# --------------------------------------------------------------------------

def merge_action_runs(V, depths, actions):
    """
    Collapses runs of consecutive action tokens sharing a depth and an action id.
    Returns (representative_of, run_size).
    """
    by_depth = {}
    for label in V:
        stream, index = parse_node(label)
        if stream == "a" and label in depths:
            by_depth.setdefault(depths[label], []).append((index, label))

    representative_of, run_size = {}, {}
    for nodes in by_depth.values():
        nodes.sort()
        run = [nodes[0]]
        for index, label in nodes[1:]:
            previous_index, _ = run[-1]
            same_action = int(actions[index]) == int(actions[previous_index])
            if index == previous_index + 1 and same_action:
                run.append((index, label))
            else:
                _close_run(run, representative_of, run_size)
                run = [(index, label)]
        _close_run(run, representative_of, run_size)
    return representative_of, run_size


def _close_run(run, representative_of, run_size):
    head = run[0][1]
    for _, label in run:
        representative_of[label] = head
    run_size[head] = len(run)


def _draw_state_node(ax, x, y, frame, zoom, edgecolor, linewidth=1.2):
    box = OffsetImage(np.asarray(frame), zoom=zoom, interpolation="nearest")
    annotation = AnnotationBbox(
        box, (x, y), frameon=True, pad=0.12,
        bboxprops=dict(edgecolor=edgecolor, linewidth=linewidth),
    )
    ax.add_artist(annotation)


def _draw_action_node(ax, x, y, action_id, edgecolor, fontsize=13, badge=None):
    """Sized in points: x is timesteps and y is depth, so data units come out stretched."""
    ax.text(x, y, action_glyph(action_id), ha="center", va="center",
            fontsize=fontsize, color="white", zorder=4,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#2b2b2b",
                      edgecolor=edgecolor, linewidth=1.2))
    ax.annotate(str(int(action_id)), (x, y), textcoords="offset points",
                xytext=(fontsize * 0.95, -fontsize * 0.95), ha="left", va="top",
                fontsize=5.5, color="#888888", zorder=5)
    if badge:
        ax.annotate(badge, (x, y), textcoords="offset points",
                    xytext=(0, -fontsize * 1.8), ha="center", va="top",
                    fontsize=7.5, color="#333333", zorder=5)


def render_attribution_tree(V, E, mats, ts_state, ts_action, frames, actions,
                            t_target, tree_diag=None, thumb_zoom=0.42,
                            crop_px=None, merge_runs=False,
                            sarfa_by_state_idx=None, sarfa_zoom=0.95, title=None):
    """
    Layered DAG: y is BFS depth, x is the real timestep, so a long-range edge
    reads as one. 'terminal' edges are not drawn -- TARGET connects to every node
    without a score, and is shown as a marked node over a shaded band.
    """
    depths = node_depths(V, E)
    representative_of, run_size = ({}, {})
    if merge_runs:
        representative_of, run_size = merge_action_runs(V, depths, actions)

    def resolve(label):
        return representative_of.get(label, label)

    def position(label):
        stream, index = parse_node(label)
        x = ts_state[index] if stream == "s" else ts_action[index]
        return float(x), -float(depths[label])

    drawn = {resolve(label) for label in V if label in depths}
    max_depth = max((depths[label] for label in drawn), default=0)
    glyph_fontsize = _glyph_fontsize(thumb_zoom, np.asarray(frames[0]).shape[0])

    fig, ax = plt.subplots(figsize=(13, 2.4 + 1.35 * (max_depth + 1)))

    all_ts = list(ts_state) + list(ts_action)
    x_min, x_max = min(all_ts), max(all_ts)
    ax.axhspan(0.55, 1.15, color=COLOR_TARGET, alpha=0.06, zorder=0)

    # Terminal edges are dotted rather than weighted: TARGET attaches to the
    # tree's root by construction, without a score.
    for node in {resolve(v) for u, v, k in E if k == "terminal"} & drawn:
        x, y = position(node)
        ax.plot([t_target, x], [0.78, y], linestyle=":", linewidth=0.6,
                color=COLOR_TARGET, alpha=0.30, zorder=1)

    real_edges = [(u, v, k) for u, v, k in E if k != "terminal"]
    weights = [abs(edge_weight(u, v, k, mats)) for u, v, k in real_edges]
    peak = max(weights) if weights else 1.0

    for (src, dst, kind), weight in zip(real_edges, weights):
        source, target = resolve(src), resolve(dst)
        if source == target or source not in drawn or target not in drawn:
            continue
        x0, y0 = position(source)
        x1, y1 = position(target)
        color = COLOR_FUSION if kind == "real" else (
            COLOR_ENC_S if parse_node(source)[0] == "s" else COLOR_ENC_A
        )
        # shrink is in points, so arrows stop at the node edge at any data scale
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", color=color, alpha=0.8,
                            linewidth=0.6 + 3.4 * (weight / peak) if peak > 0 else 0.6,
                            shrinkA=20, shrinkB=20,
                            connectionstyle="arc3,rad=0.10"),
            zorder=2,
        )

    for label in sorted(drawn, key=lambda n: depths[n]):
        stream, index = parse_node(label)
        x, y = position(label)
        ax.plot([x, x], [y, -max_depth - 0.55], linestyle=":", linewidth=0.6,
                color="#bbbbbb", zorder=1)
        if stream == "s":
            frame = np.asarray(frames[index])
            base_zoom = thumb_zoom
            if sarfa_by_state_idx and index in sarfa_by_state_idx:
                from analysis.render_saliency import overlay_heatmap  # pulls in scipy
                frame = overlay_heatmap(frame, sarfa_by_state_idx[index])
                base_zoom = sarfa_zoom
            frame, zoom = _thumb(frame, crop_px, base_zoom)
            _draw_state_node(ax, x, y, frame, zoom,
                             COLOR_ENC_S if depths[label] else COLOR_TARGET,
                             linewidth=2.0 if depths[label] == 0 else 1.2)
        else:
            size = run_size.get(label, 1)
            _draw_action_node(ax, x, y, int(actions[index]),
                              COLOR_ENC_A if depths[label] else COLOR_TARGET,
                              fontsize=glyph_fontsize,
                              badge=f"x{size}" if size > 1 else None)

    ax.plot([t_target], [0.85], marker="*", markersize=20, color=COLOR_TARGET, zorder=5)
    ax.text(t_target, 0.85, "  TARGET", ha="left", va="center", fontsize=9,
            color=COLOR_TARGET, zorder=5)

    ax.set_xlim(x_min - 1.5, x_max + 1.5)
    ax.set_ylim(-max_depth - 0.75, 1.3)
    _apply_ticks(ax, "y", [-d for d in range(max_depth + 1)],
                 [f"depth {d}" for d in range(max_depth + 1)], fontsize=8)
    ax.set_xlabel("environment timestep", fontsize=9)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    note = ("dotted: TARGET -> tree root, wired by construction and not scored. "
            "solid: attribution, width by weight")
    if tree_diag is not None:
        note += (f"    |    {tree_diag['n_orphans']}/{tree_diag['n_candidate_tokens']} "
                 "tokens never entered the tree")
    if crop_px:
        note += f"    |    frames cropped to a centered {crop_px}x{crop_px}"
    if sarfa_by_state_idx:
        note += "    |    SARFA overlays are each normalized to their own max"
    ax.set_title(title or "Attribution tree", fontsize=13)
    fig.text(0.5, 0.005, note, ha="center", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


# --------------------------------------------------------------------------
# figure 3 -- filmstrip
# --------------------------------------------------------------------------

def token_masses(a_s, a_a, a_enc_s=None, a_enc_a=None):
    """What each token sends through fusion, receives there, and sends in its encoder."""
    mass_state = np.abs(a_s).sum(axis=1) + np.abs(a_a).sum(axis=0)
    mass_action = np.abs(a_a).sum(axis=1) + np.abs(a_s).sum(axis=0)
    if a_enc_s is not None:
        mass_state = mass_state + np.abs(a_enc_s).sum(axis=1)
    if a_enc_a is not None:
        mass_action = mass_action + np.abs(a_enc_a).sum(axis=1)
    return mass_state, mass_action


def collapse_columns(actions, mass, quiet_frac=0.05, keep_last=True):
    """
    Diff-viewer accordion: contiguous runs with the same action and less than
    `quiet_frac` of the peak column mass collapse into one column. Returns
    (column_of, columns), with columns as (indices, is_collapsed).
    """
    n = len(mass)
    threshold = quiet_frac * float(np.max(mass)) if n else 0.0
    quiet = [bool(mass[k] < threshold) for k in range(n)]
    if keep_last and n:
        quiet[-1] = False

    columns, run = [], []
    for k in range(n):
        same_action = run and int(actions[k]) == int(actions[run[-1]])
        if quiet[k] and same_action and k == run[-1] + 1:
            run.append(k)
            continue
        if run:
            columns.append((run, len(run) > 1))
        run = [k] if quiet[k] else []
        if not quiet[k]:
            columns.append(([k], False))
            run = []
    if run:
        columns.append((run, len(run) > 1))

    column_of = {}
    for position, (indices, _) in enumerate(columns):
        for index in indices:
            column_of[index] = position
    return column_of, columns


def _arc(ax, x0, y0, x1, y1, height, color, linewidth, alpha):
    control = ((x0 + x1) / 2.0, (y0 + y1) / 2.0 + height)
    path = MplPath([(x0, y0), control, (x1, y1)],
                   [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3])
    ax.add_patch(PathPatch(path, fill=False, edgecolor=color,
                           linewidth=linewidth, alpha=alpha, zorder=2))


def render_filmstrip(a_s, a_a, ts_state, ts_action, frames, actions,
                     a_enc_s=None, a_enc_a=None, a_enc_s_arcs=True,
                     quiet_frac=0.05, max_arcs=40, thumb_zoom=0.42,
                     crop_px=None, window=None, title=None):
    """
    Two rails on a time axis: frames on top, action glyphs below. Arcs split the
    plane into four bands -- state encoder above, fusion between the rails,
    action encoder below. Same relations as the 2x2 matrix, read as narrative.

    `window` keeps only the last N timesteps. That is a deliberate crop, so it
    is reported in the caption; the matrix stays the uncropped view.
    """
    n_full = len(ts_state)
    cropped = window is not None and window < n_full
    if cropped:
        cut = n_full - window
        a_s, a_a = a_s[cut:, cut:], a_a[cut:, cut:]
        a_enc_s = a_enc_s[cut:, cut:] if a_enc_s is not None else None
        a_enc_a = a_enc_a[cut:, cut:] if a_enc_a is not None else None
        ts_state, ts_action = ts_state[cut:], ts_action[cut:]
        frames, actions = frames[cut:], actions[cut:]

    n_s, n_a = len(ts_state), len(ts_action)
    mass_state, mass_action = token_masses(a_s, a_a, a_enc_s, a_enc_a)

    column_mass = mass_state.copy()
    column_mass[:n_a] = column_mass[:n_a] + mass_action
    column_actions = list(actions) + [int(actions[-1]) if n_a else 0]
    column_of, columns = collapse_columns(column_actions, column_mass, quiet_frac)
    n_collapsed = sum(1 for _, collapsed in columns if collapsed)

    y_state, y_action = 1.0, 0.0
    glyph_fontsize = _glyph_fontsize(thumb_zoom, np.asarray(frames[0]).shape[0])
    fig, ax = plt.subplots(figsize=(max(8.0, 1.15 * len(columns)), 4.8))

    for position, (indices, collapsed) in enumerate(columns):
        if collapsed:
            ax.axvspan(position - 0.42, position + 0.42, ymin=0.30, ymax=0.70,
                       facecolor="#dddddd", hatch="//", edgecolor="#aaaaaa",
                       linewidth=0.0, zorder=1)
            ax.text(position, (y_state + y_action) / 2, f"x{len(indices)}",
                    ha="center", va="center", fontsize=8, zorder=4)
            continue
        index = indices[0]
        frame, zoom = _thumb(frames[index], crop_px, thumb_zoom)
        _draw_state_node(ax, position, y_state, frame, zoom, "#888888")
        if index < n_a:
            _draw_action_node(ax, position, y_action, int(actions[index]), "#888888",
                              fontsize=glyph_fontsize)

    # One budget per band, not a global top-k: encoder attribution is larger than
    # fusion attribution, so a shared ranking hides the fusion band entirely.
    # Width is likewise normalized within each band.
    bands = {}
    if a_enc_s is not None and a_enc_s_arcs:
        bands["enc_s"] = [(i, j, a_enc_s[i, j])
                          for i in range(n_s) for j in range(n_s) if i != j]
    if a_enc_a is not None:
        bands["enc_a"] = [(i, j, a_enc_a[i, j])
                          for i in range(n_a) for j in range(n_a) if i != j]
    bands["a_s"] = [(i, j, a_s[i, j]) for i in range(n_s) for j in range(n_a)]
    bands["a_a"] = [(i, j, a_a[i, j]) for i in range(n_a) for j in range(n_s)]

    budget = max(1, max_arcs // len(bands))
    candidates, peaks = [], {}
    for kind, items in bands.items():
        items.sort(key=lambda item: abs(item[2]), reverse=True)
        kept = items[:budget]
        peaks[kind] = max((abs(w) for _, _, w in kept), default=1.0)
        candidates += [(kind, i, j, w) for i, j, w in kept]

    for kind, i, j, weight in candidates:
        x0, x1 = column_of.get(i), column_of.get(j)
        if x0 is None or x1 is None or x0 == x1:
            continue
        peak = peaks[kind]
        linewidth = 0.5 + 2.8 * (abs(weight) / peak) if peak > 0 else 0.5
        span = abs(x1 - x0)
        if kind == "enc_s":
            _arc(ax, x0, y_state, x1, y_state, 0.10 + 0.05 * span,
                 COLOR_ENC_S, linewidth, 0.7)
        elif kind == "enc_a":
            _arc(ax, x0, y_action, x1, y_action, -0.10 - 0.05 * span,
                 COLOR_ENC_A, linewidth, 0.7)
        elif kind == "a_s":
            _arc(ax, x0, y_state, x1, y_action, 0.0, COLOR_FUSION, linewidth, 0.55)
        else:
            _arc(ax, x0, y_action, x1, y_state, 0.0, COLOR_FUSION, linewidth, 0.55)

    target_column = column_of.get(n_s - 1)
    if target_column is not None:
        ax.axvspan(target_column - 0.5, target_column + 0.5,
                   color=COLOR_TARGET, alpha=0.10, zorder=0)
        ax.text(target_column, y_state + 0.68, "TARGET", ha="center", va="bottom",
                fontsize=9, color=COLOR_TARGET)

    labels = []
    for indices, collapsed in columns:
        first = int(ts_state[indices[0]])
        labels.append(f"{first}+" if collapsed else str(first))
    _apply_ticks(ax, "x", range(len(columns)), labels, fontsize=8,
                 rotation=0 if len(columns) <= 12 else 90)
    ax.set_xlim(-0.8, len(columns) - 0.2)
    ax.set_ylim(y_action - 0.45, y_state + 0.78)
    _apply_ticks(ax, "y", [y_action, y_state], ["actions", "states"], fontsize=9)
    ax.set_xlabel("environment timestep", fontsize=9)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    ax.set_title(title or "Attribution filmstrip", fontsize=13)
    note = f"last {window} of {n_full} timesteps    |    " if cropped else ""
    if crop_px:
        note += f"frames cropped to a centered {crop_px}x{crop_px}    |    "
    fig.text(0.5, 0.005,
             f"{note}accordion: quiet_frac={quiet_frac}, {n_collapsed} runs collapsed"
             f"    |    top {budget} arcs per band ({', '.join(bands)}), width "
             "normalized within each band",
             ha="center", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig
