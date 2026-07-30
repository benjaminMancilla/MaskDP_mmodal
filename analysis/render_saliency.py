import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import zoom

from analysis.select_qualitative_states import load_cell, load_states


def upsample_map(raw_map, size):
    zoom_factors = (size[0] / raw_map.shape[0], size[1] / raw_map.shape[1])
    return np.clip(zoom(raw_map, zoom_factors, order=1), 0, None)


def overlay_heatmap(frame, raw_map, alpha_max=0.6):
    h, w = frame.shape[:2]
    upsampled = upsample_map(raw_map, (h, w))
    peak = upsampled.max()
    normalized = upsampled / peak if peak > 0 else upsampled
    alpha = (normalized * alpha_max)[..., None]
    red = np.zeros_like(frame, dtype=np.float32)
    red[..., 0] = 255
    blended = frame.astype(np.float32) * (1 - alpha) + red * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def find_frame(states_by_level, level, t):
    for state in states_by_level.get(level, []):
        if state["t"] == t:
            return state["obs_buffer"][-1]
    raise KeyError(f"(level={level}, t={t}) not in the recorded states")


def render_cell(output_dir, traj_name, eval_name, out_dir, map_key="answer_maps", zoom_px=4, limit=None):
    cell = load_cell(output_dir, traj_name, eval_name)
    states_by_level = load_states(output_dir, traj_name)["levels"]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(cell["levels"]) if limit is None else min(limit, len(cell["levels"]))
    for i in range(n):
        level, t = int(cell["levels"][i]), int(cell["timesteps"][i])
        frame = find_frame(states_by_level, level, t)
        blended = overlay_heatmap(frame, cell[map_key][i])
        img = Image.fromarray(blended).resize((64 * zoom_px, 64 * zoom_px), Image.NEAREST)
        img.save(out_dir / f"traj_{traj_name}__eval_{eval_name}__L{level}_t{t}.png")

    print(f"{n} frames -> {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    parser.add_argument("traj_name")
    parser.add_argument("eval_name")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--map-key", default="answer_maps", choices=["answer_maps", "dP_maps"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--zoom", type=int, default=4)
    args = parser.parse_args()

    out_dir = args.out_dir or str(Path(args.output_dir) / "renders")
    render_cell(args.output_dir, args.traj_name, args.eval_name, out_dir, args.map_key, args.zoom, args.limit)


if __name__ == "__main__":
    main()
