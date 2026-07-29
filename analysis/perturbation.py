import numpy as np
from scipy.ndimage import gaussian_filter

# Greydanus et al 2018 perturbation

def gaussian_mask(center, size=(64, 64), radius=5.0):
    H, W = size
    row, col = center
    assert 0 <= row < H and 0 <= col < W, f"center={center} out of size={size}"

    mask = np.zeros((H, W), dtype=np.float32)
    mask[int(round(row)), int(round(col))] = 1.0
    mask = gaussian_filter(mask, sigma=radius)

    peak = mask.max()
    if peak > 0:
        mask = mask / peak
    return mask


def _assert_pixel_range(frame):
    peak = float(np.asarray(frame).max())
    assert peak > 1.5, (
        f"frame.max()={peak:.4f} <= 1.5 -- looks normalized to [0, 1]. "
        "occlude()/searchlight()/perturb_batch() expects raw values in "
        "[0, 255] (uint8 o float32), never pre-normalized."
    )


def _blur_frame(frame, blur_sigma):
    return gaussian_filter(frame.astype(np.float32), sigma=(blur_sigma, blur_sigma, 0))


def occlude(frame, mask, blur_sigma=3.0):
    """
    frame * (1 - mask) + blur(frame) * mask.
    `mask` (H, W) the same in the 3 channels.
    """
    _assert_pixel_range(frame)
    frame = frame.astype(np.float32)
    blurred = _blur_frame(frame, blur_sigma)
    m = mask[..., None]
    return frame * (1.0 - m) + blurred * m


def searchlight(frame, mask, blur_sigma=3.0):
    """
    frame * mask + blur(frame) * (1 - mask).
    `mask` (H, W) the same in the 3 channels.
    """
    _assert_pixel_range(frame)
    frame = frame.astype(np.float32)
    blurred = _blur_frame(frame, blur_sigma)
    m = mask[..., None]
    return frame * m + blurred * (1.0 - m)


def grid_centers(size=(64, 64), d=4):
    H, W = size
    rows = np.arange(d // 2, H, d)
    cols = np.arange(d // 2, W, d)
    return [(int(r), int(c)) for r in rows for c in cols]


def perturb_batch(frame, d=4, radius=4.0, blur_sigma=3.0, mode="occlude"):
    """
    Generates perturbed versions of `frame` over a grid with step size `d`. 
    Returns (perturbed, centers):

        perturbed: (N, H, W, C) float32 tensor of perturbed frames, in the same 
                order as `centers`. Ready to be passed directly to the agent.
        centers: List of (row, col) coordinates for each perturbation.
    """
    assert mode in ("occlude", "searchlight"), f"mode must be 'occlude'|'searchlight', but is '{mode}'"
    _assert_pixel_range(frame)

    H, W = frame.shape[0], frame.shape[1]
    centers = grid_centers(size=(H, W), d=d)
    fn = occlude if mode == "occlude" else searchlight

    perturbed = np.empty((len(centers),) + frame.shape, dtype=np.float32)
    for i, center in enumerate(centers):
        mask = gaussian_mask(center, size=(H, W), radius=radius)
        perturbed[i] = fn(frame, mask, blur_sigma=blur_sigma)
    return perturbed, centers
