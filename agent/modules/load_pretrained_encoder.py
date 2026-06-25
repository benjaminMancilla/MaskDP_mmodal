from pathlib import Path
import torch
import torch.nn as nn


def load_drqbc_convnet(encoder, ckpt_path: str, freeze: bool = True):
    """
    Load pretrained DrQBC convnet weights into a DrQv2Encoder instance.

    Two paths:
    - Direct load  : encoder was built with obs_shape=(64,64,9); all shapes match the
                     checkpoint; no weight transformation is applied.
    - Surgery path : encoder was built with obs_shape=(64,64,3); first conv is adapted
                     by averaging over the 3 frame-stack groups.  Kept for reproducibility
                     of older single-frame experiments.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    pretrained_sd = ckpt["convnet"]

    current_sd = encoder.convnet.state_dict()

    new_sd = {}
    _surgery_applied = False

    for key in pretrained_sd:
        pretrained_w = pretrained_sd[key]
        current_w    = current_sd[key]

        if pretrained_w.shape == current_w.shape:
            new_sd[key] = pretrained_w

        elif key == "0.weight":
            # Surgery: pretrained (32, 9, 3, 3) -> current (32, 3, 3, 3)
            assert pretrained_w.shape == (32, 9, 3, 3), \
                f"Unexpected shape for first conv: {pretrained_w.shape}"
            assert current_w.shape == (32, 3, 3, 3), \
                f"Unexpected shape in MaskDP first conv: {current_w.shape}"
            new_w = pretrained_w.reshape(32, 3, 3, 3, 3).mean(dim=1)
            new_sd[key] = new_w
            _surgery_applied = True
            print(f"  [load_drqbc_convnet] SURGERY: adapted first conv "
                  f"{pretrained_w.shape} -> {new_w.shape} (mean over frame-stack groups)")

        else:
            if pretrained_w.shape == current_w.shape:
                new_sd[key] = pretrained_w
            else:
                print(f"  [load_drqbc_convnet] WARNING: skipping {key}, "
                      f"shape mismatch {pretrained_w.shape} vs {current_w.shape}")
                new_sd[key] = current_w

    missing, unexpected = encoder.convnet.load_state_dict(new_sd, strict=True)
    assert len(missing) == 0,    f"Missing keys: {missing}"
    assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"

    if _surgery_applied:
        print("  [load_drqbc_convnet] SURGERY path taken — first conv averaged over 3 frame-stack groups")
    else:
        print("  [load_drqbc_convnet] Direct load, NO surgery — all weight shapes matched")

    if freeze:
        for p in encoder.convnet.parameters():
            p.requires_grad = False
        for p in encoder.projection.parameters():
            p.requires_grad = False
        print(f"  [load_drqbc_convnet] Convnet and projection frozen.")
    else:
        print(f"  [load_drqbc_convnet] Convnet loaded, NOT frozen (fine-tune mode).")

    print(f"  [load_drqbc_convnet] Loaded from: {ckpt_path} "
          f"(trained for {ckpt.get('global_step', '?')} steps)")
    return encoder


def verify_load(ckpt_path: str, obs_shape=(64, 64, 9), feature_dim=256):
    """Quick sanity check: instantiate encoder, load weights, run a dummy forward."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from agent.modules.pixel_encoder import DrQv2Encoder

    encoder = DrQv2Encoder(obs_shape, feature_dim)
    load_drqbc_convnet(encoder, ckpt_path, freeze=True)

    dummy = torch.zeros(2, 1, *obs_shape, dtype=torch.uint8)  # (B, T, H, W, C)
    with torch.no_grad():
        out = encoder(dummy)
    print(f"  Forward OK: input {tuple(dummy.shape)} => output {tuple(out.shape)}")
    assert out.shape == (2, 1, feature_dim), f"Unexpected output shape: {out.shape}"
    print("  Verification passed.")


def load_procgen_impala(encoder, ckpt_path: str, freeze: bool = True):
    """
    Load a clean checkpoint into an ImpalaProcgenEncoder instance.
    Loads convnet and projection
    Expected ckpt format:
        {
            "convnet": <state_dict matching encoder.convnet>,
            "projection": <state_dict matching encoder.projection>,
            "feature_dim": int,
            "cnn_style": "impala",
            "source": str,            # e.g. "sgoodfriend/ppo-procgen-coinrun-easy"
            "env_id": str,            # e.g. "coinrun"
            "distribution_mode": str, # "easy" | "hard"
        }
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")

    for submodule_name in ("convnet", "projection"):
        if submodule_name not in ckpt:
            raise KeyError(
                f"  [load_procgen_impala] ckpt at {ckpt_path} has no '{submodule_name}' "
                f"key. Found keys: {list(ckpt.keys())}. Did you generate this file with "
                f"extract_procgen_ckpt.py?"
            )
        submodule = getattr(encoder, submodule_name)
        missing, unexpected = submodule.load_state_dict(ckpt[submodule_name], strict=True)
        assert len(missing) == 0, f"  [load_procgen_impala] Missing keys in '{submodule_name}': {missing}"
        assert len(unexpected) == 0, f"  [load_procgen_impala] Unexpected keys in '{submodule_name}': {unexpected}"

    if freeze:
        for p in encoder.convnet.parameters():
            p.requires_grad = False
        for p in encoder.projection.parameters():
            p.requires_grad = False
        print("  [load_procgen_impala] Convnet and projection frozen.")
    else:
        print("  [load_procgen_impala] Loaded, NOT frozen (fine-tune mode).")

    print(
        f"  [load_procgen_impala] Loaded from: {ckpt_path} "
        f"(source: {ckpt.get('source', '?')}, env_id: {ckpt.get('env_id', '?')}, "
        f"distribution_mode: {ckpt.get('distribution_mode', '?')})"
    )
    return encoder


def verify_load_procgen(ckpt_path: str, obs_shape=(64, 64, 3), feature_dim=256):
    """Quick sanity check: instantiate ImpalaProcgenEncoder, load weights, dummy forward."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from agent.modules.pixel_encoder import ImpalaProcgenEncoder

    encoder = ImpalaProcgenEncoder(obs_shape, feature_dim)
    load_procgen_impala(encoder, ckpt_path, freeze=True)

    dummy = torch.zeros(2, 1, *obs_shape, dtype=torch.uint8)  # (B, T, H, W, C)
    with torch.no_grad():
        out = encoder(dummy)
    print(f"  Forward OK: input {tuple(dummy.shape)} => output {tuple(out.shape)}")
    assert out.shape == (2, 1, feature_dim), f"Unexpected output shape: {out.shape}"
    print("  Verification passed.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--obs_shape", nargs=3, type=int, default=[64, 64, 9])
    p.add_argument(
        "--kind", choices=["drqv2", "procgen_impala"], default="drqv2",
        help="Which verification path to run.",
    )
    args = p.parse_args()
    if args.kind == "drqv2":
        verify_load(args.ckpt, obs_shape=tuple(args.obs_shape))
    else:
        verify_load_procgen(args.ckpt, obs_shape=tuple(args.obs_shape))