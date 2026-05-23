from pathlib import Path
import torch
import torch.nn as nn


def load_drqbc_convnet(encoder, ckpt_path: str, freeze: bool = True):
    """
    Load pretrained DrQBC convnet weights into a DrQv2Encoder instance.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    pretrained_sd = ckpt["convnet"]

    current_sd = encoder.convnet.state_dict()

    new_sd = {}
    for key in pretrained_sd:
        pretrained_w = pretrained_sd[key]
        current_w    = current_sd[key]

        if pretrained_w.shape == current_w.shape:
            # Layers 2,4,6  shapes match, load directly
            new_sd[key] = pretrained_w

        elif key == "0.weight":
            # First conv: pretrained (32, 9, 3, 3)
            # Average over the 3 framestack groups
            assert pretrained_w.shape == (32, 9, 3, 3), \
                f"Unexpected shape for first conv: {pretrained_w.shape}"
            assert current_w.shape == (32, 3, 3, 3), \
                f"Unexpected shape in MaskDP first conv: {current_w.shape}"
            new_w = pretrained_w.reshape(32, 3, 3, 3, 3).mean(dim=1)
            new_sd[key] = new_w
            print(f"  [load_drqbc_convnet] Adapted first conv: "
                  f"{pretrained_w.shape} → {new_w.shape} (mean over framestack groups)")

        else:
            # Bias of first conv or unexpected mismatch
            if pretrained_w.shape == current_w.shape:
                new_sd[key] = pretrained_w
            else:
                print(f"  [load_drqbc_convnet] WARNING: skipping {key}, "
                      f"shape mismatch {pretrained_w.shape} vs {current_w.shape}")
                new_sd[key] = current_w

    missing, unexpected = encoder.convnet.load_state_dict(new_sd, strict=True)
    assert len(missing) == 0,    f"Missing keys: {missing}"
    assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"

    if freeze:
        for p in encoder.convnet.parameters():
            p.requires_grad = False
        print(f"  [load_drqbc_convnet] Convnet frozen. "
              f"Projection head remains trainable.")
    else:
        print(f"  [load_drqbc_convnet] Convnet loaded, NOT frozen (fine-tune mode).")

    print(f"  [load_drqbc_convnet] Loaded from: {ckpt_path} "
          f"(trained for {ckpt.get('global_step', '?')} steps)")
    return encoder


def verify_load(ckpt_path: str, obs_shape=(64, 64, 3), feature_dim=256):
    """Quick sanity check, instantiate encoder, load weights, run a dummy forward."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from pixel_encoder import DrQv2Encoder

    encoder = DrQv2Encoder(obs_shape, feature_dim)
    load_drqbc_convnet(encoder, ckpt_path, freeze=True)

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
    args = p.parse_args()
    verify_load(args.ckpt)