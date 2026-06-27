from __future__ import annotations

import math

import torch
import torch.nn as nn


class PixelReconDecoder(nn.Module):
    """
    Pixel reconstruction decoder, encoder-agnostic
    """

    def __init__(
        self,
        d_model: int,
        tokens_per_frame: int,
        frame_hw: tuple[int, int],
        channels: int,
        c0: int = 256,
        init_spatial: int = 8,
        patch_hw: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        H, W = frame_hw
        self.H = H
        self.W = W
        self.C = channels
        self.P = tokens_per_frame

        if tokens_per_frame == 1:
            self._build_cnn_decoder(d_model, H, W, channels, c0, init_spatial)
        else:
            raise NotImplementedError(
                f"PixelReconDecoder: tokens_per_frame={tokens_per_frame} (ViT P>1) is not "
                "implemented yet.  Add a per-patch MLP head + reassembly here when upgrading "
                "from CNN to ViT (set tokens_per_frame=N and provide patch_hw)."
            )


    def _build_cnn_decoder(
        self,
        d_model: int,
        H: int,
        W: int,
        channels: int,
        c0: int,
        init_spatial: int,
    ) -> None:
        
        assert H == W, (
            f"PixelReconDecoder (CNN, P=1): expected square frame, got {H}×{W}."
        )
        n_ups = int(round(math.log2(H / init_spatial)))
        assert init_spatial * (2 ** n_ups) == H, (
            f"PixelReconDecoder: cannot reach H={H} from init_spatial={init_spatial} "
            f"with stride-2 upsamples (need init_spatial × 2^k == H)."
        )

        self.init_spatial = init_spatial
        self.c0 = c0

        self.linear = nn.Linear(d_model, init_spatial * init_spatial * c0)

        layers: list[nn.Module] = []
        in_ch = c0
        for _ in range(n_ups):
            out_ch = max(in_ch // 2, 32)
            layers += [
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch

        # Final 1×1 projection to target channels, NO activation (linear output)
        layers.append(nn.Conv2d(in_ch, channels, kernel_size=1))
        self.decoder_conv = nn.Sequential(*layers)


    def forward(self, latent_state_tokens: torch.Tensor) -> torch.Tensor:
        B, T, P, D = latent_state_tokens.shape
        if P != self.P:
            raise ValueError(
                f"PixelReconDecoder: got P={P} tokens per frame, "
                f"expected P={self.P}."
            )

        if P == 1:
            x = latent_state_tokens.reshape(B * T, D)                           # [B*T, D]
            x = self.linear(x)                                                   # [B*T, init_h²×C0]
            x = x.reshape(B * T, self.c0, self.init_spatial, self.init_spatial) # [B*T, C0, h, h]
            x = self.decoder_conv(x)                                             # [B*T, C, H, W]
            x = x.permute(0, 2, 3, 1).contiguous()                              # [B*T, H, W, C]
            x = x.reshape(B, T, self.H, self.W, self.C)                         # [B, T, H, W, C]
            return x

        raise NotImplementedError("ViT P>1 path not yet implemented.")