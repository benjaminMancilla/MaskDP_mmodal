from typing import Optional, Type

import torch
import torch.nn as nn


def layer_init(
    layer: nn.Module, init_layers_orthogonal: bool, std: float = 2 ** 0.5
) -> nn.Module:
    if not init_layers_orthogonal:
        return layer
    nn.init.orthogonal_(layer.weight, std)  # type: ignore
    nn.init.constant_(layer.bias, 0.0)  # type: ignore
    return layer


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        activation: Type[nn.Module] = nn.ReLU,
        init_layers_orthogonal: bool = False,
    ) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            activation(),
            layer_init(
                nn.Conv2d(channels, channels, 3, padding=1), init_layers_orthogonal
            ),
            activation(),
            layer_init(
                nn.Conv2d(channels, channels, 3, padding=1), init_layers_orthogonal
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual(x)


class ConvSequence(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Type[nn.Module] = nn.ReLU,
        init_layers_orthogonal: bool = False,
    ) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            layer_init(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                init_layers_orthogonal,
            ),
            nn.MaxPool2d(3, stride=2, padding=1),
            ResidualBlock(out_channels, activation, init_layers_orthogonal),
            ResidualBlock(out_channels, activation, init_layers_orthogonal),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class ImpalaCnn(nn.Module):
    """
    Ported verbatim (architecture + attribute names) from:
    sgoodfriend/rl-algo-impls, shared/module/feature_extractor.py
    https://github.com/sgoodfriend/rl-algo-impls (MIT License, commit 21ee1ab9)
    
    IMPALA-style CNN architecture (Espeholt et al., 2018), as used in the original
    Procgen baselines (Cobbe et al., 2020) and in sgoodfriend/rl-algo-impls.

    Default channel progression [16, 32, 32], 3 ConvSequence blocks. Output is
    flattened (caller is responsible for projecting to the desired feature_dim).
    """

    def __init__(
        self,
        in_channels: int,
        activation: Type[nn.Module] = nn.ReLU,
        init_layers_orthogonal: Optional[bool] = None,
    ) -> None:
        super().__init__()
        if init_layers_orthogonal is None:
            init_layers_orthogonal = False
        sequences = []
        for out_channels in [16, 32, 32]:
            sequences.append(
                ConvSequence(
                    in_channels, out_channels, activation, init_layers_orthogonal
                )
            )
            in_channels = out_channels
        sequences.extend([activation(), nn.Flatten()])
        self.seq = nn.Sequential(*sequences)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.seq(obs)