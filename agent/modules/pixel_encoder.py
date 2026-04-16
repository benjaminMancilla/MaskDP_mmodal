import torch
import torch.nn as nn


class PixelEncoder(nn.Module):
    """
    CNN placeholder for pixel based observations.
    Input:  (B, T, H, W, C)  uint8 — replay buffer format
    Output: (B, T, feature_dim)  float32
    """

    def __init__(self, obs_shape, feature_dim: int):
        """
        obs_shape: (H, W, C) — shape of ONE observation, e.g. (64, 64, 3)
        feature_dim: output dim, typicaly = n_embd of the transformer
        """
        super().__init__()
        H, W, C = obs_shape

        self.convnet = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1), nn.ReLU(),
        )

        # Dynamically compute  the output of the conv
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            conv_out = self.convnet(dummy)
            self.conv_out_dim = conv_out.view(1, -1).shape[1]

        self.projection = nn.Sequential(
            nn.Linear(self.conv_out_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, T, H, W, C) uint8  or  (B, H, W, C) uint8
        returns: (B, T, feature_dim)  or  (B, feature_dim)
        """
        squeeze = obs.ndim == 4  # single-step  case (B, H, W, C)
        if squeeze:
            obs = obs.unsqueeze(1)  # -> (B, 1, H, W, C)

        B, T, H, W, C = obs.shape
        # Normalize uint8 -> float [0, 1] and use channels-first
        x = obs.reshape(B * T, H, W, C).float() / 255.0
        x = x.permute(0, 3, 1, 2).contiguous()  # (B*T, C, H, W)

        x = self.convnet(x)
        x = x.reshape(B * T, -1)
        x = self.projection(x)                   # (B*T, feature_dim)
        x = x.reshape(B, T, -1)                  # (B, T, feature_dim)

        if squeeze:
            x = x.squeeze(1)
        return x