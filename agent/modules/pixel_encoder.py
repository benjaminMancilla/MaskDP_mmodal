import torch
import torch.nn as nn
import torchvision.models as models


class PixelEncoder(nn.Module):
    """
    CNN placeholder backbone for pixel based observations.
    Pretrained with RestNet18 (ImageNet)
    Frozen weights except the final projection.
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

        # ResNet18 with out the final classification layer
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # → (B, 512, 1, 1)
        self.conv_out_dim = 512

        # Freeze the whole backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Only the last projection is trainable
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

         # ResNet espera imágenes normalizadas con mean/std de ImageNet
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        x = self.backbone(x)
        x = x.reshape(B * T, -1)
        x = self.projection(x)                   # (B*T, feature_dim)
        x = x.reshape(B, T, -1)                  # (B, T, feature_dim)

        if squeeze:
            x = x.squeeze(1)
        return x