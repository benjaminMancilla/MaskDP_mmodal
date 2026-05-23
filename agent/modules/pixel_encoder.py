import torch
import torch.nn as nn
import torchvision.models as models


class DrQv2Encoder(nn.Module):
    """
    CNN encoder from DrQ-v2 (Yarats et al., 2021).
    Reference: https://github.com/conglu1997/v-d4rl/blob/main/drqbc/drqv2.py
    output_dim=256.
    """

    def __init__(self, obs_shape, feature_dim: int):
        super().__init__()
        H, W, C = obs_shape
        assert H == W, f"DrQv2Encoder assumes square input, got {H}x{W}"

        self.convnet = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            conv_out = self.convnet(dummy)
            self.conv_out_dim = int(conv_out.flatten(1).shape[1])

        self.projection = nn.Sequential(
            nn.Linear(self.conv_out_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )

        self.apply(_drq_weight_init)

    def forward_single(self, x_btchw_uint8: torch.Tensor) -> torch.Tensor:
        x = x_btchw_uint8.float() / 255.0 - 0.5
        h = self.convnet(x)
        h = h.flatten(1)
        return self.projection(h)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        squeeze = obs.ndim == 4
        if squeeze:
            obs = obs.unsqueeze(1)  # -> (B, 1, H, W, C)

        B, T, H, W, C = obs.shape
        x = obs.reshape(B * T, H, W, C).permute(0, 3, 1, 2).contiguous()  # (B*T, C, H, W)
        feat = self.forward_single(x)
        feat = feat.reshape(B, T, -1)

        if squeeze:
            feat = feat.squeeze(1)
        return feat


class ResNetFrozenEncoder(nn.Module):
    """
    Legacy ResNet18-ImageNet encoder, fully frozen. Kept here for reproducibility of
    older experiments and as a comparison baseline. Not recommended as the default.
    """

    def __init__(self, obs_shape, feature_dim: int):
        super().__init__()
        H, W, C = obs_shape

        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.conv_out_dim = 512
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(self.conv_out_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )
        for p in self.projection.parameters():
            p.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        squeeze = obs.ndim == 4
        if squeeze:
            obs = obs.unsqueeze(1)

        B, T, H, W, C = obs.shape
        x = obs.reshape(B * T, H, W, C).float() / 255.0
        x = x.permute(0, 3, 1, 2).contiguous()
        x = (x - self.mean) / self.std

        x = self.backbone(x)
        x = x.reshape(B * T, -1)
        x = self.projection(x)
        x = x.reshape(B, T, -1)

        if squeeze:
            x = x.squeeze(1)
        return x


def PixelEncoder(obs_shape, feature_dim: int, encoder_type: str = "drqv2"):
    """
    Returns the requested encoder module.

    encoder_type:
        - "drqv2"          : trainable DrQ-v2 style CNN (default; ~250k params)
        - "resnet_frozen"  : ResNet18-ImageNet, fully frozen (legacy baseline)
    """
    if encoder_type == "drqv2":
        return DrQv2Encoder(obs_shape, feature_dim)
    elif encoder_type == "resnet_frozen":
        return ResNetFrozenEncoder(obs_shape, feature_dim)
    else:
        raise ValueError(
            f"Unknown encoder_type='{encoder_type}'. "
            f"Choose from: 'drqv2', 'resnet_frozen'."
        )


def _drq_weight_init(m):
    """Orthogonal init for convs/linear, identity for LayerNorm. Matches DrQ-v2."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.LayerNorm):
        m.bias.data.zero_()
        m.weight.data.fill_(1.0)