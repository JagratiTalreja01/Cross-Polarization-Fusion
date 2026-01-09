import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.mlp(x)
        return x * w


class SpatialAttention(nn.Module):
    """Spatial attention using channel-wise avg/max pooling."""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        s = torch.cat([avg, mx], dim=1)
        a = self.act(self.conv(s))
        return x * a


class CrossGating(nn.Module):
    """
    Produces a gate map from source features, applied to target features.
    Gate is computed from concatenated (source, target) to allow interaction.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, source, target):
        g = self.gate(torch.cat([source, target], dim=1))
        return target * g


class CPFBlock(nn.Module):
    """
    Cross-Polarization Fusion block:
      - CA + SA on each branch
      - cross-gating VV->VH and VH->VV
      - fuse and project
    Inputs: f_vv, f_vh (B,C,H,W)
    Output: fused (B,C,H,W)
    """
    def __init__(self, channels: int):
        super().__init__()
        self.ca_vv = ChannelAttention(channels)
        self.ca_vh = ChannelAttention(channels)
        self.sa_vv = SpatialAttention()
        self.sa_vh = SpatialAttention()

        self.gate_vv_to_vh = CrossGating(channels)
        self.gate_vh_to_vv = CrossGating(channels)

        self.proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, f_vv, f_vh):
        # intra-branch attention
        f_vv_a = self.sa_vv(self.ca_vv(f_vv))
        f_vh_a = self.sa_vh(self.ca_vh(f_vh))

        # cross interaction (bidirectional)
        vh_g = self.gate_vv_to_vh(f_vv_a, f_vh_a)
        vv_g = self.gate_vh_to_vv(f_vh_a, f_vv_a)

        # fuse
        fused = self.proj(torch.cat([vv_g, vh_g], dim=1))
        return fused
