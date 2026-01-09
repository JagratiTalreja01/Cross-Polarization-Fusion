import torch
import torch.nn as nn

from models.cpf import CPFBlock


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_c, out_c)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    """
    Autoencoder-style upsampling block (NO skip connections).
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(in_c, out_c)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class CPFAutoEncoderSeg(nn.Module):
    """
    CPF + Autoencoder (segmentation):
      - Input: (B,2,H,W) [VV,VH]
      - VV stem: 1->C
      - VH stem: 1->C
      - CPF: fuse -> C
      - Encoder: C -> 2C -> 4C -> 8C
      - Decoder: 8C -> 4C -> 2C -> C
      - Output: 1 channel sigmoid mask
    No skip connections (unlike U-Net).
    """
    def __init__(self, base_c: int = 64):
        super().__init__()
        C = base_c

        # polarization-specific stems
        self.stem_vv = ConvBlock(1, C)
        self.stem_vh = ConvBlock(1, C)

        # CPF fusion at feature level
        self.cpf = CPFBlock(C)

        # Encoder
        self.enc2 = Down(C, 2 * C)
        self.enc3 = Down(2 * C, 4 * C)
        self.enc4 = Down(4 * C, 8 * C)

        # Bottleneck
        self.bottleneck = ConvBlock(8 * C, 8 * C)

        # Decoder (no skips)
        self.dec3 = Up(8 * C, 4 * C)
        self.dec2 = Up(4 * C, 2 * C)
        self.dec1 = Up(2 * C, C)

        self.out = nn.Conv2d(C, 1, kernel_size=1)

    def forward(self, x):
        vv = x[:, 0:1, :, :]
        vh = x[:, 1:2, :, :]

        f_vv = self.stem_vv(vv)     # (B,C,H,W)
        f_vh = self.stem_vh(vh)     # (B,C,H,W)

        z1 = self.cpf(f_vv, f_vh)   # (B,C,H,W)

        z2 = self.enc2(z1)          # (B,2C,H/2,W/2)
        z3 = self.enc3(z2)          # (B,4C,H/4,W/4)
        z4 = self.enc4(z3)          # (B,8C,H/8,W/8)

        b = self.bottleneck(z4)     # (B,8C,H/8,W/8)

        d3 = self.dec3(b)           # (B,4C,H/4,W/4)
        d2 = self.dec2(d3)          # (B,2C,H/2,W/2)
        d1 = self.dec1(d2)          # (B,C,H,W)

        return torch.sigmoid(self.out(d1))
