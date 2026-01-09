import torch
import torch.nn as nn

from models.cpf import CPFBlock


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class CPFUNet(nn.Module):
    """
    CPF-UNet:
      VV(1ch)->stem->C
      VH(1ch)->stem->C
      CPF(C,C)->fused C
      then UNet encoder-decoder
    Input: (B,2,H,W) with channels [VV, VH]
    Output: (B,1,H,W) sigmoid probabilities
    """
    def __init__(self, base_c: int = 64):
        super().__init__()
        C = base_c

        # polarization-specific stems
        self.stem_vv = DoubleConv(1, C)
        self.stem_vh = DoubleConv(1, C)

        # CPF fusion at feature level
        self.cpf = CPFBlock(C)

        # UNet encoder
        self.pool = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(C, 2*C)
        self.enc3 = DoubleConv(2*C, 4*C)

        # decoder
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = DoubleConv(4*C + 2*C, 2*C)
        self.dec2 = DoubleConv(2*C + C, C)

        self.out = nn.Conv2d(C, 1, kernel_size=1)

    def forward(self, x):
        # x: (B,2,H,W)
        vv = x[:, 0:1, :, :]
        vh = x[:, 1:2, :, :]

        f_vv = self.stem_vv(vv)   # (B,C,H,W)
        f_vh = self.stem_vh(vh)   # (B,C,H,W)

        e1 = self.cpf(f_vv, f_vh) # fused (B,C,H,W)

        e2 = self.enc2(self.pool(e1))      # (B,2C,H/2,W/2)
        e3 = self.enc3(self.pool(e2))      # (B,4C,H/4,W/4)

        d3 = self.up(e3)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))

        d2 = self.up(d3)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        return torch.sigmoid(self.out(d2))
