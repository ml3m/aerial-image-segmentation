"""
model.py — Pure-PyTorch U-Net for semantic segmentation.

Architecture reference:
    Ronneberger et al. (2015) "U-Net: Convolutional Networks for Biomedical
    Image Segmentation" — https://arxiv.org/abs/1505.04597

Modifications for aerial imagery:
    - Input channels = 3 (RGB)
    - Output channels = num_classes (default 6 for ISPRS Potsdam)
    - Bilinear upsampling instead of transposed convolution (smoother, fewer
      artefacts on large feature maps)
    - BatchNorm after every Conv for stable training
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import NUM_CLASSES


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):
    """Two consecutive Conv2d-BN-ReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Max-pool then DoubleConv (encoder step)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """
    Bilinear upsample, concatenate skip connection, then DoubleConv.

    The skip feature map from the encoder may be slightly larger due to
    integer rounding — we pad it with zeros to match sizes before concat.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        # in_ch = channels from previous decoder stage  (will be halved by up)
        # skip  = channels from skip connection          (= in_ch // 2)
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad x to match skip height/width if needed
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        if dh > 0 or dw > 0:
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """1×1 convolution to map feature channels → num_classes logits."""

    def __init__(self, in_ch: int, num_classes: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class UNet(nn.Module):
    """
    arguments:
        in_channels:  Number of input image channels (3 for RGB).
        num_classes:  Number of output segmentation classes.
        base_filters: Feature channels of the first encoder block.
                      Deeper blocks double this value.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = NUM_CLASSES,
        base_filters: int = 64,
    ) -> None:
        super().__init__()
        f = base_filters  # shorthand

        # Encoder
        self.inc   = DoubleConv(in_channels, f)       # 3   → 64
        self.down1 = Down(f,     f * 2)               # 64  → 128
        self.down2 = Down(f * 2, f * 4)               # 128 → 256
        self.down3 = Down(f * 4, f * 8)               # 256 → 512

        # Bottleneck
        self.down4 = Down(f * 8, f * 16)              # 512 → 1024

        # Decoder
        self.up1   = Up(f * 16 + f * 8,  f * 8)      # 1024+512 → 512
        self.up2   = Up(f * 8  + f * 4,  f * 4)      # 512+256  → 256
        self.up3   = Up(f * 4  + f * 2,  f * 2)      # 256+128  → 128
        self.up4   = Up(f * 2  + f,      f)           # 128+64   → 64

        self.outc  = OutConv(f, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)   # bottleneck

        # Decoder
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)

        return self.outc(x)   # shape: (B, num_classes, H, W)


# smoke test
if __name__ == "__main__":
    model = UNet(in_channels=3, num_classes=NUM_CLASSES)
    dummy = torch.randn(2, 3, 512, 512)
    out   = model(dummy)
    print(f"Input : {dummy.shape}")
    print(f"Output: {out.shape}")   # expected: (2, 6, 512, 512)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {total_params:,}")
