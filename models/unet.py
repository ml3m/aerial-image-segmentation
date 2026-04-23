"""
Pure-PyTorch U-Net for semantic segmentation.

Reference:
    Ronneberger et al. (2015) "U-Net: Convolutional Networks for Biomedical
    Image Segmentation" — https://arxiv.org/abs/1505.04597

Adaptations for aerial imagery:
  - Input channels = 3 (RGB)
  - Output channels = num_classes (default 6 for ISPRS Potsdam)
  - Bilinear upsampling (fewer artefacts than transposed conv)
  - BatchNorm after every conv for stable training
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive Conv-BN-ReLU blocks."""

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
    """MaxPool + DoubleConv (encoder step)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """Bilinear upsample, align with skip, DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad x to match the skip tensor's spatial size when shapes diverge
        # due to odd-sized inputs.
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        if dh > 0 or dw > 0:
            x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_ch: int, num_classes: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    Args:
        in_channels:  Input image channels (3 for RGB).
        num_classes:  Number of output classes.
        base_filters: Channels of the first encoder block (doubled per depth).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
        base_filters: int = 64,
    ) -> None:
        super().__init__()
        f = base_filters

        self.inc = DoubleConv(in_channels, f)
        self.down1 = Down(f, f * 2)
        self.down2 = Down(f * 2, f * 4)
        self.down3 = Down(f * 4, f * 8)
        self.down4 = Down(f * 8, f * 16)  # bottleneck

        self.up1 = Up(f * 16 + f * 8, f * 8)
        self.up2 = Up(f * 8 + f * 4, f * 4)
        self.up3 = Up(f * 4 + f * 2, f * 2)
        self.up4 = Up(f * 2 + f, f)

        self.outc = OutConv(f, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)


if __name__ == "__main__":
    # Smoke test: forward a batch and print the shape + param count.
    from utils.device import get_device

    device = get_device()
    model = UNet(in_channels=3, num_classes=6).to(device)
    dummy = torch.randn(2, 3, 512, 512, device=device)
    out = model(dummy)
    print(f"Input : {tuple(dummy.shape)}")
    print(f"Output: {tuple(out.shape)}  (expected: (2, 6, 512, 512))")
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {total:,}")
