"""MesoInception-4 — the Inception-flavoured variant from Afchar et al., WIFS 2018.

The first two conv blocks of Meso-4 are replaced by "Inception-like" modules that run several
convolutions in parallel and concatenate them. The paper uses DILATED 3x3 convolutions (rates
1..3) in place of larger kernels, which enlarges the receptive field cheaply. The last two conv
blocks and the FC head are identical to Meso-4. ~28k parameters.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class InceptionLayer(nn.Module):
    """Four parallel branches, concatenated on the channel axis.

    Branch 1: 1x1 conv (a filters)
    Branch 2: 1x1 conv -> 3x3 conv, dilation 1 (b filters)
    Branch 3: 1x1 conv -> 3x3 conv, dilation 2 (c filters)
    Branch 4: 1x1 conv -> 3x3 conv, dilation 3 (d filters)
    Output channels: a + b + c + d
    """

    def __init__(self, in_ch: int, a: int, b: int, c: int, d: int):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, a, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.branch2 = self._dilated_branch(in_ch, b, dilation=1)
        self.branch3 = self._dilated_branch(in_ch, c, dilation=2)
        self.branch4 = self._dilated_branch(in_ch, d, dilation=3)
        self.out_channels = a + b + c + d

    @staticmethod
    def _dilated_branch(in_ch: int, ch: int, dilation: int) -> nn.Sequential:
        # padding == dilation keeps spatial size fixed for a 3x3 kernel
        return nn.Sequential(
            nn.Conv2d(in_ch, ch, kernel_size=1),
            nn.Conv2d(ch, ch, kernel_size=3, padding=dilation, dilation=dilation),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1
        )


def _conv_block(in_ch: int, out_ch: int, kernel: int, pool: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_ch),
        nn.MaxPool2d(kernel_size=pool, stride=pool),
    )


class MesoInception4(nn.Module):
    def __init__(self, num_classes: int = 1, dropout: float = 0.5, image_size: int = 256):
        super().__init__()
        self.inception1 = InceptionLayer(3, 1, 4, 4, 2)          # -> 11 ch
        self.bn1 = nn.BatchNorm2d(self.inception1.out_channels)
        self.pool1 = nn.MaxPool2d(2, 2)                          # 256 -> 128

        self.inception2 = InceptionLayer(self.inception1.out_channels, 2, 4, 4, 2)  # -> 12 ch
        self.bn2 = nn.BatchNorm2d(self.inception2.out_channels)
        self.pool2 = nn.MaxPool2d(2, 2)                          # 128 -> 64

        self.block3 = _conv_block(self.inception2.out_channels, 16, kernel=5, pool=2)  # 64 -> 32
        self.block4 = _conv_block(16, 16, kernel=5, pool=4)                            # 32 -> 8

        feat = image_size // 32
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(16 * feat * feat, 16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.bn1(self.inception1(x)))
        x = self.pool2(self.bn2(self.inception2(x)))
        x = self.block3(x)
        x = self.block4(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
