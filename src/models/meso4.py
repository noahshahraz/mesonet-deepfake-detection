"""Meso-4 — the compact forgery detector from Afchar et al., WIFS 2018 (arXiv:1809.00888).

Architecture (paper Fig. 2 / Table 1): four Conv->ReLU->BatchNorm->MaxPool blocks operating
at the "mesoscopic" scale, followed by a small fully-connected head. ~28k parameters.

Input : 256x256x3 RGB face crop.
Output: a single logit (use BCEWithLogitsLoss). The original Keras model ends in a sigmoid;
        emitting a raw logit is numerically equivalent and more stable — see README "Differences".
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _block(in_ch: int, out_ch: int, kernel: int, pool: int) -> nn.Sequential:
    padding = kernel // 2  # 'same' padding for odd kernels
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=padding),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_ch),
        nn.MaxPool2d(kernel_size=pool, stride=pool),
    )


class Meso4(nn.Module):
    def __init__(self, num_classes: int = 1, dropout: float = 0.5, image_size: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            _block(3, 8, kernel=3, pool=2),    # 256 -> 128
            _block(8, 8, kernel=5, pool=2),    # 128 -> 64
            _block(8, 16, kernel=5, pool=2),   # 64  -> 32
            _block(16, 16, kernel=5, pool=4),  # 32  -> 8   (larger final pool, per paper)
        )
        feat = image_size // 32  # total downsample 2*2*2*4 = 32
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(16 * feat * feat, 16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
