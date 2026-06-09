"""Small pixel policy for drawing imitation."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def block(cin: int, cout: int, stride: int = 1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, kernel_size=3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class DrawingPolicy(nn.Module):
    """Predict the next brush cell from current canvas and final target image."""

    def __init__(self, grid: int = 32, channels=(32, 64, 128)):
        super().__init__()
        self.grid = grid
        self.net = nn.Sequential(
            block(2, channels[0]),
            block(channels[0], channels[1], stride=2),
            block(channels[1], channels[2], stride=2),
        )
        self.head = nn.Sequential(
            nn.Conv2d(channels[-1] + 2, channels[-1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.10),
            nn.Conv2d(channels[-1], 1, kernel_size=1),
        )

    def _coords(self, feat: torch.Tensor) -> torch.Tensor:
        b, _c, h, w = feat.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, h, device=feat.device, dtype=feat.dtype),
            torch.linspace(-1, 1, w, device=feat.device, dtype=feat.dtype),
            indexing="ij",
        )
        coords = torch.stack((xx, yy), dim=0).expand(b, -1, -1, -1)
        return torch.cat((feat, coords), dim=1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        feat = self._coords(self.net(obs))
        heat = self.head(feat)
        heat = F.interpolate(
            heat, size=(self.grid, self.grid), mode="bilinear", align_corners=False
        )
        return heat.flatten(1)

    @torch.no_grad()
    def predict_cell(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs).argmax(dim=1)
