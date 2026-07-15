"""Minimal ONNX-facing wrapper for the unchanged GCN_res model."""

from __future__ import annotations

import torch
from torch import nn


class GCNResOnnxWrapper(nn.Module):
    """Expose only logits while keeping points and adjacency as external inputs."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, points: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        _, logits = self.model(points, adj)
        return logits
