"""Small text-target adapter; the JEPA encoder and text teacher stay frozen."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SemanticAdapter(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        input_dim = config["input_dim"]
        hidden_dim = config["hidden_dim"]
        output_dim = config["output_dim"]
        self.backbone = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(hidden_dim, output_dim),
        )
        self.dimension_head = nn.Linear(output_dim, 22)
        self.feature_count_head = nn.Linear(output_dim, 4)
        self.operation_head = nn.Linear(output_dim, 4)

    def forward(self, frozen_jepa: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = F.normalize(self.backbone(frozen_jepa.float()), dim=-1)
        return {
            "embedding": embedding,
            "dimension": self.dimension_head(embedding),
            "feature_count": self.feature_count_head(embedding),
            "operation": self.operation_head(embedding),
        }
