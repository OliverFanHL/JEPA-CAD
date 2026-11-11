"""Full-width attention over original and complementary frozen STEP views."""

from __future__ import annotations

import math

import torch
from torch import nn
from geometry_features import FEATURE_NAMES


class FullWidthLocalClassifier(nn.Module):
    """Frozen two-view attention, optionally with full-file geometry evidence."""

    def __init__(self, classes: int, use_alternate: bool, heads: int = 8,
                 geometry_readout: bool = False,
                 geometry_interaction: bool = False, interaction_rank: int = 16,
                 hidden: int = 256, dropout: float = 0.25):
        super().__init__()
        if 512 % heads:
            raise ValueError("512-D STEP node width must be divisible by attention heads")
        self.use_alternate = use_alternate
        self.geometry_readout = geometry_readout
        self.geometry_interaction = geometry_interaction
        if geometry_readout and not use_alternate:
            raise ValueError("Geometry comparison requires the complementary view")
        if geometry_interaction and not geometry_readout:
            raise ValueError("Geometry interaction requires the additive geometry control")
        if geometry_interaction and not 1 <= interaction_rank <= 64:
            raise ValueError("Geometry interaction rank must be between 1 and 64")
        self.role_query_bias = nn.Parameter(torch.empty(1, 4, 512))
        self.query_norm = nn.LayerNorm(512)
        self.node_norm = nn.LayerNorm(512)
        self.cross_attention = nn.MultiheadAttention(
            512, heads, dropout=dropout, batch_first=True,
        )
        self.context_norm = nn.LayerNorm(512)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(4096),
            nn.Linear(4096, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, classes),
        )
        if geometry_readout:
            # Train-only-standardized physical descriptors use a small linear
            # residual head; the JEPA and two-view attention paths are intact.
            self.geometry_head = nn.Linear(len(FEATURE_NAMES), classes, bias=False)
            nn.init.zeros_(self.geometry_head.weight)
        nn.init.normal_(self.role_query_bias, std=0.02)
        if geometry_interaction:
            # Low-rank interaction between the same supervised JEPA readout
            # and the standardized full-file geometry. Additive control weights
            # above initialize identically in both comparison arms.
            self.geometry_factor = nn.Linear(len(FEATURE_NAMES), interaction_rank, bias=False)
            self.jepa_factor = nn.Linear(hidden, interaction_rank, bias=False)
            self.interaction_head = nn.Linear(interaction_rank, classes, bias=False)
            nn.init.zeros_(self.interaction_head.weight)

    def forward(self, batch: dict[str, torch.Tensor],
                return_attention: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        pooled = batch["pooled"]
        if pooled.ndim != 2 or pooled.shape[1] != 2048:
            raise ValueError("Expected the unchanged 2048-D frozen v7 STEP vector")
        scaled = pooled.float() * math.sqrt(2048)
        roles = scaled.reshape(len(scaled), 4, 512)
        queries = self.query_norm(roles) + self.role_query_bias
        if self.use_alternate:
            view = batch["alternative"]
            if view.ndim != 3 or view.shape != (len(scaled), 4, 512):
                raise ValueError("Expected four 512-D complementary pooled roles")
            keys = self.node_norm(view.float())
        else:
            keys = queries
        attended, weights = self.cross_attention(
            queries, keys, keys,
            need_weights=return_attention, average_attn_weights=True,
        )
        context = self.context_norm(queries + self.dropout(attended))
        # The original full 2048-D pooled vector and four full-width 512-D
        # readouts reach the same supervised head in both comparison arms.
        hidden = self.head[:-1](torch.cat((scaled, context.flatten(1)), dim=-1))
        logits = self.head[-1](hidden)
        if self.geometry_readout:
            geometry = batch["geometry"].float()
            if geometry.ndim != 2 or geometry.shape != (len(scaled), len(FEATURE_NAMES)):
                raise ValueError("Expected standardized full-file STEP geometry descriptors")
            logits = logits + self.geometry_head(self.dropout(geometry))
            if self.geometry_interaction:
                joint = torch.tanh(self.geometry_factor(geometry)) * torch.tanh(
                    self.jepa_factor(hidden)
                )
                logits = logits + self.interaction_head(self.dropout(joint))
        return logits, weights


def build_model(name: str, classes: int, settings: dict) -> FullWidthLocalClassifier:
    if name not in {"pooled_plus_brep_view", "pooled_plus_brep_view_geometry",
                    "pooled_plus_brep_view_geometry_interaction"}:
        raise ValueError(f"Unknown classifier: {name}")
    return FullWidthLocalClassifier(
        classes, use_alternate=True,
        heads=int(settings["attention_heads"]),
        geometry_readout=name in {"pooled_plus_brep_view_geometry",
                                  "pooled_plus_brep_view_geometry_interaction"},
        geometry_interaction=name == "pooled_plus_brep_view_geometry_interaction",
        interaction_rank=int(settings.get("interaction_rank", 16)),
        hidden=int(settings["hidden_width"]),
        dropout=float(settings["dropout"]),
    )
