"""Exact frozen mesh-classifier control plus two small residual readouts."""

from __future__ import annotations

import math

import torch
from torch import nn


GEOMETRY_WIDTH = 37
SURFACE_WIDTH = 84
PHYSICAL_WIDTH = GEOMETRY_WIDTH + SURFACE_WIDTH
MODEL_NAME = "pooled_plus_brep_view_geometry_mesh"


class FrozenMeshClassifier(nn.Module):
    """State-dict-compatible copy of the existing trained control model.

    Its parameters are loaded from the saved checkpoint, placed in eval mode,
    and never optimized. `hidden` is JEPA/two-view context *before* the
    geometry and surface-profile logit additions.
    """

    def __init__(self, classes: int, heads: int, hidden: int, dropout: float):
        super().__init__()
        if 512 % heads:
            raise ValueError("attention_heads must divide 512")
        self.role_query_bias = nn.Parameter(torch.empty(1, 4, 512))
        self.query_norm = nn.LayerNorm(512)
        self.node_norm = nn.LayerNorm(512)
        self.cross_attention = nn.MultiheadAttention(
            512, heads, dropout=dropout, batch_first=True,
        )
        self.context_norm = nn.LayerNorm(512)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(4096), nn.Linear(4096, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, classes),
        )
        self.geometry_head = nn.Linear(GEOMETRY_WIDTH, classes, bias=False)
        self.mesh_head = nn.Linear(SURFACE_WIDTH, classes, bias=False)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = batch["pooled"]
        alternative = batch["alternative"]
        geometry = batch["geometry"]
        surface = batch["mesh"]
        size = len(pooled)
        if (pooled.shape != (size, 2048)
                or alternative.shape != (size, 4, 512)
                or geometry.shape != (size, GEOMETRY_WIDTH)
                or surface.shape != (size, SURFACE_WIDTH)):
            raise ValueError("Frozen TMCAD input widths do not match the baseline")
        scaled = pooled.float() * math.sqrt(2048)
        queries = self.query_norm(scaled.reshape(size, 4, 512)) + self.role_query_bias
        keys = self.node_norm(alternative.float())
        attended, _ = self.cross_attention(queries, keys, keys, need_weights=False)
        context = self.context_norm(queries + self.dropout(attended))
        hidden = self.head[:-1](torch.cat((scaled, context.flatten(1)), dim=-1))
        logits = self.head[-1](hidden)
        logits = logits + self.geometry_head(self.dropout(geometry.float()))
        logits = logits + self.mesh_head(self.dropout(surface.float()))
        return logits, hidden


class ResidualAdapter(nn.Module):
    """A modest nonlinear correction; baseline logits are immutable inputs."""

    def __init__(self, variant: str, classes: int, settings: dict,
                 baseline_hidden_width: int):
        super().__init__()
        if variant not in {"physical_residual", "conditional_residual"}:
            raise ValueError(f"Unsupported adapter variant: {variant}")
        self.variant = variant
        physical_width = int(settings["physical_width"])
        context_width = int(settings["context_width"])
        fusion_width = int(settings["fusion_width"])
        dropout = float(settings["dropout"])
        if not (16 <= physical_width <= 256 and 16 <= context_width <= 128
                and 16 <= fusion_width <= 256 and 0 <= dropout < 0.8):
            raise ValueError("Adapter dimensions/dropout are outside modest bounds")
        self.physical = nn.Sequential(
            nn.Linear(PHYSICAL_WIDTH, physical_width), nn.GELU(),
            nn.Dropout(dropout),
        )
        if variant == "conditional_residual":
            self.context = nn.Sequential(
                nn.LayerNorm(baseline_hidden_width),
                nn.Linear(baseline_hidden_width, context_width),
                nn.GELU(), nn.Dropout(dropout),
            )
            fusion_in = physical_width + context_width
        else:
            self.context = None
            fusion_in = physical_width
        self.readout = nn.Sequential(
            nn.Linear(fusion_in, fusion_width), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(fusion_width, classes),
        )
        # At initialization every prediction is exactly the saved baseline.
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, baseline_logits: torch.Tensor, hidden: torch.Tensor,
                geometry: torch.Tensor, surface: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        if (geometry.shape[-1] != GEOMETRY_WIDTH
                or surface.shape[-1] != SURFACE_WIDTH):
            raise ValueError("Physical descriptor widths differ from training")
        physical = torch.cat((geometry.float(), surface.float()), dim=-1)
        encoded = self.physical(physical)
        if self.context is not None:
            encoded = torch.cat((encoded, self.context(hidden.float())), dim=-1)
        residual = self.readout(encoded)
        return baseline_logits.float() + residual, residual
