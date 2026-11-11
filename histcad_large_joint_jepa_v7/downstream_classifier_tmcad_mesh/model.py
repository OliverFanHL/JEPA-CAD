"""Full-width attention over original and complementary frozen STEP views."""

from __future__ import annotations

import math

import torch
from torch import nn
from geometry_features import FEATURE_NAMES
from mesh_features import FEATURE_NAMES as MESH_NAMES


class FullWidthLocalClassifier(nn.Module):
    """Frozen STEP views plus physical descriptors and optional surface shape."""

    def __init__(self, classes: int, use_alternate: bool, heads: int = 8,
                 geometry_readout: bool = False,
                 mesh_readout: bool = False,
                 hidden: int = 256, dropout: float = 0.25):
        super().__init__()
        if 512 % heads:
            raise ValueError("512-D STEP node width must be divisible by attention heads")
        self.use_alternate = use_alternate
        self.geometry_readout = geometry_readout
        self.mesh_readout = mesh_readout
        if geometry_readout and not use_alternate:
            raise ValueError("Geometry comparison requires the complementary view")
        if mesh_readout and not geometry_readout:
            raise ValueError("STEP surface profile requires the geometry control")
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
        if mesh_readout:
            # Zero initialization preserves the baseline prediction at start;
            # the existing JEPA and geometry arms initialize identically.
            self.mesh_head = nn.Linear(len(MESH_NAMES), classes, bias=False)
            nn.init.zeros_(self.mesh_head.weight)

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
        if self.mesh_readout:
            mesh = batch["mesh"].float()
            if mesh.ndim != 2 or mesh.shape != (len(scaled), len(MESH_NAMES)):
                raise ValueError("Expected standardized STEP surface profile")
            logits = logits + self.mesh_head(self.dropout(mesh))
        return logits, weights


def build_model(name: str, classes: int, settings: dict) -> FullWidthLocalClassifier:
    if name not in {"pooled_plus_brep_view_geometry",
                    "pooled_plus_brep_view_geometry_mesh"}:
        raise ValueError(f"Unknown classifier: {name}")
    return FullWidthLocalClassifier(
        classes, use_alternate=True,
        heads=int(settings["attention_heads"]),
        geometry_readout=True,
        mesh_readout=name == "pooled_plus_brep_view_geometry_mesh",
        hidden=int(settings["hidden_width"]),
        dropout=float(settings["dropout"]),
    )
