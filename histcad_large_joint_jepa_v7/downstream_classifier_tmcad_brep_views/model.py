"""Full-width attention over original and complementary frozen STEP views."""

from __future__ import annotations

import math

import torch
from torch import nn


class FullWidthLocalClassifier(nn.Module):
    """Same 512-D attention/head in both arms; only the key/value source differs.

    `pooled_fullwidth` attends the four original 512-D pooled roles to each
    other. `pooled_plus_brep_view` attends those same queries to the four
    frozen 512-D role tokens from a geometry-balanced STEP graph. The original
    2048-D pooled vector reaches the head unchanged in both arms. The model
    parameterization is otherwise identical, isolating the alternate view.
    """

    def __init__(self, classes: int, use_alternate: bool, heads: int = 8,
                 direct_alternate_readout: bool = False,
                 role_delta_readout: bool = False,
                 hidden: int = 256, dropout: float = 0.25):
        super().__init__()
        if 512 % heads:
            raise ValueError("512-D STEP node width must be divisible by attention heads")
        self.use_alternate = use_alternate
        if (direct_alternate_readout or role_delta_readout) and not use_alternate:
            raise ValueError("Complementary readout requires the complementary view")
        if direct_alternate_readout and role_delta_readout:
            raise ValueError("Select only one complementary readout")
        self.direct_alternate_readout = direct_alternate_readout
        self.role_delta_readout = role_delta_readout
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
        # Small linear bypass: tests whether cross-attention discards useful
        # complementary role information. No extra STEP parsing or JEPA update.
        if direct_alternate_readout:
            self.alternate_norm = nn.LayerNorm(2048)
            self.alternate_head = nn.Linear(2048, classes, bias=False)
            nn.init.zeros_(self.alternate_head.weight)
        if role_delta_readout:
            # Only the three B-Rep summary roles are compared. The fourth
            # global role is already present in the unchanged original vector
            # and dominated several observed attention readouts.
            self.role_delta_head = nn.Linear(3 * 512, classes, bias=False)
            nn.init.zeros_(self.role_delta_head.weight)
        nn.init.normal_(self.role_query_bias, std=0.02)

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
        logits = self.head(torch.cat((scaled, context.flatten(1)), dim=-1))
        if self.direct_alternate_readout:
            # Both stored 2048-D vectors have unit L2 norm. Scaling before the
            # linear head keeps the direct view on the original branch's scale.
            direct_view = view.float().reshape(len(scaled), 2048) * math.sqrt(2048)
            logits = logits + self.alternate_head(
                self.dropout(self.alternate_norm(direct_view))
            )
        if self.role_delta_readout:
            # The two saved 2048-D vectors are L2-normalized in the same way.
            # A signed, role-aligned difference exposes what the balanced graph
            # adds, without a new attention stack or trainable JEPA weights.
            view_roles = view.float() * math.sqrt(2048)
            delta = view_roles[:, :3] - roles[:, :3]
            logits = logits + self.role_delta_head(self.dropout(delta.flatten(1)))
        return logits, weights


def build_model(name: str, classes: int, settings: dict) -> FullWidthLocalClassifier:
    if name not in {"pooled_fullwidth", "pooled_plus_brep_view",
                    "pooled_plus_brep_view_direct", "pooled_plus_brep_view_role_delta"}:
        raise ValueError(f"Unknown classifier: {name}")
    return FullWidthLocalClassifier(
        classes, use_alternate=name != "pooled_fullwidth",
        heads=int(settings["attention_heads"]),
        direct_alternate_readout=name == "pooled_plus_brep_view_direct",
        role_delta_readout=name == "pooled_plus_brep_view_role_delta",
        hidden=int(settings["hidden_width"]),
        dropout=float(settings["dropout"]),
    )
