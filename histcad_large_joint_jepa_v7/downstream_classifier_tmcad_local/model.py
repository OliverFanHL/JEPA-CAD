"""Full-width pooled-role queries over frozen pre-pooling STEP node states."""

from __future__ import annotations

import math

import torch
from torch import nn


class FullWidthLocalClassifier(nn.Module):
    """Same 512-D attention/head in both arms; only the key/value source differs.

    `pooled_fullwidth` attends the four 512-D pooled roles to each other.
    `pooled_plus_local` attends those same queries to *every* frozen 512-D
    pre-pooling node of the cached STEP graph. The original 2048-D pooled
    vector is concatenated unchanged into the classifier head in both arms.
    No 512->256 projection or fixed local-token subsampling is performed.
    """

    def __init__(self, classes: int, use_local: bool, heads: int = 8,
                 hidden: int = 256, dropout: float = 0.25):
        super().__init__()
        if 512 % heads:
            raise ValueError("512-D STEP node width must be divisible by attention heads")
        self.use_local = use_local
        self.role_query_bias = nn.Parameter(torch.empty(1, 4, 512))
        self.query_norm = nn.LayerNorm(512)
        self.node_norm = nn.LayerNorm(512)
        self.node_role = nn.Embedding(3, 512)
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
        nn.init.normal_(self.role_query_bias, std=0.02)
        nn.init.normal_(self.node_role.weight, std=0.02)

    def forward(self, batch: dict[str, torch.Tensor],
                return_attention: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        pooled = batch["pooled"]
        if pooled.ndim != 2 or pooled.shape[1] != 2048:
            raise ValueError("Expected the unchanged 2048-D frozen v7 STEP vector")
        scaled = pooled.float() * math.sqrt(2048)
        roles = scaled.reshape(len(scaled), 4, 512)
        queries = self.query_norm(roles) + self.role_query_bias
        if self.use_local:
            nodes, valid, node_roles = batch["nodes"], batch["valid"], batch["roles"]
            if nodes.ndim != 3 or nodes.shape[:2] != valid.shape or nodes.shape[2] != 512:
                raise ValueError("Expected all 512-D frozen node states and their valid mask")
            if not valid.any(dim=1).all():
                raise ValueError("A part has no local STEP nodes")
            keys = self.node_norm(nodes.float()) + self.node_role(node_roles.long())
            padding_mask = ~valid
        else:
            keys = queries
            padding_mask = None
        attended, weights = self.cross_attention(
            queries, keys, keys, key_padding_mask=padding_mask,
            need_weights=return_attention, average_attn_weights=True,
        )
        context = self.context_norm(queries + self.dropout(attended))
        # The full 2048-D pooled vector and four full-width 512-D readouts
        # reach the classifier head. Its final hidden width is not a JEPA
        # embedding bottleneck; it is a supervised logits head.
        logits = self.head(torch.cat((scaled, context.flatten(1)), dim=-1))
        return logits, weights


def build_model(name: str, classes: int, settings: dict) -> FullWidthLocalClassifier:
    if name not in {"pooled_fullwidth", "pooled_plus_local"}:
        raise ValueError(f"Unknown classifier: {name}")
    return FullWidthLocalClassifier(
        classes, use_local=name == "pooled_plus_local",
        heads=int(settings["attention_heads"]),
        hidden=int(settings["hidden_width"]),
        dropout=float(settings["dropout"]),
    )
