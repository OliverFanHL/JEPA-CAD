"""Attention across the meaningful frozen v7 summary slots, not arbitrary coordinates."""

from __future__ import annotations

import math

import torch
from torch import nn

from labels import TASKS

TOKEN_NAMES = (
    "classification_query",
    "param_feature_operations",
    "param_sketch_geometry",
    "param_constraints",
    "param_all_relations",
    "param_global_history",
    "step_typed_entities",
    "step_brep_instances",
    "step_all_connectivity",
    "step_global_type",
    "joint_global",
)


class AttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int, feedforward: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(
            nn.Linear(width, feedforward), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(feedforward, width), nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_weights: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        query = self.norm1(x)
        values, weights = self.attention(
            query, query, query, need_weights=return_weights,
            average_attn_weights=False if return_weights else True,
        )
        x = x + self.dropout(values)
        x = x + self.feedforward(self.norm2(x))
        return x, weights


class RoleTokenAttentionClassifier(nn.Module):
    """Frozen-feature probe with cross-source self-attention and task heads.

    The five parametric and four STEP tokens are exactly the contiguous 512-D
    slots written by the v7 role-aware pools. The 4096-D joint vector is a
    separate global token. No token is a zero-padded or duplicated chunk.
    """

    def __init__(self, report: dict, width: int = 256, heads: int = 8, depth: int = 2,
                 feedforward: int = 768, dropout: float = 0.1):
        super().__init__()
        if width % heads:
            raise ValueError("Attention width must be divisible by number of heads")
        self.width = width
        self.param_projection = nn.Linear(512, width)
        self.step_projection = nn.Linear(512, width)
        self.joint_projection = nn.Linear(4096, width)
        self.token_norm = nn.LayerNorm(width)
        self.query_token = nn.Parameter(torch.empty(1, 1, width))
        self.role_embedding = nn.Parameter(torch.empty(1, len(TOKEN_NAMES), width))
        self.blocks = nn.ModuleList([
            AttentionBlock(width, heads, feedforward, dropout) for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(width)
        self.shared = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout))
        self.heads = nn.ModuleDict({
            task: nn.Linear(width, len(report["tasks"][task]["retained_classes"]))
            for task in TASKS
        })
        nn.init.normal_(self.query_token, std=0.02)
        nn.init.normal_(self.role_embedding, std=0.02)

    def forward(self, parametric: torch.Tensor, step: torch.Tensor, joint: torch.Tensor,
                return_attention: bool = False) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
        batch = parametric.shape[0]
        if parametric.shape != (batch, 2560) or step.shape != (batch, 2048) or joint.shape != (batch, 4096):
            raise ValueError("Expected frozen v7 widths: parametric=2560, STEP=2048, joint=4096")
        # Restore ordinary O(1) feature magnitudes from the saved L2-normalized
        # branch vectors, then use the existing role boundaries as token borders.
        p = self.param_projection(parametric.float().reshape(batch, 5, 512) * math.sqrt(2560))
        s = self.step_projection(step.float().reshape(batch, 4, 512) * math.sqrt(2048))
        j = self.joint_projection(joint.float() * math.sqrt(4096)).unsqueeze(1)
        q = self.query_token.expand(batch, -1, -1)
        x = torch.cat((q, p, s, j), dim=1)
        x = self.token_norm(x) + self.role_embedding
        weights = None
        for index, block in enumerate(self.blocks):
            x, weights = block(x, return_attention and index == len(self.blocks) - 1)
        pooled = self.shared(self.final_norm(x[:, 0]))
        logits = {task: head(pooled) for task, head in self.heads.items()}
        # Averaged final-block query attention is diagnostic, not a causal explanation.
        query_weights = weights[:, :, 0].mean(1) if weights is not None else None
        return logits, query_weights


class ParametricResidualAttentionClassifier(RoleTokenAttentionClassifier):
    """Keep a direct, regularized parametric path for the ambiguous form head.

    This does not replace the cross-source attention path and never updates the
    JEPA encoder. The auxiliary output is used only during training.
    """

    def __init__(self, report: dict, width: int = 256, heads: int = 8, depth: int = 2,
                 feedforward: int = 768, dropout: float = 0.2,
                 parametric_hidden: int = 384, parametric_dropout: float = 0.3):
        super().__init__(report, width, heads, depth, feedforward, dropout)
        self.form_parametric = nn.Sequential(
            nn.LayerNorm(2560),
            nn.Linear(2560, parametric_hidden),
            nn.GELU(),
            nn.Dropout(parametric_dropout),
            nn.Linear(parametric_hidden, len(report["tasks"]["functional_form"]["retained_classes"])),
        )
        nn.init.zeros_(self.form_parametric[-1].weight)
        nn.init.zeros_(self.form_parametric[-1].bias)

    def forward(self, parametric: torch.Tensor, step: torch.Tensor, joint: torch.Tensor,
                return_attention: bool = False) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
        logits, weights = super().forward(parametric, step, joint, return_attention)
        direct = self.form_parametric(parametric.float() * math.sqrt(2560))
        logits["functional_form"] = logits["functional_form"] + direct
        logits["functional_form_parametric"] = direct
        return logits, weights
