"""Frozen-STEP downstream heads; no JEPA weights are trainable here."""

from __future__ import annotations

import math

import torch
from torch import nn

TOKEN_NAMES = ("class_query", "typed_entities", "brep_instances", "connectivity", "global_step")


class LinearProbe(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.head = nn.Linear(2048, num_classes)

    def forward(self, embedding: torch.Tensor, return_attention: bool = False,
                structural: torch.Tensor | None = None):
        if embedding.ndim != 2 or embedding.shape[1] != 2048:
            raise ValueError("Expected frozen 2048-D STEP embeddings")
        return self.head(embedding.float() * math.sqrt(2048)), None


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

    def forward(self, tokens: torch.Tensor, return_attention: bool):
        normal = self.norm1(tokens)
        values, weights = self.attention(normal, normal, normal,
                                         need_weights=return_attention,
                                         average_attn_weights=not return_attention)
        tokens = tokens + self.dropout(values)
        tokens = tokens + self.feedforward(self.norm2(tokens))
        return tokens, weights


class StepTokenAttentionClassifier(nn.Module):
    """Attend across the actual 3 pooled STEP roles plus its global 512-D slot.

    This is attention over frozen role summaries, not over raw STEP entities.
    The four tokens are contiguous 512-D regions of the v7 STEP encoder output.
    """

    def __init__(self, num_classes: int, width: int = 256, heads: int = 8,
                 depth: int = 2, feedforward: int = 768, dropout: float = 0.2):
        super().__init__()
        if width % heads:
            raise ValueError("Attention width must be divisible by its head count")
        self.token_projection = nn.Linear(512, width)
        self.query = nn.Parameter(torch.empty(1, 1, width))
        self.roles = nn.Parameter(torch.empty(1, 5, width))
        self.token_norm = nn.LayerNorm(width)
        self.blocks = nn.ModuleList([
            AttentionBlock(width, heads, feedforward, dropout) for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(width)
        self.head = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width, num_classes),
        )
        nn.init.normal_(self.query, std=0.02)
        nn.init.normal_(self.roles, std=0.02)

    def forward(self, embedding: torch.Tensor, return_attention: bool = False,
                structural: torch.Tensor | None = None):
        if embedding.ndim != 2 or embedding.shape[1] != 2048:
            raise ValueError("Expected frozen 2048-D STEP embeddings")
        batch = len(embedding)
        parts = embedding.float().reshape(batch, 4, 512) * math.sqrt(2048)
        parts = self.token_projection(parts)
        tokens = torch.cat((self.query.expand(batch, -1, -1), parts), dim=1)
        tokens = self.token_norm(tokens) + self.roles
        weights = None
        for index, block in enumerate(self.blocks):
            tokens, weights = block(tokens, return_attention and index == len(self.blocks) - 1)
        logits = self.head(self.final_norm(tokens[:, 0]))
        query_weights = weights[:, :, 0].mean(1) if weights is not None else None
        return logits, query_weights


class ClassRoleAttentionClassifier(StepTokenAttentionClassifier):
    """Small class-specific readout of the four frozen STEP role summaries.

    The original query readout is retained. Each class can additionally attend
    to the entity, instance, connectivity, and global roles differently. This
    adds only class-by-width parameters, not another transformer or STEP input.
    """

    def __init__(self, num_classes: int, width: int = 256, heads: int = 8,
                 depth: int = 2, feedforward: int = 768, dropout: float = 0.2):
        super().__init__(num_classes, width, heads, depth, feedforward, dropout)
        self.class_queries = nn.Parameter(torch.empty(num_classes, width))
        self.class_weights = nn.Parameter(torch.empty(num_classes, width))
        self.class_bias = nn.Parameter(torch.zeros(num_classes))
        self.class_role_norm = nn.LayerNorm(width)
        self.class_role_dropout = nn.Dropout(dropout)
        self.role_scale = nn.Parameter(torch.tensor(0.25))
        nn.init.normal_(self.class_queries, std=0.02)
        nn.init.normal_(self.class_weights, std=width ** -0.5)

    def forward(self, embedding: torch.Tensor, return_attention: bool = False,
                structural: torch.Tensor | None = None):
        if embedding.ndim != 2 or embedding.shape[1] != 2048:
            raise ValueError("Expected frozen 2048-D STEP embeddings")
        batch = len(embedding)
        parts = embedding.float().reshape(batch, 4, 512) * math.sqrt(2048)
        parts = self.token_projection(parts)
        tokens = torch.cat((self.query.expand(batch, -1, -1), parts), dim=1)
        tokens = self.token_norm(tokens) + self.roles
        weights = None
        for index, block in enumerate(self.blocks):
            tokens, weights = block(tokens, return_attention and index == len(self.blocks) - 1)

        tokens = self.final_norm(tokens)
        shared_logits = self.head(tokens[:, 0])
        role_tokens = tokens[:, 1:]
        scores = torch.einsum("cw,brw->bcr", self.class_queries, role_tokens) / math.sqrt(role_tokens.shape[-1])
        class_attention = scores.softmax(dim=-1)
        pooled = torch.einsum("bcr,brw->bcw", class_attention, role_tokens)
        pooled = self.class_role_dropout(self.class_role_norm(pooled))
        class_logits = torch.einsum("bcw,cw->bc", pooled, self.class_weights) + self.class_bias
        logits = shared_logits + self.role_scale * class_logits
        query_weights = weights[:, :, 0].mean(1) if weights is not None else None
        return logits, query_weights


class LinearStructuralProbe(nn.Module):
    """Low-capacity control for information added by the cached graph descriptors."""

    def __init__(self, num_classes: int, structural_dim: int):
        super().__init__()
        if structural_dim <= 0:
            raise ValueError("The structural probe requires graph descriptors")
        self.head = nn.Linear(2048 + structural_dim, num_classes)

    def forward(self, embedding: torch.Tensor, return_attention: bool = False,
                structural: torch.Tensor | None = None):
        if embedding.ndim != 2 or embedding.shape[1] != 2048 or structural is None:
            raise ValueError("Expected frozen 2048-D STEP embeddings and graph descriptors")
        if structural.ndim != 2 or structural.shape[0] != len(embedding):
            raise ValueError("Graph descriptor batch differs from STEP embeddings")
        combined = torch.cat((embedding.float() * math.sqrt(2048), structural.float()), dim=-1)
        return self.head(combined), None


class AttentionStructuralClassifier(nn.Module):
    """Existing attention model plus a small B-Rep-descriptor residual head."""

    def __init__(self, num_classes: int, structural_dim: int, width: int = 256,
                 heads: int = 8, depth: int = 2, feedforward: int = 768,
                 dropout: float = 0.2):
        super().__init__()
        if structural_dim <= 0:
            raise ValueError("Structural attention requires graph descriptors")
        self.step = StepTokenAttentionClassifier(num_classes, width, heads, depth,
                                                 feedforward, dropout)
        self.structural_head = nn.Sequential(
            nn.LayerNorm(structural_dim), nn.Linear(structural_dim, 128),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(128, num_classes),
        )
        self.structural_scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, embedding: torch.Tensor, return_attention: bool = False,
                structural: torch.Tensor | None = None):
        if structural is None or structural.ndim != 2 or structural.shape[0] != len(embedding):
            raise ValueError("Expected one graph descriptor row per STEP embedding")
        step_logits, query_weights = self.step(embedding, return_attention=return_attention)
        logits = step_logits + self.structural_scale * self.structural_head(structural.float())
        return logits, query_weights


def make_classifier(name: str, classes: list[str], settings: dict,
                    structural_dim: int = 0) -> nn.Module:
    if name == "linear":
        return LinearProbe(len(classes))
    if name == "attention":
        return StepTokenAttentionClassifier(
            len(classes), settings["token_width"], settings["attention_heads"],
            settings["attention_layers"], settings["feedforward_width"],
            settings["dropout"],
        )
    if name == "attention_class_role":
        return ClassRoleAttentionClassifier(
            len(classes), settings["token_width"], settings["attention_heads"],
            settings["attention_layers"], settings["feedforward_width"],
            settings["dropout"],
        )
    if name == "linear_structural":
        return LinearStructuralProbe(len(classes), structural_dim)
    if name == "attention_structural":
        return AttentionStructuralClassifier(
            len(classes), structural_dim, settings["token_width"],
            settings["attention_heads"], settings["attention_layers"],
            settings["feedforward_width"], settings["dropout"],
        )
    raise ValueError(f"Unknown classifier architecture: {name}")
