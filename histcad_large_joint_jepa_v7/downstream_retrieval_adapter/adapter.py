from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class LightweightRetrievalAdapter(nn.Module):
    """Residual chunk-attention adapter for a frozen CAD embedding."""

    def __init__(self, config: dict[str, Any], dimension_targets: int = 22):
        super().__init__()
        input_dim, tokens = config["input_dim"], config["tokens"]
        if input_dim % tokens:
            raise ValueError("input_dim must be divisible by tokens")
        token_dim = input_dim // tokens
        if token_dim % config["attention_heads"]:
            raise ValueError("token dimension must be divisible by attention_heads")
        self.input_dim, self.tokens, self.token_dim = input_dim, tokens, token_dim
        self.input_norm = nn.LayerNorm(input_dim)
        self.position = nn.Parameter(torch.empty(1, tokens, token_dim))
        self.attention = nn.MultiheadAttention(
            token_dim,
            config["attention_heads"],
            dropout=config["dropout"],
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(token_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(token_dim * 2, token_dim),
        )
        self.norm2 = nn.LayerNorm(token_dim)
        self.pool_query = nn.Parameter(torch.empty(1, 1, token_dim))
        self.pool_attention = nn.MultiheadAttention(
            token_dim,
            config["attention_heads"],
            dropout=config["dropout"],
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, config["mlp_hidden_dim"]),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["mlp_hidden_dim"], input_dim),
        )
        self.residual_gate = nn.Parameter(torch.tensor(-2.0))
        self.dimension_head = nn.Linear(input_dim, dimension_targets)
        self.feature_count_head = nn.Linear(input_dim, 4)
        self.final_operation_head = nn.Linear(input_dim, 4)
        nn.init.normal_(self.position, std=0.02)
        nn.init.normal_(self.pool_query, std=0.02)

    def encode(self, frozen_embedding: torch.Tensor) -> torch.Tensor:
        source = F.normalize(frozen_embedding.float(), dim=-1)
        tokens = self.input_norm(source).reshape(-1, self.tokens, self.token_dim) + self.position
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm1(tokens + attended)
        tokens = self.norm2(tokens + self.feed_forward(tokens))
        query = self.pool_query.expand(len(tokens), -1, -1)
        pooled, _ = self.pool_attention(query, tokens, tokens, need_weights=False)
        delta = self.mlp(pooled[:, 0])
        return F.normalize(source + torch.sigmoid(self.residual_gate) * delta, dim=-1)

    def forward(self, frozen_embedding: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding = self.encode(frozen_embedding)
        return {
            "embedding": embedding,
            "dimension": self.dimension_head(embedding),
            "feature_count": self.feature_count_head(embedding),
            "final_operation": self.final_operation_head(embedding),
        }
