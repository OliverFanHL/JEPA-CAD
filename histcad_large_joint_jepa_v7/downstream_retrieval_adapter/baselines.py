from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from src.histcad_large_jepa.model import ParametricEncoder, StepEncoder, projection
from src.histcad_large_jepa.parametric_graph import PARAMETRIC_NUMERIC_DIM
from src.histcad_large_jepa.step_graph import STEP_NUMERIC_DIM
from src.histcad_large_jepa.vocab import (
    AUX_TYPES,
    NODE_TYPES,
    PARAMETRIC_RELATIONS,
    STEP_RELATIONS,
    SUBTYPES,
)


def _histogram(
    values: torch.Tensor,
    graph_index: torch.Tensor,
    graph_count: int,
    bins: int,
    hashed: bool = False,
) -> torch.Tensor:
    values = values.remainder(bins) if hashed else values
    output = torch.zeros((graph_count, bins), device=values.device)
    output.index_put_((graph_index, values), torch.ones_like(values, dtype=output.dtype), accumulate=True)
    counts = torch.bincount(graph_index, minlength=graph_count).clamp_min(1).to(output.dtype)
    return output / counts[:, None]


def _numeric_moments(
    values: torch.Tensor,
    graph_index: torch.Tensor,
    graph_count: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    values = values.float()
    weights = torch.ones_like(values) if mask is None else mask.float()
    totals = torch.zeros((graph_count, values.shape[1]), device=values.device)
    squares = torch.zeros_like(totals)
    counts = torch.zeros_like(totals)
    expanded = graph_index[:, None].expand_as(values)
    totals.scatter_add_(0, expanded, values * weights)
    squares.scatter_add_(0, expanded, values.square() * weights)
    counts.scatter_add_(0, expanded, weights)
    mean = totals / counts.clamp_min(1.0)
    variance = squares / counts.clamp_min(1.0) - mean.square()
    return torch.cat((mean, variance.clamp_min(0.0).sqrt()), dim=1)


def _log_counts(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    graph_count = len(batch["global_index"])
    node_counts = torch.bincount(batch["graph_index"], minlength=graph_count).float()
    edge_graph = batch["graph_index"][batch["edge_index"][0]]
    edge_counts = torch.bincount(edge_graph, minlength=graph_count).float()
    return torch.stack((torch.log1p(node_counts), torch.log1p(edge_counts)), dim=1)


def graph_descriptors(
    parametric: dict[str, torch.Tensor],
    step: dict[str, torch.Tensor],
    step_hash_bins: int,
) -> torch.Tensor:
    """Deterministic, label-free summaries used as AE reconstruction targets and a baseline."""
    p_count, s_count = len(parametric["global_index"]), len(step["global_index"])
    if p_count != s_count:
        raise ValueError("Parametric and STEP batches must contain the same number of graphs")
    p_edge_graph = parametric["graph_index"][parametric["edge_index"][0]]
    s_edge_graph = step["graph_index"][step["edge_index"][0]]
    pieces = (
        _histogram(parametric["node_type"], parametric["graph_index"], p_count, len(NODE_TYPES)),
        _histogram(parametric["subtype"], parametric["graph_index"], p_count, len(SUBTYPES)),
        _histogram(parametric["aux_type"], parametric["graph_index"], p_count, len(AUX_TYPES)),
        _histogram(parametric["edge_type"], p_edge_graph, p_count, len(PARAMETRIC_RELATIONS)),
        _numeric_moments(
            parametric["numeric"],
            parametric["graph_index"],
            p_count,
            parametric["numeric_mask"],
        ),
        _log_counts(parametric),
        _histogram(step["node_type"], step["graph_index"], s_count, step_hash_bins, True),
        _histogram(step["node_role"], step["graph_index"], s_count, 3),
        _histogram(step["edge_type"], s_edge_graph, s_count, len(STEP_RELATIONS)),
        _numeric_moments(step["numeric"], step["graph_index"], s_count),
        _log_counts(step),
    )
    result = torch.cat(pieces, dim=1)
    expected = descriptor_dim(step_hash_bins)
    if result.shape[1] != expected:
        raise RuntimeError(f"Descriptor width mismatch: expected {expected}, got {result.shape[1]}")
    return result


def descriptor_dim(step_hash_bins: int) -> int:
    return (
        len(NODE_TYPES)
        + len(SUBTYPES)
        + len(AUX_TYPES)
        + len(PARAMETRIC_RELATIONS)
        + 2 * PARAMETRIC_NUMERIC_DIM
        + 2
        + step_hash_bins
        + 3
        + len(STEP_RELATIONS)
        + 2 * STEP_NUMERIC_DIM
        + 2
    )


class GraphAutoencoder(nn.Module):
    """Same graph encoders and joint width as v7, trained by reconstruction instead of JEPA."""

    def __init__(self, step_entity_types: int, model_config: dict[str, Any], target_dim: int):
        super().__init__()
        p_dim = model_config["parametric_output_dim"]
        s_dim = model_config["step_output_dim"]
        embedding_dim = model_config["embedding_dim"]
        self.parametric_encoder = ParametricEncoder(
            model_config["parametric_node_dim"],
            p_dim,
            model_config["parametric_depth"],
            model_config["attention_heads"],
            model_config["parametric_pooling_slots"],
            model_config["dropout"],
            model_config["max_features"],
        )
        self.step_encoder = StepEncoder(
            step_entity_types,
            model_config["step_node_dim"],
            s_dim,
            model_config["step_depth"],
            model_config["attention_heads"],
            model_config["step_pooling_slots"],
            model_config["dropout"],
        )
        self.joint_projection = projection(
            p_dim + s_dim,
            model_config["joint_hidden_dim"],
            embedding_dim,
            model_config["dropout"],
        )
        decoder_hidden = model_config.get("autoencoder_decoder_hidden_dim", 1024)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, decoder_hidden),
            nn.GELU(),
            nn.Dropout(model_config["dropout"]),
            nn.Linear(decoder_hidden, target_dim),
        )

    def forward(
        self, parametric: dict[str, torch.Tensor], step: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        parametric_values = self.parametric_encoder(parametric)
        step_values = self.step_encoder(step)
        fused = torch.cat(
            (F.normalize(parametric_values, dim=-1), F.normalize(step_values, dim=-1)), dim=-1
        )
        raw = self.joint_projection(fused)
        return F.normalize(raw, dim=-1), self.decoder(raw)
