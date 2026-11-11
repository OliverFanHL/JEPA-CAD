from __future__ import annotations

import copy
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .parametric_graph import PARAMETRIC_NUMERIC_DIM
from .step_graph import STEP_NUMERIC_DIM
from .vocab import AUX_TYPES, NODE_TO_ID, NODE_TYPES, PARAMETRIC_RELATIONS, STEP_RELATIONS, SUBTYPES


class RelationalAttention(nn.Module):
    """Sparse multi-head attention with typed edge biases and values."""

    def __init__(self, dim: int, heads: int, relation_count: int, dropout: float, weighted: bool):
        super().__init__()
        if dim % heads:
            raise ValueError(f"Node dimension {dim} must be divisible by {heads} attention heads")
        self.heads, self.head_dim, self.scale = heads, dim // heads, (dim // heads) ** -0.5
        self.query, self.key, self.value = (
            nn.Linear(dim, dim),
            nn.Linear(dim, dim),
            nn.Linear(dim, dim),
        )
        self.relation_bias = nn.Embedding(relation_count, heads)
        self.relation_value = nn.Embedding(relation_count, dim)
        self.edge_weight_scale = nn.Parameter(torch.ones(heads)) if weighted else None
        self.output, self.dropout = nn.Linear(dim, dim), nn.Dropout(dropout)

    def forward(
        self,
        values: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_keep: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        edge_index, edge_type = edge_index[:, edge_keep], edge_type[edge_keep]
        if edge_type.numel() == 0:
            return torch.zeros_like(values)
        if edge_weight is not None:
            edge_weight = edge_weight[edge_keep]
        nodes, dim = values.shape
        receiver, sender = edge_index
        query = self.query(values).reshape(nodes, self.heads, self.head_dim)
        key = self.key(values).reshape(nodes, self.heads, self.head_dim)
        value = self.value(values).reshape(nodes, self.heads, self.head_dim)
        relation = self.relation_value(edge_type).reshape(-1, self.heads, self.head_dim)
        scores = (query[receiver] * key[sender]).sum(-1) * self.scale + self.relation_bias(
            edge_type
        )
        if edge_weight is not None and self.edge_weight_scale is not None:
            scores = scores + torch.log1p(edge_weight).unsqueeze(-1) * self.edge_weight_scale
        scatter = receiver[:, None].expand(-1, self.heads)
        maximum = torch.full(
            (nodes, self.heads),
            torch.finfo(scores.dtype).min,
            dtype=scores.dtype,
            device=scores.device,
        )
        maximum.scatter_reduce_(0, scatter, scores, reduce="amax", include_self=True)
        unnormalized = torch.exp(scores - maximum[receiver])
        denominator = torch.zeros_like(maximum)
        denominator.scatter_add_(0, scatter, unnormalized)
        attention = self.dropout(unnormalized / denominator[receiver].clamp_min(1e-8))
        messages = attention.unsqueeze(-1) * (value[sender] + relation)
        aggregated = torch.zeros(
            (nodes, self.heads, self.head_dim), dtype=values.dtype, device=values.device
        )
        aggregated.scatter_add_(0, receiver[:, None, None].expand_as(messages), messages)
        return self.output(aggregated.reshape(nodes, dim))


class RelationalBlock(nn.Module):
    def __init__(self, dim: int, heads: int, relation_count: int, dropout: float, weighted: bool):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = RelationalAttention(dim, heads, relation_count, dropout, weighted)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, values: torch.Tensor, batch: dict[str, torch.Tensor], active: torch.Tensor
    ) -> torch.Tensor:
        receiver, sender = batch["edge_index"]
        keep = active[receiver] & active[sender]
        values = values + self.attention(
            self.norm1(values),
            batch["edge_index"],
            batch["edge_type"],
            keep,
            batch.get("edge_weight"),
        )
        values = values + self.mlp(self.norm2(values))
        return values.masked_fill(~active.unsqueeze(-1), 0.0)


class RoleAwareSummaryPool(nn.Module):
    """Pool semantically distinct node groups without padding or duplicated slots."""

    def __init__(
        self,
        node_dim: int,
        output_dim: int,
        slot_names: tuple[str, ...],
        dropout: float,
    ):
        super().__init__()
        self.node_dim, self.slot_names = node_dim, slot_names
        self.slots = len(slot_names)
        expected_output = (self.slots + 1) * node_dim
        if output_dim != expected_output:
            raise ValueError(
                f"Branch output must preserve all pooled and global features: "
                f"expected {(self.slots + 1)} * {node_dim} = {expected_output}, got {output_dim}"
            )
        self.queries = nn.Parameter(torch.empty(self.slots, node_dim))
        self.absent_tokens = nn.Parameter(torch.empty(self.slots, node_dim))
        self.key, self.value = nn.Linear(node_dim, node_dim), nn.Linear(node_dim, node_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)
        nn.init.normal_(self.queries, std=0.02)
        nn.init.normal_(self.absent_tokens, std=0.02)

    def forward(
        self,
        nodes: torch.Tensor,
        graph_index: torch.Tensor,
        global_index: torch.Tensor,
        active: torch.Tensor,
        role_mask: torch.Tensor,
    ) -> torch.Tensor:
        graph_count = len(global_index)
        global_values = nodes[global_index]
        keys, values = self.key(nodes), self.value(nodes)
        scores = keys @ self.queries.T / math.sqrt(self.node_dim)
        eligible = active.unsqueeze(-1) & role_mask
        minimum = torch.finfo(scores.dtype).min
        masked_scores = scores.masked_fill(~eligible, minimum)
        scatter = graph_index[:, None].expand(-1, self.slots)
        maximum = torch.full(
            (graph_count, self.slots),
            minimum,
            dtype=scores.dtype,
            device=scores.device,
        )
        maximum.scatter_reduce_(0, scatter, masked_scores, reduce="amax", include_self=True)
        centered = torch.where(eligible, scores - maximum[graph_index], torch.zeros_like(scores))
        # exp/scatter may be promoted to float32 by autocast. Cast the normalized
        # attention path back to the value dtype so scatter_add has matching dtypes.
        weights = (torch.exp(centered.float()) * eligible).to(values.dtype)
        denominator = torch.zeros(
            (graph_count, self.slots), dtype=weights.dtype, device=weights.device
        )
        denominator.scatter_add_(0, scatter, weights)
        weights = weights / denominator[graph_index].clamp_min(1e-8)
        messages = weights.unsqueeze(-1) * values.unsqueeze(1)
        pooled = torch.zeros(
            (graph_count, self.slots, self.node_dim),
            dtype=messages.dtype,
            device=messages.device,
        )
        pooled.scatter_add_(0, graph_index[:, None, None].expand_as(messages), messages)
        present = denominator > 0
        pooled = torch.where(
            present.unsqueeze(-1), pooled, self.absent_tokens.unsqueeze(0).expand_as(pooled)
        )
        return self.norm(torch.cat((self.dropout(pooled.flatten(1)), global_values), dim=-1))

    def query_diversity(self) -> torch.Tensor:
        if self.slots < 2:
            return self.queries.sum() * 0.0
        normalized = F.normalize(self.queries, dim=-1)
        similarities = normalized @ normalized.T
        off_diagonal = ~torch.eye(self.slots, dtype=torch.bool, device=similarities.device)
        return similarities[off_diagonal].square().mean()


class ParametricEncoder(nn.Module):
    def __init__(
        self,
        node_dim: int,
        output_dim: int,
        depth: int,
        heads: int,
        pooling_slots: int,
        dropout: float,
        max_features: int,
    ):
        super().__init__()
        self.node_type = nn.Embedding(len(NODE_TYPES), node_dim, padding_idx=0)
        self.subtype = nn.Embedding(len(SUBTYPES), node_dim, padding_idx=0)
        self.aux_type = nn.Embedding(len(AUX_TYPES), node_dim)
        self.feature_position = nn.Embedding(max_features + 1, node_dim)
        self.numeric = nn.Sequential(
            nn.Linear(PARAMETRIC_NUMERIC_DIM * 2, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, node_dim),
        )
        self.blocks = nn.ModuleList(
            [
                RelationalBlock(node_dim, heads, len(PARAMETRIC_RELATIONS), dropout, False)
                for _ in range(depth)
            ]
        )
        self.node_norm = nn.LayerNorm(node_dim)
        slot_names = ("feature_operations", "sketch_geometry", "constraints", "all_relations")
        if pooling_slots != len(slot_names):
            raise ValueError(f"Parametric encoder requires {len(slot_names)} role-aware slots")
        self.pool = RoleAwareSummaryPool(node_dim, output_dim, slot_names, dropout)
        self.output_dim = output_dim

    def forward(
        self, batch: dict[str, torch.Tensor], hidden_nodes: torch.Tensor | None = None
    ) -> torch.Tensor:
        active = torch.ones_like(batch["node_type"], dtype=torch.bool)
        if hidden_nodes is not None:
            active &= ~hidden_nodes
        positions = (batch["feature_index"] + 1).clamp(0, self.feature_position.num_embeddings - 1)
        numeric = torch.cat((batch["numeric"], batch["numeric_mask"].float()), dim=-1)
        values = (
            self.node_type(batch["node_type"])
            + self.subtype(batch["subtype"])
            + self.aux_type(batch["aux_type"])
            + self.feature_position(positions)
            + self.numeric(numeric)
        ).masked_fill(~active.unsqueeze(-1), 0.0)
        for block in self.blocks:
            values = block(values, batch, active)
        values = self.node_norm(values).masked_fill(~active.unsqueeze(-1), 0.0)
        node_type = batch["node_type"]
        geometry = (
            (node_type == NODE_TO_ID["PRIMITIVE"])
            | (node_type == NODE_TO_ID["BOUNDARY_REF"])
            | (node_type == NODE_TO_ID["CONTROL_POINT"])
            | (node_type == NODE_TO_ID["KNOT"])
        )
        role_mask = torch.stack(
            (
                node_type == NODE_TO_ID["FEATURE"],
                geometry,
                node_type == NODE_TO_ID["CONSTRAINT"],
                node_type != NODE_TO_ID["GLOBAL"],
            ),
            dim=-1,
        )
        return self.pool(values, batch["graph_index"], batch["global_index"], active, role_mask)


class StepEncoder(nn.Module):
    def __init__(
        self,
        entity_types: int,
        node_dim: int,
        output_dim: int,
        depth: int,
        heads: int,
        pooling_slots: int,
        dropout: float,
    ):
        super().__init__()
        self.entity_type = nn.Embedding(entity_types, node_dim, padding_idx=0)
        self.node_role = nn.Embedding(3, node_dim)
        self.numeric = nn.Sequential(
            nn.Linear(STEP_NUMERIC_DIM, node_dim), nn.GELU(), nn.Linear(node_dim, node_dim)
        )
        self.blocks = nn.ModuleList(
            [
                RelationalBlock(node_dim, heads, len(STEP_RELATIONS), dropout, True)
                for _ in range(depth)
            ]
        )
        self.node_norm = nn.LayerNorm(node_dim)
        slot_names = ("typed_entities", "brep_instances", "all_connectivity")
        if pooling_slots != len(slot_names):
            raise ValueError(f"STEP encoder requires {len(slot_names)} role-aware slots")
        self.pool = RoleAwareSummaryPool(node_dim, output_dim, slot_names, dropout)
        self.output_dim = output_dim

    def forward(
        self, batch: dict[str, torch.Tensor], hidden_nodes: torch.Tensor | None = None
    ) -> torch.Tensor:
        active = torch.ones_like(batch["node_type"], dtype=torch.bool)
        if hidden_nodes is not None:
            active &= ~hidden_nodes
        values = (
            self.entity_type(batch["node_type"])
            + self.node_role(batch["node_role"])
            + self.numeric(batch["numeric"])
        ).masked_fill(~active.unsqueeze(-1), 0.0)
        for block in self.blocks:
            values = block(values, batch, active)
        values = self.node_norm(values).masked_fill(~active.unsqueeze(-1), 0.0)
        role = batch["node_role"]
        role_mask = torch.stack((role == 1, role == 2, role != 0), dim=-1)
        return self.pool(values, batch["graph_index"], batch["global_index"], active, role_mask)


def projection(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, output_dim),
    )


class WideJointJEPA(nn.Module):
    def __init__(
        self, num_step_entity_types: int, config: dict[str, Any], dimension_targets: int = 22
    ):
        super().__init__()
        param_dim, step_dim = config["parametric_output_dim"], config["step_output_dim"]
        embedding_dim, hidden = config["embedding_dim"], config["joint_hidden_dim"]
        self.parametric_encoder = ParametricEncoder(
            config["parametric_node_dim"],
            param_dim,
            config["parametric_depth"],
            config["attention_heads"],
            config["parametric_pooling_slots"],
            config["dropout"],
            config["max_features"],
        )
        self.step_encoder = StepEncoder(
            num_step_entity_types,
            config["step_node_dim"],
            step_dim,
            config["step_depth"],
            config["attention_heads"],
            config["step_pooling_slots"],
            config["dropout"],
        )
        self.joint_projection = projection(
            param_dim + step_dim, hidden, embedding_dim, config["dropout"]
        )
        self.joint_predictor = projection(embedding_dim, hidden, embedding_dim, config["dropout"])
        self.parametric_predictor = projection(
            param_dim, param_dim * 2, param_dim, config["dropout"]
        )
        self.step_predictor = projection(step_dim, step_dim * 2, step_dim, config["dropout"])
        self.parametric_alignment, self.step_alignment = (
            nn.Linear(param_dim, 256),
            nn.Linear(step_dim, 256),
        )
        self.dimension_head = nn.Linear(embedding_dim, dimension_targets)
        count = config.get("correlation_permutations", 16)

        def permutation_bank(dimensions: int, seed: int) -> torch.Tensor:
            generator = torch.Generator().manual_seed(seed)
            return torch.stack(
                [torch.randperm(dimensions, generator=generator) for _ in range(count)]
            )

        self.register_buffer(
            "joint_permutation_bank", permutation_bank(embedding_dim, 1701), persistent=True
        )
        self.register_buffer(
            "parametric_permutation_bank", permutation_bank(param_dim, 1702), persistent=True
        )
        self.register_buffer(
            "step_permutation_bank", permutation_bank(step_dim, 1703), persistent=True
        )
        self.target_parametric_encoder = copy.deepcopy(self.parametric_encoder)
        self.target_step_encoder = copy.deepcopy(self.step_encoder)
        self.target_joint_projection = copy.deepcopy(self.joint_projection)
        for module in self.target_modules:
            for parameter in module.parameters():
                parameter.requires_grad = False

    @property
    def target_modules(self) -> tuple[nn.Module, ...]:
        return (
            self.target_parametric_encoder,
            self.target_step_encoder,
            self.target_joint_projection,
        )

    def train(self, mode: bool = True) -> WideJointJEPA:
        super().train(mode)
        for module in self.target_modules:
            module.eval()
        return self

    @staticmethod
    def fuse(parametric: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        return torch.cat((F.normalize(parametric, dim=-1), F.normalize(step, dim=-1)), dim=-1)

    @staticmethod
    def _slot_diversity(raw: torch.Tensor, pool: RoleAwareSummaryPool) -> torch.Tensor:
        summaries = raw[:, : pool.slots * pool.node_dim].reshape(
            len(raw), pool.slots, pool.node_dim
        )
        summaries = F.normalize(summaries, dim=-1)
        similarities = summaries @ summaries.transpose(1, 2)
        off_diagonal = ~torch.eye(
            pool.slots, dtype=torch.bool, device=similarities.device
        ).unsqueeze(0)
        output_diversity = similarities[off_diagonal.expand_as(similarities)].square().mean()
        return (output_diversity + pool.query_diversity()) / 2.0

    def forward(
        self,
        parametric: dict[str, torch.Tensor],
        step: dict[str, torch.Tensor],
        parametric_mask: torch.Tensor,
        step_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        parametric_raw = self.parametric_encoder(parametric, parametric_mask)
        step_raw = self.step_encoder(step, step_mask)
        joint_raw = self.joint_projection(self.fuse(parametric_raw, step_raw))
        with torch.no_grad():
            parametric_target_raw = self.target_parametric_encoder(parametric, None)
            step_target_raw = self.target_step_encoder(step, None)
            joint_target_raw = self.target_joint_projection(
                self.fuse(parametric_target_raw, step_target_raw)
            )
        return {
            "joint": F.normalize(joint_raw, dim=-1),
            "joint_raw": joint_raw,
            "joint_prediction": F.normalize(self.joint_predictor(joint_raw), dim=-1),
            "joint_target": F.normalize(joint_target_raw, dim=-1).detach(),
            "parametric_raw": parametric_raw,
            "step_raw": step_raw,
            "parametric_prediction": F.normalize(self.parametric_predictor(parametric_raw), dim=-1),
            "step_prediction": F.normalize(self.step_predictor(step_raw), dim=-1),
            "parametric_target": F.normalize(parametric_target_raw, dim=-1).detach(),
            "step_target": F.normalize(step_target_raw, dim=-1).detach(),
            "parametric_alignment": F.normalize(self.parametric_alignment(parametric_raw), dim=-1),
            "step_alignment": F.normalize(self.step_alignment(step_raw), dim=-1),
            "dimension_prediction": self.dimension_head(joint_raw),
            "slot_diversity": (
                self._slot_diversity(parametric_raw, self.parametric_encoder.pool)
                + self._slot_diversity(step_raw, self.step_encoder.pool)
            )
            / 2.0,
        }

    @torch.no_grad()
    def embed(
        self, parametric: dict[str, torch.Tensor], step: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        parametric_raw, step_raw = self.parametric_encoder(parametric), self.step_encoder(step)
        joint_raw = self.joint_projection(self.fuse(parametric_raw, step_raw))
        return {
            "joint": F.normalize(joint_raw, dim=-1),
            "joint_raw": joint_raw,
            "parametric": F.normalize(parametric_raw, dim=-1),
            "step": F.normalize(step_raw, dim=-1),
            "dimension_prediction": self.dimension_head(joint_raw),
        }

    @torch.no_grad()
    def update_targets(self, momentum: float) -> None:
        for online, target in (
            (self.parametric_encoder, self.target_parametric_encoder),
            (self.step_encoder, self.target_step_encoder),
            (self.joint_projection, self.target_joint_projection),
        ):
            for source, destination in zip(online.parameters(), target.parameters(), strict=True):
                destination.data.mul_(momentum).add_(source.data, alpha=1.0 - momentum)

    def correlation_permutations(self, step: int) -> dict[str, torch.Tensor]:
        index = step % len(self.joint_permutation_bank)
        return {
            "joint": self.joint_permutation_bank[index],
            "parametric": self.parametric_permutation_bank[index],
            "step": self.step_permutation_bank[index],
        }


def parameter_report(model: WideJointJEPA) -> dict[str, Any]:
    return {
        "online_trainable_parameters": sum(
            item.numel() for item in model.parameters() if item.requires_grad
        ),
        "ema_frozen_parameters": sum(
            item.numel() for item in model.parameters() if not item.requires_grad
        ),
        "parametric_output_dim": model.parametric_encoder.output_dim,
        "step_output_dim": model.step_encoder.output_dim,
        "joint_input_dim": model.parametric_encoder.output_dim + model.step_encoder.output_dim,
        "embedding_dim": model.dimension_head.in_features,
        "fusion": "L2-normalized branch concatenation followed by learned nonlinear joint projection",
        "parametric_summary_slots": model.parametric_encoder.pool.slot_names,
        "step_summary_slots": model.step_encoder.pool.slot_names,
        "branch_vector_padding": False,
        "initialization": "random; no earlier experiment checkpoint is accepted or loaded",
        "per_part_gate": False,
    }
