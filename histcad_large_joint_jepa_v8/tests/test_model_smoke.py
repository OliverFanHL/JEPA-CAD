from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from histcad_large_jepa.config import load_config  # noqa: E402
from histcad_large_jepa.model import WideJointJEPA  # noqa: E402
from histcad_large_jepa.vocab import NODE_TO_ID  # noqa: E402


def self_edges(nodes: int) -> torch.Tensor:
    indices = torch.arange(nodes)
    return torch.stack((indices, indices))


def test_production_dimensions_are_source_first() -> None:
    config = load_config(ROOT / "configs" / "full_dataset.yaml")
    model = config["model"]
    assert model["parametric_output_dim"] == 5 * model["parametric_node_dim"] == 2560
    assert model["step_output_dim"] == 4 * model["step_node_dim"] == 2048
    assert model["parametric_output_dim"] + model["step_output_dim"] == 4608
    assert model["embedding_dim"] == 4096
    assert config["training"]["epochs"] == 20


def test_role_aware_model_forward_shapes() -> None:
    config = {
        "parametric_node_dim": 16,
        "parametric_pooling_slots": 4,
        "parametric_output_dim": 80,
        "parametric_depth": 1,
        "step_node_dim": 16,
        "step_pooling_slots": 3,
        "step_output_dim": 64,
        "step_depth": 1,
        "attention_heads": 4,
        "joint_hidden_dim": 32,
        "embedding_dim": 32,
        "dropout": 0.0,
        "max_features": 8,
        "correlation_permutations": 2,
    }
    model = WideJointJEPA(num_step_entity_types=8, config=config, dimension_targets=22)
    parametric_nodes = torch.tensor(
        [
            NODE_TO_ID["GLOBAL"],
            NODE_TO_ID["FEATURE"],
            NODE_TO_ID["PRIMITIVE"],
            NODE_TO_ID["CONSTRAINT"],
        ]
    )
    parametric = {
        "node_type": parametric_nodes,
        "subtype": torch.zeros(4, dtype=torch.long),
        "aux_type": torch.zeros(4, dtype=torch.long),
        "numeric": torch.zeros(4, 24),
        "numeric_mask": torch.zeros(4, 24, dtype=torch.bool),
        "feature_index": torch.tensor([-1, 0, 0, 0]),
        "edge_index": self_edges(4),
        "edge_type": torch.zeros(4, dtype=torch.long),
        "graph_index": torch.zeros(4, dtype=torch.long),
        "global_index": torch.tensor([0]),
    }
    step = {
        "node_type": torch.tensor([0, 1, 2, 3]),
        "node_role": torch.tensor([0, 1, 2, 2]),
        "numeric": torch.zeros(4, 12),
        "edge_index": self_edges(4),
        "edge_type": torch.zeros(4, dtype=torch.long),
        "edge_weight": torch.ones(4),
        "graph_index": torch.zeros(4, dtype=torch.long),
        "global_index": torch.tensor([0]),
    }
    outputs = model(
        parametric,
        step,
        torch.zeros(4, dtype=torch.bool),
        torch.zeros(4, dtype=torch.bool),
    )
    assert outputs["parametric_raw"].shape == (1, 80)
    assert outputs["step_raw"].shape == (1, 64)
    assert outputs["joint"].shape == (1, 32)
    assert outputs["slot_diversity"].ndim == 0
