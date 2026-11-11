from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))
sys.path.insert(0, str(ROOT))

from baselines import descriptor_dim, graph_descriptors  # noqa: E402

from src.histcad_large_jepa.parametric_graph import PARAMETRIC_NUMERIC_DIM  # noqa: E402
from src.histcad_large_jepa.step_graph import STEP_NUMERIC_DIM  # noqa: E402


def test_graph_descriptors_have_fixed_width_and_finite_values() -> None:
    parametric = {
        "node_type": torch.tensor([1, 2, 1, 3]),
        "subtype": torch.tensor([2, 3, 2, 4]),
        "aux_type": torch.tensor([0, 1, 0, 2]),
        "numeric": torch.arange(4 * PARAMETRIC_NUMERIC_DIM).reshape(4, -1).float(),
        "numeric_mask": torch.ones((4, PARAMETRIC_NUMERIC_DIM), dtype=torch.bool),
        "edge_index": torch.tensor([[0, 1, 2, 3], [0, 0, 2, 2]]),
        "edge_type": torch.tensor([0, 1, 0, 1]),
        "graph_index": torch.tensor([0, 0, 1, 1]),
        "global_index": torch.tensor([0, 2]),
    }
    step = {
        "node_type": torch.tensor([1, 17, 2, 18]),
        "node_role": torch.tensor([0, 1, 0, 2]),
        "numeric": torch.arange(4 * STEP_NUMERIC_DIM).reshape(4, -1).float(),
        "edge_index": torch.tensor([[0, 1, 2, 3], [0, 0, 2, 2]]),
        "edge_type": torch.tensor([0, 1, 0, 1]),
        "graph_index": torch.tensor([0, 0, 1, 1]),
        "global_index": torch.tensor([0, 2]),
    }
    result = graph_descriptors(parametric, step, step_hash_bins=16)
    assert result.shape == (2, descriptor_dim(16))
    assert torch.isfinite(result).all()
