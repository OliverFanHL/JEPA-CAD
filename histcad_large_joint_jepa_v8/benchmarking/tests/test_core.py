from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cad_benchmark.io import Representation  # noqa: E402
from cad_benchmark.metrics import (  # noqa: E402
    class_separation,
    controlled_retrieval_metrics,
    dimension_reference_neighbors,
    latent_diagnostics,
    retrieval_metrics,
    semantic_study_metrics,
)
from cad_benchmark.projection import FixedProjector  # noqa: E402

from benchmarking.baselines.contrastive.train import graph_to_device  # noqa: E402


def test_fixed_projector_is_deterministic_and_capacity_matched() -> None:
    values = torch.randn(12, 19)
    left = FixedProjector(19, 8, 17)(values)
    right = FixedProjector(19, 8, 17)(values)
    assert left.shape == (12, 8)
    assert torch.allclose(left, right)
    assert torch.allclose(left.norm(dim=1), torch.ones(12), atol=1e-5)


def test_representation_rejects_misalignment() -> None:
    try:
        Representation("bad", "test", torch.zeros(3, 2), ["a", "b"], ["train"] * 3)
    except ValueError as error:
        assert "count" in str(error)
    else:
        raise AssertionError("Misaligned representation should fail")


def test_diagnostics_and_retrieval_are_finite() -> None:
    generator = torch.Generator().manual_seed(3)
    values = torch.randn((24, 6), generator=generator)
    labels = {"class": [str(index // 4) for index in range(24)]}
    targets = torch.randn((24, 3), generator=generator)
    diagnostics = latent_diagnostics(values, 24, 3)
    retrieval = retrieval_metrics(values, labels, targets, 8)
    assert diagnostics["effective_rank_entropy"] > 1
    assert retrieval["samples"] == 24
    assert 0 <= retrieval["labels"]["class"]["hit_at_1"] <= 1


def test_class_separation_handles_noncontiguous_test_classes() -> None:
    values = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    result = class_separation(values, [1, 1, 4, 4])
    assert result["mean_centroid_margin"] > 0


def test_existing_downstream_metric_contracts() -> None:
    generator = torch.Generator().manual_seed(11)
    values = torch.randn((24, 8), generator=generator)
    targets = torch.randn((24, 3), generator=generator)
    labels = {
        "operation_signature": [f"op-{i // 4}" for i in range(24)],
        "feature_sequence": [f"seq-{i // 4}" for i in range(24)],
        "final_operation": [f"final-{i // 8}" for i in range(24)],
        "feature_count": [str(i // 4) for i in range(24)],
        "feature_count_bucket": [str(i // 4) for i in range(24)],
        "primitive_profile": [f"primitive-{i // 4}" for i in range(24)],
        "dominant_constraint": [f"constraint-{i // 8}" for i in range(24)],
        "functional_type": [f"functional part {i // 4}" for i in range(24)],
        "canonical_functional_type": [f"functional {i // 4} part" for i in range(24)],
    }
    references = dimension_reference_neighbors(targets, 8)
    retrieval = controlled_retrieval_metrics(values, targets, labels, references, 8)
    semantic = semantic_study_metrics(values, targets, labels, references, 8)
    assert "dimension_neighbor_retrieval" in retrieval
    assert "canonical_functional_type_evaluation_only" in retrieval
    assert "function_at_1_eligible" in semantic
    assert "geometry_structure_at_1" in semantic


def test_contrastive_graph_transfer_preserves_auxiliary_metadata() -> None:
    metadata = ["sample-a", "sample-b"]
    moved = graph_to_device(
        {"node_values": torch.tensor([1.0, 2.0]), "metadata": metadata},
        torch.device("cpu"),
    )
    assert torch.equal(moved["node_values"], torch.tensor([1.0, 2.0]))
    assert moved["metadata"] is metadata
