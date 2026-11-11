from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F


def _aligned_embedding(payload: dict[str, Any], name: str, uids: list[str]) -> torch.Tensor:
    lookup = {uid: index for index, uid in enumerate(payload["uids"])}
    missing = [uid for uid in uids if uid not in lookup]
    if missing:
        raise ValueError(f"Representation {name!r} is missing {len(missing)} test UIDs")
    values = payload["embeddings"][name][[lookup[uid] for uid in uids]].float()
    return F.normalize(values, dim=-1)


def _neighbors(values: torch.Tensor, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    indices, scores = [], []
    for start in range(0, len(values), chunk_size):
        stop = min(start + chunk_size, len(values))
        similarity = values[start:stop] @ values.T
        rows = torch.arange(stop - start)
        similarity[rows, torch.arange(start, stop)] = -torch.inf
        chunk_scores, chunk_indices = similarity.topk(10, dim=1)
        indices.append(chunk_indices)
        scores.append(chunk_scores)
    return torch.cat(indices), torch.cat(scores)


def _categorical(values: list[str], retrieved: torch.Tensor) -> float:
    matches = [
        bool(values[index]) and values[index] == values[int(retrieved[index, 0])]
        for index in range(len(values))
    ]
    return sum(matches) / max(1, len(matches))


def _rank_statistics(values: torch.Tensor) -> tuple[float, int]:
    singular = torch.linalg.svdvals(values - values.mean(0))
    variance = singular.square()
    effective_rank = float(variance.sum().square() / variance.square().sum().clamp_min(1e-12))
    cumulative = variance.cumsum(0) / variance.sum().clamp_min(1e-12)
    dimensions_95 = int(torch.searchsorted(cumulative, torch.tensor(0.95)).item() + 1)
    return effective_rank, dimensions_95


def representation_metrics(
    values: torch.Tensor,
    dimension_targets: torch.Tensor,
    operation_signatures: list[str],
    functional_labels: list[str],
    chunk_size: int,
    tie_tolerance: float,
    near_tie_tolerance: float,
) -> tuple[dict[str, float | int], torch.Tensor, torch.Tensor]:
    retrieved, scores = _neighbors(values, chunk_size)
    target = dimension_targets[:, None, :]
    neighbor_dimensions = dimension_targets[retrieved[:, :5]]
    difference = (target - neighbor_dimensions).abs().mean(-1)
    unique, counts = torch.unique(values, dim=0, return_counts=True)
    del unique
    effective_rank, dimensions_95 = _rank_statistics(values)
    gaps = scores[:, :-1] - scores[:, 1:]
    metrics: dict[str, float | int] = {
        "dimension_mae_at_1": float(difference[:, 0].mean()),
        "dimension_mae_at_5": float(difference.mean()),
        "operation_hit_at_1": _categorical(operation_signatures, retrieved),
        "functional_hit_at_1_evaluation_only": _categorical(functional_labels, retrieved),
        "exact_duplicate_samples": int(counts[counts > 1].sum()),
        "top10_tie_queries": int((gaps.abs() <= tie_tolerance).any(1).sum()),
        "top10_near_tie_queries": int((gaps.abs() <= near_tie_tolerance).any(1).sum()),
        "median_top1_top2_margin": float((scores[:, 0] - scores[:, 1]).median()),
        "effective_rank": effective_rank,
        "dimensions_95pct_variance": dimensions_95,
    }
    return metrics, retrieved, scores


def evaluate(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["paths"]["output_dir"])
    payload = torch.load(
        output_dir / "representations.pt", map_location="cpu", weights_only=False, mmap=True
    )
    test_indices = [
        index for index, item in enumerate(payload["metadata"]) if item["split"] == "test"
    ]
    uids = [payload["uids"][index] for index in test_indices]
    dimensions = payload["dimension_targets"][test_indices].float()
    dimensions = (dimensions - payload["dimension_target_mean"]) / payload["dimension_target_std"]
    operations = [
        payload["structural_metadata"][index]["operation_signature"] for index in test_indices
    ]
    labels = [payload["metadata"][index].get("functional_type", "") for index in test_indices]
    representations: dict[str, torch.Tensor] = {
        f"v7_{name}": F.normalize(payload["embeddings"][name][test_indices].float(), dim=-1)
        for name in ("parametric", "step", "joint")
    }
    for baseline_name, baseline_path in config["evaluation"].get("optional_baselines", {}).items():
        baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
        embedding_name = (
            "joint"
            if "joint" in baseline.get("embeddings", {})
            else next(iter(baseline["embeddings"]))
        )
        representations[baseline_name] = _aligned_embedding(baseline, embedding_name, uids)
    settings = config["evaluation"]
    result: dict[str, Any] = {
        "protocol": {
            "test_samples": len(test_indices),
            "functional_labels": "evaluation_only",
            "architecture_selection": "validation_only",
        },
        "representations": {},
    }
    joint_neighbors = joint_scores = None
    for name, values in representations.items():
        metrics, neighbors, scores = representation_metrics(
            values,
            dimensions,
            operations,
            labels,
            settings["retrieval_chunk_size"],
            settings["tie_tolerance"],
            settings["near_tie_tolerance"],
        )
        result["representations"][name] = {"dimensions": values.shape[1], **metrics}
        if name == "v7_joint":
            joint_neighbors, joint_scores = neighbors, scores
    (output_dir / "retrieval_benchmark.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    rows = [
        {"representation": name, **metrics} for name, metrics in result["representations"].items()
    ]
    with (output_dir / "retrieval_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    if joint_neighbors is not None and joint_scores is not None:
        with (output_dir / "retrieval_examples.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "query_uid",
                    "neighbor_uid",
                    "cosine_similarity",
                    "same_operation",
                    "same_functional_label",
                ),
            )
            writer.writeheader()
            for query in range(min(settings["examples"], len(uids))):
                candidate = int(joint_neighbors[query, 0])
                writer.writerow(
                    {
                        "query_uid": uids[query],
                        "neighbor_uid": uids[candidate],
                        "cosine_similarity": float(joint_scores[query, 0]),
                        "same_operation": operations[query] == operations[candidate],
                        "same_functional_label": bool(labels[query])
                        and labels[query] == labels[candidate],
                    }
                )
    return result
