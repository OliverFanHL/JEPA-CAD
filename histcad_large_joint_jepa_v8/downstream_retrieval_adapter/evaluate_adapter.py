from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.histcad_large_jepa.benchmark import (  # noqa: E402, I001
    _dimension_reference_neighbors,
    retrieval_metrics,
)


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (ROOT / candidate).resolve()


def canonical_label(value: str) -> str:
    value = re.sub(r"\([^)]*\)", "", value.lower())
    value = re.sub(r"\b(?:with|and|for|of|the|a|an)\b", " ", value)
    return " ".join(sorted(set(re.findall(r"[a-z0-9]+", value))))


def split_labels(
    metadata: list[dict[str, Any]], structural: list[dict[str, Any]], indices: list[int]
) -> dict[str, list[str]]:
    labels = {
        "operation": [structural[index]["operation_signature"] for index in indices],
        "feature_sequence": [">".join(structural[index]["feature_types"]) for index in indices],
        "final_operation": [
            structural[index]["operation_signature"].rsplit(":", 1)[-1]
            if structural[index]["operation_signature"]
            else ""
            for index in indices
        ],
        "feature_count": [
            str(structural[index]["num_features"])
            if structural[index]["num_features"] < 4
            else "4+"
            for index in indices
        ],
        "functional": [metadata[index].get("functional_type", "") for index in indices],
        "primitive_profile": [
            ">".join(
                f"{name}:{count}"
                for name, count in sorted(structural[index]["primitive_counts"].items())
            )
            for index in indices
        ],
        "dominant_constraint": [
            max(
                structural[index]["constraint_counts"],
                key=structural[index]["constraint_counts"].get,
            )
            if structural[index]["constraint_counts"]
            else ""
            for index in indices
        ],
    }
    labels["canonical"] = [canonical_label(value) for value in labels["functional"]]
    return labels


def compact(row: dict[str, Any]) -> dict[str, float | int]:
    return {
        "dimension_recall_at_10": row["dimension_neighbor_retrieval"]["recall_at_10"],
        "dimension_success_at_5": row["dimension_neighbor_retrieval"]["query_success_at_5"],
        "dimension_mae_at_1": row["dimension_retrieval"]["mae_at_1"],
        "operation_history_nn_accuracy": row["operation_signature"]["hit_at_1"],
        "operation_history_success_at_5": row["operation_signature"]["hit_at_5"],
        "feature_sequence_nn_accuracy": row["feature_type_sequence"]["hit_at_1"],
        "final_operation_nn_accuracy": row["final_boolean_operation"]["hit_at_1"],
        "feature_count_nn_accuracy": row["feature_count_bucket"]["hit_at_1"],
        "primitive_profile_nn_accuracy": row["primitive_profile"]["hit_at_1"],
        "dominant_constraint_nn_accuracy": row["dominant_constraint_type"]["hit_at_1"],
        "canonical_function_nn_accuracy": row["canonical_functional_type_evaluation_only"][
            "hit_at_1"
        ],
        "canonical_function_success_at_5": row["canonical_functional_type_evaluation_only"][
            "hit_at_5"
        ],
        "near_tie_queries": row["ties"]["queries_with_top10_gap_below_1e_4"],
        "median_top1_top2_margin": row["ties"]["median_top1_top2_margin"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate JEPA, autoencoder, and nonlearned CAD retrieval representations"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument(
        "--exclude-adapter",
        action="store_true",
        help="Omit the secondary adapter ablation even if its representations exist",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    torch.set_num_threads(8)
    source = torch.load(
        resolve(config["paths"]["representations"]),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    output_dir = resolve(config["paths"]["output_dir"])
    baseline_path = output_dir / "baseline_representations.pt"
    if not baseline_path.is_file():
        raise FileNotFoundError(
            f"Baseline representations not found: {baseline_path}. Run train_baselines.py first."
        )
    baselines = torch.load(
        baseline_path, map_location="cpu", weights_only=False, mmap=True
    )
    if source["uids"] != baselines["uids"]:
        raise ValueError("Baseline representations are not UID-aligned with the v7 source")
    representations: dict[str, torch.Tensor] = {
        "pretrained_jepa_joint": source["embeddings"]["joint"],
        **baselines["embeddings"],
    }
    methods: dict[str, Any] = {
        "pretrained_jepa_joint": {
            "family": "proposed_representation",
            "training": "v7 JEPA pretraining on the train split; checkpoint selected on validation",
            "retrieval_labels_used": False,
        },
        **baselines["methods"],
    }
    adapted_path = output_dir / "adapted_representations.pt"
    adapter_checkpoint = None
    if adapted_path.is_file() and not args.exclude_adapter:
        adapted = torch.load(adapted_path, map_location="cpu", weights_only=False, mmap=True)
        if source["uids"] != adapted["uids"]:
            raise ValueError("Adapted representations are not UID-aligned with the v7 source")
        representations["adapted_jepa_joint"] = adapted["embeddings"]["adapted_joint"]
        adapter_checkpoint = adapted["adapter_checkpoint"]
        methods["adapted_jepa_joint"] = {
            "family": "secondary_downstream_ablation",
            "training": "frozen JEPA plus task-specific adapter trained on train and selected on validation",
            "retrieval_labels_used": False,
        }
    result: dict[str, Any] = {
        "protocol": {
            "objective": "compare representation-learning methods for CAD retrieval",
            "primary_comparison": [
                "pretrained_jepa_joint",
                "graph_autoencoder_joint",
                "pca_graph_statistics",
                "input_graph_statistics",
            ],
            "secondary_comparison": (
                ["pretrained_jepa_joint", "adapted_jepa_joint"]
                if "adapted_jepa_joint" in representations
                else []
            ),
            "retrieval": "within-split leave-self-out cosine nearest-neighbor retrieval",
            "shared_split_source": config["paths"]["cache"],
            "shared_metrics": "histcad_large_jepa.benchmark.retrieval_metrics",
            "source": "v7 joint 4096-D extracted from validation-selected best.pt",
            "baseline_checkpoint": baselines["autoencoder_checkpoint"],
            "adapter_checkpoint": adapter_checkpoint,
            "evaluation_labels_used_for_representation_training": False,
            "test_used_for_model_selection": False,
            "methods": methods,
        },
        "splits": {},
    }
    settings = config["evaluation"]
    for split in ("val", "test"):
        indices = [index for index, item in enumerate(source["metadata"]) if item["split"] == split]
        targets = (source["dimension_targets"][indices] - source["dimension_target_mean"]) / source[
            "dimension_target_std"
        ]
        labels = split_labels(source["metadata"], source["structural_metadata"], indices)
        references = _dimension_reference_neighbors(
            targets,
            settings["retrieval_chunk_size"],
            settings["dimension_reference_neighbors"],
        )
        rows = {}
        for name, values in representations.items():
            rows[name] = retrieval_metrics(
                values[indices],
                targets,
                labels,
                references,
                settings["retrieval_chunk_size"],
            )
            print(json.dumps({"evaluated": name, "split": split}), flush=True)
        summaries = {name: compact(row) for name, row in rows.items()}
        secondary = {}
        if "adapted_jepa_joint" in summaries:
            direct_compact = summaries["pretrained_jepa_joint"]
            adapted_compact = summaries["adapted_jepa_joint"]
            secondary["adapted_minus_pretrained_jepa"] = {
                name: adapted_compact[name] - direct_compact[name] for name in direct_compact
            }
        result["splits"][split] = {
            "samples": len(indices),
            "representations": rows,
            "summary": summaries,
            "secondary_adapter_analysis": secondary,
        }
    (output_dir / "benchmark_evaluation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    first = next(iter(result["splits"]["test"]["summary"].values()))
    fields = ("split", "representation", "family", *first)
    with (output_dir / "benchmark_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split in ("val", "test"):
            for name in representations:
                writer.writerow(
                    {
                        "split": split,
                        "representation": name,
                        "family": methods[name]["family"],
                        **result["splits"][split]["summary"][name],
                    }
                )
    print(json.dumps({"status": "complete", "test": result["splits"]["test"]["summary"]}))


if __name__ == "__main__":
    main()
