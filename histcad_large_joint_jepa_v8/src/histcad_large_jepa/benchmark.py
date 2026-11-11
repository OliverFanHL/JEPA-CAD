from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .schema import CONSTRAINT_TYPES, PRIMITIVE_TYPES


def _path(project_root: Path, value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (project_root / candidate).resolve()


def _stored_values(payload: dict[str, Any], storage: str, key: str) -> torch.Tensor:
    if storage == "embeddings_dict":
        return payload["embeddings"][key]
    return payload[key]


def _canonical_label(value: str) -> str:
    value = re.sub(r"\([^)]*\)", "", value.lower())
    value = re.sub(r"\b(?:with|and|for|of|the|a|an)\b", " ", value)
    return " ".join(sorted(set(re.findall(r"[a-z0-9]+", value))))


def _neighbors(values: torch.Tensor, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    values = F.normalize(values.float(), dim=-1)
    indices, scores = [], []
    for start in range(0, len(values), chunk_size):
        stop = min(start + chunk_size, len(values))
        similarity = values[start:stop] @ values.T
        rows = torch.arange(stop - start)
        similarity[rows, torch.arange(start, stop)] = -torch.inf
        chunk_scores, chunk_indices = similarity.topk(10, dim=1)
        scores.append(chunk_scores)
        indices.append(chunk_indices)
    return torch.cat(indices), torch.cat(scores)


def _categorical_retrieval(labels: list[str], neighbors: torch.Tensor) -> dict[str, float | int]:
    counts = Counter(labels)
    eligible = [index for index, label in enumerate(labels) if label and counts[label] > 1]
    result: dict[str, float | int] = {"eligible_queries": len(eligible)}
    for k in (1, 5, 10):
        match_rows = [
            [labels[int(candidate)] == labels[query] for candidate in neighbors[query, :k]]
            for query in eligible
        ]
        query_success = [any(row) for row in match_rows]
        result[f"hit_at_{k}"] = sum(query_success) / max(1, len(query_success))
        result[f"recall_at_{k}_at_least_one_relevant"] = result[f"hit_at_{k}"]
        result[f"precision_at_{k}"] = sum(sum(row) / k for row in match_rows) / max(
            1, len(match_rows)
        )
        majority_correct = []
        for query in eligible:
            predicted = Counter(labels[int(candidate)] for candidate in neighbors[query, :k])
            majority_correct.append(predicted.most_common(1)[0][0] == labels[query])
        result[f"knn_majority_accuracy_at_{k}"] = sum(majority_correct) / max(
            1, len(majority_correct)
        )
    reciprocal_ranks = []
    for query in eligible:
        rank = next(
            (
                rank
                for rank, candidate in enumerate(neighbors[query, :10], start=1)
                if labels[int(candidate)] == labels[query]
            ),
            None,
        )
        reciprocal_ranks.append(0.0 if rank is None else 1.0 / rank)
    result["mean_reciprocal_rank_at_10"] = sum(reciprocal_ranks) / max(1, len(reciprocal_ranks))
    return result


def _dimension_reference_neighbors(
    targets: torch.Tensor, chunk_size: int, reference_size: int
) -> torch.Tensor:
    references = []
    for start in range(0, len(targets), chunk_size):
        stop = min(start + chunk_size, len(targets))
        distance = torch.cdist(targets[start:stop].float(), targets.float(), p=1)
        rows = torch.arange(stop - start)
        distance[rows, torch.arange(start, stop)] = torch.inf
        references.append(distance.topk(reference_size, dim=1, largest=False).indices)
    return torch.cat(references)


def _dimension_neighbor_metrics(
    neighbors: torch.Tensor, references: torch.Tensor
) -> dict[str, float | int]:
    result: dict[str, float | int] = {
        "reference_neighbors_per_query": references.shape[1],
        "exact_dimension_nearest_neighbor_accuracy": float(
            (neighbors[:, 0] == references[:, 0]).float().mean()
        ),
    }
    for k in (1, 5, 10):
        relevant = (neighbors[:, :k, None] == references[:, None, :]).any(-1)
        result[f"precision_at_{k}"] = float(relevant.float().mean())
        result[f"recall_at_{k}"] = float((relevant.sum(1) / references.shape[1]).float().mean())
        result[f"query_success_at_{k}"] = float(relevant.any(1).float().mean())
    ranks = torch.where(
        neighbors[:, :, None] == references[:, None, :1],
        torch.arange(1, neighbors.shape[1] + 1)[None, :, None],
        neighbors.shape[1] + 1,
    ).amin((1, 2))
    result["exact_dimension_neighbor_mrr_at_10"] = float(
        torch.where(
            ranks <= 10, 1.0 / ranks.float(), torch.zeros_like(ranks, dtype=torch.float)
        ).mean()
    )
    return result


def retrieval_metrics(
    values: torch.Tensor,
    targets: torch.Tensor,
    labels: dict[str, list[str]],
    dimension_references: torch.Tensor,
    chunk_size: int,
) -> dict[str, Any]:
    normalized = F.normalize(values.float(), dim=-1)
    neighbors, scores = _neighbors(normalized, chunk_size)
    dimension: dict[str, float] = {}
    for k in (1, 5, 10):
        differences = (targets[neighbors[:, :k]] - targets[:, None]).abs().mean(-1)
        dimension[f"mae_at_{k}"] = float(differences.mean())
        dimension[f"best_mae_at_{k}"] = float(differences.min(1).values.mean())
    keys = [hashlib.sha1(row.numpy().tobytes()).hexdigest() for row in normalized]
    duplicate_counts = Counter(keys)
    gaps = scores[:, :-1] - scores[:, 1:]
    return {
        "dimensions": values.shape[1],
        "dimension_retrieval": dimension,
        "dimension_neighbor_retrieval": _dimension_neighbor_metrics(
            neighbors, dimension_references
        ),
        "operation_signature": _categorical_retrieval(labels["operation"], neighbors),
        "feature_type_sequence": _categorical_retrieval(labels["feature_sequence"], neighbors),
        "final_boolean_operation": _categorical_retrieval(labels["final_operation"], neighbors),
        "feature_count_bucket": _categorical_retrieval(labels["feature_count"], neighbors),
        "primitive_profile": _categorical_retrieval(labels["primitive_profile"], neighbors),
        "dominant_constraint_type": _categorical_retrieval(
            labels["dominant_constraint"], neighbors
        ),
        "functional_type_evaluation_only": _categorical_retrieval(labels["functional"], neighbors),
        "canonical_functional_type_evaluation_only": _categorical_retrieval(
            labels["canonical"], neighbors
        ),
        "ties": {
            "samples_in_exact_duplicate_groups": sum(
                count for count in duplicate_counts.values() if count > 1
            ),
            "queries_with_top10_tie_at_1e_7": int((gaps.abs() <= 1e-7).any(1).sum()),
            "queries_with_top10_gap_below_1e_4": int((gaps.abs() <= 1e-4).any(1).sum()),
            "median_top1_top2_margin": float((scores[:, 0] - scores[:, 1]).median()),
        },
    }


def _classification_metrics(
    logits: torch.Tensor, targets: torch.Tensor, classes: int
) -> dict[str, float]:
    prediction = logits.argmax(1)
    recalls, f1_values = [], []
    for label in range(classes):
        true_positive = int(((prediction == label) & (targets == label)).sum())
        false_positive = int(((prediction == label) & (targets != label)).sum())
        false_negative = int(((prediction != label) & (targets == label)).sum())
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        recalls.append(recall)
        f1_values.append(2 * precision * recall / max(1e-12, precision + recall))
    return {
        "accuracy": float((prediction == targets).float().mean()),
        "balanced_accuracy": sum(recalls) / classes,
        "macro_f1": sum(f1_values) / classes,
        "top3_accuracy": float(
            (logits.topk(min(3, classes), dim=1).indices == targets[:, None]).any(1).float().mean()
        ),
    }


def prototype_suite(
    values: torch.Tensor,
    uids: list[str],
    train_uids: list[str],
    validation_uids: list[str],
    test_uids: list[str],
    tasks: dict[str, dict[str, Any]],
    chunk_size: int,
) -> dict[str, Any]:
    lookup = {uid: index for index, uid in enumerate(uids)}
    available_train = [uid for uid in train_uids if uid in lookup]
    sums = {
        name: torch.zeros((len(task["classes"]), values.shape[1])) for name, task in tasks.items()
    }
    counts = {name: torch.zeros(len(task["classes"])) for name, task in tasks.items()}
    for start in range(0, len(available_train), chunk_size):
        chunk_uids = available_train[start : start + chunk_size]
        chunk = F.normalize(values[[lookup[uid] for uid in chunk_uids]].float(), dim=-1)
        for name, task in tasks.items():
            target = torch.tensor([task["mapping"].get(uid, -1) for uid in chunk_uids])
            valid = target >= 0
            if valid.any():
                sums[name].index_add_(0, target[valid], chunk[valid])
                counts[name].index_add_(0, target[valid], torch.ones(int(valid.sum())))
    centroids = {
        name: F.normalize(sums[name] / counts[name].clamp_min(1).unsqueeze(1), dim=-1)
        for name in tasks
    }

    def evaluate_split(split_uids: list[str], name: str) -> tuple[dict[str, float], int]:
        task = tasks[name]
        selected = [uid for uid in split_uids if uid in lookup and uid in task["mapping"]]
        features = F.normalize(values[[lookup[uid] for uid in selected]].float(), dim=-1)
        target = torch.tensor([task["mapping"][uid] for uid in selected])
        return _classification_metrics(
            features @ centroids[name].T, target, len(task["classes"])
        ), len(selected)

    result = {}
    for name, task in tasks.items():
        validation, validation_count = evaluate_split(validation_uids, name)
        test, test_count = evaluate_split(test_uids, name)
        test_targets = [task["mapping"][uid] for uid in test_uids if uid in task["mapping"]]
        majority = max(Counter(test_targets).values()) / max(1, len(test_targets))
        result[name] = {
            "classes": task["classes"],
            "train_samples": int(counts[name].sum()),
            "validation_samples": validation_count,
            "test_samples": test_count,
            "validation": validation,
            "test": test,
            "majority_test_accuracy": majority,
            "random_chance_accuracy": 1 / len(task["classes"]),
        }
    return result


def _tasks(
    metadata: list[dict[str, Any]], structural: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    train_indices = [index for index, item in enumerate(metadata) if item["split"] == "train"]
    functional_counts = Counter(
        metadata[index].get("functional_type", "") for index in train_indices
    )
    functional_classes = [
        label for label, count in functional_counts.most_common(20) if label and count >= 50
    ]
    dominant_counts: Counter[str] = Counter()
    dominant_by_uid: dict[str, str] = {}
    for index, item in enumerate(metadata):
        counts = structural[index]["constraint_counts"]
        if counts:
            dominant = max(counts, key=counts.get)
            dominant_by_uid[item["uid"]] = dominant
            if item["split"] == "train":
                dominant_counts[dominant] += 1
    dominant_classes = [name for name, _ in dominant_counts.most_common(8)]
    operation_classes = ["NewBody", "Join", "Cut", "Intersect"]
    feature_classes = ["1", "2", "3", "4+"]

    def feature_bucket(count: int) -> str:
        return str(count) if count < 4 else "4+"

    class_sets = {
        "functional_top20_evaluation_only": functional_classes,
        "final_boolean_operation": operation_classes,
        "feature_count_bucket": feature_classes,
        "dominant_constraint_type": dominant_classes,
    }
    raw_labels: dict[str, dict[str, str]] = {name: {} for name in class_sets}
    for index, item in enumerate(metadata):
        uid, signature = item["uid"], structural[index]["operation_signature"]
        raw_labels["functional_top20_evaluation_only"][uid] = item.get("functional_type", "")
        raw_labels["final_boolean_operation"][uid] = (
            signature.rsplit(":", 1)[-1] if signature else ""
        )
        raw_labels["feature_count_bucket"][uid] = feature_bucket(structural[index]["num_features"])
        raw_labels["dominant_constraint_type"][uid] = dominant_by_uid.get(uid, "")
    tasks = {}
    for name, classes in class_sets.items():
        class_to_id = {label: index for index, label in enumerate(classes)}
        tasks[name] = {
            "classes": classes,
            "mapping": {
                uid: class_to_id[label]
                for uid, label in raw_labels[name].items()
                if label in class_to_id
            },
        }
    return tasks


def _handcrafted(payload: dict[str, Any]) -> torch.Tensor:
    targets = (payload["dimension_targets"] - payload["dimension_target_mean"]) / payload[
        "dimension_target_std"
    ]
    rows = []
    for metadata in payload["structural_metadata"]:
        counts = [metadata["primitive_counts"].get(name, 0) for name in PRIMITIVE_TYPES]
        counts.extend(metadata["constraint_counts"].get(name, 0) for name in CONSTRAINT_TYPES)
        rows.append([math.log1p(value) for value in counts])
    return torch.cat((targets[:, :19].float(), torch.tensor(rows).float()), dim=1)


def _write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (output_dir / "retrieval_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = (
            "representation",
            "dimension_mae_at_1",
            "dimension_best_mae_at_5",
            "operation_hit_at_1",
            "canonical_functional_hit_at_5",
            "exact_duplicate_samples",
            "near_tie_queries",
            "median_margin",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, row in result["retrieval"].items():
            writer.writerow(
                {
                    "representation": name,
                    "dimension_mae_at_1": row["dimension_retrieval"]["mae_at_1"],
                    "dimension_best_mae_at_5": row["dimension_retrieval"]["best_mae_at_5"],
                    "operation_hit_at_1": row["operation_signature"]["hit_at_1"],
                    "canonical_functional_hit_at_5": row[
                        "canonical_functional_type_evaluation_only"
                    ]["hit_at_5"],
                    "exact_duplicate_samples": row["ties"]["samples_in_exact_duplicate_groups"],
                    "near_tie_queries": row["ties"]["queries_with_top10_gap_below_1e_4"],
                    "median_margin": row["ties"]["median_top1_top2_margin"],
                }
            )
    with (output_dir / "prototype_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = (
            "representation",
            "pretraining_scope",
            "task",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "top3_accuracy",
            "test_samples",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for representation, suite in result["prototype_classification"].items():
            for task, row in suite["tasks"].items():
                writer.writerow(
                    {
                        "representation": representation,
                        "pretraining_scope": suite["pretraining_scope"],
                        "task": task,
                        **row["test"],
                        "test_samples": row["test_samples"],
                    }
                )
    with (output_dir / "intuitive_retrieval_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = (
            "representation",
            "dimension_neighbor_recall_at_10",
            "dimension_neighbor_success_at_5",
            "operation_history_nn_accuracy",
            "operation_history_recall_at_5",
            "feature_sequence_nn_accuracy",
            "final_operation_nn_accuracy",
            "feature_count_nn_accuracy",
            "primitive_profile_nn_accuracy",
            "dominant_constraint_nn_accuracy",
            "canonical_function_nn_accuracy",
            "canonical_function_recall_at_5",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, row in result["retrieval"].items():
            writer.writerow(
                {
                    "representation": name,
                    "dimension_neighbor_recall_at_10": row["dimension_neighbor_retrieval"][
                        "recall_at_10"
                    ],
                    "dimension_neighbor_success_at_5": row["dimension_neighbor_retrieval"][
                        "query_success_at_5"
                    ],
                    "operation_history_nn_accuracy": row["operation_signature"]["hit_at_1"],
                    "operation_history_recall_at_5": row["operation_signature"]["hit_at_5"],
                    "feature_sequence_nn_accuracy": row["feature_type_sequence"]["hit_at_1"],
                    "final_operation_nn_accuracy": row["final_boolean_operation"]["hit_at_1"],
                    "feature_count_nn_accuracy": row["feature_count_bucket"]["hit_at_1"],
                    "primitive_profile_nn_accuracy": row["primitive_profile"]["hit_at_1"],
                    "dominant_constraint_nn_accuracy": row["dominant_constraint_type"]["hit_at_1"],
                    "canonical_function_nn_accuracy": row[
                        "canonical_functional_type_evaluation_only"
                    ]["hit_at_1"],
                    "canonical_function_recall_at_5": row[
                        "canonical_functional_type_evaluation_only"
                    ]["hit_at_5"],
                }
            )


def run_benchmarks(config: dict[str, Any], project_root: Path) -> dict[str, Any]:
    torch.set_num_threads(8)
    current_path = _path(project_root, config["current_representations"])
    checkpoint_path = _path(project_root, config["current_checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_provenance = {
        "path": str(checkpoint_path),
        "format_version": checkpoint["format_version"],
        "epoch": checkpoint["epoch"],
        "best_validation_loss": checkpoint["best_validation_loss"],
        "selection": "minimum validation objective",
        "representation_extraction": "train.py reloads best.pt before representations.pt extraction",
    }
    del checkpoint
    gc.collect()
    current = torch.load(current_path, map_location="cpu", weights_only=False)
    metadata, structural, uids = (
        current["metadata"],
        current["structural_metadata"],
        current["uids"],
    )
    split_uids = {
        split: [item["uid"] for item in metadata if item["split"] == split]
        for split in ("train", "val", "test")
    }
    test_indices = [index for index, item in enumerate(metadata) if item["split"] == "test"]
    test_uids = split_uids["test"]
    test_targets = (
        current["dimension_targets"][test_indices] - current["dimension_target_mean"]
    ) / current["dimension_target_std"]
    test_labels = {
        "operation": [structural[index]["operation_signature"] for index in test_indices],
        "feature_sequence": [
            ">".join(structural[index]["feature_types"]) for index in test_indices
        ],
        "final_operation": [
            structural[index]["operation_signature"].rsplit(":", 1)[-1]
            if structural[index]["operation_signature"]
            else ""
            for index in test_indices
        ],
        "feature_count": [
            str(structural[index]["num_features"])
            if structural[index]["num_features"] < 4
            else "4+"
            for index in test_indices
        ],
        "functional": [metadata[index].get("functional_type", "") for index in test_indices],
        "primitive_profile": [
            ">".join(
                f"{name}:{count}"
                for name, count in sorted(structural[index]["primitive_counts"].items())
            )
            for index in test_indices
        ],
        "dominant_constraint": [
            max(
                structural[index]["constraint_counts"],
                key=structural[index]["constraint_counts"].get,
            )
            if structural[index]["constraint_counts"]
            else ""
            for index in test_indices
        ],
    }
    test_labels["canonical"] = [_canonical_label(label) for label in test_labels["functional"]]
    dimension_references = _dimension_reference_neighbors(
        test_targets,
        config["retrieval_chunk_size"],
        config["dimension_reference_neighbors"],
    )
    tasks = _tasks(metadata, structural)
    result: dict[str, Any] = {
        "protocol": {
            "test_samples": len(test_uids),
            "labels_used_in_pretraining": False,
            "current_checkpoint": checkpoint_provenance,
            "dimension_target_names": list(current["dimension_target_names"]),
            "prototype_head": "nearest class centroid; frozen representations",
            "caveat": "Earlier optional baselines may use different pretraining subsets; v7 uses the full train split",
        },
        "retrieval": {},
        "prototype_classification": {},
    }

    current_test = {
        name: current["embeddings"][name][test_indices].float()
        for name in ("parametric", "step", "joint")
    }
    current_test["direct_concat"] = torch.cat(
        (
            F.normalize(current_test["parametric"], dim=-1),
            F.normalize(current_test["step"], dim=-1),
        ),
        dim=-1,
    )
    for key, values in current_test.items():
        name = f"v7_{key}_{values.shape[1]}_full"
        result["retrieval"][name] = retrieval_metrics(
            values,
            test_targets,
            test_labels,
            dimension_references,
            config["retrieval_chunk_size"],
        )
        print(json.dumps({"retrieval_complete": name}), flush=True)
    for key in ("parametric", "step", "joint"):
        name = f"v7_{key}_{current['embeddings'][key].shape[1]}_full"
        suite = prototype_suite(
            current["embeddings"][key],
            uids,
            split_uids["train"],
            split_uids["val"],
            split_uids["test"],
            tasks,
            config["prototype_chunk_size"],
        )
        result["prototype_classification"][name] = {"pretraining_scope": "full", "tasks": suite}
        print(json.dumps({"prototype_complete": name}), flush=True)
    handcrafted = _handcrafted(current)
    result["prototype_classification"]["handcrafted_parametric_statistics_44"] = {
        "pretraining_scope": "not pretrained",
        "tasks": prototype_suite(
            handcrafted,
            uids,
            split_uids["train"],
            split_uids["val"],
            split_uids["test"],
            tasks,
            config["prototype_chunk_size"],
        ),
    }
    del handcrafted, current_test, current
    gc.collect()

    baseline_groups: dict[Path, list[dict[str, Any]]] = {}
    for specification in config["baselines"]:
        baseline_groups.setdefault(_path(project_root, specification["path"]), []).append(
            specification
        )
    for path, specifications in baseline_groups.items():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        lookup = {uid: index for index, uid in enumerate(payload["uids"])}
        aligned_test = [lookup[uid] for uid in test_uids]
        available_train = [uid for uid in split_uids["train"] if uid in lookup]
        for specification in specifications:
            values = _stored_values(payload, specification["storage"], specification["key"])
            test_values = values[aligned_test].float()
            name = specification["name"]
            result["retrieval"][name] = retrieval_metrics(
                test_values,
                test_targets,
                test_labels,
                dimension_references,
                config["retrieval_chunk_size"],
            )
            suite = prototype_suite(
                values,
                payload["uids"],
                available_train,
                split_uids["val"],
                split_uids["test"],
                tasks,
                config["prototype_chunk_size"],
            )
            result["prototype_classification"][name] = {
                "pretraining_scope": specification["pretraining_scope"],
                "downstream_train_samples_available": len(available_train),
                "tasks": suite,
            }
            print(json.dumps({"baseline_complete": name}), flush=True)
        del payload
        gc.collect()
    output_dir = _path(project_root, config["output_dir"])
    _write_outputs(result, output_dir)
    return result
