from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .io import atomic_json, write_csv
from .metrics import class_separation, latent_diagnostics, retrieval_metrics
from .projection import FixedProjector
from .registry import Registry
from .training import encode_labels, knn_classification, train_probe


def _split_indices(splits: list[str], valid: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    output = {}
    for split in ("train", "val", "test"):
        indices = torch.tensor([i for i, value in enumerate(splits) if value == split])
        output[split] = indices if valid is None else indices[valid[indices]]
    if any(not len(indices) for indices in output.values()):
        raise ValueError("Every frozen evaluation requires non-empty train/val/test splits")
    return output


def run(config: dict[str, Any], registry: Registry, force: bool = False) -> dict[str, Any]:
    settings = config["frozen_representation"]
    output = Path(config["paths"]["output_dir"]) / "frozen_representation" / "results.json"
    if output.is_file() and not force:
        return json.loads(output.read_text(encoding="utf-8"))
    result: dict[str, Any] = {
        "protocol": {
            "retrieval": "direct cosine on the held-out test split; self matches excluded",
            "knn": f"k={settings['knn_k']} against train only",
            "linear_probe": "deterministic frozen projection followed by one linear layer; validation macro-F1 selection",
            "adapter_probe": "same projected width and hidden width for every representation",
            "test_used_for_selection": False,
        },
        "representations": {},
        "shared_coverage": registry.coverage_summary(),
    }
    summary_rows = []
    canonical_labels: dict[str, dict[str, list[Any]]] = {}
    for name in registry.names("frozen"):
        representation = registry.load(name)
        dataset = registry.specs[name]["dataset"]
        canonical_labels.setdefault(dataset, representation.labels)
        labels = canonical_labels[dataset]
        dataset_settings = settings["datasets"][dataset]
        label_name = dataset_settings["label"]
        if label_name not in labels:
            raise ValueError(f"{name}: missing configured label {label_name}")
        target, classes = encode_labels(
            labels[label_name], representation.splits,
            dataset_settings["minimum_train_class_count"],
        )
        valid = target >= 0
        split_indices = _split_indices(representation.splits, valid)
        projected = FixedProjector(representation.dim, settings["common_width"], config["seed"])(representation.values)
        test_indices = split_indices["test"]
        test_labels = {
            key: [values[int(index)] for index in test_indices]
            for key, values in labels.items()
            if key in dataset_settings["retrieval_labels"]
        }
        targets = representation.targets.get(dataset_settings.get("target", ""))
        test_targets = None if targets is None else targets[test_indices].float()
        row: dict[str, Any] = {
            "family": representation.family,
            "dimensions": representation.dim,
            "classes": classes,
            "latent": latent_diagnostics(representation.values, settings["diagnostic_max_rows"], config["seed"]),
            "retrieval": retrieval_metrics(
                representation.values[test_indices], test_labels, test_targets,
                settings["retrieval_chunk_size"],
            ),
            "knn": knn_classification(
                representation.values, target, split_indices, settings["knn_k"],
                settings["knn_chunk_size"],
            ),
            "class_separation_test": class_separation(
                representation.values[test_indices], target[test_indices].tolist()
            ),
            "fixed_projection": {
                "input_dimensions": representation.dim,
                "output_dimensions": settings["common_width"],
                "trainable_parameters": 0,
                "seed": config["seed"],
            },
        }
        row["linear_probe"], _ = train_probe(
            projected, target, split_indices, len(classes), "linear", settings["probe"], config["seed"]
        )
        row["adapter_probe"], _ = train_probe(
            projected, target, split_indices, len(classes), "adapter", settings["probe"], config["seed"]
        )
        result["representations"][name] = row
        summary_rows.append({
            "representation": name,
            "family": representation.family,
            "dimensions": representation.dim,
            "effective_rank": row["latent"]["effective_rank_entropy"],
            "retrieval_margin": row["retrieval"]["duplicates_and_ties"]["median_top1_top2_margin"],
            "knn_test_macro_f1": row["knn"]["test"]["macro_f1"],
            "linear_test_macro_f1": row["linear_probe"]["test"]["macro_f1"],
            "adapter_test_macro_f1": row["adapter_probe"]["test"]["macro_f1"],
        })
        print(json.dumps({"stage": "frozen", "complete": name}), flush=True)
    atomic_json(output, result)
    write_csv(output.with_name("summary.csv"), summary_rows)
    return result
