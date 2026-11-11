from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .io import atomic_json, write_csv
from .projection import FixedProjector
from .registry import Registry
from .training import knn_classification, train_probe


def _indices(splits: list[str]) -> dict[str, torch.Tensor]:
    return {name: torch.tensor([i for i, value in enumerate(splits) if value == name]) for name in ("train", "val", "test")}


def _existing_controls(paths: list[Path]) -> list[dict[str, Any]]:
    output = []
    for path in paths:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        output.append({"path": str(path), "payload": payload, "measured": True})
    return output


def run(config: dict[str, Any], registry: Registry, force: bool = False) -> dict[str, Any]:
    settings = config["tmcad_classification"]
    output = Path(config["paths"]["output_dir"]) / "tmcad_classification" / "results.json"
    if output.is_file() and not force:
        return json.loads(output.read_text(encoding="utf-8"))
    names = registry.names("classification")
    registry.validate_alignment("classification")
    result: dict[str, Any] = {
        "protocol": {
            "split": "exact saved TMCAD train/validation/test assignment",
            "supervision": "identical class labels for all representations",
            "model_selection": "validation macro-F1 only; test evaluated after selection",
            "frozen_probe": "capacity-matched linear head after deterministic frozen projection",
            "controlled_adapter": "identical projected input width, hidden width, optimizer, budget, and class weighting",
        },
        "representations": {},
        "shared_coverage": registry.coverage_summary(),
        "task_specific_supervised_controls": _existing_controls(settings["existing_control_results"]),
    }
    rows = []
    for name in names:
        representation = registry.load(name)
        target = torch.tensor(representation.labels["class"], dtype=torch.long)
        split_indices = _indices(representation.splits)
        classes = int(target.max()) + 1
        projected = FixedProjector(representation.dim, settings["common_width"], config["seed"])(representation.values)
        linear, _ = train_probe(projected, target, split_indices, classes, "linear", settings["training"], config["seed"])
        adapter, _ = train_probe(projected, target, split_indices, classes, "adapter", settings["training"], config["seed"])
        row = {
            "family": representation.family,
            "input_dimensions": representation.dim,
            "common_width": settings["common_width"],
            "knn": knn_classification(representation.values, target, split_indices, settings["knn_k"], settings["knn_chunk_size"]),
            "frozen_linear_probe": linear,
            "controlled_adapter": adapter,
        }
        result["representations"][name] = row
        rows.append({
            "representation": name,
            "family": representation.family,
            "frozen_probe_test_macro_f1": linear["test"]["macro_f1"],
            "adapter_test_macro_f1": adapter["test"]["macro_f1"],
            "adapter_validation_macro_f1": adapter["validation"]["macro_f1"],
            "adapter_trainable_parameters": adapter["trainable_parameters"],
        })
        print(json.dumps({"stage": "classification", "complete": name}), flush=True)
    atomic_json(output, result)
    write_csv(output.with_name("summary.csv"), rows)
    return result
