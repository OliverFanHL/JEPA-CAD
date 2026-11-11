from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .cache import (
    PARAMETRIC_EDGE_KEYS,
    PARAMETRIC_NODE_KEYS,
    STEP_EDGE_KEYS,
    STEP_NODE_KEYS,
    unpack_graph,
)
from .parametric_graph import collate_parametric_graphs
from .step_graph import collate_step_graphs

PARAMETRIC_GLOBAL_TARGET_INDICES = (0, *range(7, 22))


class PairedGraphDataset(Dataset):
    def __init__(
        self,
        cache: str | Path | dict[str, Any],
        split: str,
        target_mean: torch.Tensor | None = None,
        target_std: torch.Tensor | None = None,
    ):
        self.payload = (
            cache
            if isinstance(cache, dict)
            else torch.load(cache, map_location="cpu", weights_only=False, mmap=True)
        )
        if self.payload.get("format_version") != 7:
            raise ValueError("Expected a self-contained v7 cache; rebuild with run.py --cache-only")
        self.indices = [
            index for index, item in enumerate(self.payload["metadata"]) if item["split"] == split
        ]
        train_count = self.payload["split_sizes"]["train"]
        train_targets = self.payload["dimension_targets"][:train_count]
        self.target_mean = train_targets.mean(0) if target_mean is None else target_mean
        self.target_std = train_targets.std(0).clamp_min(0.05) if target_std is None else target_std

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict[str, Any]:
        index = self.indices[position]
        parametric = unpack_graph(
            self.payload["parametric_graphs"], index, PARAMETRIC_NODE_KEYS, PARAMETRIC_EDGE_KEYS
        )
        parametric.update(
            uid=self.payload["uids"][index], metadata=self.payload["structural_metadata"][index]
        )
        standardized = (
            self.payload["dimension_targets"][index] - self.target_mean
        ) / self.target_std
        parametric["numeric"] = parametric["numeric"].clone()
        parametric["numeric_mask"] = parametric["numeric_mask"].clone()
        parametric["numeric"][0, 4:20] = standardized[list(PARAMETRIC_GLOBAL_TARGET_INDICES)]
        parametric["numeric_mask"][0, 4:20] = True
        step = unpack_graph(self.payload["step_graphs"], index, STEP_NODE_KEYS, STEP_EDGE_KEYS)
        return {
            "index": index,
            "parametric": parametric,
            "step": step,
            "dimension_target": standardized,
        }


def collate_paired(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "indices": torch.tensor([item["index"] for item in items]),
        "parametric": collate_parametric_graphs([item["parametric"] for item in items]),
        "step": collate_step_graphs([item["step"] for item in items]),
        "dimension_target": torch.stack([item["dimension_target"] for item in items]),
    }


def make_node_mask(
    batch: dict[str, torch.Tensor], ratio: float, generator: torch.Generator
) -> torch.Tensor:
    result = torch.zeros_like(batch["graph_index"], dtype=torch.bool)
    global_nodes = torch.zeros_like(result)
    global_nodes[batch["global_index"]] = True
    for graph in range(len(batch["global_index"])):
        candidates = torch.where((batch["graph_index"] == graph) & ~global_nodes)[0]
        if len(candidates) < 2:
            continue
        count = max(1, min(len(candidates) - 1, round(len(candidates) * ratio)))
        chosen = candidates[torch.randperm(len(candidates), generator=generator)[:count]]
        result[chosen] = True
    return result


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    def move(mapping: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in mapping.items()
        }

    return {
        "indices": batch["indices"].to(device),
        "parametric": move(batch["parametric"]),
        "step": move(batch["step"]),
        "dimension_target": batch["dimension_target"].to(device),
    }
