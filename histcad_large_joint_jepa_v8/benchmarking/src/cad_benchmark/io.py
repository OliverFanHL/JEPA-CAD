from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch


def sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class Representation:
    name: str
    family: str
    values: torch.Tensor
    uids: list[str]
    splits: list[str]
    labels: dict[str, list[Any]] = field(default_factory=dict)
    targets: dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError(f"{self.name}: embeddings must be a rank-2 tensor")
        count = len(self.values)
        if len(self.uids) != count or len(self.splits) != count:
            raise ValueError(f"{self.name}: UID/split count does not match embeddings")
        if len(set(self.uids)) != count:
            raise ValueError(f"{self.name}: UIDs are not unique")
        for key, values in self.labels.items():
            if len(values) != count:
                raise ValueError(f"{self.name}: label {key!r} is not row-aligned")
        for key, values in self.targets.items():
            if len(values) != count:
                raise ValueError(f"{self.name}: target {key!r} is not row-aligned")
        check = self.values if count <= 4096 else self.values[torch.linspace(0, count - 1, 4096).long()]
        if not torch.isfinite(check.float()).all():
            raise ValueError(f"{self.name}: embeddings contain non-finite values")

    @property
    def dim(self) -> int:
        return int(self.values.shape[1])

    def indices(self, split: str) -> torch.Tensor:
        return torch.tensor([i for i, value in enumerate(self.splits) if value == split])

    def aligned(self, uids: list[str]) -> torch.Tensor:
        lookup = {uid: index for index, uid in enumerate(self.uids)}
        missing = [uid for uid in uids if uid not in lookup]
        if missing:
            raise ValueError(f"{self.name}: {len(missing)} requested UIDs are missing")
        return self.values[torch.tensor([lookup[uid] for uid in uids])]


def _histcad_labels(payload: dict[str, Any]) -> dict[str, list[Any]]:
    structural = payload["structural_metadata"]
    metadata = payload["metadata"]
    functional = [item.get("functional_type", "") for item in metadata]
    canonical = []
    for value in functional:
        value = re.sub(r"\([^)]*\)", "", value.lower())
        value = re.sub(r"\b(?:with|and|for|of|the|a|an)\b", " ", value)
        canonical.append(" ".join(sorted(set(re.findall(r"[a-z0-9]+", value)))))
    return {
        "functional_type": functional,
        "canonical_functional_type": canonical,
        "operation_signature": [item["operation_signature"] for item in structural],
        "final_operation": [
            item["operation_signature"].rsplit(":", 1)[-1]
            if item["operation_signature"] else ""
            for item in structural
        ],
        "feature_count": [str(item["num_features"]) for item in structural],
        "feature_count_bucket": [
            str(item["num_features"]) if item["num_features"] < 4 else "4+"
            for item in structural
        ],
        "feature_sequence": [">".join(item["feature_types"]) for item in structural],
        "primitive_profile": [
            ">".join(f"{name}:{count}" for name, count in sorted(item["primitive_counts"].items()))
            for item in structural
        ],
        "dominant_constraint": [
            max(item["constraint_counts"], key=item["constraint_counts"].get)
            if item["constraint_counts"] else ""
            for item in structural
        ],
    }


def load_representation(spec: dict[str, Any]) -> Representation:
    kind = spec["format"]
    if kind == "histcad_aligned_pt":
        path = Path(spec["artifact"])
        source_path = Path(spec["reference_artifact"])
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        source = torch.load(source_path, map_location="cpu", weights_only=False, mmap=True)
        values = payload[spec.get("storage", "embeddings")][spec["key"]]
        if payload["uids"] != source["uids"]:
            raise ValueError(f"{spec['name']}: artifact is not aligned with its reference")
        return Representation(
            spec["name"], spec["family"], values, list(source["uids"]),
            [item["split"] for item in source["metadata"]], _histcad_labels(source),
            {"dimensions": (source["dimension_targets"] - source["dimension_target_mean"])
             / source["dimension_target_std"]},
            {"artifact": str(path), "reference_artifact": str(source_path),
             "key": spec["key"], **spec.get("provenance", {})},
        )
    if kind in {"histcad_pt", "generic_pt"}:
        path = Path(spec["artifact"])
        # payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        import inspect

        load_kwargs = {
            "map_location": "cpu",
        }

        torch_load_params = inspect.signature(torch.load).parameters

        if "weights_only" in torch_load_params:
            load_kwargs["weights_only"] = False

        if "mmap" in torch_load_params:
            load_kwargs["mmap"] = True

        payload = torch.load(path, **load_kwargs)
        storage = spec.get("storage", "embeddings")
        values = payload[storage][spec["key"]] if storage else payload[spec["key"]]
        uids = list(payload["uids"])
        if kind == "histcad_pt":
            splits = [item["split"] for item in payload["metadata"]]
            labels = _histcad_labels(payload)
            targets = {
                "dimensions": (
                    payload["dimension_targets"] - payload["dimension_target_mean"]
                ) / payload["dimension_target_std"]
            }
        else:
            splits = list(payload[spec.get("splits_key", "splits")])
            labels = dict(payload.get("labels", {}))
            targets = dict(payload.get("targets", {}))
        return Representation(
            spec["name"], spec["family"], values, uids, splits, labels, targets,
            {"artifact": str(path), "key": spec["key"], **spec.get("provenance", {})},
        )
    if kind == "tmcad_npy":
        metadata_path = Path(spec["metadata"])
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        values = torch.from_numpy(np.asarray(np.load(spec["array"], mmap_mode="c")))
        return Representation(
            spec["name"], spec["family"], values,
            list(metadata["uids"]), list(metadata["splits"]),
            {"class": list(metadata["labels"])}, {},
            {"artifact": str(spec["array"]), "metadata": str(metadata_path),
             "classes": metadata["classes"], **spec.get("provenance", {})},
        )
    raise ValueError(f"Unknown representation format: {kind}")


def save_bundle(path: Path, representation: Representation) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": "cad_benchmark_embeddings_v1",
        "uids": representation.uids,
        "splits": representation.splits,
        "labels": representation.labels,
        "targets": representation.targets,
        "embeddings": {representation.name: representation.values},
        "metadata": representation.metadata,
    }, path)
