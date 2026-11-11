from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .io import Representation, load_representation


class Registry:
    def __init__(self, config: dict[str, Any], apply_shared_coverage: bool = False):
        self.specs = {item["name"]: item for item in config["representations"]}
        if len(self.specs) != len(config["representations"]):
            raise ValueError("Representation names must be unique")
        self.coverage: dict[str, dict[str, Any]] = {}
        if apply_shared_coverage:
            for dataset, settings in config.get("shared_coverage", {}).items():
                path = Path(settings["file"])
                if not path.is_file():
                    continue
                manifest = json.loads(path.read_text(encoding="utf-8"))
                omitted = manifest["omitted_uids"]
                if (manifest["dataset"] != dataset or len(set(omitted)) != len(omitted)
                        or manifest["source_rows"] != manifest["included_rows"] + len(omitted)
                        or manifest["omitted_rows"] != len(omitted)):
                    raise ValueError(f"Invalid shared coverage manifest: {path}")
                self.coverage[dataset] = {**manifest, "file": str(path), "_omitted": set(omitted)}

    def names(self, task: str, available_only: bool = True) -> list[str]:
        output = []
        for name, spec in self.specs.items():
            if task not in spec.get("tasks", []):
                continue
            if available_only and not self.available(name):
                continue
            output.append(name)
        return output

    def available(self, name: str) -> bool:
        spec = self.specs[name]
        paths = [spec.get(key) for key in ("artifact", "reference_artifact", "metadata", "array") if spec.get(key)]
        return bool(paths) and all(Path(path).is_file() for path in paths)

    def load(self, name: str) -> Representation:
        if name not in self.specs:
            raise KeyError(name)
        if not self.available(name):
            raise FileNotFoundError(f"Representation {name!r} is not materialized")
        representation = load_representation(self.specs[name])
        coverage = self.coverage.get(self.specs[name]["dataset"])
        if not coverage or not coverage["_omitted"]:
            return representation
        retained = [index for index, uid in enumerate(representation.uids)
                    if uid not in coverage["_omitted"]]
        if len(retained) == len(representation.uids):
            return representation
        indices = torch.tensor(retained, dtype=torch.long)
        return Representation(
            representation.name, representation.family, representation.values[indices],
            [representation.uids[index] for index in retained],
            [representation.splits[index] for index in retained],
            {key: [values[index] for index in retained]
             for key, values in representation.labels.items()},
            {key: values[indices] for key, values in representation.targets.items()},
            representation.metadata,
        )

    def coverage_summary(self) -> dict[str, dict[str, Any]]:
        return {
            dataset: {key: value for key, value in entry.items()
                      if key not in {"_omitted", "omitted_uids"}}
            for dataset, entry in self.coverage.items()
        }

    def inventory(self) -> list[dict[str, Any]]:
        rows = []
        for name, spec in self.specs.items():
            available = self.available(name)
            row = {
                "name": name,
                "family": spec["family"],
                "dataset": spec["dataset"],
                "available": available,
                "required": bool(spec.get("required", False)),
                "reuse": bool(spec.get("reuse", False)),
                "tasks": spec.get("tasks", []),
                "comparability": spec.get("comparability", ""),
                "prepare": spec.get("prepare"),
            }
            if available:
                representation = load_representation(spec)
                omitted = self.coverage.get(spec["dataset"], {}).get("_omitted", set())
                row.update(rows=sum(uid not in omitted for uid in representation.uids),
                           source_rows=len(representation.uids), dimensions=representation.dim)
            rows.append(row)
        return rows

    def validate_alignment(self, task: str) -> None:
        names = self.names(task)
        if not names:
            raise ValueError(f"No available representations for task {task}")
        datasets: dict[str, list[str]] = {}
        for name in names:
            datasets.setdefault(self.specs[name]["dataset"], []).append(name)
        for dataset, group in datasets.items():
            reference = self.load(group[0])
            for name in group[1:]:
                candidate = self.load(name)
                if candidate.uids != reference.uids or candidate.splits != reference.splits:
                    raise ValueError(
                        f"{task}/{dataset}: {name} is not exactly UID/split aligned with {group[0]}"
                    )
