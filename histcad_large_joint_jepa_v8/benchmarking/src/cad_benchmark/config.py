from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _resolve_paths(value: Any, base: Path, key: str = "") -> Any:
    if isinstance(value, dict):
        return {name: _resolve_paths(item, base, name) for name, item in value.items()}
    if isinstance(value, list):
        return [_resolve_paths(item, base, key) for item in value]
    if isinstance(value, str) and (
        key.endswith(("path", "file", "dir", "root", "metadata", "array", "checkpoint", "results", "artifact", "csv"))
        or key in {"artifact", "output", "teacher_targets", "training_summary", "metrics_csv"}
    ):
        candidate = Path(value)
        return candidate if candidate.is_absolute() else (base / candidate).resolve()
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Benchmark configuration must be a mapping: {path}")
    config = _resolve_paths(copy.deepcopy(raw), path.parent)
    config["_config_path"] = path
    config["_root"] = path.parent
    return config


def output_path(config: dict[str, Any], *parts: str) -> Path:
    path = Path(config["paths"]["output_dir"]).joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
