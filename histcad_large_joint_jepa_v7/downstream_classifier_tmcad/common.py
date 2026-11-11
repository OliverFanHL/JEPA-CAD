"""Configuration and frozen-v7 import helpers for the TMCAD experiment."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent


def load_config(path: Path) -> dict:
    path = path.resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = path.parent
    config["paths"] = {
        name: (Path(value) if Path(value).is_absolute() else base / value).resolve()
        for name, value in config["paths"].items()
    }
    return config


def v7_modules(config: dict):
    root = config["paths"]["v7_root"]
    if not (root / "src/histcad_large_jepa/model.py").is_file():
        raise FileNotFoundError(f"JEPA-v7 source not found at {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from step_statements import install_step_matcher

    from src.histcad_large_jepa import step_graph
    from src.histcad_large_jepa.model import StepEncoder

    install_step_matcher(step_graph)
    return StepEncoder, step_graph.collate_step_graphs, step_graph.parse_step_graph


def read_pretrained_metadata(config: dict) -> tuple[dict, tuple[str, ...]]:
    checkpoint_path = config["paths"]["checkpoint"]
    cache_path = config["paths"]["histcad_graph_cache"]
    if checkpoint_path.name != "best.pt":
        raise ValueError("Frozen extraction must use the v7 best.pt checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False, mmap=True)
    if checkpoint.get("format_version") != 7 or cache.get("format_version") != 7:
        raise ValueError("The source checkpoint and graph cache must both be JEPA-v7")
    vocabulary = tuple(cache["step_entity_types"])
    if tuple(checkpoint["step_entity_types"]) != vocabulary:
        raise ValueError("The frozen checkpoint and graph cache use different STEP vocabularies")
    summary = {"model_config": checkpoint["model_config"], "epoch": int(checkpoint["epoch"]),
               "checkpoint": str(checkpoint_path), "vocabulary_size": len(vocabulary)}
    return summary, vocabulary


def device_for(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
