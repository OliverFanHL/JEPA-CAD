"""Paths and reproducibility for the standalone local-token downstream run."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent


def load_config(path: Path) -> dict:
    path = path.resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["paths"] = {
        key: (Path(value) if Path(value).is_absolute() else path.parent / value).resolve()
        for key, value in config["paths"].items()
    }
    return config


def device_for(value: str) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if value == "auto" else torch.device(value)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def source_manifest(config: dict) -> tuple[dict, Path, str]:
    path = config["paths"]["tmcad_processed"] / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"TMCAD prepared manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != "tmcad_v7_step_only_1":
        raise ValueError("Expected the prepared v7 TMCAD STEP manifest")
    if str(config["paths"]["checkpoint"]) != manifest.get("source_checkpoint"):
        raise ValueError("The prepared TMCAD manifest was made for a different v7 checkpoint")
    samples = manifest.get("samples", [])
    if not samples or any(sample.get("split") not in {"train", "val", "test"} for sample in samples):
        raise ValueError("Prepared TMCAD manifest is empty or has invalid splits")
    return manifest, path, file_sha256(path)
