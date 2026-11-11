"""Aligned frozen JEPA vectors and train/validation-only text targets."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import torch
import yaml
from sentence_transformers import SentenceTransformer
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
OPERATIONS = {"NewBody": 0, "Join": 1, "Cut": 2, "Intersect": 3}


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def config() -> dict:
    filename = os.environ.get("HISTCAD_STUDY_CONFIG", "config.yaml")
    return yaml.safe_load((ROOT / filename).read_text(encoding="utf-8"))


def teacher_text(value: str, mode: str) -> str:
    value = value.strip()
    if mode == "raw":
        return value
    if mode != "strip_dimensions_v1":
        raise ValueError(f"Unknown text preprocessing mode: {mode}")
    # Functional descriptions often contain exact sizes. Remove parenthetical
    # dimension phrases and standalone numerical measurements to isolate shape
    # and function words from explicit size supervision.
    value = re.sub(r"\([^)]*\)", " ", value)
    value = re.sub(r"\b\d+(?:[.,]\d+)?\s*(?:mm|cm|meters?|inches?|in|ft)?\b", " ", value, flags=re.I)
    value = re.sub(r"\b(?:mm|cm|meters?|inches?|ft)\b", " ", value, flags=re.I)
    return re.sub(r"\s+", " ", value).strip()


def payload(settings: dict) -> dict:
    source = torch.load(
        resolve(settings["paths"]["representations"]),
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    if source.get("format_version") != 7 or source["embeddings"]["joint"].shape[1] != 4096:
        raise ValueError("Expected aligned 4096-D v7 representation cache")
    return source


def split_indices(source: dict) -> dict[str, torch.Tensor]:
    return {
        name: torch.tensor(
            [index for index, item in enumerate(source["metadata"]) if item["split"] == name],
            dtype=torch.long,
        )
        for name in ("train", "val", "test")
    }


def geometry_targets(source: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dimensions = (source["dimension_targets"] - source["dimension_target_mean"]) / source[
        "dimension_target_std"
    ]
    structural = source["structural_metadata"]
    feature = torch.tensor([min(max(item["num_features"], 1), 4) - 1 for item in structural])
    operation = torch.tensor(
        [
            OPERATIONS.get(item["operation_signature"].rsplit(":", 1)[-1], 0)
            if item["operation_signature"]
            else 0
            for item in structural
        ]
    )
    return dimensions.float(), feature.long(), operation.long()


def prepare_text_targets(settings: dict, source: dict, splits: dict[str, torch.Tensor]) -> Path:
    """Text never enters the frozen JEPA encoder or test-time CAD retrieval."""
    output = resolve(settings["paths"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    path = output / "functional_text_teacher.pt"
    if path.exists():
        existing = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        if (
            existing["uids"] == source["uids"]
            and existing["model_id"] == settings["text_teacher"]["model_id"]
            and existing.get("preprocess", "raw") == settings["text_teacher"]["preprocess"]
        ):
            return path
    train_val = torch.cat((splits["train"], splits["val"])).tolist()
    descriptions = [
        teacher_text(source["metadata"][index].get("functional_type", ""), settings["text_teacher"]["preprocess"])
        for index in train_val
    ]
    unique = list(dict.fromkeys(text for text in descriptions if text))
    teacher_settings = settings["text_teacher"]
    model = SentenceTransformer(
        teacher_settings["model_id"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        local_files_only=True,
    )
    model.max_seq_length = teacher_settings["max_seq_length"]
    print(f"Encoding {len(unique):,} unique train/validation functional descriptions", flush=True)
    vectors = model.encode(
        unique,
        batch_size=teacher_settings["batch_size"],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    lookup = {description: index for index, description in enumerate(unique)}
    targets = torch.zeros(
        (len(source["uids"]), settings["model"]["output_dim"]), dtype=torch.float16
    )
    index_array = np.array([lookup.get(description, -1) for description in descriptions])
    valid = index_array >= 0
    targets[torch.tensor(train_val)[valid]] = torch.from_numpy(vectors[index_array[valid]]).half()
    targets = F.normalize(targets.float(), dim=-1).half()
    torch.save(
        {
            "uids": source["uids"],
            "model_id": teacher_settings["model_id"],
            "preprocess": teacher_settings["preprocess"],
            "text_field": "Functional Type",
            "targets": targets,
            "train_val_nonempty": int(valid.sum()),
            "train_val_total": len(train_val),
            "unique_texts": len(unique),
        },
        path,
    )
    (output / "teacher_audit.json").write_text(
        json.dumps(
            {
                "model_id": teacher_settings["model_id"],
                "preprocess": teacher_settings["preprocess"],
                "text_field": "Functional Type",
                "train_val_nonempty": int(valid.sum()),
                "train_val_total": len(train_val),
                "unique_texts": len(unique),
                "test_text_encoded": False,
                "jepa_encoder_frozen": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path
