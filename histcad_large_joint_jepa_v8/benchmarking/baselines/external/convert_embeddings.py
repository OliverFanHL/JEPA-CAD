"""Validate and convert externally extracted embeddings into the benchmark contract."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--values", type=Path, required=True, help=".npy rank-2 embedding array")
    parser.add_argument("--manifest", type=Path, required=True, help="JSON with uids and splits")
    parser.add_argument("--metadata", type=Path, required=True, help="JSON provenance contract")
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    provenance = json.loads(args.metadata.read_text(encoding="utf-8"))
    required = {
        "source_repository", "source_revision", "checkpoint", "checkpoint_sha256",
        "preprocessing", "training_data", "pretraining_objective", "parameter_count",
    }
    missing = sorted(key for key in required if provenance.get(key) in (None, ""))
    if missing:
        raise ValueError(f"External provenance is missing: {missing}")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", str(provenance["checkpoint_sha256"])):
        raise ValueError("checkpoint_sha256 must be a 64-character hexadecimal digest")
    if not isinstance(provenance["parameter_count"], int) or provenance["parameter_count"] <= 0:
        raise ValueError("parameter_count must be a positive integer")
    values = torch.from_numpy(np.asarray(np.load(args.values, mmap_mode="c")))
    uids, splits = list(manifest["uids"]), list(manifest["splits"])
    if values.ndim != 2 or len(values) != len(uids) or len(splits) != len(uids):
        raise ValueError("Embedding rows, UIDs, and splits must align exactly")
    if len(set(uids)) != len(uids) or set(splits) != {"train", "val", "test"}:
        raise ValueError("UIDs must be unique and all three splits must be present")
    if not torch.isfinite(values.float()).all():
        raise ValueError("External embeddings contain non-finite values")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": "cad_benchmark_embeddings_v1",
        "uids": uids, "splits": splits,
        "labels": manifest.get("labels", {}), "targets": manifest.get("targets", {}),
        "embeddings": {args.name: values}, "metadata": provenance,
    }, args.output)
    print(f"Saved validated {args.name} embeddings: {args.output}")


if __name__ == "__main__":
    main()
