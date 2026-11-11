"""Memory-mapped frozen JEPA features with variable-length local STEP tokens."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from common import source_manifest
from torch.utils.data import Dataset


class FrozenLocalFeatures(Dataset):
    def __init__(self, config: dict, split: str, include_local: bool = True):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {split}")
        root = config["paths"]["feature_cache"]
        path = root / "metadata.json"
        if not path.is_file():
            raise FileNotFoundError("Run the frozen extraction stage before classifier training")
        self.metadata = json.loads(path.read_text(encoding="utf-8"))
        manifest, _, manifest_hash = source_manifest(config)
        meta = self.metadata
        if (meta.get("format") != "tmcad_frozen_v7_local_tokens_1"
                or meta.get("manifest_sha256") != manifest_hash
                or meta.get("checkpoint") != str(config["paths"]["checkpoint"])):
            raise ValueError("Local-token cache does not match this TMCAD manifest/checkpoint")
        if (meta["classes"] != manifest["classes"]
                or meta["uids"] != [item["uid"] for item in manifest["samples"]]):
            raise ValueError("Local-token cache labels or sample order differ from the manifest")
        self.root = root
        self.split = split
        self.include_local = include_local
        self.indices = [index for index, name in enumerate(meta["splits"]) if name == split]
        if not self.indices:
            raise ValueError(f"No {split} rows in local-token cache")
        self._arrays = None

    def _open(self) -> dict[str, np.ndarray]:
        if self._arrays is None:
            self._arrays = {
                name: np.load(self.root / f"{name}.npy", mmap_mode="r")
                for name in ("pooled", "nodes", "roles", "type_ids", "offsets")
            }
            values = self._arrays
            meta = self.metadata
            if (values["pooled"].shape != (meta["sample_count"], 2048)
                    or values["nodes"].shape != (meta["node_count"], 512)
                    or values["roles"].shape != (meta["node_count"],)
                    or values["type_ids"].shape != (meta["node_count"],)
                    or values["offsets"].shape != (meta["sample_count"] + 1,)):
                raise ValueError("Frozen local-token cache arrays have inconsistent shapes")
            if int(values["offsets"][0]) != 0 or int(values["offsets"][-1]) != meta["node_count"]:
                raise ValueError("Frozen local-token cache offsets are inconsistent")
        return self._arrays

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_arrays"] = None  # Each Windows worker opens its own read-only memmaps.
        return state

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        arrays = self._open()
        sample = self.indices[index]
        start, end = (int(arrays["offsets"][sample]), int(arrays["offsets"][sample + 1]))
        if end <= start:
            raise ValueError(f"Empty frozen node sequence at sample {sample}")
        if self.include_local:
            nodes = np.array(arrays["nodes"][start:end], dtype=np.float32)
            roles = np.array(arrays["roles"][start:end], dtype=np.int64)
            type_ids = np.array(arrays["type_ids"][start:end], dtype=np.int64)
        else:
            nodes = np.zeros((1, 512), dtype=np.float32)
            roles = np.zeros(1, dtype=np.int64)
            type_ids = np.full(1, -1, dtype=np.int64)
        return {
            "index": sample,
            "pooled": np.array(arrays["pooled"][sample], dtype=np.float32),
            "nodes": nodes,
            "roles": roles,
            "type_ids": type_ids,
            "label": self.metadata["labels"][sample],
        }


def collate_local(batch: list[dict]) -> dict[str, torch.Tensor]:
    count = len(batch)
    maximum = max(len(item["nodes"]) for item in batch)
    nodes = torch.zeros((count, maximum, 512), dtype=torch.float32)
    roles = torch.zeros((count, maximum), dtype=torch.long)
    type_ids = torch.full((count, maximum), -1, dtype=torch.long)
    valid = torch.zeros((count, maximum), dtype=torch.bool)
    for row, item in enumerate(batch):
        length = len(item["nodes"])
        nodes[row, :length] = torch.from_numpy(item["nodes"])
        roles[row, :length] = torch.from_numpy(item["roles"])
        type_ids[row, :length] = torch.from_numpy(item["type_ids"])
        valid[row, :length] = True
    return {
        "index": torch.tensor([item["index"] for item in batch], dtype=torch.long),
        "pooled": torch.stack([torch.from_numpy(item["pooled"]) for item in batch]),
        "nodes": nodes,
        "roles": roles,
        "type_ids": type_ids,
        "valid": valid,
        "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
    }
