"""Fixed-shape frozen original and complementary STEP summary views."""

from __future__ import annotations

import json

import numpy as np
import torch
from common import source_manifest
from geometry_features import FEATURE_NAMES, geometry_cache
from torch.utils.data import Dataset


class FrozenViews(Dataset):
    def __init__(self, config: dict, split: str, include_alternative: bool = True,
                 include_geometry: bool = False):
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split: {split}")
        root = config["paths"]["feature_cache"]
        metadata_path = root / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError("Run frozen two-view extraction before classification")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        manifest, _, manifest_hash = source_manifest(config)
        meta = self.metadata
        if (meta.get("format") != "tmcad_frozen_v7_brep_views_1"
                or meta.get("manifest_sha256") != manifest_hash
                or meta.get("checkpoint") != str(config["paths"]["checkpoint"])
                or meta.get("alternative_cap") != config["alternative"]["max_step_instances"]):
            raise ValueError("Frozen two-view cache does not match configuration/manifest")
        if (meta["classes"] != manifest["classes"]
                or meta["uids"] != [item["uid"] for item in manifest["samples"]]):
            raise ValueError("Frozen two-view sample order differs from the prepared manifest")
        self.root = root
        self.include_alternative = include_alternative
        self.include_geometry = include_geometry
        self.geometry_path, expected_geometry = geometry_cache(config)
        if include_geometry:
            metadata_path = self.geometry_path.with_name("geometry_metadata.json")
            if not self.geometry_path.is_file() or not metadata_path.is_file():
                raise FileNotFoundError("Run full-file geometry extraction before geometry classification")
            saved_geometry = json.loads(metadata_path.read_text(encoding="utf-8"))
            if any(saved_geometry.get(key) != value for key, value in expected_geometry.items()):
                raise ValueError("Geometry descriptors do not match TMCAD manifest/parser")
        self.indices = [i for i, name in enumerate(meta["splits"]) if name == split]
        if not self.indices:
            raise ValueError(f"No {split} samples")
        self._pooled = None
        self._alternative = None
        self._geometry = None

    def _open(self) -> None:
        if self._pooled is not None:
            return
        self._pooled = np.load(self.root / "pooled.npy", mmap_mode="r")
        self._alternative = np.load(self.root / "alternative.npy", mmap_mode="r")
        n = self.metadata["sample_count"]
        if self._pooled.shape != (n, 2048) or self._alternative.shape != (n, 4, 512):
            raise ValueError("Frozen two-view arrays have invalid widths or sample counts")
        if self.include_geometry:
            self._geometry = np.load(self.geometry_path, mmap_mode="r")
            if self._geometry.shape != (n, len(FEATURE_NAMES)):
                raise ValueError("Geometry descriptor cache has invalid shape")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_pooled"] = None
        state["_alternative"] = None
        state["_geometry"] = None
        return state

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        self._open()
        sample = self.indices[index]
        alternative = (np.array(self._alternative[sample], dtype=np.float32)
                       if self.include_alternative else np.zeros((4, 512), dtype=np.float32))
        geometry = (np.array(self._geometry[sample], dtype=np.float32)
                    if self.include_geometry else np.zeros(len(FEATURE_NAMES), dtype=np.float32))
        return {
            "index": torch.tensor(sample, dtype=torch.long),
            "pooled": torch.from_numpy(np.array(self._pooled[sample], dtype=np.float32)),
            "alternative": torch.from_numpy(alternative),
            "geometry": torch.from_numpy(geometry),
            "label": torch.tensor(self.metadata["labels"][sample], dtype=torch.long),
        }
