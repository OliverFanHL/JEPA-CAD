"""Validated, read-only TMCAD frozen-JEPA and physical-feature inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parent
GEOMETRY_WIDTH = 37
SURFACE_WIDTH = 84
SOURCE_FORMAT = "tmcad_frozen_v7_brep_views_1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict:
    path = path.resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["paths"] = {
        name: (Path(value) if Path(value).is_absolute()
               else path.parent / value).resolve()
        for name, value in config["paths"].items()
    }
    return config


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def device_for(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


class TMCADSource:
    """Exact existing TMCAD split; no original 256-D classifier is required."""

    def __init__(self, config: dict):
        self.config = config
        paths = config["paths"]
        for name in ("manifest", "geometry_cache", "geometry_metadata",
                     "mesh_cache", "mesh_metadata"):
            if not paths[name].is_file():
                raise FileNotFoundError(f"Required {name}: {paths[name]}")
        root = paths["frozen_views"]
        for name in ("metadata.json", "pooled.npy", "alternative.npy"):
            if not (root / name).is_file():
                raise FileNotFoundError(root / name)
        self.manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        self.metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        geometry_meta = json.loads(paths["geometry_metadata"].read_text(encoding="utf-8"))
        mesh_meta = json.loads(paths["mesh_metadata"].read_text(encoding="utf-8"))
        if self.manifest.get("format") != "tmcad_v7_step_only_1":
            raise ValueError("Unexpected TMCAD manifest format")
        samples = self.manifest["samples"]
        self.classes = list(self.manifest["classes"])
        self.count = len(samples)
        uids = [sample["uid"] for sample in samples]
        manifest_hash = sha256(paths["manifest"])
        meta = self.metadata
        if (meta.get("format") != SOURCE_FORMAT
                or meta.get("manifest_sha256") != manifest_hash
                or meta.get("sample_count") != self.count
                or meta.get("classes") != self.classes
                or meta.get("uids") != uids
                or meta.get("splits") != [sample["split"] for sample in samples]
                or meta.get("checkpoint") != self.manifest.get("source_checkpoint")
                or meta.get("alternative_cap") != 256):
            raise ValueError("Frozen JEPA views are inconsistent with the TMCAD manifest")
        self.labels = np.asarray(meta["labels"], dtype=np.int64)
        if (self.labels.shape != (self.count,)
                or np.any(self.labels < 0)
                or np.any(self.labels >= len(self.classes))
                or any(self.classes[int(label)] != sample["label"]
                       for label, sample in zip(self.labels, samples, strict=True))):
            raise ValueError("Cached labels differ from the TMCAD manifest")
        for name, descriptor, fmt, width in (
            ("geometry", geometry_meta, "tmcad_full_step_geometry_v1", GEOMETRY_WIDTH),
            ("mesh", mesh_meta, "tmcad_occ_surface_profile_v2", SURFACE_WIDTH),
        ):
            if (descriptor.get("format") != fmt
                    or descriptor.get("manifest_sha256") != manifest_hash
                    or descriptor.get("sample_count") != self.count
                    or descriptor.get("uids") != uids
                    or len(descriptor.get("feature_names", [])) != width):
                raise ValueError(f"{name} descriptor cache does not match the split")
        if (float(mesh_meta.get("relative_deflection", -1)) != 0.01
                or int(mesh_meta.get("max_triangles_per_part", -1)) != 150000):
            raise ValueError("Unexpected cached surface-profile extraction settings")
        self.pooled = np.load(root / "pooled.npy", mmap_mode="r")
        self.alternative = np.load(root / "alternative.npy", mmap_mode="r")
        self.geometry = np.load(paths["geometry_cache"], mmap_mode="r")
        self.mesh = np.load(paths["mesh_cache"], mmap_mode="r")
        for name, actual, expected in (
            ("pooled", self.pooled.shape, (self.count, 2048)),
            ("alternative", self.alternative.shape, (self.count, 4, 512)),
            ("geometry", self.geometry.shape, (self.count, GEOMETRY_WIDTH)),
            ("mesh", self.mesh.shape, (self.count, SURFACE_WIDTH)),
        ):
            if actual != expected:
                raise ValueError(f"{name} cache shape {actual}; expected {expected}")
        splits = np.asarray(meta["splits"])
        self.indices = {name: np.flatnonzero(splits == name)
                        for name in ("train", "val", "test")}
        if any(len(indices) == 0 for indices in self.indices.values()):
            raise ValueError("A TMCAD split is empty")
        self.provenance = {
            "format": "tmcad_512_196_inputs_v1",
            "manifest_sha256": manifest_hash,
            "frozen_view_metadata_sha256": sha256(root / "metadata.json"),
            "geometry_cache_sha256": sha256(paths["geometry_cache"]),
            "mesh_cache_sha256": sha256(paths["mesh_cache"]),
        }

    def sample_path(self, global_index: int) -> Path:
        root = Path(self.manifest["dataset_root"]).resolve()
        path = (root / self.metadata["source_relpaths"][global_index]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("STEP sample path escaped TMCAD dataset root")
        return path


class InputRows(Dataset):
    def __init__(self, source: TMCADSource, split: str):
        if split not in source.indices:
            raise ValueError(f"Unknown split: {split}")
        self.source = source
        self.indices = source.indices[split]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, local_index: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[local_index])
        source = self.source
        return {
            "index": torch.tensor(index, dtype=torch.long),
            "pooled": torch.from_numpy(np.array(source.pooled[index], dtype=np.float32)),
            "alternative": torch.from_numpy(np.array(source.alternative[index], dtype=np.float32)),
            "geometry": torch.from_numpy(np.array(source.geometry[index], dtype=np.float32)),
            "mesh": torch.from_numpy(np.array(source.mesh[index], dtype=np.float32)),
            "label": torch.tensor(int(source.labels[index]), dtype=torch.long),
        }
