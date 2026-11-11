"""Strict, read-only access to the existing TMCAD split and frozen caches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from model import (GEOMETRY_WIDTH, MODEL_NAME, SURFACE_WIDTH,
                   FrozenMeshClassifier)
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
CACHE_FORMAT = "tmcad_mesh_residual_frozen_inputs_v1"


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
        name: (Path(value) if Path(value).is_absolute() else path.parent / value).resolve()
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
    def __init__(self, config: dict):
        self.config = config
        paths = config["paths"]
        required = ("baseline_checkpoint", "manifest", "geometry_cache",
                    "geometry_metadata", "mesh_cache", "mesh_metadata")
        for key in required:
            if not paths[key].is_file():
                raise FileNotFoundError(f"Required input {key}: {paths[key]}")
        feature_root = paths["frozen_views"]
        for name in ("metadata.json", "pooled.npy", "alternative.npy"):
            if not (feature_root / name).is_file():
                raise FileNotFoundError(feature_root / name)
        self.manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        self.metadata = json.loads((feature_root / "metadata.json").read_text(encoding="utf-8"))
        geometry_metadata = json.loads(paths["geometry_metadata"].read_text(encoding="utf-8"))
        mesh_metadata = json.loads(paths["mesh_metadata"].read_text(encoding="utf-8"))
        if self.manifest.get("format") != "tmcad_v7_step_only_1":
            raise ValueError("Unexpected TMCAD manifest format")
        samples = self.manifest["samples"]
        self.classes = list(self.manifest["classes"])
        self.count = len(samples)
        expected_uids = [sample["uid"] for sample in samples]
        manifest_hash = sha256(paths["manifest"])
        meta = self.metadata
        if (meta.get("format") != "tmcad_frozen_v7_brep_views_1"
                or meta.get("manifest_sha256") != manifest_hash
                or meta.get("sample_count") != self.count
                or meta.get("classes") != self.classes
                or meta.get("uids") != expected_uids
                or meta.get("splits") != [sample["split"] for sample in samples]
                or meta.get("checkpoint") != self.manifest.get("source_checkpoint")
                or meta.get("alternative_cap") != 256):
            raise ValueError("Frozen two-view cache differs from manifest or v7 model")
        self.labels = np.asarray(meta["labels"], dtype=np.int64)
        if (self.labels.shape != (self.count,)
                or np.any(self.labels < 0) or np.any(self.labels >= len(self.classes))
                or any(self.classes[int(label)] != sample["label"]
                       for label, sample in zip(self.labels, samples))):
            raise ValueError("Frozen labels differ from the TMCAD manifest")
        for name, descriptor_meta, expected_format, width in (
            ("geometry", geometry_metadata, "tmcad_full_step_geometry_v1", GEOMETRY_WIDTH),
            ("mesh", mesh_metadata, "tmcad_occ_surface_profile_v2", SURFACE_WIDTH),
        ):
            if (descriptor_meta.get("format") != expected_format
                    or descriptor_meta.get("manifest_sha256") != manifest_hash
                    or descriptor_meta.get("sample_count") != self.count
                    or descriptor_meta.get("uids") != expected_uids
                    or len(descriptor_meta.get("feature_names", [])) != width):
                raise ValueError(f"{name} descriptors differ from the prepared split")
        if (float(mesh_metadata.get("relative_deflection", -1)) != 0.01
                or int(mesh_metadata.get("max_triangles_per_part", -1)) != 150000):
            raise ValueError("Expected the current v7 mesh descriptor settings")
        self.pooled = np.load(feature_root / "pooled.npy", mmap_mode="r")
        self.alternative = np.load(feature_root / "alternative.npy", mmap_mode="r")
        self.geometry = np.load(paths["geometry_cache"], mmap_mode="r")
        self.mesh = np.load(paths["mesh_cache"], mmap_mode="r")
        for name, actual, expected in (
            ("pooled", self.pooled.shape, (self.count, 2048)),
            ("alternative", self.alternative.shape, (self.count, 4, 512)),
            ("geometry", self.geometry.shape, (self.count, GEOMETRY_WIDTH)),
            ("mesh", self.mesh.shape, (self.count, SURFACE_WIDTH)),
        ):
            if actual != expected:
                raise ValueError(f"{name} array shape {actual}; expected {expected}")
        self.indices = {
            split: np.flatnonzero(np.asarray(meta["splits"]) == split)
            for split in ("train", "val", "test")
        }
        if any(len(value) == 0 for value in self.indices.values()):
            raise ValueError("A TMCAD split is empty")
        self.provenance = {
            "format": CACHE_FORMAT,
            "baseline_checkpoint_sha256": sha256(paths["baseline_checkpoint"]),
            "manifest_sha256": manifest_hash,
            "frozen_view_metadata_sha256": sha256(feature_root / "metadata.json"),
            "geometry_cache_sha256": sha256(paths["geometry_cache"]),
            "mesh_cache_sha256": sha256(paths["mesh_cache"]),
        }

    def baseline(self, device: torch.device) -> FrozenMeshClassifier:
        saved = torch.load(self.config["paths"]["baseline_checkpoint"],
                           map_location="cpu", weights_only=False)
        if (saved.get("architecture") != MODEL_NAME
                or saved.get("classes") != self.classes
                or saved.get("manifest_sha256") != self.provenance["manifest_sha256"]
                or saved.get("feature_format") != self.metadata["format"]
                or saved.get("source_checkpoint") != self.metadata["checkpoint"]
                or saved.get("mesh_sha256") != self.provenance["mesh_cache_sha256"]):
            raise ValueError("Saved baseline classifier is not compatible with these caches")
        settings = saved["settings"]
        model = FrozenMeshClassifier(
            len(self.classes), int(settings["attention_heads"]),
            int(settings["hidden_width"]), float(settings["dropout"]),
        )
        model.load_state_dict(saved["state_dict"], strict=True)
        model.requires_grad_(False)
        return model.to(device).eval()

    def sample_path(self, global_index: int) -> Path:
        root = Path(self.manifest["dataset_root"]).resolve()
        path = (root / self.metadata["source_relpaths"][global_index]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("STEP sample path escaped the TMCAD directory")
        return path


class InputRows(Dataset):
    def __init__(self, source: TMCADSource, split: str):
        if split not in source.indices:
            raise ValueError(f"Unknown split {split}")
        self.source = source
        self.indices = source.indices[split]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, local_index: int) -> dict[str, torch.Tensor]:
        i = int(self.indices[local_index])
        source = self.source
        return {
            "index": torch.tensor(i, dtype=torch.long),
            "pooled": torch.from_numpy(np.array(source.pooled[i], dtype=np.float32)),
            "alternative": torch.from_numpy(np.array(source.alternative[i], dtype=np.float32)),
            "geometry": torch.from_numpy(np.array(source.geometry[i], dtype=np.float32)),
            "mesh": torch.from_numpy(np.array(source.mesh[i], dtype=np.float32)),
            "label": torch.tensor(int(source.labels[i]), dtype=torch.long),
        }


def _cache_paths(source: TMCADSource, split: str) -> tuple[Path, Path]:
    root = source.config["paths"]["processed"]
    return root / f"{split}_frozen_inputs.npz", root / f"{split}_frozen_inputs.json"


@torch.inference_mode()
def frozen_split(source: TMCADSource, split: str, device: torch.device) -> dict[str, np.ndarray]:
    """Materialize only a requested split; train() never extracts test rows."""
    cache_path, metadata_path = _cache_paths(source, split)
    expected = {**source.provenance, "split": split,
                "sample_count": int(len(source.indices[split])),
                "classes": source.classes}
    if cache_path.is_file() and metadata_path.is_file():
        saved = json.loads(metadata_path.read_text(encoding="utf-8"))
        if saved == expected:
            with np.load(cache_path, allow_pickle=False) as handle:
                result = {name: handle[name] for name in
                          ("baseline_logits", "hidden", "geometry", "mesh", "label", "index")}
            _validate_split(result, source, split)
            print(f"Reusing immutable {split} frozen baseline inputs: {cache_path}", flush=True)
            return result
    model = source.baseline(device)
    dataset = InputRows(source, split)
    settings = source.config["baseline"]
    loader = DataLoader(dataset, batch_size=int(settings["batch_size"]),
                        shuffle=False, num_workers=int(settings["num_workers"]),
                        pin_memory=device.type == "cuda")
    collected: dict[str, list[np.ndarray]] = {
        name: [] for name in ("baseline_logits", "hidden", "geometry", "mesh", "label", "index")
    }
    for batch_number, original in enumerate(loader, 1):
        batch = {name: value.to(device, non_blocking=True)
                 for name, value in original.items() if name not in {"index", "label"}}
        logits, hidden = model(batch)
        collected["baseline_logits"].append(logits.float().cpu().numpy())
        collected["hidden"].append(hidden.float().cpu().numpy())
        for name in ("geometry", "mesh", "label", "index"):
            collected[name].append(original[name].numpy())
        if batch_number % 25 == 0 or batch_number == len(loader):
            print(f"Frozen baseline {split}: {batch_number}/{len(loader)} batches", flush=True)
    result = {name: np.concatenate(chunks) for name, chunks in collected.items()}
    _validate_split(result, source, split)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(cache_path.stem + ".partial.npz")
    np.savez_compressed(temporary, **result)
    temporary.replace(cache_path)
    save_json(metadata_path, expected)
    print(f"Saved {split} frozen baseline inputs: {cache_path}", flush=True)
    return result


def _validate_split(result: dict[str, np.ndarray], source: TMCADSource,
                    split: str) -> None:
    n = len(source.indices[split])
    expected = {
        "baseline_logits": (n, len(source.classes)),
        "hidden": (n, 256), "geometry": (n, GEOMETRY_WIDTH),
        "mesh": (n, SURFACE_WIDTH), "label": (n,), "index": (n,),
    }
    if any(result[name].shape != shape for name, shape in expected.items()):
        raise ValueError(f"Cached {split} baseline inputs have incorrect array shapes")
    if (not np.array_equal(result["index"], source.indices[split])
            or not np.array_equal(result["label"], source.labels[source.indices[split]])):
        raise ValueError(f"Cached {split} labels/order differ from the TMCAD manifest")
    if any(not np.isfinite(result[name]).all() for name in
           ("baseline_logits", "hidden", "geometry", "mesh")):
        raise ValueError(f"Nonfinite values in the {split} frozen input cache")
