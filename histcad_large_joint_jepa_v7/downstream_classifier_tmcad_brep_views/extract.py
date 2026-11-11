"""Frozen v7 pooled features from original and geometry-balanced STEP graphs."""

from __future__ import annotations

import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
from balanced_step_graph import parse_balanced_step_graph
from common import device_for, save_json, source_manifest
from step_statements import PARSER_VERSION
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


class PairedGraphs(Dataset):
    def __init__(self, samples: list[dict], prepared: Path, source_root: Path,
                 alternative_cache: Path, vocabulary: dict[str, int], cap: int,
                 original_parser_version: str):
        self.samples, self.prepared = samples, prepared
        self.source_root, self.alternative_cache = source_root, alternative_cache
        self.vocabulary, self.cap = vocabulary, cap
        self.original_parser_version = original_parser_version

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[dict, dict]:
        sample = self.samples[index]
        original_payload = torch.load(self.prepared / sample["graph_relpath"],
                                      map_location="cpu", weights_only=True)
        if original_payload.get("parser_version") != self.original_parser_version:
            raise ValueError(f"Stale original STEP graph for {sample['uid']}")
        digest = sample["sha256"]
        path = self.alternative_cache / digest[:2] / f"{digest}.pt"
        if path.is_file():
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if (payload.get("selection") != "geometry_balanced_v1"
                    or payload.get("cap") != self.cap
                    or payload.get("parser_version") != PARSER_VERSION):
                raise ValueError(f"Stale alternate STEP graph: {path}")
            alternative = payload["graph"]
        else:
            alternative = parse_balanced_step_graph(
                self.source_root / sample["source_relpath"], self.vocabulary, self.cap,
            )
            if alternative["num_brep_candidates"] != sample["num_brep_candidates"]:
                raise ValueError(f"Alternate parser candidate count differs for {sample['uid']}")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.stem + f".{index}.partial.pt")
            torch.save({"selection": "geometry_balanced_v1", "cap": self.cap,
                        "parser_version": PARSER_VERSION, "graph": alternative}, temporary)
            temporary.replace(path)
        return original_payload["graph"], alternative


def _collate_pairs(batch: list[tuple[dict, dict]], collate):
    original, alternative = zip(*batch, strict=True)
    return collate(list(original)), collate(list(alternative))


def _frozen_encoder(config: dict, device: torch.device):
    root = config["paths"]["v7_root"]
    if not (root / "src/histcad_large_jepa/model.py").is_file():
        raise FileNotFoundError(f"JEPA-v7 source missing: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.histcad_large_jepa.model import StepEncoder
    from src.histcad_large_jepa.step_graph import collate_step_graphs

    checkpoint = torch.load(config["paths"]["checkpoint"], map_location="cpu",
                            weights_only=False, mmap=True)
    graph_cache = torch.load(config["paths"]["histcad_graph_cache"], map_location="cpu",
                             weights_only=False, mmap=True)
    if checkpoint.get("format_version") != 7 or graph_cache.get("format_version") != 7:
        raise ValueError("Expected JEPA-v7 checkpoint and graph cache")
    vocabulary = list(graph_cache["step_entity_types"])
    if list(checkpoint["step_entity_types"]) != vocabulary:
        raise ValueError("Frozen STEP vocabularies differ")
    settings = checkpoint["model_config"]
    if (settings["step_node_dim"], settings["step_output_dim"], settings["step_pooling_slots"]) != (512, 2048, 3):
        raise ValueError("Expected v7's 512-D nodes and 2048-D STEP output")
    encoder = StepEncoder(
        len(vocabulary), settings["step_node_dim"], settings["step_output_dim"],
        settings["step_depth"], settings["attention_heads"],
        settings["step_pooling_slots"], settings["dropout"],
    )
    weights = {key.removeprefix("step_encoder."): value
               for key, value in checkpoint["model"].items() if key.startswith("step_encoder.")}
    encoder.load_state_dict(weights, strict=True)
    encoder.eval().requires_grad_(False)
    encoder.to(device)
    epoch = int(checkpoint["epoch"])
    return encoder, collate_step_graphs, vocabulary, epoch


def extract_features(config: dict) -> Path:
    manifest, _, manifest_hash = source_manifest(config)
    samples = manifest["samples"]
    cap = int(config["alternative"]["max_step_instances"])
    if config["alternative"]["selection"] != "geometry_balanced_v1":
        raise ValueError("Only the audited geometry_balanced_v1 selection is implemented")
    if cap != 256:
        raise ValueError("This controlled view experiment holds the v7-trained 256-instance cap fixed")
    if int(config["extraction"]["num_workers"]) != 0:
        raise ValueError("Alternate graph caching requires num_workers=0 to avoid duplicate-file races")
    feature_root = config["paths"]["feature_cache"]
    feature_root.mkdir(parents=True, exist_ok=True)
    names = ("pooled.npy", "alternative.npy")
    paths = {name: feature_root / name for name in names}
    metadata_path = feature_root / "metadata.json"
    if metadata_path.is_file():
        old = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (old.get("format") == "tmcad_frozen_v7_brep_views_1"
                and old.get("manifest_sha256") == manifest_hash
                and old.get("checkpoint") == str(config["paths"]["checkpoint"])
                and old.get("alternative_cap") == cap
                and old.get("selection") == config["alternative"]["selection"]
                and old.get("storage_dtype") == config["extraction"]["storage_dtype"]
                and all(path.is_file() for path in paths.values())):
            if (np.load(paths["pooled.npy"], mmap_mode="r").shape == (len(samples), 2048)
                    and np.load(paths["alternative.npy"], mmap_mode="r").shape == (len(samples), 4, 512)):
                print("Frozen two-view cache already exists; reusing it.", flush=True)
                return metadata_path
    device = device_for(config["extraction"]["device"])
    encoder, collate, vocabulary, epoch = _frozen_encoder(config, device)
    dataset = PairedGraphs(
        samples, config["paths"]["tmcad_processed"], Path(manifest["dataset_root"]),
        feature_root / "balanced_graphs_256", dict(zip(vocabulary, range(len(vocabulary)), strict=True)),
        cap, manifest["parser_version"],
    )
    loader = DataLoader(dataset, batch_size=int(config["extraction"]["graph_batch_size"]),
                        shuffle=False, num_workers=0, collate_fn=partial(_collate_pairs, collate=collate),
                        pin_memory=device.type == "cuda")
    dtype_name = config["extraction"]["storage_dtype"]
    if dtype_name not in {"float16", "float32"}:
        raise ValueError("storage_dtype must be float16 or float32")
    storage_dtype = np.float16 if dtype_name == "float16" else np.float32
    temporary = {name: path.with_name(path.stem + ".partial" + path.suffix)
                 for name, path in paths.items()}
    count = len(samples)
    pooled = np.lib.format.open_memmap(temporary["pooled.npy"], mode="w+", dtype=storage_dtype,
                                        shape=(count, 2048))
    views = np.lib.format.open_memmap(temporary["alternative.npy"], mode="w+", dtype=storage_dtype,
                                       shape=(count, 4, 512))
    cursor = 0
    print(f"Frozen dual-view extraction: {count:,} TMCAD parts, two 256-instance graphs/part, "
          f"device={device}, v7_epoch={epoch}", flush=True)
    with torch.inference_mode():
        for batch_number, (original, alternative) in enumerate(loader, start=1):
            original = {key: value.to(device, non_blocking=True) for key, value in original.items()}
            alternative = {key: value.to(device, non_blocking=True) for key, value in alternative.items()}
            source = F.normalize(encoder(original).float(), dim=-1)
            complement = F.normalize(encoder(alternative).float(), dim=-1)
            size = len(source)
            if len(complement) != size or not torch.isfinite(source).all() or not torch.isfinite(complement).all():
                raise ValueError(f"Invalid frozen two-view embedding in batch {batch_number}")
            pooled[cursor:cursor + size] = source.cpu().numpy()
            views[cursor:cursor + size] = complement.reshape(size, 4, 512).cpu().numpy()
            cursor += size
            every = int(config["extraction"]["progress_every"])
            if batch_number % every == 0 or cursor == count:
                print(f"Frozen two-view extraction {cursor:,}/{count:,}", flush=True)
    if cursor != count:
        raise RuntimeError("Frozen extraction stopped before the end of the manifest")
    for array in (pooled, views):
        array.flush()
    del array, pooled, views
    for name, path in temporary.items():
        path.replace(paths[name])
    save_json(metadata_path, {
        "format": "tmcad_frozen_v7_brep_views_1",
        "manifest_sha256": manifest_hash,
        "checkpoint": str(config["paths"]["checkpoint"]),
        "checkpoint_epoch": epoch,
        "sample_count": count,
        "pooled_width": 2048,
        "alternative_shape": [4, 512],
        "alternative_cap": cap,
        "selection": "geometry_balanced_v1",
        "storage_dtype": dtype_name,
        "classes": manifest["classes"],
        "uids": [item["uid"] for item in samples],
        "labels": [manifest["classes"].index(item["label"]) for item in samples],
        "splits": [item["split"] for item in samples],
        "source_relpaths": [item["source_relpath"] for item in samples],
        "step_entity_types": vocabulary,
        "alternative_definition": "Four 512-D frozen pooled role tokens from a complementary 256-instance geometry-balanced STEP graph; not raw nodes.",
        "pooled_definition": "Original, unchanged normalized 2048-D v7 STEP graph embedding.",
    })
    print(f"Saved independent two-view features: {metadata_path}", flush=True)
    return metadata_path
