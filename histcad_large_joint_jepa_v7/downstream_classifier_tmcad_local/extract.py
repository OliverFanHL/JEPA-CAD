"""Save every frozen post-GNN STEP node, before v7 role-aware pooling.

The pooled 2048-D vector is also saved unchanged. The forward hook observes
StepEncoder.node_norm, immediately upstream of its four-slot summary pool;
it neither changes encoder weights nor duplicates the encoder computation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from common import device_for, save_json, source_manifest
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


class PreparedGraphs(Dataset):
    def __init__(self, samples: list[dict], processed: Path, parser_version: str):
        self.samples, self.processed = samples, processed
        self.parser_version = parser_version

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        payload = torch.load(self.processed / self.samples[index]["graph_relpath"],
                             map_location="cpu", weights_only=True)
        if payload.get("parser_version") != self.parser_version:
            raise ValueError(f"Stale prepared STEP graph at sample {index}")
        return payload["graph"]


def _encoder(config: dict, device: torch.device):
    v7_root = config["paths"]["v7_root"]
    if not (v7_root / "src/histcad_large_jepa/model.py").is_file():
        raise FileNotFoundError(f"JEPA-v7 source is missing: {v7_root}")
    if str(v7_root) not in sys.path:
        sys.path.insert(0, str(v7_root))
    from src.histcad_large_jepa.model import StepEncoder
    from src.histcad_large_jepa.step_graph import collate_step_graphs

    checkpoint = torch.load(config["paths"]["checkpoint"], map_location="cpu",
                            weights_only=False, mmap=True)
    graph_cache = torch.load(config["paths"]["histcad_graph_cache"], map_location="cpu",
                             weights_only=False, mmap=True)
    if checkpoint.get("format_version") != 7 or graph_cache.get("format_version") != 7:
        raise ValueError("Expected v7 checkpoint and v7 HistCAD graph cache")
    vocabulary = list(graph_cache["step_entity_types"])
    if list(checkpoint["step_entity_types"]) != vocabulary:
        raise ValueError("Checkpoint and graph-cache STEP vocabularies differ")
    settings = checkpoint["model_config"]
    if (settings["step_node_dim"], settings["step_output_dim"], settings["step_pooling_slots"]) != (512, 2048, 3):
        raise ValueError("This experiment expects v7's 512-D nodes and 4 x 512-D pooled STEP output")
    model = StepEncoder(
        len(vocabulary), settings["step_node_dim"], settings["step_output_dim"],
        settings["step_depth"], settings["attention_heads"],
        settings["step_pooling_slots"], settings["dropout"],
    )
    weights = {key.removeprefix("step_encoder."): value
               for key, value in checkpoint["model"].items() if key.startswith("step_encoder.")}
    model.load_state_dict(weights, strict=True)
    epoch = int(checkpoint["epoch"])
    del checkpoint, graph_cache, weights
    model.eval().requires_grad_(False)
    model.to(device)
    return model, collate_step_graphs, vocabulary, epoch


def _paths(config: dict) -> dict[str, Path]:
    root = config["paths"]["feature_cache"]
    return {name: root / name for name in
            ("metadata.json", "pooled.npy", "nodes.npy", "roles.npy", "type_ids.npy", "offsets.npy")}


def extract_features(config: dict) -> Path:
    manifest, _, manifest_hash = source_manifest(config)
    samples = manifest["samples"]
    paths = _paths(config)
    paths["metadata.json"].parent.mkdir(parents=True, exist_ok=True)
    if paths["metadata.json"].is_file():
        import json
        existing = json.loads(paths["metadata.json"].read_text(encoding="utf-8"))
        if (existing.get("manifest_sha256") == manifest_hash
                and existing.get("checkpoint") == str(config["paths"]["checkpoint"])
                and existing.get("storage_dtype") == config["extraction"]["storage_dtype"]
                and all(path.is_file() for path in paths.values())):
            shapes = {name: np.load(path, mmap_mode="r").shape
                      for name, path in paths.items() if path.suffix == ".npy"}
            if (shapes["pooled.npy"] == (len(samples), 2048)
                    and shapes["nodes.npy"] == (existing.get("node_count"), 512)
                    and shapes["roles.npy"] == (existing.get("node_count"),)
                    and shapes["type_ids.npy"] == (existing.get("node_count"),)
                    and shapes["offsets.npy"] == (len(samples) + 1,)):
                print("Frozen local-token cache already exists; reusing it.", flush=True)
                return paths["metadata.json"]

    node_counts = np.fromiter(
        (1 + int(item["num_step_types"]) + int(item["num_brep_instances"]) for item in samples),
        dtype=np.int64, count=len(samples),
    )
    if (node_counts <= 1).any():
        raise ValueError("At least one prepared STEP graph has no local nodes")
    offsets = np.empty(len(samples) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(node_counts, out=offsets[1:])
    total_nodes = int(offsets[-1])
    dtype_name = config["extraction"]["storage_dtype"]
    if dtype_name not in {"float16", "float32"}:
        raise ValueError("storage_dtype must be float16 or float32")
    storage_dtype = np.float16 if dtype_name == "float16" else np.float32
    device = device_for(config["extraction"]["device"])
    encoder, collate, vocabulary, epoch = _encoder(config, device)
    loader = DataLoader(
        PreparedGraphs(samples, config["paths"]["tmcad_processed"], manifest["parser_version"]),
        batch_size=int(config["extraction"]["graph_batch_size"]), shuffle=False,
        num_workers=int(config["extraction"]["num_workers"]), collate_fn=collate,
        pin_memory=device.type == "cuda",
    )
    temporary = {name: path.with_name(path.stem + ".partial" + path.suffix)
                 for name, path in paths.items() if path.suffix == ".npy"}
    pooled = np.lib.format.open_memmap(temporary["pooled.npy"], mode="w+", dtype=storage_dtype,
                                        shape=(len(samples), 2048))
    nodes = np.lib.format.open_memmap(temporary["nodes.npy"], mode="w+", dtype=storage_dtype,
                                       shape=(total_nodes, 512))
    roles = np.lib.format.open_memmap(temporary["roles.npy"], mode="w+", dtype=np.uint8,
                                       shape=(total_nodes,))
    type_ids = np.lib.format.open_memmap(temporary["type_ids.npy"], mode="w+", dtype=np.int32,
                                          shape=(total_nodes,))
    captures: dict[str, torch.Tensor] = {}

    def capture_nodes(_module, _inputs, output):
        captures["nodes"] = output

    hook = encoder.node_norm.register_forward_hook(capture_nodes)
    cursor = 0
    print(f"Frozen v7 extraction: {len(samples):,} parts, {total_nodes:,} unpooled nodes, "
          f"2048-D pooled + 512-D/node, device={device}, checkpoint_epoch={epoch}", flush=True)
    try:
        with torch.inference_mode():
            for batch_number, batch in enumerate(loader, start=1):
                graph = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                part_count = len(graph["global_index"])
                end = cursor + part_count
                local_start, local_end = int(offsets[cursor]), int(offsets[end])
                if len(graph["node_type"]) != local_end - local_start:
                    raise ValueError(f"Cached node count differs from manifest at sample {cursor}")
                pooled_values = F.normalize(encoder(graph).float(), dim=-1)
                local_values = captures.pop("nodes", None)
                if local_values is None or local_values.shape != (local_end - local_start, 512):
                    raise RuntimeError("Pre-pool node hook did not capture the expected encoder states")
                if not torch.isfinite(pooled_values).all() or not torch.isfinite(local_values).all():
                    raise ValueError(f"Non-finite frozen features in batch {batch_number}")
                pooled[cursor:end] = pooled_values.cpu().numpy()
                nodes[local_start:local_end] = local_values.float().cpu().numpy()
                roles[local_start:local_end] = graph["node_role"].cpu().numpy()
                type_ids[local_start:local_end] = graph["node_type"].cpu().numpy()
                cursor = end
                every = int(config["extraction"]["progress_every"])
                if batch_number % every == 0 or cursor == len(samples):
                    print(f"Frozen extraction {cursor:,}/{len(samples):,}", flush=True)
    finally:
        hook.remove()
    if cursor != len(samples):
        raise RuntimeError("Frozen extraction ended before the complete manifest")
    for array in (pooled, nodes, roles, type_ids):
        array.flush()
    del array
    del pooled, nodes, roles, type_ids
    np.save(temporary["offsets.npy"], offsets)
    for name, temporary_path in temporary.items():
        temporary_path.replace(paths[name])
    metadata = {
        "format": "tmcad_frozen_v7_local_tokens_1",
        "manifest_sha256": manifest_hash,
        "checkpoint": str(config["paths"]["checkpoint"]),
        "checkpoint_epoch": epoch,
        "sample_count": len(samples),
        "node_count": total_nodes,
        "pooled_width": 2048,
        "node_width": 512,
        "storage_dtype": dtype_name,
        "classes": manifest["classes"],
        "uids": [item["uid"] for item in samples],
        "labels": [manifest["classes"].index(item["label"]) for item in samples],
        "splits": [item["split"] for item in samples],
        "source_relpaths": [item["source_relpath"] for item in samples],
        "step_entity_types": vocabulary,
        "node_definition": "Post-relational-block, post-node-norm, pre-role-pooling StepEncoder states; every cached graph node retained.",
        "pooled_definition": "Original 4 x 512-D v7 STEP role summary, L2-normalized; no branch projection or resizing.",
    }
    save_json(paths["metadata.json"], metadata)
    print(f"Saved independent pooled + local-token cache: {paths['metadata.json']}", flush=True)
    return paths["metadata.json"]
