"""Extract frozen JEPA-v7 STEP embeddings for valid TMCAD files."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from common import device_for, read_pretrained_metadata, sha256_file, v7_modules
from step_statements import PARSER_VERSION
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


class CachedStepDataset(Dataset):
    def __init__(self, manifest: dict, processed_dir: Path):
        self.samples = manifest["samples"]
        self.processed_dir = processed_dir

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        path = self.processed_dir / self.samples[index]["graph_relpath"]
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("parser_version") != PARSER_VERSION:
            raise ValueError(f"Stale STEP graph cache: {path}")
        return payload["graph"]


def extract_features(config: dict) -> Path:
    processed = config["paths"]["processed_dir"]
    manifest_path = processed / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Run the prepare stage first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("parser_version") != PARSER_VERSION:
        raise ValueError("Manifest uses the old STEP parser; rerun the prepare stage")
    metadata, vocabulary = read_pretrained_metadata(config)
    if manifest["source_checkpoint"] != metadata["checkpoint"]:
        raise ValueError("Manifest and frozen checkpoint do not match")
    StepEncoder, collate_step_graphs, _ = v7_modules(config)
    model_config = metadata["model_config"]
    encoder = StepEncoder(
        len(vocabulary), model_config["step_node_dim"], model_config["step_output_dim"],
        model_config["step_depth"], model_config["attention_heads"],
        model_config["step_pooling_slots"], model_config["dropout"],
    )
    checkpoint = torch.load(config["paths"]["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    frozen_state = {key.removeprefix("step_encoder."): value for key, value in checkpoint["model"].items()
                    if key.startswith("step_encoder.")}
    encoder.load_state_dict(frozen_state, strict=True)
    del checkpoint, frozen_state
    encoder.eval()
    encoder.requires_grad_(False)
    device = device_for(config["extraction"]["device"])
    encoder.to(device)
    loader = DataLoader(CachedStepDataset(manifest, processed),
                        batch_size=int(config["extraction"]["batch_size"]), shuffle=False,
                        num_workers=int(config["extraction"]["num_workers"]),
                        collate_fn=collate_step_graphs, pin_memory=device.type == "cuda")
    count = len(manifest["samples"])
    width = int(model_config["step_output_dim"])
    if width != 2048 or model_config["step_pooling_slots"] != 3 or model_config["step_node_dim"] != 512:
        raise ValueError("This downstream attention model expects the v7 4 x 512-D STEP summary")
    vectors = torch.empty((count, width), dtype=torch.float16)
    cursor = 0
    print(f"Frozen STEP extraction: {count:,} parts, device={device}, width={width}, "
          f"v7_epoch={metadata['epoch']}", flush=True)
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader, start=1):
            graph = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            embedding = F.normalize(encoder(graph).float(), dim=-1)
            if not torch.isfinite(embedding).all():
                raise ValueError(f"Non-finite STEP embedding in extraction batch {batch_number}")
            size = len(graph["global_index"])
            vectors[cursor:cursor + size] = embedding.cpu().half()
            cursor += size
            if batch_number % int(config["extraction"]["progress_every_batches"]) == 0 or cursor == count:
                print(f"Frozen STEP extraction {cursor:,}/{count:,}", flush=True)
    if cursor != count:
        raise RuntimeError("Frozen extraction count differs from manifest length")
    classes = manifest["classes"]
    payload = {"format": "tmcad_frozen_v7_step_1", "source_checkpoint": metadata["checkpoint"],
               "source_epoch": metadata["epoch"], "manifest_sha256": sha256_file(manifest_path),
               "class_names": classes, "uids": [x["uid"] for x in manifest["samples"]],
               "source_relpaths": [x["source_relpath"] for x in manifest["samples"]],
               "sha256": [x["sha256"] for x in manifest["samples"]],
               "splits": [x["split"] for x in manifest["samples"]],
               "labels": torch.tensor([classes.index(x["label"]) for x in manifest["samples"]], dtype=torch.long),
               "unknown_type_fraction": [x["unknown_type_fraction"] for x in manifest["samples"]],
               "brep_truncated": [x["brep_truncated"] for x in manifest["samples"]],
               "embeddings": vectors}
    output = processed / "frozen_step_embeddings.pt"
    torch.save(payload, output)
    print(f"Saved frozen embeddings: {output}", flush=True)
    return output
