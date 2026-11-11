"""Train or extract the controlled same-encoder graph-contrastive baseline.

This is intentionally not named GC-CAD: it uses the HistCAD v7 graph encoders and cache so the
pretraining objective can be compared while architecture, input data, split, and embedding width stay
fixed. It uses two independently node-masked views and symmetric NT-Xent.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

from src.histcad_large_jepa.data import (  # noqa: E402
    PairedGraphDataset,
    collate_paired,
    make_node_mask,
)
from src.histcad_large_jepa.model import ParametricEncoder, StepEncoder, projection  # noqa: E402


class GraphContrastiveEncoder(nn.Module):
    def __init__(self, entity_types: int, config: dict[str, Any]):
        super().__init__()
        self.config = config
        self.parametric_encoder = ParametricEncoder(
            config["parametric_node_dim"], config["parametric_output_dim"],
            config["parametric_depth"], config["attention_heads"],
            config["parametric_pooling_slots"], config["dropout"], config["max_features"],
        )
        self.step_encoder = StepEncoder(
            entity_types, config["step_node_dim"], config["step_output_dim"],
            config["step_depth"], config["attention_heads"],
            config["step_pooling_slots"], config["dropout"],
        )
        self.joint_projection = projection(
            config["parametric_output_dim"] + config["step_output_dim"],
            config["joint_hidden_dim"], config["embedding_dim"], config["dropout"],
        )

    def forward(self, parametric: dict[str, torch.Tensor], step: dict[str, torch.Tensor],
                parametric_mask: torch.Tensor | None = None,
                step_mask: torch.Tensor | None = None) -> torch.Tensor:
        p = self.parametric_encoder(parametric, parametric_mask)
        s = self.step_encoder(step, step_mask)
        fused = torch.cat((F.normalize(p, dim=1), F.normalize(s, dim=1)), dim=1)
        return F.normalize(self.joint_projection(fused), dim=1)


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def device_for(value: str) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if value == "auto" else torch.device(value)


def loader(
    dataset: PairedGraphDataset,
    settings: dict[str, Any],
    shuffle: bool,
    seed: int,
    batch_size: int | None = None,
) -> DataLoader:
    workers = int(settings["num_workers"])
    options: dict[str, Any] = {
        "batch_size": batch_size or settings["batch_size"],
        "shuffle": shuffle,
        "num_workers": workers,
        "collate_fn": collate_paired,
        "generator": torch.Generator().manual_seed(seed),
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        options["prefetch_factor"] = int(settings.get("prefetch_factor", 2))
    return DataLoader(dataset, **options)


def amp_dtype(device: torch.device, enabled: bool) -> torch.dtype | None:
    if device.type != "cuda" or not enabled:
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def autocast(device: torch.device, dtype: torch.dtype | None):
    if dtype is None:
        return nullcontext()
    return torch.autocast("cuda", dtype=dtype)


def graph_to_device(graph: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move graph tensors while preserving auxiliary list-valued metadata."""
    non_blocking = device.type == "cuda"
    return {
        key: value.to(device, non_blocking=non_blocking) if isinstance(value, torch.Tensor) else value
        for key, value in graph.items()
    }


def log_event(path: Path, event: dict[str, Any]) -> None:
    line = json.dumps(event, sort_keys=True)
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def nt_xent(left: torch.Tensor, right: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = left @ right.T / temperature
    target = torch.arange(len(left), device=left.device)
    return (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)) / 2


def two_views(model: GraphContrastiveEncoder, host: dict[str, Any], device: torch.device,
              ratio: float, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    # Pinned-memory transfers can overlap the CPU mask construction.  The target and index tensors
    # are deliberately not copied because this objective does not consume them.
    parametric = graph_to_device(host["parametric"], device)
    step = graph_to_device(host["step"], device)
    non_blocking = device.type == "cuda"
    masks = [
        (make_node_mask(host["parametric"], ratio, generator).to(device, non_blocking=non_blocking),
         make_node_mask(host["step"], ratio, generator).to(device, non_blocking=non_blocking))
        for _ in range(2)
    ]
    left = model(parametric, step, masks[0][0], masks[0][1])
    right = model(parametric, step, masks[1][0], masks[1][1])
    return left, right


@torch.inference_mode()
def validate(model: GraphContrastiveEncoder, data: DataLoader, device: torch.device,
             settings: dict[str, Any], seed: int, dtype: torch.dtype | None) -> float:
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    total = torch.zeros((), device=device, dtype=torch.float32)
    count = 0
    for host in data:
        with autocast(device, dtype):
            left, right = two_views(model, host, device, settings["mask_ratio"], generator)
            loss = nt_xent(left, right, settings["temperature"])
        total.add_(loss.detach().float() * len(left))
        count += len(left)
    return (total / count).item()


@torch.inference_mode()
def extract(model: GraphContrastiveEncoder, loaders: dict[str, DataLoader], payload: dict[str, Any],
            device: torch.device, progress_path: Path) -> torch.Tensor:
    model.eval()
    values = torch.empty((len(payload["uids"]), model.config["embedding_dim"]), dtype=torch.float16)
    for split, data in loaders.items():
        seen = 0
        for host in data:
            parametric = graph_to_device(host["parametric"], device)
            step = graph_to_device(host["step"], device)
            embedding = model(parametric, step)
            values[host["indices"]] = embedding.cpu().half()
            seen += len(host["indices"])
        log_event(progress_path, {"event": "extraction_complete", "split": split, "samples": seen})
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--random-only", action="store_true", help="Extract an untrained same-architecture control")
    args = parser.parse_args()
    benchmark = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base = args.config.resolve().parent
    section = benchmark["baseline_training"]["contrastive"]
    paths = benchmark["paths"]
    cache_path = resolve(base, paths["histcad_cache"])
    source_path = resolve(base, paths["jepa_representations"])
    output_dir = resolve(base, paths["output_dir"]) / "baselines" / ("random" if args.random_only else "contrastive")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "representations.pt"
    if output_path.is_file():
        print(f"Reusing existing representation artifact: {output_path}")
        return
    progress_path = output_dir / "progress.jsonl"
    progress_path.write_text("", encoding="utf-8")
    seed = benchmark["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    payload = torch.load(cache_path, map_location="cpu", weights_only=False, mmap=True)
    source = torch.load(source_path, map_location="cpu", weights_only=False, mmap=True)
    if payload.get("format_version") != 7 or payload["uids"] != source["uids"]:
        raise ValueError("Expected the UID-aligned v7 cache and representation metadata")
    datasets = {"train": PairedGraphDataset(payload, "train")}
    datasets.update({
        split: PairedGraphDataset(payload, split, datasets["train"].target_mean, datasets["train"].target_std)
        for split in ("val", "test")
    })
    loaders = {split: loader(data, section, False, seed) for split, data in datasets.items()}
    model = GraphContrastiveEncoder(len(payload["step_entity_types"]), source["model_config"])
    device = device_for(section["device"])
    model.to(device)
    dtype = amp_dtype(device, section["mixed_precision"])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    log_event(progress_path, {
        "event": "run_started",
        "mode": "random_extraction" if args.random_only else "contrastive_training",
        "device": str(device),
        "amp_dtype": str(dtype).removeprefix("torch.") if dtype is not None else None,
        "batch_size": section["batch_size"],
        "train_samples": len(datasets["train"]),
        "validation_samples": len(datasets["val"]),
        "trainable_parameters": parameter_count,
        "num_workers": section["num_workers"],
    })
    history, total_steps, elapsed = [], 0, 0.0
    checkpoint_path = output_dir / "best.pt"
    if not args.random_only:
        train_loader = loader(datasets["train"], section, True, seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=section["learning_rate"], weight_decay=section["weight_decay"])
        best, stale, selected_checkpoint = math.inf, 0, False
        generator = torch.Generator().manual_seed(seed + 1)
        monitor_every = max(1, int(section.get("monitor_every_steps", 1000)))
        for epoch in range(1, section["epochs"] + 1):
            started = time.perf_counter()
            model.train()
            total = torch.zeros((), device=device, dtype=torch.float32)
            window_total = torch.zeros((), device=device, dtype=torch.float32)
            count = 0
            window_count = 0
            for step, host in enumerate(train_loader, start=1):
                optimizer.zero_grad(set_to_none=True)
                with autocast(device, dtype):
                    left, right = two_views(model, host, device, section["mask_ratio"], generator)
                    loss = nt_xent(left, right, section["temperature"])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), section["gradient_clip"])
                optimizer.step()
                weighted_loss = loss.detach().float() * len(left)
                total.add_(weighted_loss)
                window_total.add_(weighted_loss)
                batch_samples = len(left)
                count += batch_samples
                window_count += batch_samples
                total_steps += 1
                if step == 1 or step % monitor_every == 0 or step == len(train_loader):
                    now = time.perf_counter()
                    epoch_seconds = now - started
                    event: dict[str, Any] = {
                        "event": "train_progress",
                        "epoch": epoch,
                        "epochs": section["epochs"],
                        "step": step,
                        "steps": len(train_loader),
                        "samples": count,
                        "loss": round((window_total / window_count).item(), 6),
                        "samples_per_second": round(count / max(epoch_seconds, 1e-9), 2),
                        "eta_seconds": round((len(train_loader) - step) * epoch_seconds / step),
                    }
                    if device.type == "cuda":
                        event["peak_cuda_memory_gib"] = round(
                            torch.cuda.max_memory_allocated(device) / (1024 ** 3), 3,
                        )
                    log_event(progress_path, event)
                    window_total = torch.zeros((), device=device, dtype=torch.float32)
                    window_count = 0
            train_loss = (total / count).item()
            log_event(progress_path, {
                "event": "validation_started", "epoch": epoch, "batches": len(loaders["val"]),
            })
            val = validate(model, loaders["val"], device, section, seed + 10_000 + epoch, dtype)
            seconds = time.perf_counter() - started
            elapsed += seconds
            history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val, "seconds": seconds})
            if val < best - section["minimum_improvement"]:
                best, stale, selected_checkpoint = val, 0, True
                state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                torch.save({"format_version": "controlled_graph_contrastive_v1", "state_dict": state_dict,
                            "model_config": source["model_config"], "step_entity_types": payload["step_entity_types"],
                            "epoch": epoch, "validation_loss": val}, checkpoint_path)
                del state_dict
            else:
                stale += 1
            log_event(progress_path, {
                "event": "epoch_complete", **history[-1], "best_validation_loss": best,
                "patience_used": stale, "optimizer_steps": total_steps,
            })
            if stale >= section["patience"]:
                log_event(progress_path, {
                    "event": "early_stopping", "epoch": epoch, "patience": section["patience"],
                })
                break
        if not selected_checkpoint:
            raise RuntimeError("No contrastive checkpoint selected")
        selected = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(selected["state_dict"])
        del selected
    extraction_batch_size = int(section.get("extraction_batch_size", section["batch_size"]))
    extraction_loaders = {
        split: loader(data, section, False, seed, batch_size=extraction_batch_size)
        for split, data in datasets.items()
    }
    values = extract(model, extraction_loaders, payload, device, progress_path)
    torch.save({
        "format_version": "cad_benchmark_embeddings_v1",
        "uids": source["uids"],
        "splits": [item["split"] for item in source["metadata"]],
        "labels": {
            "functional_type": [item.get("functional_type", "") for item in source["metadata"]],
            "operation_signature": [item["operation_signature"] for item in source["structural_metadata"]],
            "final_operation": [item["operation_signature"].rsplit(":", 1)[-1] if item["operation_signature"] else "" for item in source["structural_metadata"]],
            "feature_count": [str(min(4, item["num_features"])) for item in source["structural_metadata"]],
        },
        "targets": {"dimensions": (source["dimension_targets"] - source["dimension_target_mean"]) / source["dimension_target_std"]},
        "embeddings": {"random_joint" if args.random_only else "contrastive_joint": values},
        "metadata": {
            "objective": "none" if args.random_only else "symmetric NT-Xent between independently node-masked graph views",
            "architecture": "same HistCAD v7 parametric encoder, STEP encoder, joint projection, and 4096-D width",
            "is_gc_cad_reproduction": False,
            "seed": seed,
        },
    }, output_path)
    summary = {
        "method": "random_same_architecture" if args.random_only else "controlled_graph_contrastive",
        "trainable_parameters": parameter_count,
        "epochs_completed": len(history), "optimizer_steps": total_steps,
        "measured_training_seconds": elapsed if history else None,
        "monitoring_log": str(progress_path),
        "history": history,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log_event(progress_path, {"event": "run_complete", "output": str(output_path)})


if __name__ == "__main__":
    main()
