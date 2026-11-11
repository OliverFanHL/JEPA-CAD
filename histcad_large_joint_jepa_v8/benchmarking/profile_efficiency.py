"""Measure embedding forward time using saved models; never trains a model.

Run from benchmarking: python profile_efficiency.py --device auto
The resulting profile is consumed by: python run.py efficiency --force
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

from downstream_retrieval_adapter.baselines import GraphAutoencoder, descriptor_dim, graph_descriptors  # noqa: E402
from src.histcad_large_jepa.data import PairedGraphDataset, collate_paired, to_device  # noqa: E402
from src.histcad_large_jepa.model import WideJointJEPA  # noqa: E402
from baselines.contrastive.train import GraphContrastiveEncoder  # noqa: E402
from src.cad_benchmark.config import load_config  # noqa: E402
from src.cad_benchmark.io import atomic_json  # noqa: E402


def _load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def _forward(name: str, config: dict, payload: dict, device: torch.device):
    spec = config["efficiency_sources"][name]
    if name in {"pca_graph_statistics", "input_graph_statistics"}:
        artifact_path = Path(config["_root"]) / "../downstream_retrieval_adapter/outputs/baseline_representations.pt"
        artifact = _load(artifact_path.resolve())
        mean = artifact["descriptor_mean"]
        std = artifact["descriptor_std"]
        components = artifact["pca_components"]
        bins = mean.numel() - descriptor_dim(0)
        if bins < 1:
            raise ValueError("Saved descriptor width cannot identify the STEP hash-bin count")
        if name == "pca_graph_statistics":
            def forward(batch):
                descriptor = graph_descriptors(batch["parametric"], batch["step"], bins)
                return ((descriptor - mean) / std) @ components
        else:
            def forward(batch):
                descriptor = graph_descriptors(batch["parametric"], batch["step"], bins)
                return (descriptor - mean) / std
        return forward, None, "cpu"

    checkpoint_path = Path(spec["checkpoint"]) if spec.get("checkpoint") else None
    if name == "random":
        contrastive = _load(Path(config["efficiency_sources"]["contrastive"]["checkpoint"]))
        torch.manual_seed(config["seed"])
        model = GraphContrastiveEncoder(len(payload["step_entity_types"]), contrastive["model_config"])
        del contrastive
    else:
        checkpoint = _load(checkpoint_path)
        if name == "jepa":
            model = WideJointJEPA(
                len(payload["step_entity_types"]), checkpoint["model_config"],
                len(payload["dimension_target_names"]),
            )
            model.load_state_dict(checkpoint["model"])
        elif name == "graph_autoencoder":
            model = GraphAutoencoder(
                len(payload["step_entity_types"]), checkpoint["model_config"],
                checkpoint["descriptor_mean"].numel(),
            )
            model.load_state_dict(checkpoint["model"])
        elif name == "contrastive":
            model = GraphContrastiveEncoder(
                len(payload["step_entity_types"]), checkpoint["model_config"]
            )
            model.load_state_dict(checkpoint["state_dict"])
        else:
            raise ValueError(name)
        del checkpoint
    model.eval().to(device)
    if name == "jepa":
        return lambda batch: model.embed(batch["parametric"], batch["step"])["joint"], checkpoint_path, str(device)
    if name == "graph_autoencoder":
        return lambda batch: model(batch["parametric"], batch["step"])[0], checkpoint_path, str(device)
    return lambda batch: model(batch["parametric"], batch["step"]), checkpoint_path, str(device)


def _profile(forward, batches: list[dict], device: torch.device, warmup: int, repeats: int) -> float:
    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.inference_mode():
        for index in range(warmup):
            batch = to_device(batches[index % len(batches)], device)
            forward(batch)
        synchronize()
        elapsed = 0.0
        samples = 0
        for _ in range(repeats):
            for host in batches:
                batch = to_device(host, device)
                synchronize()
                started = time.perf_counter()
                forward(batch)
                synchronize()
                elapsed += time.perf_counter() - started
                samples += len(host["indices"])
    return elapsed / samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if min(args.samples, args.batch_size, args.repeats) < 1 or args.warmup < 0:
        parser.error("samples, batch-size, and repeats must be positive; warmup must be nonnegative")
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    payload = _load(Path(config["paths"]["histcad_cache"]))
    excluded = set()
    coverage_path = Path(config["shared_coverage"]["histcad"]["file"])
    if coverage_path.is_file():
        excluded = set(json.loads(coverage_path.read_text(encoding="utf-8"))["omitted_uids"])
    test = PairedGraphDataset(payload, "test")
    positions = [position for position, index in enumerate(test.indices)
                 if payload["uids"][index] not in excluded][:args.samples]
    if not positions:
        raise RuntimeError("No covered HistCAD test samples are available to profile")
    loader = DataLoader(Subset(test, positions), batch_size=args.batch_size,
                        shuffle=False, num_workers=0, collate_fn=collate_paired)
    batches = list(loader)
    output = Path(config["paths"]["output_dir"]) / "efficiency" / "embedding_profile.json"
    existing = json.loads(output.read_text(encoding="utf-8")) if output.is_file() else {}
    result = {"protocol": "forward_only_existing_models_v1", "dataset": "histcad",
              "split": "test", "covered_rows": len(positions), "batch_size": args.batch_size,
              "warmup_batches": args.warmup, "repeats": args.repeats,
              "excludes": ["checkpoint loading", "graph collation", "host-to-device transfer"],
              "methods": {key: value for key, value in existing.get("methods", {}).items()
                          if key.startswith("brepnet_official")}}
    for name in ("jepa", "graph_autoencoder", "contrastive", "random",
                 "input_graph_statistics", "pca_graph_statistics"):
        forward, checkpoint_path, method_device = _forward(name, config, payload, device)
        run_device = device if method_device != "cpu" else torch.device("cpu")
        seconds = _profile(forward, batches, run_device, args.warmup, args.repeats)
        result["methods"][name] = {
            "embedding_seconds_per_sample": seconds,
            "checkpoint_bytes": checkpoint_path.stat().st_size if checkpoint_path else None,
            "checkpoint_mtime_ns": checkpoint_path.stat().st_mtime_ns if checkpoint_path else None,
            "device": method_device,
            "samples": len(positions),
            "batch_size": args.batch_size,
        }
        print(json.dumps({"method": name, "embedding_seconds_per_sample": seconds}), flush=True)
        del forward
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        atomic_json(output, result)
    print(json.dumps({"profile": str(output), "note": "BRepNet requires timing its official raw-STEP extractor"}), flush=True)


if __name__ == "__main__":
    main()
