from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(ROOT))

from baselines import GraphAutoencoder, descriptor_dim, graph_descriptors  # noqa: E402

from src.histcad_large_jepa.data import PairedGraphDataset, collate_paired, to_device  # noqa: E402


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (ROOT / candidate).resolve()


def make_loader(
    dataset: PairedGraphDataset, settings: dict[str, Any], shuffle: bool, seed: int
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=settings["batch_size"],
        shuffle=shuffle,
        num_workers=settings["num_workers"],
        collate_fn=collate_paired,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=torch.cuda.is_available() and settings["device"] in {"auto", "cuda"},
        persistent_workers=settings["num_workers"] > 0,
    )


def select_device(name: str) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)


@torch.inference_mode()
def collect_descriptors(
    loaders: dict[str, DataLoader], count: int, bins: int
) -> torch.Tensor:
    output = torch.empty((count, descriptor_dim(bins)))
    for split, loader in loaders.items():
        seen = 0
        for batch in loader:
            values = graph_descriptors(batch["parametric"], batch["step"], bins)
            output[batch["indices"]] = values
            seen += len(values)
        print(json.dumps({"descriptors": split, "samples": seen}), flush=True)
    return output


@torch.inference_mode()
def validation_loss(
    model: GraphAutoencoder,
    loader: DataLoader,
    descriptors: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> float:
    model.eval()
    total, samples = 0.0, 0
    for host_batch in loader:
        batch = to_device(host_batch, device)
        _, reconstruction = model(batch["parametric"], batch["step"])
        target = ((descriptors[host_batch["indices"]] - mean) / std).to(device)
        loss = F.smooth_l1_loss(reconstruction, target)
        total += float(loss) * len(target)
        samples += len(target)
    return total / samples


@torch.inference_mode()
def extract(
    model: GraphAutoencoder,
    loaders: dict[str, DataLoader],
    count: int,
    embedding_dim: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    output = torch.empty((count, embedding_dim), dtype=torch.float16)
    for split, loader in loaders.items():
        seen = 0
        for host_batch in loader:
            batch = to_device(host_batch, device)
            embedding, _ = model(batch["parametric"], batch["step"])
            output[host_batch["indices"]] = embedding.cpu().to(torch.float16)
            seen += len(embedding)
        print(json.dumps({"extracted": split, "samples": seen}), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Train controlled non-JEPA retrieval baselines")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    settings = config["baselines"]["training"]
    device = select_device(settings["device"])
    payload = torch.load(
        resolve(config["paths"]["cache"]), map_location="cpu", weights_only=False, mmap=True
    )
    source = torch.load(
        resolve(config["paths"]["representations"]),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    if payload.get("format_version") != 7 or source.get("format_version") != 7:
        raise ValueError("Baselines require the self-contained v7 cache and representations")
    if payload["uids"] != source["uids"]:
        raise ValueError("Cache and JEPA representations are not UID-aligned")
    if [item["split"] for item in payload["metadata"]] != [
        item["split"] for item in source["metadata"]
    ]:
        raise ValueError("Cache and JEPA representations do not use the same split assignment")
    datasets = {"train": PairedGraphDataset(payload, "train")}
    for split in ("val", "test"):
        datasets[split] = PairedGraphDataset(
            payload, split, datasets["train"].target_mean, datasets["train"].target_std
        )
    extraction_loaders = {
        split: make_loader(dataset, settings, False, seed) for split, dataset in datasets.items()
    }
    train_loader = make_loader(datasets["train"], settings, True, seed)
    bins = config["baselines"]["descriptor"]["step_entity_hash_bins"]
    descriptors = collect_descriptors(extraction_loaders, len(payload["uids"]), bins)
    train_indices = torch.tensor(datasets["train"].indices)
    descriptor_mean = descriptors[train_indices].mean(0)
    descriptor_std = descriptors[train_indices].std(0).clamp_min(
        config["baselines"]["descriptor"]["minimum_std"]
    )
    model_config = dict(source["model_config"])
    model_config["autoencoder_decoder_hidden_dim"] = config["baselines"]["model"][
        "decoder_hidden_dim"
    ]
    model = GraphAutoencoder(
        len(payload["step_entity_types"]), model_config, descriptors.shape[1]
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    output_dir = resolve(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_loss, stale, history = math.inf, 0, []
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        total, samples = 0.0, 0
        for step, host_batch in enumerate(train_loader, start=1):
            batch = to_device(host_batch, device)
            target = ((descriptors[host_batch["indices"]] - descriptor_mean) / descriptor_std).to(
                device
            )
            optimizer.zero_grad(set_to_none=True)
            _, reconstruction = model(batch["parametric"], batch["step"])
            loss = F.smooth_l1_loss(reconstruction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"])
            optimizer.step()
            total += float(loss.detach()) * len(target)
            samples += len(target)
            if step % settings["monitor_every_steps"] == 0 or step == len(train_loader):
                print(json.dumps({"epoch": epoch, "step": step, "train_loss": total / samples}), flush=True)
        val_loss = validation_loss(
            model, extraction_loaders["val"], descriptors, descriptor_mean, descriptor_std, device
        )
        history.append({"epoch": epoch, "train_loss": total / samples, "validation_loss": val_loss})
        improved = val_loss < best_loss - settings["minimum_validation_improvement"]
        if improved:
            best_loss, stale = val_loss, 0
            torch.save(
                {
                    "format_version": 1,
                    "model": model.state_dict(),
                    "model_config": model_config,
                    "descriptor_mean": descriptor_mean,
                    "descriptor_std": descriptor_std,
                    "step_entity_hash_bins": bins,
                    "epoch": epoch,
                    "validation_loss": val_loss,
                },
                output_dir / "best_graph_autoencoder.pt",
            )
        else:
            stale += 1
        print(json.dumps({"epoch_complete": epoch, "validation_loss": val_loss, "best": best_loss}), flush=True)
        if stale >= settings["patience"]:
            break
    checkpoint = torch.load(
        output_dir / "best_graph_autoencoder.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    autoencoder = extract(
        model,
        extraction_loaders,
        len(payload["uids"]),
        model_config["embedding_dim"],
        device,
    )
    standardized_float = (descriptors - descriptor_mean) / descriptor_std
    pca_dim = min(
        config["baselines"]["descriptor"]["pca_dim"],
        standardized_float.shape[1],
        len(train_indices) - 1,
    )
    _, _, pca_components = torch.pca_lowrank(
        standardized_float[train_indices],
        q=pca_dim,
        center=False,
        niter=config["baselines"]["descriptor"]["pca_power_iterations"],
    )
    pca_descriptors = (standardized_float @ pca_components).to(torch.float16)
    standardized_descriptors = standardized_float.to(torch.float16)
    torch.save(
        {
            "format_version": 1,
            "uids": payload["uids"],
            "metadata": payload["metadata"],
            "embeddings": {
                "graph_autoencoder_joint": autoencoder,
                "input_graph_statistics": standardized_descriptors,
                "pca_graph_statistics": pca_descriptors,
            },
            "methods": {
                "graph_autoencoder_joint": {
                    "family": "learned_baseline",
                    "training": "same train split; validation reconstruction selects checkpoint",
                    "test_access": False,
                    "retrieval_labels_used": False,
                    "architecture": "same graph encoders and joint embedding width as v7 JEPA",
                },
                "input_graph_statistics": {
                    "family": "nonlearned_baseline",
                    "training": "train split used only for feature standardization",
                    "test_access": False,
                    "retrieval_labels_used": False,
                },
                "pca_graph_statistics": {
                    "family": "classical_unsupervised_baseline",
                    "training": "PCA fit on standardized train-split graph statistics only",
                    "test_access": False,
                    "retrieval_labels_used": False,
                },
            },
            "autoencoder_checkpoint": str((output_dir / "best_graph_autoencoder.pt").resolve()),
            "descriptor_mean": descriptor_mean,
            "descriptor_std": descriptor_std,
            "pca_components": pca_components,
        },
        output_dir / "baseline_representations.pt",
    )
    summary = {
        "best_epoch": checkpoint["epoch"],
        "best_validation_loss": checkpoint["validation_loss"],
        "epochs_completed": len(history),
        "train_samples": len(datasets["train"]),
        "validation_samples": len(datasets["val"]),
        "test_samples_accessed_during_training": 0,
        "retrieval_labels_used_for_training": False,
        "embedding_dim": model_config["embedding_dim"],
        "descriptor_dim": descriptors.shape[1],
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
    }
    (output_dir / "baseline_training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "complete", **{k: v for k, v in summary.items() if k != "history"}}))


if __name__ == "__main__":
    main()
