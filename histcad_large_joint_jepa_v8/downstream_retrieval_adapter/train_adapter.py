from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from adapter import LightweightRetrievalAdapter  # noqa: E402, I001


OPERATIONS = {"NewBody": 0, "Join": 1, "Cut": 2, "Intersect": 3}


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (ROOT / candidate).resolve()


def device_from(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def feature_bucket(count: int) -> int:
    return min(max(int(count), 1), 4) - 1


def labels(structural: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    feature = torch.tensor([feature_bucket(item["num_features"]) for item in structural])
    operation = torch.tensor(
        [
            OPERATIONS.get(item["operation_signature"].rsplit(":", 1)[-1], 0)
            if item["operation_signature"]
            else 0
            for item in structural
        ]
    )
    return feature, operation


def positive_neighbors(
    targets: torch.Tensor,
    feature: torch.Tensor,
    operation: torch.Tensor,
    projections: int,
    window: int,
    feature_penalty: float,
    operation_penalty: float,
    seed: int,
) -> torch.Tensor:
    """Approximate composite nearest neighbors without an O(N^2) search."""
    generator = torch.Generator().manual_seed(seed)
    vectors = [torch.eye(targets.shape[1])[index] for index in range(min(3, targets.shape[1]))]
    vectors.extend(torch.randn(targets.shape[1], generator=generator) for _ in range(projections))
    best_score = torch.full((len(targets),), torch.inf)
    best_index = torch.zeros(len(targets), dtype=torch.long)
    for vector in vectors:
        order = torch.argsort(targets @ vector)
        for offset in range(1, window + 1):
            left, right = order[:-offset], order[offset:]
            score = (targets[left] - targets[right]).abs().mean(1)
            score = score + feature_penalty * (feature[left] != feature[right]).float()
            score = score + operation_penalty * (operation[left] != operation[right]).float()
            improve_left = score < best_score[left]
            best_score[left[improve_left]] = score[improve_left]
            best_index[left[improve_left]] = right[improve_left]
            improve_right = score < best_score[right]
            best_score[right[improve_right]] = score[improve_right]
            best_index[right[improve_right]] = left[improve_right]
    return best_index


def fixed_negatives(count: int, positives: torch.Tensor, number: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    negatives = torch.randint(0, count, (count, number), generator=generator)
    rows = torch.arange(count)[:, None]
    invalid = (negatives == rows) | (negatives == positives[:, None])
    while invalid.any():
        negatives[invalid] = torch.randint(0, count, (int(invalid.sum()),), generator=generator)
        invalid = (negatives == rows) | (negatives == positives[:, None])
    return negatives


def loss_batch(
    model: LightweightRetrievalAdapter,
    embeddings: torch.Tensor,
    targets: torch.Tensor,
    feature: torch.Tensor,
    operation: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    indices: torch.Tensor,
    temperature: float,
    weights: dict[str, float],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    positive_indices, negative_indices = positives[indices], negatives[indices]
    query_source = embeddings[indices].to(device)
    positive_source = embeddings[positive_indices].to(device)
    negative_source = embeddings[negative_indices.flatten()].to(device)
    count, negative_count = len(indices), negatives.shape[1]
    combined = torch.cat((query_source, positive_source, negative_source), dim=0)
    adapted = model.encode(combined)
    query = adapted[:count]
    positive = adapted[count : 2 * count]
    negative = adapted[2 * count :].reshape(count, negative_count, -1)
    positive_score = (query * positive).sum(-1, keepdim=True)
    negative_score = (query[:, None] * negative).sum(-1)
    retrieval = F.cross_entropy(
        torch.cat((positive_score, negative_score), dim=1) / temperature,
        torch.zeros(count, dtype=torch.long, device=device),
    )
    dimension_prediction = model.dimension_head(query)
    feature_prediction = model.feature_count_head(query)
    operation_prediction = model.final_operation_head(query)
    dimension = F.smooth_l1_loss(dimension_prediction, targets[indices].to(device))
    feature_loss = F.cross_entropy(feature_prediction, feature[indices].to(device))
    operation_loss = F.cross_entropy(operation_prediction, operation[indices].to(device))
    anchor = (1.0 - (query * F.normalize(query_source, dim=-1)).sum(-1)).mean()
    terms = {
        "retrieval": retrieval,
        "dimension": dimension,
        "feature_count": feature_loss,
        "final_operation": operation_loss,
        "anchor": anchor,
    }
    loss = sum(weights[name] * value for name, value in terms.items())
    return loss, {"loss": float(loss.detach()), **{k: float(v.detach()) for k, v in terms.items()}}


@torch.no_grad()
def validation_loss(
    model: LightweightRetrievalAdapter,
    embeddings: torch.Tensor,
    targets: torch.Tensor,
    feature: torch.Tensor,
    operation: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    settings: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    batches = 0
    for (indices,) in DataLoader(
        TensorDataset(torch.arange(len(embeddings))),
        batch_size=settings["batch_size"],
        shuffle=False,
    ):
        _, metrics = loss_batch(
            model,
            embeddings,
            targets,
            feature,
            operation,
            positives,
            negatives,
            indices,
            settings["temperature"],
            settings["loss_weights"],
            device,
        )
        batches += 1
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + value
    return {name: value / batches for name, value in totals.items()}


@torch.no_grad()
def extract(
    model: LightweightRetrievalAdapter,
    embeddings: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    output = []
    for start in range(0, len(embeddings), batch_size):
        output.append(model.encode(embeddings[start : start + batch_size].to(device)).cpu())
    return torch.cat(output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a lightweight frozen-embedding retrieval adapter"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(8)
    device = device_from(config["training"]["device"])
    output_dir = resolve(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(
        resolve(config["paths"]["representations"]),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    checkpoint_path = resolve(config["paths"]["source_checkpoint"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"v7 best checkpoint not found: {checkpoint_path}")
    expected_dim = config["model"]["input_dim"]
    if payload.get("format_version") != 7:
        raise ValueError("The downstream adapter requires v7 representations")
    if payload.get("model_config", {}).get("embedding_dim") != expected_dim:
        raise ValueError("Adapter input_dim does not match the v7 checkpoint embedding dimension")
    if payload.get("source_checkpoint_name") != checkpoint_path.name:
        raise ValueError("representations.pt was not extracted from a best.pt checkpoint")
    embeddings = payload["embeddings"]["joint"]
    if embeddings.shape[1] != expected_dim:
        raise ValueError("Adapter input_dim does not match representations.pt")
    targets = (payload["dimension_targets"] - payload["dimension_target_mean"]) / payload[
        "dimension_target_std"
    ]
    feature, operation = labels(payload["structural_metadata"])
    split_indices = {
        split: torch.tensor(
            [index for index, item in enumerate(payload["metadata"]) if item["split"] == split]
        )
        for split in ("train", "val")
    }
    uids, metadata = payload["uids"], payload["metadata"]
    source_checkpoint_epoch = payload["source_checkpoint_epoch"]
    del payload
    gc.collect()

    split_data = {}
    settings = config["training"]
    for split, indices in split_indices.items():
        split_targets, split_feature, split_operation = (
            targets[indices],
            feature[indices],
            operation[indices],
        )
        positives = positive_neighbors(
            split_targets,
            split_feature,
            split_operation,
            settings["positive_projections"],
            settings["positive_window"],
            settings["positive_feature_penalty"],
            settings["positive_operation_penalty"],
            seed + (0 if split == "train" else 10),
        )
        negatives = fixed_negatives(
            len(indices),
            positives,
            settings["negatives_per_query"],
            seed + (1 if split == "train" else 11),
        )
        split_data[split] = (
            embeddings[indices],
            split_targets,
            split_feature,
            split_operation,
            positives,
            negatives,
        )
        print(json.dumps({"prepared": split, "samples": len(indices)}), flush=True)

    model = LightweightRetrievalAdapter(config["model"], targets.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    train_loader = DataLoader(
        TensorDataset(torch.arange(len(split_indices["train"]))),
        batch_size=settings["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    history, best_loss, stale = [], math.inf, 0
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        totals: dict[str, float] = {}
        interval = time.time()
        for step, (indices,) in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = loss_batch(
                model,
                *split_data["train"],
                indices,
                settings["temperature"],
                settings["loss_weights"],
                device,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + value
            if step % settings["monitor_every_steps"] == 0 or step == len(train_loader):
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": step,
                            "steps": len(train_loader),
                            "train_loss": round(totals["loss"] / step, 5),
                            "retrieval_loss": round(totals["retrieval"] / step, 5),
                            "seconds": round(time.time() - interval, 1),
                        }
                    ),
                    flush=True,
                )
                interval = time.time()
        validation = validation_loss(model, *split_data["val"], settings, device)
        row = {
            "epoch": epoch,
            "train": {name: value / len(train_loader) for name, value in totals.items()},
            "validation": validation,
        }
        history.append(row)
        improved = validation["loss"] < best_loss
        if improved:
            best_loss, stale = validation["loss"], 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_config": config["model"],
                    "training_config": settings,
                    "epoch": epoch,
                    "validation_loss": best_loss,
                    "source_checkpoint_epoch": source_checkpoint_epoch,
                },
                output_dir / "best_adapter.pt",
            )
        else:
            stale += 1
        print(
            json.dumps(
                {
                    "epoch_complete": epoch,
                    "validation_loss": round(validation["loss"], 6),
                    "best_validation_loss": round(best_loss, 6),
                    "checkpoint": "best" if improved else "unchanged",
                }
            ),
            flush=True,
        )
        if stale >= settings["patience"]:
            break

    best = torch.load(output_dir / "best_adapter.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    adapted = extract(model, embeddings, settings["batch_size"], device).to(torch.float16)
    torch.save(
        {
            "format_version": 1,
            "uids": uids,
            "metadata": metadata,
            "embeddings": {
                "direct_joint": embeddings,
                "adapted_joint": adapted,
            },
            "adapter_checkpoint": str((output_dir / "best_adapter.pt").resolve()),
            "source_checkpoint": str(checkpoint_path),
        },
        output_dir / "adapted_representations.pt",
    )
    summary = {
        "source_representation": "v7 joint 4096-D from best.pt; frozen",
        "embedding_dim": config["model"]["input_dim"],
        "adapter_trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_epoch": best["epoch"],
        "best_validation_loss": best["validation_loss"],
        "epochs_completed": len(history),
        "train_samples": len(split_indices["train"]),
        "validation_samples": len(split_indices["val"]),
        "test_samples_accessed_during_training": 0,
        "functional_labels_used_for_adapter_training": False,
        "history": history,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        json.dumps({"status": "complete", **{k: v for k, v in summary.items() if k != "history"}})
    )


if __name__ == "__main__":
    main()
