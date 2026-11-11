"""Supervised downstream heads only; JEPA outputs are precomputed and frozen."""

from __future__ import annotations

import csv

import numpy as np
import torch
from common import device_for, file_sha256, save_json, set_seed
from dataset import FrozenViews
from metrics import classification_metrics
from model import build_model
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


def _loader(dataset: FrozenViews, settings: dict, shuffle: bool,
            seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=int(settings["batch_size"]), shuffle=shuffle,
        num_workers=int(settings["num_workers"]),
        pin_memory=settings["device"] == "cuda" or (settings["device"] == "auto" and torch.cuda.is_available()),
        generator=generator,
    )


def _device_batch(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True)
            for name, value in batch.items() if name in {"pooled", "alternative", "geometry", "mesh", "label"}}


@torch.inference_mode()
def infer(model: nn.Module, loader: DataLoader, device: torch.device,
          weights: torch.Tensor | None = None, label_smoothing: float = 0.0,
          return_attention: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | None]:
    model.eval()
    probabilities, truth, indices = [], [], []
    weighted_loss, count = 0.0, 0
    for original in loader:
        batch = _device_batch(original, device)
        logits, _ = model(batch, return_attention=return_attention)
        size = len(batch["label"])
        if weights is not None:
            loss = F.cross_entropy(logits.float(), batch["label"], weight=weights,
                                   label_smoothing=label_smoothing)
            weighted_loss += float(loss) * size
        probabilities.append(logits.float().softmax(-1).cpu().numpy())
        truth.append(batch["label"].cpu().numpy())
        indices.append(original["index"].numpy())
        count += size
    return (np.concatenate(probabilities), np.concatenate(truth),
            np.concatenate(indices), weighted_loss / count if weights is not None else None)


def train_models(config: dict) -> None:
    settings = config["classifier"]
    classes = FrozenViews(config, "train", include_alternative=False).metadata["classes"]
    output = config["paths"]["output"]
    device = device_for(settings["device"])
    for name in settings["models"]:
        set_seed(int(config["seed"]))
        include_alternative = True
        include_geometry = True
        include_mesh = name == "pooled_plus_brep_view_geometry_mesh"
        train_set = FrozenViews(config, "train", include_alternative, include_geometry, include_mesh)
        val_set = FrozenViews(config, "val", include_alternative, include_geometry, include_mesh)
        mesh_sha256 = file_sha256(train_set.mesh_path) if include_mesh else None
        train_loader = _loader(train_set, settings, True, int(config["seed"]))
        val_loader = _loader(val_set, settings, False, int(config["seed"]))
        labels = torch.tensor([train_set.metadata["labels"][i] for i in train_set.indices], dtype=torch.long)
        counts = torch.bincount(labels, minlength=len(classes)).float()
        if (counts == 0).any():
            raise ValueError("At least one TMCAD class has no training samples")
        weights = (counts.mean() / counts).sqrt().clamp(0.5, 2.0).to(device)
        model = build_model(name, len(classes), settings).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]),
                                      weight_decay=float(settings["weight_decay"]))
        epochs = int(settings["epochs"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=float(settings["minimum_learning_rate"]),
        )
        model_dir = output / name
        model_dir.mkdir(parents=True, exist_ok=True)
        history: list[dict] = []
        best_f1, best_epoch, stale = -1.0, 0, 0
        print(f"Training {name}: train={len(train_set):,} val={len(val_set):,} "
              f"device={device} trainable={sum(p.numel() for p in model.parameters()):,}", flush=True)
        for epoch in range(1, epochs + 1):
            model.train()
            loss_total, count = 0.0, 0
            for original in train_loader:
                batch = _device_batch(original, device)
                logits, _ = model(batch)
                loss = F.cross_entropy(logits.float(), batch["label"], weight=weights,
                                       label_smoothing=float(settings["label_smoothing"]))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clip"]))
                optimizer.step()
                size = len(batch["label"])
                loss_total += float(loss) * size
                count += size
            val_prob, val_truth, _, val_loss = infer(
                model, val_loader, device, weights, float(settings["label_smoothing"]),
            )
            validation = classification_metrics(val_prob, val_truth, classes)
            row = {"epoch": epoch, "train_loss": loss_total / count,
                   "val_loss": val_loss, "val_accuracy": validation["accuracy"],
                   "val_balanced_accuracy": validation["balanced_accuracy"],
                   "val_macro_f1": validation["macro_f1"],
                   "learning_rate": optimizer.param_groups[0]["lr"]}
            history.append(row)
            print(f"{name} epoch={epoch:02d}/{epochs} train={row['train_loss']:.4f} "
                  f"val={val_loss:.4f} val_acc={validation['accuracy']:.4f} "
                  f"val_macro_F1={validation['macro_f1']:.4f} "
                  f"lr={row['learning_rate']:.2e}", flush=True)
            if validation["macro_f1"] > best_f1 + 1e-4:
                best_f1, best_epoch, stale = validation["macro_f1"], epoch, 0
                payload = {
                    "architecture": name, "state_dict": model.state_dict(),
                    "settings": settings, "classes": classes, "epoch": epoch,
                    "validation_macro_f1": best_f1,
                    "manifest_sha256": train_set.metadata["manifest_sha256"],
                    "source_checkpoint": train_set.metadata["checkpoint"],
                    "feature_format": train_set.metadata["format"],
                    "mesh_sha256": mesh_sha256,
                }
                temporary = model_dir / "best.partial.pt"
                torch.save(payload, temporary)
                temporary.replace(model_dir / "best.pt")
            else:
                stale += 1
            scheduler.step()
            if epoch >= int(settings["min_epochs"]) and stale >= int(settings["patience"]):
                print(f"{name} early stop at {epoch}; selected {best_epoch}", flush=True)
                break
        with (model_dir / "training_history.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        save_json(model_dir / "training_summary.json", {
            "best_epoch": best_epoch, "best_validation_macro_f1": best_f1,
            "completed_epochs": len(history),
            "train_class_counts": dict(zip(classes, counts.int().tolist(), strict=True)),
            "selection": "Validation macro-F1 only; no test labels used during training",
            "frozen_source_checkpoint": train_set.metadata["checkpoint"],
        })
