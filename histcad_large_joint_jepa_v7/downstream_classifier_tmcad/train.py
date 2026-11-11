"""Train only downstream heads on frozen STEP embeddings; select on validation."""

from __future__ import annotations

import csv

import numpy as np
import torch
from classifier import make_classifier
from common import device_for, save_json, set_seed, sha256_file
from metrics import classification_metrics
from structural_features import load_structural_features
from torch import nn
from torch.nn import functional as F


def load_features(config: dict) -> dict:
    path = config["paths"]["processed_dir"] / "frozen_step_embeddings.pt"
    if not path.is_file():
        raise FileNotFoundError("Run prepare and extract before training")
    features = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if features.get("format") != "tmcad_frozen_v7_step_1":
        raise ValueError("Unexpected frozen embedding cache format")
    if features["source_checkpoint"] != str(config["paths"]["checkpoint"]):
        raise ValueError("The embedding cache was made with another v7 checkpoint")
    if features["manifest_sha256"] != sha256_file(config["paths"]["processed_dir"] / "manifest.json"):
        raise ValueError("Manifest changed after extraction; regenerate the frozen embeddings")
    if features["embeddings"].shape != (len(features["uids"]), 2048):
        raise ValueError("Expected one 2048-D STEP vector per TMCAD sample")
    if features["class_names"] != list(config["dataset"]["classes"]):
        raise ValueError("Configured class order differs from extraction")
    if any(name.endswith("_structural") for name in config["classifier"]["models"]):
        values, names = load_structural_features(config, features)
        if values.shape[0] != len(features["uids"]):
            raise ValueError("Structural and frozen STEP feature counts differ")
        features["structural_features"] = values
        features["structural_feature_names"] = names
    return features


def split_indices(features: dict) -> dict[str, torch.Tensor]:
    return {split: torch.tensor([i for i, label in enumerate(features["splits"]) if label == split],
                                dtype=torch.long)
            for split in ("train", "val", "test")}


@torch.inference_mode()
def infer(model: nn.Module, features: dict, indices: torch.Tensor,
          device: torch.device, batch_size: int, weights: torch.Tensor | None = None,
          label_smoothing: float = 0.0, return_attention: bool = False):
    model.eval()
    probabilities, attention = [], []
    loss_total = 0.0
    for start in range(0, len(indices), batch_size):
        rows = indices[start:start + batch_size]
        x = features["embeddings"][rows].float().to(device, non_blocking=True)
        structural = (features["structural_features"][rows].to(device, non_blocking=True)
                      if "structural_features" in features else None)
        target = features["labels"][rows].to(device, non_blocking=True)
        logits, query_weights = model(x, return_attention=return_attention,
                                      structural=structural)
        probabilities.append(logits.float().softmax(-1).cpu().numpy())
        if weights is not None:
            loss_total += float(F.cross_entropy(logits.float(), target, weight=weights,
                                                label_smoothing=label_smoothing).item()) * len(rows)
        if query_weights is not None:
            attention.append(query_weights.float().cpu().numpy())
    result = np.concatenate(probabilities)
    return result, (loss_total / len(indices) if weights is not None else None), (np.concatenate(attention) if attention else None)


def train_classifiers(config: dict) -> None:
    features = load_features(config)
    indices = split_indices(features)
    classes = features["class_names"]
    settings = config["classifier"]
    output = config["paths"]["output_dir"]
    device = device_for(settings["device"])
    set_seed(int(config["seed"]))
    train_labels = features["labels"][indices["train"]]
    counts = torch.bincount(train_labels, minlength=len(classes)).float()
    if (counts == 0).any():
        raise ValueError("At least one training class is empty")
    class_weights = (counts.mean() / counts).sqrt().clamp(0.5, 2.0).to(device)
    for name in settings["models"]:
        structural_dim = len(features.get("structural_feature_names", []))
        model = make_classifier(name, classes, settings, structural_dim).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["learning_rate"]),
                                      weight_decay=float(settings["weight_decay"]))
        epochs = int(settings["epochs"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=float(settings["minimum_learning_rate"]))
        rng = torch.Generator().manual_seed(int(config["seed"]))
        batch_size = int(settings["batch_size"])
        model_dir = output / name
        model_dir.mkdir(parents=True, exist_ok=True)
        history = []
        best_f1, best_epoch, stale = -1.0, 0, 0
        print(f"Training {name}: frozen STEP width=2048, train={len(indices['train']):,}, "
              f"val={len(indices['val']):,}, device={device}, parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
        for epoch in range(1, epochs + 1):
            model.train()
            shuffled = indices["train"][torch.randperm(len(indices["train"]), generator=rng)]
            total_loss = 0.0
            for start in range(0, len(shuffled), batch_size):
                rows = shuffled[start:start + batch_size]
                x = features["embeddings"][rows].float().to(device, non_blocking=True)
                structural = (features["structural_features"][rows].to(device, non_blocking=True)
                              if "structural_features" in features else None)
                target = features["labels"][rows].to(device, non_blocking=True)
                logits, _ = model(x, structural=structural)
                loss = F.cross_entropy(logits.float(), target, weight=class_weights,
                                       label_smoothing=float(settings["label_smoothing"]))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clip"]))
                optimizer.step()
                total_loss += float(loss.item()) * len(rows)
            validation_prob, validation_loss, _ = infer(
                model, features, indices["val"], device, batch_size, class_weights,
                float(settings["label_smoothing"]))
            validation = classification_metrics(validation_prob,
                                                features["labels"][indices["val"]].numpy(), classes)
            learning_rate = optimizer.param_groups[0]["lr"]
            row = {"epoch": epoch, "train_loss": total_loss / len(shuffled),
                   "val_loss": validation_loss, "val_accuracy": validation["accuracy"],
                   "val_balanced_accuracy": validation["balanced_accuracy"],
                   "val_macro_f1": validation["macro_f1"], "learning_rate": learning_rate}
            history.append(row)
            print(f"{name} epoch={epoch:02d}/{epochs} train_loss={row['train_loss']:.4f} "
                  f"val_loss={validation_loss:.4f} val_acc={validation['accuracy']:.4f} "
                  f"val_macro_F1={validation['macro_f1']:.4f} lr={learning_rate:.2e}", flush=True)
            if validation["macro_f1"] > best_f1 + 1e-4:
                best_f1, best_epoch, stale = validation["macro_f1"], epoch, 0
                torch.save({"state_dict": model.state_dict(), "architecture": name,
                            "classifier_settings": settings, "class_names": classes,
                            "structural_feature_names": features.get("structural_feature_names", []),
                            "source_checkpoint": features["source_checkpoint"],
                            "manifest_sha256": features["manifest_sha256"],
                            "epoch": epoch, "validation_macro_f1": best_f1}, model_dir / "best.pt")
            else:
                stale += 1
            scheduler.step()
            if epoch >= int(settings["min_epochs"]) and stale >= int(settings["patience"]):
                print(f"{name} early stop at epoch {epoch}; selected epoch {best_epoch}", flush=True)
                break
        with (model_dir / "training_history.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        save_json(model_dir / "training_summary.json", {
            "best_epoch": best_epoch, "best_validation_macro_f1": best_f1,
            "completed_epochs": len(history), "train_class_counts": dict(zip(classes, counts.int().tolist(), strict=True)),
            "frozen_checkpoint": features["source_checkpoint"],
            "selection": "validation macro-F1 only; test untouched during training",
        })
