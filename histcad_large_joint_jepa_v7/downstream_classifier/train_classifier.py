"""Multi-task linear probes; pretrained v7 embeddings never receive gradients."""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.nn import functional as F

from labels import TASKS


class LinearProbe(nn.Module):
    def __init__(self, width: int, report: dict):
        super().__init__()
        self.heads = nn.ModuleDict({
            task: nn.Linear(width, len(report["tasks"][task]["retained_classes"]))
            for task in TASKS
        })

    def forward(self, vectors: torch.Tensor) -> dict[str, torch.Tensor]:
        return {task: head(vectors) for task, head in self.heads.items()}


class MLPProbe(nn.Module):
    """A small nonlinear readout; the JEPA encoder remains frozen."""

    def __init__(self, width: int, report: dict, hidden: int = 512):
        super().__init__()
        self.readout = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Dropout(0.1))
        self.heads = nn.ModuleDict({
            task: nn.Linear(hidden, len(report["tasks"][task]["retained_classes"]))
            for task in TASKS
        })

    def forward(self, vectors: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.readout(vectors)
        return {task: head(hidden) for task, head in self.heads.items()}


def _device(name: str) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)


def _split_indices(source: dict) -> dict[str, torch.Tensor]:
    return {
        name: torch.tensor([i for i, item in enumerate(source["metadata"]) if item["split"] == name], dtype=torch.long)
        for name in ("train", "val", "test")
    }


def _class_weights(targets: torch.Tensor, train_indices: torch.Tensor, report: dict, device: torch.device) -> dict[str, torch.Tensor]:
    weights = {}
    for column, task in enumerate(TASKS):
        n = len(report["tasks"][task]["retained_classes"])
        counts = torch.bincount(targets[train_indices, column][targets[train_indices, column] >= 0], minlength=n).float()
        relative = (counts.mean() / counts.clamp_min(1)).sqrt().clamp(0.25, 4.0)
        weights[task] = (relative / relative.mean()).to(device)
    return weights


@torch.inference_mode()
def _predict(model: nn.Module, vectors: torch.Tensor, indices: torch.Tensor, device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    model.eval()
    result: dict[str, list[np.ndarray]] = {task: [] for task in TASKS}
    scale = math.sqrt(vectors.shape[1])
    for start in range(0, len(indices), batch_size):
        batch = vectors[indices[start:start + batch_size]].float().to(device, non_blocking=True) * scale
        logits = model(batch)
        for task in TASKS:
            result[task].append(logits[task].argmax(-1).cpu().numpy())
    return {task: np.concatenate(chunks) for task, chunks in result.items()}


def _metrics(predicted: dict[str, np.ndarray], target: torch.Tensor, indices: torch.Tensor, report: dict) -> dict:
    expected = target[indices].numpy()
    result = {}
    for column, task in enumerate(TASKS):
        valid = expected[:, column] >= 0
        y_true = expected[valid, column]
        y_pred = predicted[task][valid]
        classes = report["tasks"][task]["retained_classes"]
        n = len(classes)
        if not len(y_true):
            raise ValueError(f"No labelled evaluation samples for {task}")
        majority = Counter(y_true.tolist()).most_common(1)[0][0]
        result[task] = {
            "samples": int(len(y_true)),
            "coverage": float(valid.mean()),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(n)), average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, labels=list(range(n)), average="weighted", zero_division=0)),
            "majority_accuracy": float((y_true == majority).mean()),
            "majority_macro_f1": float(f1_score(y_true, np.full_like(y_true, majority), labels=list(range(n)), average="macro", zero_division=0)),
            "per_class_recall": {
                name: float((y_pred[y_true == j] == j).mean()) if (y_true == j).any() else None
                for j, name in enumerate(classes)
            },
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(n))).tolist(),
            "class_order": classes,
        }
        if task == "functional_form":
            parent_by_class = report["tasks"][task]["parent_by_class"]
            parents = sorted(set(parent_by_class.values()))
            lookup = np.asarray([parents.index(parent_by_class[name]) for name in classes])
            true_parent, pred_parent = lookup[y_true], lookup[y_pred]
            result[task]["parent_class_order"] = parents
            result[task]["parent_accuracy"] = float(accuracy_score(true_parent, pred_parent))
            result[task]["parent_macro_f1"] = float(f1_score(true_parent, pred_parent, labels=list(range(len(parents))), average="macro", zero_division=0))
            result[task]["parent_confusion_matrix"] = confusion_matrix(true_parent, pred_parent, labels=list(range(len(parents)))).tolist()
    result["mean_macro_f1"] = float(np.mean([result[task]["macro_f1"] for task in TASKS]))
    return result


def _train_one(
    name: str, architecture: str, vectors: torch.Tensor, source: dict, targets: torch.Tensor,
    report: dict, settings: dict, output_dir: Path,
) -> dict:
    device = _device(settings["device"])
    indices = _split_indices(source)
    model = (LinearProbe(vectors.shape[1], report) if architecture == "linear" else MLPProbe(vectors.shape[1], report)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    class_weights = _class_weights(targets, indices["train"], report, device)
    generator = torch.Generator().manual_seed(settings["seed"])
    best_score, stale, best_epoch = -1.0, 0, 0
    history = []
    batch_size = settings["batch_size"]
    scale = math.sqrt(vectors.shape[1])
    checkpoint_path = output_dir / f"{name}_{architecture}_probe.pt"
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        shuffled = indices["train"][torch.randperm(len(indices["train"]), generator=generator)]
        sum_loss, steps = 0.0, 0
        for start in range(0, len(shuffled), batch_size):
            row_indices = shuffled[start:start + batch_size]
            inputs = vectors[row_indices].float().to(device, non_blocking=True) * scale
            labels = targets[row_indices].to(device, non_blocking=True)
            if not torch.isfinite(inputs).all():
                raise ValueError("Frozen representation contains non-finite coordinates")
            logits = model(inputs)
            losses = []
            for column, task in enumerate(TASKS):
                valid = labels[:, column] >= 0
                if valid.any():
                    losses.append(F.cross_entropy(logits[task][valid], labels[valid, column], weight=class_weights[task]))
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            sum_loss += float(loss.item())
            steps += 1
        val_prediction = _predict(model, vectors, indices["val"], device, batch_size)
        val_metrics = _metrics(val_prediction, targets, indices["val"], report)
        score = val_metrics["mean_macro_f1"]
        history.append({"epoch": epoch, "training_loss": sum_loss / steps, "validation_mean_macro_f1": score,
                        "validation_task_macro_f1": {task: val_metrics[task]["macro_f1"] for task in TASKS}})
        print(f"{name} epoch {epoch}/{settings['epochs']} train_loss={sum_loss/steps:.4f} val_mean_macro_F1={score:.4f} "
              + " ".join(f"{task}={val_metrics[task]['macro_f1']:.3f}" for task in TASKS), flush=True)
        if score > best_score + 1e-5:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({"state_dict": model.state_dict(), "representation": name, "architecture": architecture,
                        "width": vectors.shape[1], "epoch": epoch, "validation_mean_macro_f1": score,
                        "taxonomy_version": report["taxonomy_version"],
                        "classes": {task: report["tasks"][task]["retained_classes"] for task in TASKS}}, checkpoint_path)
        else:
            stale += 1
            if stale >= settings["patience"]:
                print(f"{name}: early stop at epoch {epoch}; best epoch {best_epoch}", flush=True)
                break
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["state_dict"])
    model.to(device)
    validation = _metrics(_predict(model, vectors, indices["val"], device, batch_size), targets, indices["val"], report)
    # Test is touched once, only after all selection decisions for this representation.
    test_prediction = _predict(model, vectors, indices["test"], device, batch_size)
    test = _metrics(test_prediction, targets, indices["test"], report)
    np.savez_compressed(output_dir / f"{name}_{architecture}_test_predictions.npz", uids=np.asarray([source["uids"][i] for i in indices["test"].tolist()]),
                        **{f"pred_{task}": test_prediction[task] for task in TASKS}, true=targets[indices["test"]].numpy())
    return {"best_epoch": best_epoch, "validation": validation, "test": test, "history": history,
            "checkpoint": str(checkpoint_path), "input_width": vectors.shape[1]}


def train_all(config: dict, output_dir: Path, source: dict, targets: torch.Tensor, report: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    settings = {**config["classification"], "seed": seed}
    results = {
        "protocol": {
            "source_checkpoint": source["source_checkpoint"],
            "source_checkpoint_epoch": source["source_checkpoint_epoch"],
            "split": "unchanged v7 train/val/test",
            "model_selection": "validation mean macro-F1 only; test evaluated after selecting each frozen probe",
            "feature_extractor_trainable": False,
            "classifier": "one linear head per task on L2-normalized frozen embedding",
            "note": "Functional/form labels are rule-based from generated text; other tasks are input-derived CAD attributes, not independent human labels.",
        },
        "representations": {},
    }
    experiments = [(name, "linear") for name in settings["representations"]]
    if settings.get("joint_mlp", False):
        experiments.append(("joint", "mlp"))
    for name, architecture in experiments:
        if name not in source["embeddings"]:
            raise KeyError(f"Missing frozen representation branch: {name}")
        vectors = source["embeddings"][name]
        key = name if architecture == "linear" else f"{name}_{architecture}"
        print(f"Training frozen {name} {architecture} probe on {len(vectors):,} x {vectors.shape[1]:,} features", flush=True)
        results["representations"][key] = _train_one(name, architecture, vectors, source, targets, report, settings, output_dir)
        (output_dir / "classification_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_summary(results, output_dir)
    print("Final test mean macro-F1:", {name: round(value["test"]["mean_macro_f1"], 4) for name, value in results["representations"].items()})


def write_summary(results: dict, output_dir: Path) -> None:
    with (output_dir / "classification_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("representation", "task", "test_n", "coverage", "test_accuracy", "test_balanced_accuracy", "test_macro_f1", "majority_accuracy", "parent_macro_f1", "validation_macro_f1", "best_epoch"))
        writer.writeheader()
        for name, run in results["representations"].items():
            for task in TASKS:
                test, val = run["test"][task], run["validation"][task]
                writer.writerow({"representation": name, "task": task, "test_n": test["samples"],
                                 "coverage": test["coverage"], "test_accuracy": test["accuracy"],
                                 "test_balanced_accuracy": test["balanced_accuracy"], "test_macro_f1": test["macro_f1"],
                                 "majority_accuracy": test["majority_accuracy"],
                                 "parent_macro_f1": test.get("parent_macro_f1", ""),
                                 "validation_macro_f1": val["macro_f1"], "best_epoch": run["best_epoch"]})
