"""Held-out comparison of full-width pooled versus pooled + local JEPA nodes."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from common import device_for, save_json, source_manifest
from dataset import FrozenLocalFeatures, collate_local
from metrics import classification_metrics
from model import build_model
from train import _device_batch, _loader, infer


def _plot_confusion(matrix: list[list[int]], classes: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(matrix, dtype=float)
    values /= np.maximum(values.sum(axis=1, keepdims=True), 1)
    figure, axis = plt.subplots(figsize=(10, 9))
    image = axis.imshow(values, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(np.arange(len(classes)), classes, rotation=55, ha="right")
    axis.set_yticks(np.arange(len(classes)), classes)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    figure.colorbar(image, ax=axis, fraction=0.045)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


@torch.inference_mode()
def _top_local_nodes(model, example: dict, vocabulary: list[str],
                     device: torch.device) -> str:
    if not model.use_local:
        return ""
    original = collate_local([example])
    batch = _device_batch(original, device)
    model.eval()
    _, weights = model(batch, return_attention=True)
    if weights is None:
        return ""
    importance = weights[0].float().mean(0).cpu().numpy()
    valid = original["valid"][0].numpy()
    positions = np.flatnonzero(valid)
    top = sorted(positions, key=lambda i: -importance[i])[:8]
    roles = ("global", "type", "instance")
    entries = []
    for position in top:
        type_id = int(original["type_ids"][0, position])
        name = vocabulary[type_id] if 0 < type_id < len(vocabulary) else "OOV/PAD"
        role = roles[int(original["roles"][0, position])]
        entries.append(f"{position}:{role}:{name}:{importance[position]:.4f}")
    return "|".join(entries)


def _qualitative_cases(model, dataset: FrozenLocalFeatures, probabilities: np.ndarray,
                       truth: np.ndarray, config: dict, model_dir: Path,
                       device: torch.device) -> None:
    manifest, _, _ = source_manifest(config)
    root = Path(manifest["dataset_root"])
    classes = dataset.metadata["classes"]
    predictions = probabilities.argmax(1)
    confidence = probabilities.max(1)
    rows = []
    for class_id, class_name in enumerate(classes):
        own = np.flatnonzero(truth == class_id)
        for kind, candidates in (("correct", own[predictions[own] == class_id]),
                                 ("error", own[predictions[own] != class_id])):
            if not len(candidates):
                continue
            local = sorted(candidates, key=lambda i: (-confidence[i], dataset.metadata["uids"][dataset.indices[i]]))[0]
            global_index = dataset.indices[local]
            sample = dataset[local]
            rows.append({
                "case": kind,
                "uid": dataset.metadata["uids"][global_index],
                "step_path": str(root / dataset.metadata["source_relpaths"][global_index]),
                "true_class": class_name,
                "predicted_class": classes[int(predictions[local])],
                "confidence": float(confidence[local]),
                "top_attended_frozen_nodes": _top_local_nodes(
                    model, sample, dataset.metadata["step_entity_types"], device,
                ),
            })
    if rows:
        with (model_dir / "qualitative_cases.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def evaluate_models(config: dict) -> None:
    settings = config["classifier"]
    output = config["paths"]["output"]
    device = device_for(settings["device"])
    comparison = []
    for name in settings["models"]:
        include_local = name == "pooled_plus_local"
        validation_set = FrozenLocalFeatures(config, "val", include_local)
        test_set = FrozenLocalFeatures(config, "test", include_local)
        classes = test_set.metadata["classes"]
        model_dir = output / name
        path = model_dir / "best.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Train {name} before evaluating: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if (checkpoint.get("architecture") != name
                or checkpoint.get("classes") != classes
                or checkpoint.get("manifest_sha256") != test_set.metadata["manifest_sha256"]
                or checkpoint.get("source_checkpoint") != test_set.metadata["checkpoint"]
                or checkpoint.get("feature_format") != test_set.metadata["format"]):
            raise ValueError(f"{name} checkpoint does not match frozen features or class order")
        model = build_model(name, len(classes), checkpoint["settings"]).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        val_loader = _loader(validation_set, settings, False, int(config["seed"]))
        test_loader = _loader(test_set, settings, False, int(config["seed"]))
        val_prob, val_truth, val_indices, _ = infer(model, val_loader, device)
        test_prob, test_truth, test_indices, _ = infer(model, test_loader, device)
        if (val_indices.tolist() != validation_set.indices
                or test_indices.tolist() != test_set.indices):
            raise ValueError("Evaluation loader changed sample order")
        validation = classification_metrics(val_prob, val_truth, classes)
        test = classification_metrics(test_prob, test_truth, classes)
        result = {
            "architecture": name,
            "selected_epoch": int(checkpoint["epoch"]),
            "input": ("2048-D frozen pooled STEP + every frozen 512-D pre-pooling graph node"
                      if include_local else "unchanged 2048-D frozen pooled STEP only"),
            "frozen_checkpoint": test_set.metadata["checkpoint"],
            "validation": validation,
            "test": test,
        }
        save_json(model_dir / "evaluation.json", result)
        with (model_dir / "per_class_test.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("class", "train_n", "val_n", "test_n", "precision", "recall", "f1"))
            for class_id, class_name in enumerate(classes):
                train_n = sum(test_set.metadata["labels"][i] == class_id
                              for i, split in enumerate(test_set.metadata["splits"]) if split == "train")
                val_n = sum(test_set.metadata["labels"][i] == class_id for i in validation_set.indices)
                item = test["per_class"][class_name]
                writer.writerow((class_name, train_n, val_n, item["support"],
                                 item["precision"], item["recall"], item["f1"]))
        with (model_dir / "test_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("uid", "true_class", "predicted_class", "confidence", "top3_classes"))
            for local, global_index in enumerate(test_indices):
                order = np.argsort(test_prob[local])[::-1]
                writer.writerow((test_set.metadata["uids"][int(global_index)],
                                 classes[int(test_truth[local])], classes[int(order[0])],
                                 float(test_prob[local, order[0]]),
                                 "|".join(classes[int(j)] for j in order[:3])))
        _plot_confusion(test["confusion_matrix"], classes, model_dir / "confusion_matrix.png")
        _qualitative_cases(model, test_set, test_prob, test_truth, config, model_dir, device)
        comparison.append({
            "model": name, "selected_epoch": checkpoint["epoch"],
            "val_accuracy": validation["accuracy"], "val_macro_f1": validation["macro_f1"],
            "test_accuracy": test["accuracy"], "test_balanced_accuracy": test["balanced_accuracy"],
            "test_macro_f1": test["macro_f1"], "test_top3_accuracy": test["top_3_accuracy"],
        })
        print(f"{name}: val_macro_F1={validation['macro_f1']:.4f} "
              f"test_accuracy={test['accuracy']:.4f} test_macro_F1={test['macro_f1']:.4f}", flush=True)
    with (output / "model_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
