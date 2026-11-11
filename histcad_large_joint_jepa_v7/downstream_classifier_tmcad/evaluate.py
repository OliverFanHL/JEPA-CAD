"""Held-out classification, frozen-feature 1-NN, and inspectable test cases."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from classifier import TOKEN_NAMES, make_classifier
from common import device_for, save_json
from metrics import classification_metrics
from torch.nn import functional as F
from train import infer, load_features, split_indices


def _plot_confusion(matrix: list[list[int]], classes: list[str], destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(matrix)
    normalized = values / np.maximum(values.sum(1, keepdims=True), 1)
    figure, axis = plt.subplots(figsize=(10, 9))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set_xticks(np.arange(len(classes)), classes, rotation=55, ha="right")
    axis.set_yticks(np.arange(len(classes)), classes)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Held-out test confusion matrix, row-normalized")
    figure.colorbar(image, ax=axis, fraction=0.045)
    for row in range(len(classes)):
        for column in range(len(classes)):
            value = normalized[row, column]
            if value >= 0.04 or row == column:
                axis.text(column, row, f"{value:.2f}", ha="center", va="center",
                          color="white" if value > 0.55 else "#253449", fontsize=8)
    figure.tight_layout()
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def _centroid_analysis(features: dict, indices: dict[str, torch.Tensor],
                       classes: list[str], output: Path) -> None:
    """Show which class means overlap in the original frozen STEP space."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers = []
    for class_id in range(len(classes)):
        rows = indices["train"][features["labels"][indices["train"]] == class_id]
        center = features["embeddings"][rows].float().mean(0)
        centers.append(F.normalize(center, dim=0))
    similarities = (torch.stack(centers) @ torch.stack(centers).T).numpy()
    save_json(output / "frozen_step_class_centroids.json", {
        "classes": classes, "cosine_similarity": similarities.tolist(),
        "interpretation": "Training-class centroid cosine in the frozen STEP representation; not a classifier score."})
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(similarities, cmap="viridis", vmin=0, vmax=1)
    axis.set_xticks(np.arange(len(classes)), classes, rotation=55, ha="right")
    axis.set_yticks(np.arange(len(classes)), classes)
    axis.set_title("Frozen STEP: training-class centroid similarity")
    figure.colorbar(image, ax=axis, fraction=0.045)
    figure.tight_layout()
    figure.savefig(output / "frozen_step_class_centroids.png", dpi=150)
    plt.close(figure)


def _nearest_train(features: dict, indices: dict[str, torch.Tensor], device: torch.device,
                   chunk_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    train_ids = indices["train"]
    test_ids = indices["test"]
    reference = F.normalize(features["embeddings"][train_ids].float().to(device), dim=1)
    nearest_rows, similarities = [], []
    for start in range(0, len(test_ids), chunk_size):
        query = F.normalize(features["embeddings"][test_ids[start:start + chunk_size]].float().to(device), dim=1)
        values, columns = (query @ reference.T).max(1)
        nearest_rows.append(train_ids[columns.cpu()].numpy())
        similarities.append(values.cpu().numpy())
    return np.concatenate(nearest_rows), np.concatenate(similarities)


def _quality_breakdown(predicted: np.ndarray, truth: np.ndarray, test_ids: torch.Tensor,
                       features: dict, warning_fraction: float) -> dict:
    unknown = np.asarray(features["unknown_type_fraction"])[test_ids.numpy()]
    truncated = np.asarray(features["brep_truncated"])[test_ids.numpy()]
    correct = predicted == truth
    result = {}
    for name, mask in (("low_oov", unknown <= warning_fraction),
                       ("high_oov", unknown > warning_fraction),
                       ("brep_not_truncated", ~truncated), ("brep_truncated", truncated)):
        result[name] = {"samples": int(mask.sum()),
                        "accuracy": float(correct[mask].mean()) if mask.any() else None}
    result["per_class"] = {}
    for class_id, class_name in enumerate(features["class_names"]):
        own = truth == class_id
        groups = {}
        for name, mask in (("low_oov", unknown <= warning_fraction),
                           ("high_oov", unknown > warning_fraction),
                           ("brep_not_truncated", ~truncated), ("brep_truncated", truncated)):
            selected = own & mask
            groups[name] = {"samples": int(selected.sum()),
                            "accuracy": float(correct[selected].mean()) if selected.any() else None}
        result["per_class"][class_name] = groups
    return result


def _write_qualitative(model_dir: Path, config: dict, features: dict,
                       test_ids: torch.Tensor, probabilities: np.ndarray,
                       neighbors: np.ndarray, similarities: np.ndarray) -> None:
    classes = features["class_names"]
    truth = features["labels"][test_ids].numpy()
    forecast = probabilities.argmax(1)
    confidence = probabilities.max(1)
    dataset_root = config["paths"]["dataset_root"]
    cases = []
    limit = int(config["qualitative"]["cases_per_class"])
    for label_id, name in enumerate(classes):
        own = np.flatnonzero(truth == label_id)
        for kind, candidates in (("correct", own[forecast[own] == label_id]),
                                 ("error", own[forecast[own] != label_id])):
            ordered = sorted(candidates, key=lambda i: (-confidence[i], features["uids"][int(test_ids[i])]))
            for local in ordered[:limit]:
                global_id = int(test_ids[local])
                neighbor_id = int(neighbors[local])
                cases.append({"case_type": kind, "uid": features["uids"][global_id],
                              "step_path": str(dataset_root / features["source_relpaths"][global_id]),
                              "true_class": name, "predicted_class": classes[int(forecast[local])],
                              "confidence": float(confidence[local]),
                              "nearest_train_uid": features["uids"][neighbor_id],
                              "nearest_train_step_path": str(dataset_root / features["source_relpaths"][neighbor_id]),
                              "nearest_train_class": classes[int(features["labels"][neighbor_id])],
                              "nearest_train_cosine": float(similarities[local])})
    with (model_dir / "qualitative_cases.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)


def evaluate_classifiers(config: dict) -> dict:
    features = load_features(config)
    indices = split_indices(features)
    classes = features["class_names"]
    settings = config["classifier"]
    device = device_for(settings["device"])
    output = config["paths"]["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    _centroid_analysis(features, indices, classes, output)
    # Same frozen-neighbor baseline for both downstream heads.
    neighbors, similarities = _nearest_train(features, indices, device)
    truth = features["labels"][indices["test"]].numpy()
    neighbor_labels = features["labels"][torch.from_numpy(neighbors)].numpy()
    nn_result = {"test_1nn_accuracy": float(np.mean(neighbor_labels == truth)),
                 "per_class_1nn_accuracy": {
                     name: float(np.mean(neighbor_labels[truth == j] == j))
                     for j, name in enumerate(classes)
                 },
                 "interpretation": "Cosine 1-NN from held-out test to training vectors, without a trained head."}
    save_json(output / "frozen_step_1nn.json", nn_result)
    with (output / "frozen_step_1nn_examples.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("test_uid", "true_class", "nearest_train_uid", "nearest_train_class", "cosine", "correct"))
        for local, global_id in enumerate(indices["test"].tolist()):
            nearest = int(neighbors[local])
            writer.writerow((features["uids"][global_id], classes[truth[local]],
                             features["uids"][nearest], classes[neighbor_labels[local]],
                             float(similarities[local]), bool(truth[local] == neighbor_labels[local])))
    comparison = []
    for name in settings["models"]:
        model_dir = output / name
        checkpoint_path = model_dir / "best.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Train {name} before evaluating it: {checkpoint_path}")
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved["class_names"] != classes or saved["manifest_sha256"] != features["manifest_sha256"]:
            raise ValueError(f"{name} checkpoint does not match extracted TMCAD labels/split")
        if saved["source_checkpoint"] != features["source_checkpoint"]:
            raise ValueError(f"{name} checkpoint was trained from different frozen features")
        structural_names = features.get("structural_feature_names", [])
        if saved.get("structural_feature_names", []) != structural_names:
            raise ValueError(f"{name} checkpoint uses a different structural feature schema")
        model = make_classifier(name, classes, saved["classifier_settings"], len(structural_names))
        model.load_state_dict(saved["state_dict"], strict=True)
        model.to(device)
        model.eval()
        validation_prob, _, _ = infer(model, features, indices["val"], device,
                                      int(settings["batch_size"]))
        test_prob, _, attention = infer(model, features, indices["test"], device,
                                        int(settings["batch_size"]), return_attention=name.startswith("attention"))
        validation = classification_metrics(validation_prob,
                                            features["labels"][indices["val"]].numpy(), classes)
        test = classification_metrics(test_prob, truth, classes)
        test["transfer_quality_breakdown"] = _quality_breakdown(
            test_prob.argmax(1), truth, indices["test"], features,
            float(config["dataset"]["warn_unknown_type_fraction"]))
        result = {"architecture": name, "selected_epoch": saved["epoch"],
                  "source_checkpoint": features["source_checkpoint"],
                  "input": ("frozen 2048-D JEPA-v7 STEP branch plus cached typed B-Rep/connectivity descriptors"
                            if name.endswith("_structural") else
                            "frozen 2048-D JEPA-v7 STEP branch only; no HistCAD JSON or joint vector"),
                  "structural_feature_names": structural_names if name.endswith("_structural") else [],
                  "validation": validation, "test": test}
        save_json(model_dir / "evaluation.json", result)
        with (model_dir / "per_class_test.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("class", "train_n", "val_n", "test_n", "precision", "recall", "f1",
                             "frozen_step_1nn_accuracy"))
            for class_id, class_name in enumerate(classes):
                row = test["per_class"][class_name]
                train_n = int((features["labels"][indices["train"]] == class_id).sum())
                val_n = int((features["labels"][indices["val"]] == class_id).sum())
                writer.writerow((class_name, train_n, val_n, row["support"], row["precision"], row["recall"],
                                 row["f1"], nn_result["per_class_1nn_accuracy"][class_name]))
        with (model_dir / "test_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("uid", "true_class", "predicted_class", "confidence", "top3_classes"))
            for local, global_id in enumerate(indices["test"].tolist()):
                order = np.argsort(test_prob[local])[::-1]
                writer.writerow((features["uids"][global_id], classes[truth[local]],
                                 classes[int(order[0])], float(test_prob[local, order[0]]),
                                 "|".join(classes[int(j)] for j in order[:3])))
        _plot_confusion(test["confusion_matrix"], classes, model_dir / "confusion_matrix.png")
        _write_qualitative(model_dir, config, features, indices["test"],
                           test_prob, neighbors, similarities)
        if attention is not None:
            save_json(model_dir / "attention_diagnostic.json", {
                "roles": TOKEN_NAMES,
                "mean_query_attention": dict(zip(TOKEN_NAMES, attention.mean(0).tolist(), strict=True)),
                "caution": "Attention weights are diagnostic, not causal attribution."})
        comparison.append({"model": name, "selected_epoch": saved["epoch"],
                           "validation_accuracy": validation["accuracy"],
                           "validation_macro_f1": validation["macro_f1"],
                           "test_accuracy": test["accuracy"],
                           "test_balanced_accuracy": test["balanced_accuracy"],
                           "test_macro_f1": test["macro_f1"],
                           "test_weighted_f1": test["weighted_f1"],
                           "test_top3_accuracy": test["top_3_accuracy"],
                           "frozen_step_1nn_accuracy": nn_result["test_1nn_accuracy"]})
        print(f"{name}: test_accuracy={test['accuracy']:.4f} macro_F1={test['macro_f1']:.4f} "
              f"balanced_accuracy={test['balanced_accuracy']:.4f}", flush=True)
    with (output / "model_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    return {row["model"]: row for row in comparison}
