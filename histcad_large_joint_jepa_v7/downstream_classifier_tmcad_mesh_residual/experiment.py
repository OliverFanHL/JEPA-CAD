"""Frozen-control nonlinear surface-readout experiment, validation-selected."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from data import TMCADSource, device_for, frozen_split, save_json
from metrics import classification_metrics, probabilities
from model import ResidualAdapter
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


VARIANTS = ("physical_residual", "conditional_residual")


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    if columns is None:
        if not rows:
            raise ValueError(f"No rows for {path}")
        columns = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class AdapterRows(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray]):
        self.values = {
            name: torch.from_numpy(np.ascontiguousarray(arrays[name]))
            for name in ("baseline_logits", "hidden", "geometry", "mesh", "label")
        }

    def __len__(self) -> int:
        return len(self.values["label"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {name: value[index] for name, value in self.values.items()}


def _loader(arrays: dict[str, np.ndarray], settings: dict, shuffle: bool,
            seed: int) -> DataLoader:
    return DataLoader(
        AdapterRows(arrays), batch_size=int(settings["batch_size"]),
        shuffle=shuffle, num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


def _device_batch(batch: dict, device: torch.device) -> dict:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def _forward(model: ResidualAdapter, batch: dict
             ) -> tuple[torch.Tensor, torch.Tensor]:
    return model(batch["baseline_logits"], batch["hidden"],
                 batch["geometry"], batch["mesh"])


@torch.inference_mode()
def _infer(model: ResidualAdapter, arrays: dict[str, np.ndarray],
           settings: dict, device: torch.device,
           weights: torch.Tensor | None = None) -> tuple[np.ndarray, float | None]:
    model.eval()
    logits, total_loss, total = [], 0.0, 0
    for original in _loader(arrays, settings, False, 0):
        batch = _device_batch(original, device)
        output, residual = _forward(model, batch)
        logits.append(output.float().cpu().numpy())
        if weights is not None:
            loss = F.cross_entropy(
                output.float(), batch["label"], weight=weights,
                label_smoothing=float(settings["label_smoothing"]),
            ) + float(settings["residual_l2"]) * residual.square().mean()
            size = len(batch["label"])
            total_loss += float(loss) * size
            total += size
    return np.concatenate(logits), (total_loss / total if total else None)


def _baseline_validation(config: dict, source: TMCADSource,
                         arrays: dict[str, np.ndarray]) -> dict:
    score = classification_metrics(
        probabilities(arrays["baseline_logits"]), arrays["label"], source.classes,
    )
    expected = float(config["baseline"]["expected_validation_macro_f1"])
    tolerance = float(config["baseline"]["validation_tolerance"])
    if abs(score["macro_f1"] - expected) > tolerance:
        raise ValueError(
            f"Frozen baseline validation macro-F1 {score['macro_f1']:.6f} does not "
            f"reproduce the existing {expected:.6f} within {tolerance:.6f}; "
            "check checkpoint, caches, and baseline architecture before training"
        )
    return score


def _checkpoint_path(output: Path, variant: str) -> Path:
    return output / variant / "best.pt"


def _save_checkpoint(path: Path, model: ResidualAdapter, variant: str,
                     source: TMCADSource, settings: dict, epoch: int,
                     val_macro_f1: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "tmcad_frozen_mesh_residual_v1",
        "variant": variant,
        "state_dict": {name: tensor.detach().cpu().clone()
                       for name, tensor in model.state_dict().items()},
        "settings": settings, "classes": source.classes,
        "provenance": source.provenance,
        "epoch": epoch, "validation_macro_f1": val_macro_f1,
    }
    temporary = path.with_name("best.partial.pt")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_checkpoint(path: Path, variant: str, source: TMCADSource,
                     hidden_width: int, device: torch.device) -> tuple[ResidualAdapter, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Train the {variant} adapter first: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload.get("format") != "tmcad_frozen_mesh_residual_v1"
            or payload.get("variant") != variant
            or payload.get("classes") != source.classes
            or payload.get("provenance") != source.provenance):
        raise ValueError(f"{variant} checkpoint does not match this frozen baseline/split")
    model = ResidualAdapter(variant, len(source.classes), payload["settings"],
                            hidden_width)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval(), payload


def _validate_settings(settings: dict) -> None:
    variants = list(settings["variants"])
    if not variants or len(variants) != len(set(variants)) or any(
        variant not in VARIANTS for variant in variants
    ):
        raise ValueError("Configure one or both supported residual variants")
    if (not 1 <= int(settings["epochs"]) <= 100
            or not 1 <= int(settings["batch_size"]) <= 1024
            or not 1 <= int(settings["patience"]) <= int(settings["epochs"])
            or not 0 <= float(settings["residual_l2"]) <= 1
            or not 0 <= float(settings["minimum_validation_macro_f1_gain"]) <= 0.05
            or not 0 <= float(settings["minimum_conditional_over_physical_gain"]) <= 0.05):
        raise ValueError("Invalid residual training/selection settings")


def train(config: dict) -> None:
    settings = config["adapter"]
    _validate_settings(settings)
    output = config["paths"]["output"]
    for variant in settings["variants"]:
        path = _checkpoint_path(output, variant)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite a completed adapter: {path}")
    source = TMCADSource(config)
    device = device_for(config["baseline"]["device"])
    train_arrays = frozen_split(source, "train", device)
    val_arrays = frozen_split(source, "val", device)
    baseline_val = _baseline_validation(config, source, val_arrays)
    hidden_width = int(train_arrays["hidden"].shape[1])
    counts = torch.bincount(torch.from_numpy(train_arrays["label"]),
                            minlength=len(source.classes)).float()
    if (counts == 0).any():
        raise ValueError("No training examples for one or more classes")
    weights = (counts.mean() / counts).sqrt().clamp(0.5, 2.0).to(device)
    save_json(output / "setup.json", {
        "train_count": len(train_arrays["label"]),
        "val_count": len(val_arrays["label"]),
        "train_class_counts": dict(zip(source.classes, counts.int().tolist(), strict=True)),
        "frozen_baseline_validation": baseline_val,
        "baseline_checkpoint": str(config["paths"]["baseline_checkpoint"]),
        "provenance": source.provenance,
        "selection": "Validation macro-F1 only. Test input is not opened during train().",
    })
    for variant in settings["variants"]:
        path = _checkpoint_path(output, variant)
        _seed(int(config["seed"]))
        model = ResidualAdapter(variant, len(source.classes), settings,
                                hidden_width).to(device)
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=float(settings["learning_rate"]),
                                      weight_decay=float(settings["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(settings["epochs"]),
            eta_min=float(settings["minimum_learning_rate"]),
        )
        best_f1, best_epoch, stale = baseline_val["macro_f1"], 0, 0
        _save_checkpoint(path, model, variant, source, settings, 0, best_f1)
        history = [{
            "epoch": 0, "train_loss": "", "val_loss": "",
            "val_accuracy": baseline_val["accuracy"],
            "val_macro_f1": best_f1,
            "coupling_recall": baseline_val["per_class"]["coupling"]["recall"],
            "pulley_recall": baseline_val["per_class"]["pulley"]["recall"],
            "learning_rate": float(settings["learning_rate"]),
        }]
        print(f"{variant}: frozen baseline val_macro_F1={best_f1:.4f} "
              f"trainable={sum(p.numel() for p in model.parameters()):,} "
              f"device={device}", flush=True)
        for epoch in range(1, int(settings["epochs"]) + 1):
            model.train()
            total_loss, total = 0.0, 0
            for original in _loader(train_arrays, settings, True, int(config["seed"]) + epoch):
                batch = _device_batch(original, device)
                logits, residual = _forward(model, batch)
                loss = F.cross_entropy(
                    logits.float(), batch["label"], weight=weights,
                    label_smoothing=float(settings["label_smoothing"]),
                ) + float(settings["residual_l2"]) * residual.square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), float(settings["gradient_clip"]))
                optimizer.step()
                size = len(batch["label"])
                total_loss += float(loss.detach()) * size
                total += size
            val_logits, val_loss = _infer(model, val_arrays, settings, device, weights)
            val_metrics = classification_metrics(
                probabilities(val_logits), val_arrays["label"], source.classes,
            )
            row = {
                "epoch": epoch, "train_loss": total_loss / total,
                "val_loss": val_loss, "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "coupling_recall": val_metrics["per_class"]["coupling"]["recall"],
                "pulley_recall": val_metrics["per_class"]["pulley"]["recall"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(row)
            print(f"{variant} epoch={epoch:02d}/{settings['epochs']} "
                  f"train={row['train_loss']:.4f} val={val_loss:.4f} "
                  f"val_macro_F1={row['val_macro_f1']:.4f} "
                  f"coupling_R={row['coupling_recall']:.3f} "
                  f"pulley_R={row['pulley_recall']:.3f} "
                  f"lr={row['learning_rate']:.2e}", flush=True)
            if val_metrics["macro_f1"] > best_f1 + 1e-4:
                best_f1, best_epoch, stale = val_metrics["macro_f1"], epoch, 0
                _save_checkpoint(path, model, variant, source, settings, epoch, best_f1)
            else:
                stale += 1
            scheduler.step()
            if epoch >= int(settings["min_epochs"]) and stale >= int(settings["patience"]):
                print(f"{variant}: early stop at {epoch}, best epoch={best_epoch}", flush=True)
                break
        _write_csv(path.parent / "training_history.csv", history)
        save_json(path.parent / "training_summary.json", {
            "best_epoch": best_epoch, "best_validation_macro_f1": best_f1,
            "baseline_validation_macro_f1": baseline_val["macro_f1"],
            "completed_epochs": len(history) - 1,
            "test_used": False,
        })


def _class_rows(metrics_by_model: dict[str, dict], classes: list[str]) -> list[dict]:
    return [
        {"model": model, "class": class_name,
         "support": scores["per_class"][class_name]["support"],
         "precision": scores["per_class"][class_name]["precision"],
         "recall": scores["per_class"][class_name]["recall"],
         "f1": scores["per_class"][class_name]["f1"]}
        for model, scores in metrics_by_model.items() for class_name in classes
    ]


def _plot_confusion(matrix: list[list[int]], classes: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(matrix, dtype=float)
    values /= np.maximum(values.sum(axis=1, keepdims=True), 1)
    figure, axis = plt.subplots(figsize=(10, 9))
    try:
        image = axis.imshow(values, cmap="Blues", vmin=0, vmax=1)
        axis.set_xticks(np.arange(len(classes)), classes, rotation=55, ha="right")
        axis.set_yticks(np.arange(len(classes)), classes)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        figure.colorbar(image, ax=axis, fraction=0.045)
        figure.tight_layout()
        figure.savefig(path, dpi=150)
    finally:
        plt.close(figure)


def _test_rows(source: TMCADSource, test: dict[str, np.ndarray],
               baseline_prob: np.ndarray, selected_prob: np.ndarray) -> list[dict]:
    rows = []
    classes = source.classes
    for local, global_index in enumerate(test["index"]):
        truth = int(test["label"][local])
        base = int(baseline_prob[local].argmax())
        selected = int(selected_prob[local].argmax())
        outcome = ("corrected" if base != truth and selected == truth else
                   "worsened" if base == truth and selected != truth else
                   "still_wrong" if selected != truth else "still_correct")
        rows.append({
            "uid": source.metadata["uids"][int(global_index)],
            "step_path": str(source.sample_path(int(global_index))),
            "true_class": classes[truth],
            "baseline_prediction": classes[base],
            "selected_prediction": classes[selected],
            "baseline_confidence": float(baseline_prob[local, base]),
            "selected_confidence": float(selected_prob[local, selected]),
            "outcome": outcome,
        })
    return rows


def evaluate(config: dict) -> None:
    settings = config["adapter"]
    _validate_settings(settings)
    source = TMCADSource(config)
    device = device_for(config["baseline"]["device"])
    val = frozen_split(source, "val", device)
    baseline_val = _baseline_validation(config, source, val)
    output = config["paths"]["output"]
    val_metrics = {"frozen_baseline": baseline_val}
    for variant in settings["variants"]:
        model, payload = _load_checkpoint(
            _checkpoint_path(output, variant), variant, source,
            val["hidden"].shape[1], device,
        )
        logits, _ = _infer(model, val, settings, device)
        score = classification_metrics(probabilities(logits), val["label"], source.classes)
        if abs(score["macro_f1"] - float(payload["validation_macro_f1"])) > 0.002:
            raise ValueError(f"{variant} validation score does not reproduce its saved checkpoint")
        val_metrics[variant] = score
    ordered = sorted(settings["variants"],
                     key=lambda variant: val_metrics[variant]["macro_f1"], reverse=True)
    selected = ordered[0]
    baseline_f1 = baseline_val["macro_f1"]
    if val_metrics[selected]["macro_f1"] < baseline_f1 + float(
        settings["minimum_validation_macro_f1_gain"]
    ):
        selected = "frozen_baseline"
    elif (selected == "conditional_residual" and "physical_residual" in val_metrics
          and val_metrics["physical_residual"]["macro_f1"] >= baseline_f1 + float(
              settings["minimum_validation_macro_f1_gain"])
          and val_metrics[selected]["macro_f1"] < val_metrics["physical_residual"]["macro_f1"]
              + float(settings["minimum_conditional_over_physical_gain"])):
        selected = "physical_residual"
    # Only the frozen control and validation-selected variant are exposed to
    # the held-out test split. Unselected variants cannot influence selection.
    test = frozen_split(source, "test", device)
    baseline_test_prob = probabilities(test["baseline_logits"])
    baseline_test = classification_metrics(baseline_test_prob, test["label"], source.classes)
    if selected == "frozen_baseline":
        selected_test_prob = baseline_test_prob
        selected_test = baseline_test
    else:
        model, _ = _load_checkpoint(
            _checkpoint_path(output, selected), selected, source,
            test["hidden"].shape[1], device,
        )
        selected_logits, _ = _infer(model, test, settings, device)
        selected_test_prob = probabilities(selected_logits)
        selected_test = classification_metrics(
            selected_test_prob, test["label"], source.classes,
        )
    test_metrics = {"frozen_baseline": baseline_test}
    if selected != "frozen_baseline":
        test_metrics[selected] = selected_test
    _write_csv(output / "validation_comparison.csv", [
        {"model": name, "selected": name == selected,
         "accuracy": score["accuracy"], "balanced_accuracy": score["balanced_accuracy"],
         "macro_f1": score["macro_f1"], "coupling_recall": score["per_class"]["coupling"]["recall"],
         "pulley_recall": score["per_class"]["pulley"]["recall"]}
        for name, score in val_metrics.items()
    ])
    _write_csv(output / "validation_per_class.csv", _class_rows(val_metrics, source.classes))
    _write_csv(output / "test_comparison.csv", [
        {"model": name, "accuracy": score["accuracy"],
         "balanced_accuracy": score["balanced_accuracy"],
         "macro_f1": score["macro_f1"],
         "top_3_accuracy": score["top_3_accuracy"],
         "coupling_recall": score["per_class"]["coupling"]["recall"],
         "pulley_recall": score["per_class"]["pulley"]["recall"]}
        for name, score in test_metrics.items()
    ])
    _write_csv(output / "test_per_class.csv", _class_rows(test_metrics, source.classes))
    predictions = _test_rows(source, test, baseline_test_prob, selected_test_prob)
    _write_csv(output / "test_predictions.csv", predictions)
    changed = [row for row in predictions
               if row["baseline_prediction"] != row["selected_prediction"]]
    _write_csv(output / "changed_predictions.csv", changed, list(predictions[0]))
    qualitative = []
    for class_name in ("coupling", "pulley"):
        for outcome in ("corrected", "worsened", "still_wrong"):
            subset = [row for row in predictions
                      if row["true_class"] == class_name and row["outcome"] == outcome]
            qualitative.extend(sorted(
                subset, key=lambda row: (-row["selected_confidence"], row["uid"])
            )[:6])
    _write_csv(output / "qualitative_review.csv", qualitative, list(predictions[0]))
    _plot_confusion(baseline_test["confusion_matrix"], source.classes,
                    output / "baseline_confusion.png")
    _plot_confusion(selected_test["confusion_matrix"], source.classes,
                    output / "selected_confusion.png")
    save_json(output / "evaluation.json", {
        "selected_by_validation": selected,
        "selection_rule": "Minimum validation macro-F1 gain over baseline; conditional variant must also beat the physical-only variant by configured margin. No test-based model selection.",
        "validation": val_metrics,
        "test": test_metrics,
        "changed_test_predictions": len(changed),
        "corrected_test_predictions": sum(row["outcome"] == "corrected" for row in predictions),
        "worsened_test_predictions": sum(row["outcome"] == "worsened" for row in predictions),
        "provenance": source.provenance,
    })
    print(f"Selected by validation: {selected}. "
          f"Test macro-F1 {baseline_test['macro_f1']:.4f} -> {selected_test['macro_f1']:.4f}; "
          f"coupling recall {baseline_test['per_class']['coupling']['recall']:.3f} -> "
          f"{selected_test['per_class']['coupling']['recall']:.3f}; "
          f"pulley recall {baseline_test['per_class']['pulley']['recall']:.3f} -> "
          f"{selected_test['per_class']['pulley']['recall']:.3f}", flush=True)
    print(f"Saved results: {output}", flush=True)
