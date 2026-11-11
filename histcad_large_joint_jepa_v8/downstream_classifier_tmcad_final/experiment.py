"""Final TMCAD classifier: frozen STEP control + 121-D physical residual."""

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


FINAL_NAME = "physical_residual"
CHECKPOINT_FORMAT = "tmcad_final_frozen_step_physical_residual_v1"


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
            for name in ("baseline_logits", "geometry", "mesh", "label")
        }

    def __len__(self) -> int:
        return len(self.values["label"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {name: value[index] for name, value in self.values.items()}


def _loader(arrays: dict[str, np.ndarray], settings: dict,
            shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(
        AdapterRows(arrays), batch_size=int(settings["batch_size"]),
        shuffle=shuffle, num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


def _on_device(batch: dict, device: torch.device) -> dict:
    return {name: tensor.to(device, non_blocking=True)
            for name, tensor in batch.items()}


def _forward(model: ResidualAdapter, batch: dict
             ) -> tuple[torch.Tensor, torch.Tensor]:
    return model(batch["baseline_logits"], batch["geometry"], batch["mesh"])


@torch.inference_mode()
def _infer(model: ResidualAdapter, arrays: dict[str, np.ndarray],
           settings: dict, device: torch.device,
           weights: torch.Tensor | None = None) -> tuple[np.ndarray, float | None]:
    model.eval()
    logits, total_loss, total = [], 0.0, 0
    for original in _loader(arrays, settings, False, 0):
        batch = _on_device(original, device)
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
                         val: dict[str, np.ndarray]) -> dict:
    score = classification_metrics(
        probabilities(val["baseline_logits"]), val["label"], source.classes,
    )
    expected = float(config["baseline"]["expected_validation_macro_f1"])
    tolerance = float(config["baseline"]["validation_tolerance"])
    if abs(score["macro_f1"] - expected) > tolerance:
        raise ValueError(
            f"Frozen control val macro-F1 {score['macro_f1']:.6f} does not "
            f"match {expected:.6f} within {tolerance:.6f}; inspect inputs/model"
        )
    return score


def _check_settings(settings: dict) -> None:
    if (not 1 <= int(settings["epochs"]) <= 100
            or not 1 <= int(settings["batch_size"]) <= 1024
            or not 1 <= int(settings["patience"]) <= int(settings["epochs"])
            or not 1 <= int(settings["min_epochs"]) <= int(settings["epochs"])
            or not 0 < float(settings["minimum_learning_rate"]) <= float(settings["learning_rate"])
            or not 0 <= float(settings["residual_l2"]) <= 1
            or not 0 <= float(settings["minimum_validation_macro_f1_gain"]) <= 0.05):
        raise ValueError("Invalid final classifier training/selection settings")


def _checkpoint_path(config: dict) -> Path:
    return config["paths"]["output"] / FINAL_NAME / "best.pt"


def _save_checkpoint(path: Path, model: ResidualAdapter,
                     source: TMCADSource, settings: dict,
                     epoch: int, val_macro_f1: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "state_dict": {name: tensor.detach().cpu().clone()
                       for name, tensor in model.state_dict().items()},
        "settings": settings,
        "classes": source.classes,
        "provenance": source.provenance,
        "epoch": epoch,
        "validation_macro_f1": val_macro_f1,
    }
    temporary = path.with_name("best.partial.pt")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_checkpoint(path: Path, source: TMCADSource,
                     device: torch.device) -> tuple[ResidualAdapter, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Run training first: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload.get("format") != CHECKPOINT_FORMAT
            or payload.get("classes") != source.classes
            or payload.get("provenance") != source.provenance):
        raise ValueError("Final classifier checkpoint is incompatible with current frozen inputs")
    model = ResidualAdapter(len(source.classes), payload["settings"])
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval(), payload


def train(config: dict) -> None:
    settings = config["adapter"]
    _check_settings(settings)
    checkpoint_path = _checkpoint_path(config)
    if checkpoint_path.exists():
        raise FileExistsError(f"Refusing to overwrite the final classifier: {checkpoint_path}")
    source = TMCADSource(config)
    device = device_for(config["baseline"]["device"])
    train_data = frozen_split(source, "train", device)
    val_data = frozen_split(source, "val", device)
    baseline = _baseline_validation(config, source, val_data)
    counts = torch.bincount(torch.from_numpy(train_data["label"]),
                            minlength=len(source.classes)).float()
    if (counts == 0).any():
        raise ValueError("At least one class has no training examples")
    weights = (counts.mean() / counts).sqrt().clamp(0.5, 2.0).to(device)
    _seed(int(config["seed"]))
    model = ResidualAdapter(len(source.classes), settings).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(settings["learning_rate"]),
                                  weight_decay=float(settings["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(settings["epochs"]),
        eta_min=float(settings["minimum_learning_rate"]),
    )
    output = config["paths"]["output"]
    save_json(output / "setup.json", {
        "architecture": FINAL_NAME,
        "train_count": len(train_data["label"]),
        "val_count": len(val_data["label"]),
        "train_class_counts": dict(zip(source.classes, counts.int().tolist(), strict=True)),
        "baseline_validation": baseline,
        "frozen_baseline_checkpoint": str(config["paths"]["baseline_checkpoint"]),
        "provenance": source.provenance,
        "selection": "Best epoch by validation macro-F1; no test rows used in training",
    })
    best_f1, best_epoch, stale = baseline["macro_f1"], 0, 0
    _save_checkpoint(checkpoint_path, model, source, settings, 0, best_f1)
    history = [{
        "epoch": 0, "train_loss": "", "val_loss": "",
        "val_accuracy": baseline["accuracy"], "val_macro_f1": best_f1,
        "coupling_recall": baseline["per_class"]["coupling"]["recall"],
        "pulley_recall": baseline["per_class"]["pulley"]["recall"],
        "learning_rate": float(settings["learning_rate"]),
    }]
    print(f"Final physical residual: train={len(train_data['label']):,} "
          f"val={len(val_data['label']):,} frozen-control val_macro_F1={best_f1:.4f} "
          f"trainable={sum(p.numel() for p in model.parameters()):,} "
          f"device={device}", flush=True)
    for epoch in range(1, int(settings["epochs"]) + 1):
        model.train()
        loss_total, count = 0.0, 0
        for original in _loader(train_data, settings, True, int(config["seed"]) + epoch):
            batch = _on_device(original, device)
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
            loss_total += float(loss.detach()) * size
            count += size
        val_logits, val_loss = _infer(model, val_data, settings, device, weights)
        val = classification_metrics(
            probabilities(val_logits), val_data["label"], source.classes,
        )
        row = {
            "epoch": epoch, "train_loss": loss_total / count,
            "val_loss": val_loss, "val_accuracy": val["accuracy"],
            "val_macro_f1": val["macro_f1"],
            "coupling_recall": val["per_class"]["coupling"]["recall"],
            "pulley_recall": val["per_class"]["pulley"]["recall"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(f"epoch={epoch:02d}/{settings['epochs']} "
              f"train={row['train_loss']:.4f} val={val_loss:.4f} "
              f"val_macro_F1={row['val_macro_f1']:.4f} "
              f"coupling_R={row['coupling_recall']:.3f} "
              f"pulley_R={row['pulley_recall']:.3f} "
              f"lr={row['learning_rate']:.2e}", flush=True)
        if val["macro_f1"] > best_f1 + 1e-4:
            best_f1, best_epoch, stale = val["macro_f1"], epoch, 0
            _save_checkpoint(checkpoint_path, model, source, settings, epoch, best_f1)
        else:
            stale += 1
        scheduler.step()
        if epoch >= int(settings["min_epochs"]) and stale >= int(settings["patience"]):
            print(f"Early stop at epoch {epoch}; selected epoch {best_epoch}", flush=True)
            break
    _write_csv(checkpoint_path.parent / "training_history.csv", history)
    save_json(checkpoint_path.parent / "training_summary.json", {
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_f1,
        "baseline_validation_macro_f1": baseline["macro_f1"],
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


def _prediction_rows(source: TMCADSource, test: dict[str, np.ndarray],
                     baseline_prob: np.ndarray, final_prob: np.ndarray) -> list[dict]:
    rows = []
    classes = source.classes
    for local, global_index in enumerate(test["index"]):
        truth = int(test["label"][local])
        base = int(baseline_prob[local].argmax())
        final = int(final_prob[local].argmax())
        outcome = ("corrected" if base != truth and final == truth else
                   "worsened" if base == truth and final != truth else
                   "still_wrong" if final != truth else "still_correct")
        rows.append({
            "uid": source.metadata["uids"][int(global_index)],
            "step_path": str(source.sample_path(int(global_index))),
            "true_class": classes[truth],
            "baseline_prediction": classes[base],
            "final_prediction": classes[final],
            "baseline_confidence": float(baseline_prob[local, base]),
            "final_confidence": float(final_prob[local, final]),
            "outcome": outcome,
        })
    return rows


def evaluate(config: dict) -> None:
    settings = config["adapter"]
    _check_settings(settings)
    source = TMCADSource(config)
    device = device_for(config["baseline"]["device"])
    val_data = frozen_split(source, "val", device)
    baseline_val = _baseline_validation(config, source, val_data)
    adapter, payload = _load_checkpoint(_checkpoint_path(config), source, device)
    adapted_val_logits, _ = _infer(adapter, val_data, settings, device)
    adapted_val = classification_metrics(
        probabilities(adapted_val_logits), val_data["label"], source.classes,
    )
    if abs(adapted_val["macro_f1"] - float(payload["validation_macro_f1"])) > 0.002:
        raise ValueError("Saved final checkpoint's validation score did not reproduce")
    selected = (FINAL_NAME if adapted_val["macro_f1"] >= baseline_val["macro_f1"]
                + float(settings["minimum_validation_macro_f1_gain"])
                else "frozen_baseline")
    # Held-out test is materialized only after this validation-only decision.
    test_data = frozen_split(source, "test", device)
    base_prob = probabilities(test_data["baseline_logits"])
    base_test = classification_metrics(base_prob, test_data["label"], source.classes)
    if selected == FINAL_NAME:
        final_logits, _ = _infer(adapter, test_data, settings, device)
        final_prob = probabilities(final_logits)
        final_test = classification_metrics(final_prob, test_data["label"], source.classes)
    else:
        final_prob, final_test = base_prob, base_test
    output = config["paths"]["output"]
    validation_models = {"frozen_baseline": baseline_val, FINAL_NAME: adapted_val}
    test_models = {"frozen_baseline": base_test}
    if selected == FINAL_NAME:
        test_models[FINAL_NAME] = final_test
    _write_csv(output / "validation_comparison.csv", [
        {"model": name, "selected": name == selected,
         "accuracy": score["accuracy"], "balanced_accuracy": score["balanced_accuracy"],
         "macro_f1": score["macro_f1"],
         "coupling_recall": score["per_class"]["coupling"]["recall"],
         "pulley_recall": score["per_class"]["pulley"]["recall"]}
        for name, score in validation_models.items()
    ])
    _write_csv(output / "validation_per_class.csv", _class_rows(validation_models, source.classes))
    _write_csv(output / "test_comparison.csv", [
        {"model": name, "accuracy": score["accuracy"],
         "balanced_accuracy": score["balanced_accuracy"],
         "macro_f1": score["macro_f1"],
         "top_3_accuracy": score["top_3_accuracy"],
         "coupling_recall": score["per_class"]["coupling"]["recall"],
         "pulley_recall": score["per_class"]["pulley"]["recall"]}
        for name, score in test_models.items()
    ])
    _write_csv(output / "test_per_class.csv", _class_rows(test_models, source.classes))
    predictions = _prediction_rows(source, test_data, base_prob, final_prob)
    _write_csv(output / "test_predictions.csv", predictions)
    changed = [row for row in predictions
               if row["baseline_prediction"] != row["final_prediction"]]
    _write_csv(output / "changed_predictions.csv", changed, list(predictions[0]))
    qualitative = []
    for class_name in ("coupling", "pulley"):
        for outcome in ("corrected", "worsened", "still_wrong"):
            subset = [row for row in predictions
                      if row["true_class"] == class_name and row["outcome"] == outcome]
            qualitative.extend(sorted(
                subset, key=lambda row: (-row["final_confidence"], row["uid"])
            )[:6])
    _write_csv(output / "qualitative_review.csv", qualitative, list(predictions[0]))
    _plot_confusion(base_test["confusion_matrix"], source.classes,
                    output / "baseline_confusion.png")
    _plot_confusion(final_test["confusion_matrix"], source.classes,
                    output / "final_confusion.png")
    save_json(output / "evaluation.json", {
        "selected_by_validation": selected,
        "selection_rule": "Physical residual is used only if validation macro-F1 beats the frozen control by the configured minimum; test is not used for selection.",
        "validation": validation_models,
        "test": test_models,
        "changed_test_predictions": len(changed),
        "corrected_test_predictions": sum(row["outcome"] == "corrected" for row in predictions),
        "worsened_test_predictions": sum(row["outcome"] == "worsened" for row in predictions),
        "provenance": source.provenance,
    })
    # Render the same held-out predictions used above. This is part of the
    # finalized evaluation, not a separate qualitative selection protocol.
    from visualize_final import render_final_results
    render_final_results(
        source, output, base_test, final_test, selected, predictions,
        config["qualitative"],
    )
    print(f"Validation-selected: {selected}. Test macro-F1 "
          f"{base_test['macro_f1']:.4f} -> {final_test['macro_f1']:.4f}; "
          f"coupling recall {base_test['per_class']['coupling']['recall']:.3f} -> "
          f"{final_test['per_class']['coupling']['recall']:.3f}; "
          f"pulley recall {base_test['per_class']['pulley']['recall']:.3f} -> "
          f"{final_test['per_class']['pulley']['recall']:.3f}", flush=True)
    print(f"Saved final classifier evaluation: {output}", flush=True)
