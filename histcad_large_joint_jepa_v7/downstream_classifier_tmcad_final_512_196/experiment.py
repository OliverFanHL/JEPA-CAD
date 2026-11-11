"""Fresh 512-D downstream classifier, 196-D residual, and held-out evaluation."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from data import InputRows, TMCADSource, device_for, save_json, sha256
from metrics import classification_metrics, probabilities
from model import AttentionClassifier, ResidualAdapter


BASE_FORMAT = "tmcad_fresh_attention_512_v1"
RESIDUAL_FORMAT = "tmcad_fresh_512_physical_196_v1"


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        if not rows:
            raise ValueError(f"No rows for {path}")
        fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _check(config: dict) -> None:
    base = config["classifier"]
    adapter = config["adapter"]
    if (int(base["hidden_width"]) != 256
            or int(adapter["physical_width"]) != 96
            or int(adapter["fusion_width"]) != 96):
        raise ValueError("This experiment requires 4096→512 and 121→96→96")
    for stage in (base, adapter):
        if (not 1 <= int(stage["epochs"]) <= 100
                or not 1 <= int(stage["batch_size"]) <= 1024
                or not 1 <= int(stage["patience"]) <= int(stage["epochs"])
                or not 1 <= int(stage["min_epochs"]) <= int(stage["epochs"])
                or not 0 < float(stage["minimum_learning_rate"])
                <= float(stage["learning_rate"])):
            raise ValueError("Invalid training schedule or batch size")
    if (not 0 <= float(adapter["residual_l2"]) <= 1
            or not 0 <= float(adapter["minimum_validation_macro_f1_gain"]) <= 0.05):
        raise ValueError("Invalid residual regularization or selection margin")


def _base_path(config: dict) -> Path:
    return config["paths"]["output"] / "base_512" / "best.pt"


def _residual_path(config: dict) -> Path:
    return config["paths"]["output"] / "physical_196" / "best.pt"


def _on_device(batch: dict, device: torch.device) -> dict:
    return {name: tensor.to(device, non_blocking=True)
            for name, tensor in batch.items()}


def _input_loader(source: TMCADSource, split: str, settings: dict,
                  shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(
        InputRows(source, split), batch_size=int(settings["batch_size"]),
        shuffle=shuffle, num_workers=0, pin_memory=False,
        generator=torch.Generator().manual_seed(seed),
    )


def _class_weights(source: TMCADSource, device: torch.device) -> torch.Tensor:
    counts = torch.bincount(
        torch.from_numpy(source.labels[source.indices["train"]]),
        minlength=len(source.classes),
    ).float()
    if (counts == 0).any():
        raise ValueError("A TMCAD class has no training samples")
    return (counts.mean() / counts).sqrt().clamp(0.5, 2.0).to(device)


def _save_base(path: Path, model: AttentionClassifier, source: TMCADSource,
               settings: dict, epoch: int, val_f1: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": BASE_FORMAT, "classes": source.classes,
        "provenance": source.provenance, "settings": settings,
        "epoch": epoch, "validation_macro_f1": val_f1,
        "state_dict": {name: value.detach().cpu().clone()
                       for name, value in model.state_dict().items()},
    }
    temporary = path.with_name("best.partial.pt")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_base(config: dict, source: TMCADSource,
               device: torch.device) -> tuple[AttentionClassifier, dict]:
    path = _base_path(config)
    if not path.is_file():
        raise FileNotFoundError(f"Train the 512-D classifier first: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload.get("format") != BASE_FORMAT
            or payload.get("classes") != source.classes
            or payload.get("provenance") != source.provenance
            or payload.get("settings") != config["classifier"]):
        raise ValueError("512-D classifier checkpoint does not match the current inputs/config")
    stage = payload["settings"]
    model = AttentionClassifier(
        len(source.classes), int(stage["attention_heads"]),
        int(stage["hidden_width"]), float(stage["dropout"]),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.requires_grad_(False)
    return model.to(device).eval(), payload


@torch.inference_mode()
def _extract_base(model: AttentionClassifier, source: TMCADSource,
                  split: str, settings: dict, device: torch.device,
                  weights: torch.Tensor | None = None
                  ) -> tuple[dict[str, np.ndarray], float | None]:
    model.eval()
    arrays: dict[str, list[np.ndarray]] = {
        name: [] for name in ("baseline_logits", "geometry", "mesh", "label", "index")
    }
    loss_total, count = 0.0, 0
    for original in _input_loader(source, split, settings, False, 0):
        batch = _on_device(original, device)
        logits, _ = model(batch)
        arrays["baseline_logits"].append(logits.float().cpu().numpy())
        for name in ("geometry", "mesh", "label", "index"):
            arrays[name].append(original[name].numpy())
        if weights is not None:
            loss = F.cross_entropy(
                logits.float(), batch["label"], weight=weights,
                label_smoothing=float(settings["label_smoothing"]),
            )
            size = len(batch["label"])
            loss_total += float(loss) * size
            count += size
    result = {name: np.concatenate(chunks) for name, chunks in arrays.items()}
    if (not np.array_equal(result["index"], source.indices[split])
            or not np.array_equal(result["label"],
                                  source.labels[source.indices[split]])):
        raise ValueError(f"{split} row ordering or labels changed")
    if any(not np.isfinite(result[name]).all() for name in
           ("baseline_logits", "geometry", "mesh")):
        raise ValueError(f"Nonfinite {split} features/logits")
    return result, (loss_total / count if count else None)


class _AdapterRows(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray]):
        self.values = {name: torch.from_numpy(np.ascontiguousarray(arrays[name]))
                       for name in ("baseline_logits", "geometry", "mesh", "label")}

    def __len__(self) -> int:
        return len(self.values["label"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {name: tensor[index] for name, tensor in self.values.items()}


def _adapter_loader(arrays: dict[str, np.ndarray], settings: dict,
                    shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(
        _AdapterRows(arrays), batch_size=int(settings["batch_size"]),
        shuffle=shuffle, num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


@torch.inference_mode()
def _infer_adapter(model: ResidualAdapter, arrays: dict[str, np.ndarray],
                   settings: dict, device: torch.device,
                   weights: torch.Tensor | None = None
                   ) -> tuple[np.ndarray, float | None]:
    model.eval()
    output, total_loss, count = [], 0.0, 0
    for original in _adapter_loader(arrays, settings, False, 0):
        batch = _on_device(original, device)
        logits, residual = model(batch["baseline_logits"],
                                 batch["geometry"], batch["mesh"])
        output.append(logits.float().cpu().numpy())
        if weights is not None:
            loss = F.cross_entropy(
                logits.float(), batch["label"], weight=weights,
                label_smoothing=float(settings["label_smoothing"]),
            ) + float(settings["residual_l2"]) * residual.square().mean()
            size = len(batch["label"])
            total_loss += float(loss) * size
            count += size
    return np.concatenate(output), (total_loss / count if count else None)


def _save_residual(path: Path, model: ResidualAdapter, source: TMCADSource,
                   settings: dict, base_path: Path,
                   epoch: int, val_f1: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": RESIDUAL_FORMAT, "classes": source.classes,
        "provenance": source.provenance, "settings": settings,
        "base_checkpoint_sha256": sha256(base_path),
        "epoch": epoch, "validation_macro_f1": val_f1,
        "state_dict": {name: value.detach().cpu().clone()
                       for name, value in model.state_dict().items()},
    }
    temporary = path.with_name("best.partial.pt")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_residual(config: dict, source: TMCADSource,
                   device: torch.device) -> tuple[ResidualAdapter, dict]:
    path = _residual_path(config)
    if not path.is_file():
        raise FileNotFoundError(f"Train the 196-D residual first: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload.get("format") != RESIDUAL_FORMAT
            or payload.get("classes") != source.classes
            or payload.get("provenance") != source.provenance
            or payload.get("settings") != config["adapter"]
            or payload.get("base_checkpoint_sha256") != sha256(_base_path(config))):
        raise ValueError("196-D residual checkpoint does not match base/config")
    model = ResidualAdapter(len(source.classes), payload["settings"])
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval(), payload


def train_base(config: dict) -> None:
    _check(config)
    path = _base_path(config)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite the trained 512-D base: {path}")
    source = TMCADSource(config)
    device = device_for(config["device"])
    stage = config["classifier"]
    weights = _class_weights(source, device)
    _seed(int(config["seed"]))
    model = AttentionClassifier(
        len(source.classes), int(stage["attention_heads"]),
        int(stage["hidden_width"]), float(stage["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(stage["learning_rate"]),
        weight_decay=float(stage["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(stage["epochs"]),
        eta_min=float(stage["minimum_learning_rate"]),
    )
    loader = _input_loader(source, "train", stage, True, int(config["seed"]))
    output = config["paths"]["output"]
    save_json(output / "setup.json", {
        "train_count": len(source.indices["train"]),
        "val_count": len(source.indices["val"]),
        "test_count_not_used_for_training": len(source.indices["test"]),
        "train_class_counts": {
            name: int(np.count_nonzero(source.labels[source.indices["train"]] == i))
            for i, name in enumerate(source.classes)
        },
        "representation": "2048 frozen original STEP + 4x512 frozen complementary role summaries + 37+84 physical descriptors",
        "base_head": "Linear(4096, 512) + GELU + dropout + 10-way logits",
        "physical_residual": "121 -> 196 -> 196 -> 10",
        "provenance": source.provenance,
        "selection": "Validation macro-F1 only; test is not extracted during either training stage",
    })
    print(f"Fresh base: device={device} train={len(source.indices['train']):,} "
          f"val={len(source.indices['val']):,} parameters={sum(p.numel() for p in model.parameters()):,}",
          flush=True)
    best, best_epoch, stale, history = -1.0, 0, 0, []
    for epoch in range(1, int(stage["epochs"]) + 1):
        model.train()
        total_loss, count = 0.0, 0
        for original in loader:
            batch = _on_device(original, device)
            logits, _ = model(batch)
            loss = F.cross_entropy(
                logits.float(), batch["label"], weight=weights,
                label_smoothing=float(stage["label_smoothing"]),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(stage["gradient_clip"]))
            optimizer.step()
            size = len(batch["label"])
            total_loss += float(loss.detach()) * size
            count += size
        val_data, val_loss = _extract_base(
            model, source, "val", stage, device, weights,
        )
        metrics = classification_metrics(
            probabilities(val_data["baseline_logits"]), val_data["label"],
            source.classes,
        )
        row = {
            "epoch": epoch, "train_loss": total_loss / count,
            "val_loss": val_loss, "val_accuracy": metrics["accuracy"],
            "val_macro_f1": metrics["macro_f1"],
            "coupling_recall": metrics["per_class"]["coupling"]["recall"],
            "pulley_recall": metrics["per_class"]["pulley"]["recall"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(f"base epoch={epoch:02d}/{stage['epochs']} "
              f"train={row['train_loss']:.4f} val={val_loss:.4f} "
              f"macro_F1={row['val_macro_f1']:.4f} "
              f"lr={row['learning_rate']:.2e}", flush=True)
        if metrics["macro_f1"] > best + 1e-4:
            best, best_epoch, stale = metrics["macro_f1"], epoch, 0
            _save_base(path, model, source, stage, epoch, best)
        else:
            stale += 1
        scheduler.step()
        if epoch >= int(stage["min_epochs"]) and stale >= int(stage["patience"]):
            print(f"Base early stop: epoch {epoch}; selected {best_epoch}", flush=True)
            break
    _write_csv(path.parent / "training_history.csv", history)
    save_json(path.parent / "training_summary.json", {
        "best_epoch": best_epoch, "best_validation_macro_f1": best,
        "completed_epochs": len(history), "test_used": False,
    })


def train_adapter(config: dict) -> None:
    _check(config)
    path = _residual_path(config)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite the trained 196-D residual: {path}")
    source = TMCADSource(config)
    device = device_for(config["device"])
    base, _ = _load_base(config, source, device)
    base_settings = config["classifier"]
    train_data, _ = _extract_base(base, source, "train", base_settings, device)
    val_data, _ = _extract_base(base, source, "val", base_settings, device)
    base_val = classification_metrics(
        probabilities(val_data["baseline_logits"]), val_data["label"], source.classes,
    )
    stage = config["adapter"]
    weights = _class_weights(source, device)
    _seed(int(config["seed"]))
    model = ResidualAdapter(len(source.classes), stage).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(stage["learning_rate"]),
        weight_decay=float(stage["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(stage["epochs"]),
        eta_min=float(stage["minimum_learning_rate"]),
    )
    best, best_epoch, stale = base_val["macro_f1"], 0, 0
    _save_residual(path, model, source, stage, _base_path(config), 0, best)
    history = [{
        "epoch": 0, "train_loss": "", "val_loss": "",
        "val_accuracy": base_val["accuracy"], "val_macro_f1": best,
        "coupling_recall": base_val["per_class"]["coupling"]["recall"],
        "pulley_recall": base_val["per_class"]["pulley"]["recall"],
        "learning_rate": float(stage["learning_rate"]),
    }]
    print(f"196-D residual: train={len(train_data['label']):,} "
          f"val={len(val_data['label']):,} base_val_macro_F1={best:.4f} "
          f"trainable={sum(p.numel() for p in model.parameters()):,}", flush=True)
    for epoch in range(1, int(stage["epochs"]) + 1):
        model.train()
        total_loss, count = 0.0, 0
        for original in _adapter_loader(
                train_data, stage, True, int(config["seed"]) + epoch):
            batch = _on_device(original, device)
            logits, residual = model(batch["baseline_logits"],
                                     batch["geometry"], batch["mesh"])
            loss = F.cross_entropy(
                logits.float(), batch["label"], weight=weights,
                label_smoothing=float(stage["label_smoothing"]),
            ) + float(stage["residual_l2"]) * residual.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(stage["gradient_clip"]))
            optimizer.step()
            size = len(batch["label"])
            total_loss += float(loss.detach()) * size
            count += size
        val_logits, val_loss = _infer_adapter(
            model, val_data, stage, device, weights,
        )
        val = classification_metrics(probabilities(val_logits),
                                     val_data["label"], source.classes)
        row = {
            "epoch": epoch, "train_loss": total_loss / count,
            "val_loss": val_loss, "val_accuracy": val["accuracy"],
            "val_macro_f1": val["macro_f1"],
            "coupling_recall": val["per_class"]["coupling"]["recall"],
            "pulley_recall": val["per_class"]["pulley"]["recall"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(f"residual epoch={epoch:02d}/{stage['epochs']} "
              f"train={row['train_loss']:.4f} val={val_loss:.4f} "
              f"macro_F1={row['val_macro_f1']:.4f} "
              f"lr={row['learning_rate']:.2e}", flush=True)
        if val["macro_f1"] > best + 1e-4:
            best, best_epoch, stale = val["macro_f1"], epoch, 0
            _save_residual(path, model, source, stage, _base_path(config), epoch, best)
        else:
            stale += 1
        scheduler.step()
        if epoch >= int(stage["min_epochs"]) and stale >= int(stage["patience"]):
            print(f"Residual early stop: epoch {epoch}; selected {best_epoch}", flush=True)
            break
    _write_csv(path.parent / "training_history.csv", history)
    save_json(path.parent / "training_summary.json", {
        "best_epoch": best_epoch, "best_validation_macro_f1": best,
        "base_validation_macro_f1": base_val["macro_f1"],
        "completed_epochs": len(history) - 1, "test_used": False,
    })


def _class_rows(models: dict[str, dict], classes: list[str]) -> list[dict]:
    return [{
        "model": name, "class": part,
        "support": score["per_class"][part]["support"],
        "precision": score["per_class"][part]["precision"],
        "recall": score["per_class"][part]["recall"],
        "f1": score["per_class"][part]["f1"],
    } for name, score in models.items() for part in classes]


def _plot_confusion(matrix: list[list[int]], classes: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

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
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=150)
    finally:
        plt.close(figure)


def _prediction_rows(source: TMCADSource, data: dict[str, np.ndarray],
                     base_probability: np.ndarray,
                     final_probability: np.ndarray) -> list[dict]:
    rows = []
    for local, index in enumerate(data["index"]):
        truth = int(data["label"][local])
        base = int(base_probability[local].argmax())
        final = int(final_probability[local].argmax())
        outcome = ("corrected" if base != truth and final == truth else
                   "worsened" if base == truth and final != truth else
                   "still_wrong" if final != truth else "still_correct")
        rows.append({
            "uid": source.metadata["uids"][int(index)],
            "step_path": str(source.sample_path(int(index))),
            "true_class": source.classes[truth],
            "baseline_prediction": source.classes[base],
            "final_prediction": source.classes[final],
            "baseline_confidence": float(base_probability[local, base]),
            "final_confidence": float(final_probability[local, final]),
            "outcome": outcome,
        })
    return rows


def evaluate(config: dict) -> None:
    _check(config)
    source = TMCADSource(config)
    device = device_for(config["device"])
    base, base_checkpoint = _load_base(config, source, device)
    residual, residual_checkpoint = _load_residual(config, source, device)
    val_data, _ = _extract_base(base, source, "val", config["classifier"], device)
    base_val = classification_metrics(
        probabilities(val_data["baseline_logits"]), val_data["label"], source.classes,
    )
    final_val_logits, _ = _infer_adapter(residual, val_data, config["adapter"], device)
    final_val = classification_metrics(
        probabilities(final_val_logits), val_data["label"], source.classes,
    )
    if (abs(base_val["macro_f1"] - float(base_checkpoint["validation_macro_f1"])) > 0.002
            or abs(final_val["macro_f1"]
                   - float(residual_checkpoint["validation_macro_f1"])) > 0.002):
        raise ValueError("Saved checkpoints do not reproduce validation scores")
    gain = float(config["adapter"]["minimum_validation_macro_f1_gain"])
    selected = ("physical_residual" if final_val["macro_f1"]
                >= base_val["macro_f1"] + gain else "base_512")
    # All model decisions above are fixed before materializing any test row.
    test_data, _ = _extract_base(base, source, "test", config["classifier"], device)
    base_prob = probabilities(test_data["baseline_logits"])
    base_test = classification_metrics(base_prob, test_data["label"], source.classes)
    final_logits, _ = _infer_adapter(residual, test_data, config["adapter"], device)
    final_prob = probabilities(final_logits)
    final_test = classification_metrics(final_prob, test_data["label"], source.classes)
    output = config["paths"]["output"]
    comparisons = {
        "base_512": base_test, "physical_residual_196": final_test,
    }
    _write_csv(output / "validation_comparison.csv", [
        {"model": name, "selected": ("physical_residual" if name == "physical_residual_196"
                                        else "base_512") == selected,
         "accuracy": score["accuracy"], "balanced_accuracy": score["balanced_accuracy"],
         "macro_f1": score["macro_f1"], "top_3_accuracy": score["top_3_accuracy"]}
        for name, score in {"base_512": base_val,
                            "physical_residual_196": final_val}.items()
    ])
    _write_csv(output / "test_comparison.csv", [
        {"model": name, "selected_by_validation":
         ("physical_residual" if name == "physical_residual_196" else "base_512") == selected,
         "accuracy": score["accuracy"],
         "balanced_accuracy": score["balanced_accuracy"],
         "macro_f1": score["macro_f1"],
         "weighted_f1": score["weighted_f1"],
         "top_3_accuracy": score["top_3_accuracy"],
         "coupling_recall": score["per_class"]["coupling"]["recall"],
         "pulley_recall": score["per_class"]["pulley"]["recall"]}
        for name, score in comparisons.items()
    ])
    _write_csv(output / "validation_per_class.csv", _class_rows(
        {"base_512": base_val, "physical_residual_196": final_val}, source.classes,
    ))
    _write_csv(output / "test_per_class.csv", _class_rows(comparisons, source.classes))
    predictions = _prediction_rows(source, test_data, base_prob, final_prob)
    _write_csv(output / "test_predictions.csv", predictions)
    _write_csv(output / "changed_predictions.csv", [
        row for row in predictions
        if row["baseline_prediction"] != row["final_prediction"]
    ], list(predictions[0]))
    _plot_confusion(base_test["confusion_matrix"], source.classes,
                    output / "base_512_confusion.png")
    _plot_confusion(final_test["confusion_matrix"], source.classes,
                    output / "physical_196_confusion.png")
    save_json(output / "evaluation.json", {
        "selected_by_validation": selected,
        "selection_rule": "196-D residual requires configured validation macro-F1 gain; test is reporting only",
        "validation": {"base_512": base_val,
                       "physical_residual_196": final_val},
        "test": comparisons, "provenance": source.provenance,
        "base_checkpoint": str(_base_path(config)),
        "residual_checkpoint": str(_residual_path(config)),
        "corrected_test_predictions": sum(row["outcome"] == "corrected"
                                          for row in predictions),
        "worsened_test_predictions": sum(row["outcome"] == "worsened"
                                         for row in predictions),
    })
    from visualize import render_final_results
    render_final_results(source, output, base_test, final_test,
                         selected, predictions, config["qualitative"])
    print(f"Validation selected {selected}. Test macro-F1 "
          f"base={base_test['macro_f1']:.4f} physical={final_test['macro_f1']:.4f}; "
          f"results={output}", flush=True)
