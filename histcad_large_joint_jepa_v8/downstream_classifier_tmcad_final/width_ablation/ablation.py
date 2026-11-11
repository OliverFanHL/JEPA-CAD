"""Two-stage, frozen-JEPA 256-versus-512 supervised-head ablation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

PARENT = Path(__file__).resolve().parent.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from data import InputRows, TMCADSource, device_for, load_config, save_json, sha256  # noqa: E402
from experiment import (_class_rows, _infer as adapter_infer,
                        _loader as adapter_loader, _on_device,
                        _plot_confusion, _seed, _write_csv)  # noqa: E402
from metrics import classification_metrics, probabilities  # noqa: E402
from model import FrozenMeshClassifier, ResidualAdapter  # noqa: E402


BASE_FORMAT = "tmcad_final_width_ablation_base_v1"
RESIDUAL_FORMAT = "tmcad_final_width_ablation_residual_v1"


def load_ablation(path: Path) -> tuple[dict, dict]:
    path = path.resolve()
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    base_path = (path.parent / settings["base_config"]).resolve()
    settings["output"] = (path.parent / settings["output"]).resolve()
    if list(settings["widths"]) != [256, 512]:
        raise ValueError("This comparison must contain the predeclared [256, 512] widths")
    margin = float(settings["minimum_validation_gain_for_512"])
    if not 0 <= margin <= 0.05:
        raise ValueError("Invalid validation-only width selection margin")
    stage = settings["classifier"]
    if (not 1 <= int(stage["epochs"]) <= 100
            or not 1 <= int(stage["batch_size"]) <= 1024
            or not 1 <= int(stage["patience"]) <= int(stage["epochs"])
            or not 1 <= int(stage["min_epochs"]) <= int(stage["epochs"])
            or not 0 < float(stage["minimum_learning_rate"]) <= float(stage["learning_rate"])):
        raise ValueError("Invalid classifier stage settings")
    return load_config(base_path), settings


def _folder(settings: dict, width: int) -> Path:
    return settings["output"] / f"width_{width}"


def _base_path(settings: dict, width: int) -> Path:
    return _folder(settings, width) / "base_best.pt"


def _residual_path(settings: dict, width: int) -> Path:
    return _folder(settings, width) / "residual_best.pt"


def _loader(source: TMCADSource, split: str, stage: dict,
            shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(
        InputRows(source, split), batch_size=int(stage["batch_size"]),
        shuffle=shuffle, num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


@torch.inference_mode()
def _base_infer(model: FrozenMeshClassifier, loader: DataLoader,
                device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    scores, labels, indices = [], [], []
    for original in loader:
        batch = _on_device(original, device)
        logits, _ = model(batch)
        scores.append(logits.float().cpu().numpy())
        labels.append(original["label"].numpy())
        indices.append(original["index"].numpy())
    return np.concatenate(scores), np.concatenate(labels), np.concatenate(indices)


def _save_base(path: Path, model: FrozenMeshClassifier, source: TMCADSource,
               stage: dict, width: int, epoch: int, score: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": BASE_FORMAT, "width": width, "classes": source.classes,
        "provenance": source.provenance, "settings": stage,
        "epoch": epoch, "validation_macro_f1": score,
        "state_dict": {name: value.detach().cpu().clone()
                       for name, value in model.state_dict().items()},
    }
    temporary = path.with_name("base_best.partial.pt")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_base(path: Path, source: TMCADSource, width: int,
               device: torch.device) -> tuple[FrozenMeshClassifier, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Train the width-{width} classifier first: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload.get("format") != BASE_FORMAT
            or payload.get("width") != width
            or payload.get("classes") != source.classes
            or payload.get("provenance") != source.provenance):
        raise ValueError(f"Width-{width} base checkpoint is incompatible")
    stage = payload["settings"]
    model = FrozenMeshClassifier(
        len(source.classes), int(stage["attention_heads"]), width,
        float(stage["dropout"]),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.requires_grad_(False)
    return model.to(device).eval(), payload


@torch.inference_mode()
def _extract(model: FrozenMeshClassifier, source: TMCADSource,
             split: str, stage: dict, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    arrays: dict[str, list[np.ndarray]] = {
        name: [] for name in ("baseline_logits", "geometry", "mesh", "label", "index")
    }
    for original in _loader(source, split, stage, False, 0):
        batch = _on_device(original, device)
        logits, _ = model(batch)
        arrays["baseline_logits"].append(logits.float().cpu().numpy())
        for name in ("geometry", "mesh", "label", "index"):
            arrays[name].append(original[name].numpy())
    result = {name: np.concatenate(chunks) for name, chunks in arrays.items()}
    if (not np.array_equal(result["index"], source.indices[split])
            or not np.array_equal(result["label"], source.labels[source.indices[split]])):
        raise ValueError(f"Width-ablation {split} row order changed")
    if any(not np.isfinite(result[name]).all() for name in
           ("baseline_logits", "geometry", "mesh")):
        raise ValueError(f"Nonfinite width-ablation {split} features")
    return result


def _save_residual(path: Path, model: ResidualAdapter, source: TMCADSource,
                   settings: dict, width: int, base_path: Path,
                   epoch: int, score: float) -> None:
    payload = {
        "format": RESIDUAL_FORMAT, "width": width, "classes": source.classes,
        "provenance": source.provenance, "settings": settings,
        "base_checkpoint_sha256": sha256(base_path),
        "epoch": epoch, "validation_macro_f1": score,
        "state_dict": {name: value.detach().cpu().clone()
                       for name, value in model.state_dict().items()},
    }
    temporary = path.with_name("residual_best.partial.pt")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_residual(path: Path, source: TMCADSource, width: int,
                   base_path: Path, device: torch.device) -> tuple[ResidualAdapter, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Train the width-{width} physical residual first: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload.get("format") != RESIDUAL_FORMAT
            or payload.get("width") != width
            or payload.get("classes") != source.classes
            or payload.get("provenance") != source.provenance
            or payload.get("base_checkpoint_sha256") != sha256(base_path)):
        raise ValueError(f"Width-{width} residual checkpoint is incompatible")
    model = ResidualAdapter(len(source.classes), payload["settings"])
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval(), payload


def _train_base(source: TMCADSource, config: dict, settings: dict,
                width: int, device: torch.device,
                weights: torch.Tensor) -> None:
    stage = settings["classifier"]
    _seed(int(config["seed"]))
    model = FrozenMeshClassifier(
        len(source.classes), int(stage["attention_heads"]), width,
        float(stage["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(stage["learning_rate"]),
                                  weight_decay=float(stage["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(stage["epochs"]),
        eta_min=float(stage["minimum_learning_rate"]),
    )
    train_loader = _loader(source, "train", stage, True, int(config["seed"]))
    val_loader = _loader(source, "val", stage, False, int(config["seed"]))
    best_f1, best_epoch, stale = -1.0, 0, 0
    history = []
    print(f"Width {width} base: train={len(source.indices['train']):,} "
          f"val={len(source.indices['val']):,} "
          f"parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
    for epoch in range(1, int(stage["epochs"]) + 1):
        model.train()
        total_loss, count = 0.0, 0
        for original in train_loader:
            batch = _on_device(original, device)
            logits, _ = model(batch)
            loss = F.cross_entropy(logits.float(), batch["label"], weight=weights,
                                   label_smoothing=float(stage["label_smoothing"]))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(stage["gradient_clip"]))
            optimizer.step()
            size = len(batch["label"])
            total_loss += float(loss.detach()) * size
            count += size
        val_logits, val_labels, val_indices = _base_infer(model, val_loader, device)
        if not np.array_equal(val_indices, source.indices["val"]):
            raise ValueError("Validation sample order changed")
        val_metrics = classification_metrics(
            probabilities(val_logits), val_labels, source.classes,
        )
        row = {
            "epoch": epoch, "train_loss": total_loss / count,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "coupling_recall": val_metrics["per_class"]["coupling"]["recall"],
            "pulley_recall": val_metrics["per_class"]["pulley"]["recall"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(f"width={width} base epoch={epoch:02d}/{stage['epochs']} "
              f"train={row['train_loss']:.4f} val_macro_F1={row['val_macro_f1']:.4f} "
              f"lr={row['learning_rate']:.2e}", flush=True)
        if val_metrics["macro_f1"] > best_f1 + 1e-4:
            best_f1, best_epoch, stale = val_metrics["macro_f1"], epoch, 0
            _save_base(_base_path(settings, width), model, source, stage,
                       width, epoch, best_f1)
        else:
            stale += 1
        scheduler.step()
        if epoch >= int(stage["min_epochs"]) and stale >= int(stage["patience"]):
            print(f"Width {width} base early stop at {epoch}; best={best_epoch}", flush=True)
            break
    _write_csv(_folder(settings, width) / "base_history.csv", history)
    save_json(_folder(settings, width) / "base_summary.json", {
        "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
        "completed_epochs": len(history), "test_used": False,
    })


def _train_residual(source: TMCADSource, config: dict, settings: dict,
                    width: int, device: torch.device,
                    weights: torch.Tensor) -> None:
    base, _ = _load_base(_base_path(settings, width), source, width, device)
    stage = settings["classifier"]
    train_data = _extract(base, source, "train", stage, device)
    val_data = _extract(base, source, "val", stage, device)
    adapter_settings = config["adapter"]
    base_val = classification_metrics(
        probabilities(val_data["baseline_logits"]), val_data["label"], source.classes,
    )
    _seed(int(config["seed"]))
    model = ResidualAdapter(len(source.classes), adapter_settings).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(adapter_settings["learning_rate"]),
                                  weight_decay=float(adapter_settings["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(adapter_settings["epochs"]),
        eta_min=float(adapter_settings["minimum_learning_rate"]),
    )
    path = _residual_path(settings, width)
    best_f1, best_epoch, stale = base_val["macro_f1"], 0, 0
    _save_residual(path, model, source, adapter_settings, width,
                   _base_path(settings, width), 0, best_f1)
    history = [{
        "epoch": 0, "train_loss": "", "val_loss": "",
        "val_accuracy": base_val["accuracy"], "val_macro_f1": best_f1,
        "coupling_recall": base_val["per_class"]["coupling"]["recall"],
        "pulley_recall": base_val["per_class"]["pulley"]["recall"],
        "learning_rate": float(adapter_settings["learning_rate"]),
    }]
    print(f"Width {width} physical residual: initial val_macro_F1={best_f1:.4f} "
          f"trainable={sum(p.numel() for p in model.parameters()):,}", flush=True)
    for epoch in range(1, int(adapter_settings["epochs"]) + 1):
        model.train()
        total_loss, count = 0.0, 0
        for original in adapter_loader(train_data, adapter_settings, True,
                                       int(config["seed"]) + epoch):
            batch = _on_device(original, device)
            logits, residual = model(batch["baseline_logits"],
                                     batch["geometry"], batch["mesh"])
            loss = F.cross_entropy(
                logits.float(), batch["label"], weight=weights,
                label_smoothing=float(adapter_settings["label_smoothing"]),
            ) + float(adapter_settings["residual_l2"]) * residual.square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(adapter_settings["gradient_clip"]))
            optimizer.step()
            size = len(batch["label"])
            total_loss += float(loss.detach()) * size
            count += size
        val_logits, val_loss = adapter_infer(model, val_data, adapter_settings,
                                             device, weights)
        val = classification_metrics(probabilities(val_logits), val_data["label"],
                                     source.classes)
        row = {
            "epoch": epoch, "train_loss": total_loss / count,
            "val_loss": val_loss, "val_accuracy": val["accuracy"],
            "val_macro_f1": val["macro_f1"],
            "coupling_recall": val["per_class"]["coupling"]["recall"],
            "pulley_recall": val["per_class"]["pulley"]["recall"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(f"width={width} residual epoch={epoch:02d}/{adapter_settings['epochs']} "
              f"train={row['train_loss']:.4f} val={val_loss:.4f} "
              f"val_macro_F1={row['val_macro_f1']:.4f} "
              f"lr={row['learning_rate']:.2e}", flush=True)
        if val["macro_f1"] > best_f1 + 1e-4:
            best_f1, best_epoch, stale = val["macro_f1"], epoch, 0
            _save_residual(path, model, source, adapter_settings, width,
                           _base_path(settings, width), epoch, best_f1)
        else:
            stale += 1
        scheduler.step()
        if epoch >= int(adapter_settings["min_epochs"]) and stale >= int(adapter_settings["patience"]):
            print(f"Width {width} residual early stop at {epoch}; best={best_epoch}", flush=True)
            break
    _write_csv(_folder(settings, width) / "residual_history.csv", history)
    save_json(_folder(settings, width) / "residual_summary.json", {
        "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
        "base_val_macro_f1": base_val["macro_f1"],
        "completed_epochs": len(history) - 1, "test_used": False,
    })


def train(config: dict, settings: dict) -> None:
    for width in settings["widths"]:
        if (_base_path(settings, width).exists()
                or _residual_path(settings, width).exists()):
            raise FileExistsError(
                f"Existing width-{width} checkpoints would be overwritten; "
                "choose another ablation output directory"
            )
    source = TMCADSource(config)
    device = device_for(config["baseline"]["device"])
    counts = torch.bincount(torch.from_numpy(source.labels[source.indices["train"]]),
                            minlength=len(source.classes)).float()
    if (counts == 0).any():
        raise ValueError("A TMCAD class has no training samples")
    weights = (counts.mean() / counts).sqrt().clamp(0.5, 2.0).to(device)
    save_json(settings["output"] / "setup.json", {
        "widths": settings["widths"],
        "train_count": len(source.indices["train"]),
        "val_count": len(source.indices["val"]),
        "train_class_counts": dict(zip(source.classes, counts.int().tolist(), strict=True)),
        "frozen_input_provenance": source.provenance,
        "selection": "Validation-only width selection; no test extraction during training",
    })
    for width in settings["widths"]:
        _train_base(source, config, settings, width, device, weights)
        _train_residual(source, config, settings, width, device, weights)


def _prediction_rows(source: TMCADSource,
                     outputs: dict[int, np.ndarray], truth: np.ndarray,
                     indices: np.ndarray) -> list[dict]:
    p256, p512 = outputs[256], outputs[512]
    rows = []
    for local, global_index in enumerate(indices):
        actual = int(truth[local])
        first, second = int(p256[local].argmax()), int(p512[local].argmax())
        outcome = ("512_corrected" if first != actual and second == actual else
                   "512_worsened" if first == actual and second != actual else
                   "both_wrong" if second != actual else "both_correct")
        rows.append({
            "uid": source.metadata["uids"][int(global_index)],
            "step_path": str(source.sample_path(int(global_index))),
            "true_class": source.classes[actual],
            "prediction_256": source.classes[first],
            "prediction_512": source.classes[second],
            "confidence_256": float(p256[local, first]),
            "confidence_512": float(p512[local, second]),
            "outcome": outcome,
        })
    return rows


def evaluate(config: dict, settings: dict) -> None:
    source = TMCADSource(config)
    device = device_for(config["baseline"]["device"])
    validation = {}
    models = {}
    for width in settings["widths"]:
        base, base_payload = _load_base(_base_path(settings, width), source, width, device)
        residual, residual_payload = _load_residual(
            _residual_path(settings, width), source, width,
            _base_path(settings, width), device,
        )
        val_data = _extract(base, source, "val", settings["classifier"], device)
        base_score = classification_metrics(
            probabilities(val_data["baseline_logits"]), val_data["label"], source.classes,
        )
        residual_logits, _ = adapter_infer(residual, val_data, config["adapter"], device)
        residual_score = classification_metrics(
            probabilities(residual_logits), val_data["label"], source.classes,
        )
        if (abs(base_score["macro_f1"] - float(base_payload["validation_macro_f1"])) > 0.002
                or abs(residual_score["macro_f1"]
                       - float(residual_payload["validation_macro_f1"])) > 0.002):
            raise ValueError(f"Width-{width} validation scores do not reproduce saved checkpoints")
        validation[width] = {"base": base_score, "with_physical_residual": residual_score}
        models[width] = (base, residual)
    selected = (512 if validation[512]["with_physical_residual"]["macro_f1"]
                >= validation[256]["with_physical_residual"]["macro_f1"]
                + float(settings["minimum_validation_gain_for_512"]) else 256)
    # The decision above is fixed before either test score is computed.
    test_metrics = {}
    test_probabilities = {}
    test_truth = None
    test_indices = None
    for width in settings["widths"]:
        base, residual = models[width]
        test_data = _extract(base, source, "test", settings["classifier"], device)
        if test_truth is None:
            test_truth, test_indices = test_data["label"], test_data["index"]
        elif (not np.array_equal(test_truth, test_data["label"])
              or not np.array_equal(test_indices, test_data["index"])):
            raise ValueError("Width arms have different test rows")
        base_score = classification_metrics(
            probabilities(test_data["baseline_logits"]), test_data["label"], source.classes,
        )
        residual_logits, _ = adapter_infer(residual, test_data, config["adapter"], device)
        physical_probability = probabilities(residual_logits)
        physical_score = classification_metrics(
            physical_probability, test_data["label"], source.classes,
        )
        test_metrics[width] = {"base": base_score, "with_physical_residual": physical_score}
        test_probabilities[width] = physical_probability
        _plot_confusion(physical_score["confusion_matrix"], source.classes,
                        _folder(settings, width) / "test_confusion.png")
    output = settings["output"]
    comparison = [
        {"width": width, "stage": stage, "selected_width": width == selected,
         "val_accuracy": validation[width][stage]["accuracy"],
         "val_macro_f1": validation[width][stage]["macro_f1"],
         "test_accuracy": test_metrics[width][stage]["accuracy"],
         "test_balanced_accuracy": test_metrics[width][stage]["balanced_accuracy"],
         "test_macro_f1": test_metrics[width][stage]["macro_f1"],
         "test_top3_accuracy": test_metrics[width][stage]["top_3_accuracy"],
         "test_coupling_recall": test_metrics[width][stage]["per_class"]["coupling"]["recall"],
         "test_pulley_recall": test_metrics[width][stage]["per_class"]["pulley"]["recall"]}
        for width in settings["widths"]
        for stage in ("base", "with_physical_residual")
    ]
    _write_csv(output / "width_comparison.csv", comparison)
    _write_csv(output / "validation_per_class.csv", [
        {"width": width, "stage": stage, **row}
        for width in settings["widths"]
        for stage, scores in validation[width].items()
        for row in _class_rows({f"width_{width}_{stage}": scores}, source.classes)
    ])
    _write_csv(output / "test_per_class.csv", [
        {"width": width, "stage": stage, **row}
        for width in settings["widths"]
        for stage, scores in test_metrics[width].items()
        for row in _class_rows({f"width_{width}_{stage}": scores}, source.classes)
    ])
    predictions = _prediction_rows(source, test_probabilities, test_truth, test_indices)
    _write_csv(output / "test_predictions_256_vs_512.csv", predictions)
    qualitative = []
    for class_name in ("coupling", "pulley"):
        for outcome in ("512_corrected", "512_worsened", "both_wrong"):
            group = [row for row in predictions
                     if row["true_class"] == class_name and row["outcome"] == outcome]
            qualitative.extend(sorted(
                group, key=lambda row: (-row["confidence_512"], row["uid"])
            )[:6])
    _write_csv(output / "qualitative_review.csv", qualitative, list(predictions[0]))
    save_json(output / "evaluation.json", {
        "selected_width_by_validation": selected,
        "selection_rule": "512 requires the configured validation macro-F1 gain over 256 after the identical physical residual; test metrics are reporting only.",
        "validation": {str(width): value for width, value in validation.items()},
        "test": {str(width): value for width, value in test_metrics.items()},
        "provenance": source.provenance,
    })
    print(f"Validation selected {selected}-D supervised hidden width. "
          f"Test macro-F1: 256={test_metrics[256]['with_physical_residual']['macro_f1']:.4f}, "
          f"512={test_metrics[512]['with_physical_residual']['macro_f1']:.4f}. "
          f"Results: {output}", flush=True)
