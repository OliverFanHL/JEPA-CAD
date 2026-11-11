"""Validation-only pulley decision-bias test for the frozen mesh classifier.

The model and inputs are unchanged. This one-parameter diagnostic asks whether
pulley's low recall is partly a class decision-boundary problem. Coupling is
not adjusted: its validation precision is already below recall and errors
are distributed over several classes.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from common import ROOT, device_for, file_sha256, load_config, save_json
from dataset import FrozenViews
from evaluate import _plot_confusion, _qualitative_cases
from metrics import classification_metrics
from model import build_model
from train import _device_batch, _loader


MODEL = "pooled_plus_brep_view_geometry_mesh"


@torch.inference_mode()
def _logits(model, dataset: FrozenViews, settings: dict,
            device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    loader = _loader(dataset, settings, False, 0)
    scores, labels, indices = [], [], []
    model.eval()
    for original in loader:
        batch = _device_batch(original, device)
        output, _ = model(batch)
        scores.append(output.float().cpu().numpy())
        labels.append(batch["label"].cpu().numpy())
        indices.append(original["index"].numpy())
    order = np.concatenate(indices)
    if order.tolist() != dataset.indices:
        raise ValueError("Evaluation loader changed sample order")
    return np.concatenate(scores), np.concatenate(labels)


def _probability(logits: np.ndarray) -> np.ndarray:
    centered = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    weights = np.exp(centered)
    return (weights / weights.sum(axis=1, keepdims=True)).astype(np.float32)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to save: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "outputs/default" / MODEL / "best.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/pulley_calibration")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing calibration: {output}")
    config = load_config(args.config)
    calibration = config["calibration"]
    classes_to_adjust = calibration["class"]
    if classes_to_adjust != "pulley":
        raise ValueError("This targeted diagnostic adjusts pulley only")
    max_bias = float(calibration["max_logit_bias"])
    steps = int(calibration["steps"])
    minimum_gain = float(calibration["minimum_val_macro_f1_gain"])
    if not (0 < max_bias <= 3 and 2 <= steps <= 121 and 0 <= minimum_gain <= 0.05):
        raise ValueError("Invalid pulley-bias calibration range")
    settings = config["classifier"]
    device = device_for(settings["device"])
    validation = FrozenViews(config, "val", True, True, True)
    test = FrozenViews(config, "test", True, True, True)
    classes = test.metadata["classes"]
    target = classes.index("pulley")
    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    if (checkpoint.get("architecture") != MODEL
            or checkpoint.get("classes") != classes
            or checkpoint.get("manifest_sha256") != test.metadata["manifest_sha256"]
            or checkpoint.get("source_checkpoint") != test.metadata["checkpoint"]
            or checkpoint.get("feature_format") != test.metadata["format"]
            or checkpoint.get("mesh_sha256") != file_sha256(test.mesh_path)):
        raise ValueError("Checkpoint does not match TMCAD split or frozen feature caches")
    model = build_model(MODEL, len(classes), checkpoint["settings"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()

    val_logits, val_truth = _logits(model, validation, settings, device)
    base_val = classification_metrics(_probability(val_logits), val_truth, classes)
    sweep = []
    best_bias, best_f1 = 0.0, base_val["macro_f1"]
    for bias in np.linspace(0.0, max_bias, steps):
        adjusted = val_logits.copy()
        adjusted[:, target] += float(bias)
        metrics = classification_metrics(_probability(adjusted), val_truth, classes)
        sweep.append({
            "bias": float(bias), "val_macro_f1": metrics["macro_f1"],
            "val_accuracy": metrics["accuracy"],
            "val_pulley_precision": metrics["per_class"]["pulley"]["precision"],
            "val_pulley_recall": metrics["per_class"]["pulley"]["recall"],
            "val_pulley_f1": metrics["per_class"]["pulley"]["f1"],
        })
        # Earliest point wins ties, so the selected adjustment is minimal.
        if metrics["macro_f1"] > best_f1 + 1e-6:
            best_f1, best_bias = metrics["macro_f1"], float(bias)
    if best_f1 < base_val["macro_f1"] + minimum_gain:
        best_bias = 0.0
    adjusted_val_logits = val_logits.copy()
    adjusted_val_logits[:, target] += best_bias
    calibrated_val = classification_metrics(_probability(adjusted_val_logits), val_truth, classes)

    # Test is opened only after the bias has been selected on validation.
    test_logits, test_truth = _logits(model, test, settings, device)
    base_test_prob = _probability(test_logits)
    base_test = classification_metrics(base_test_prob, test_truth, classes)
    adjusted_test_logits = test_logits.copy()
    adjusted_test_logits[:, target] += best_bias
    calibrated_test_prob = _probability(adjusted_test_logits)
    calibrated_test = classification_metrics(calibrated_test_prob, test_truth, classes)

    output.mkdir(parents=True)
    _write_csv(output / "validation_bias_sweep.csv", sweep)
    _write_csv(output / "comparison.csv", [
        {"method": method, "pulley_logit_bias": bias,
         "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"],
         "test_accuracy": heldout["accuracy"], "test_balanced_accuracy": heldout["balanced_accuracy"],
         "test_macro_f1": heldout["macro_f1"], "test_top3_accuracy": heldout["top_3_accuracy"]}
        for method, bias, val, heldout in (
            ("unchanged_checkpoint", 0.0, base_val, base_test),
            ("validation_calibrated_pulley", best_bias, calibrated_val, calibrated_test),
        )
    ])
    _write_csv(output / "per_class_test.csv", [
        {"class": name, "support": base_test["per_class"][name]["support"],
         "base_precision": base_test["per_class"][name]["precision"],
         "base_recall": base_test["per_class"][name]["recall"],
         "base_f1": base_test["per_class"][name]["f1"],
         "calibrated_precision": calibrated_test["per_class"][name]["precision"],
         "calibrated_recall": calibrated_test["per_class"][name]["recall"],
         "calibrated_f1": calibrated_test["per_class"][name]["f1"]}
        for name in classes
    ])
    original_prediction = base_test_prob.argmax(axis=1)
    calibrated_prediction = calibrated_test_prob.argmax(axis=1)
    changed = np.flatnonzero(original_prediction != calibrated_prediction)
    if len(changed):
        _write_csv(output / "changed_test_predictions.csv", [
            {
                "uid": test.metadata["uids"][test.indices[int(row)]],
                "true_class": classes[int(test_truth[row])],
                "original_prediction": classes[int(original_prediction[row])],
                "calibrated_prediction": classes[int(calibrated_prediction[row])],
                "original_pulley_probability": float(base_test_prob[row, target]),
                "calibrated_pulley_probability": float(calibrated_test_prob[row, target]),
            }
            for row in changed
        ])
    _plot_confusion(calibrated_test["confusion_matrix"], classes, output / "confusion_matrix.png")
    _qualitative_cases(model, test, calibrated_test_prob, test_truth, config, output, device)
    save_json(output / "evaluation.json", {
        "class_adjusted": "pulley", "selected_logit_bias": best_bias,
        "changed_test_predictions": int(len(changed)),
        "selection": "Validation macro-F1 only, fixed nonnegative grid; test not used for bias selection",
        "minimum_val_macro_f1_gain": minimum_gain,
        "checkpoint": str(args.checkpoint.resolve()),
        "base_validation": base_val, "calibrated_validation": calibrated_val,
        "base_test": base_test, "calibrated_test": calibrated_test,
    })
    print(f"Pulley logit bias={best_bias:.3f} "
          f"val_macro_F1={base_val['macro_f1']:.4f}->{calibrated_val['macro_f1']:.4f} "
          f"test_macro_F1={base_test['macro_f1']:.4f}->{calibrated_test['macro_f1']:.4f}",
          flush=True)
    print(f"Saved validation-only pulley calibration: {output}", flush=True)


if __name__ == "__main__":
    main()
