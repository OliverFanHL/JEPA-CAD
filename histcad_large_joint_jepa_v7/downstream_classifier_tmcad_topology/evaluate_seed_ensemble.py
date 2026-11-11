"""Equal-weight evaluation of pre-existing frozen-JEPA geometry classifiers.

No model is trained, tuned, or selected using test data. The seed list and
probability-averaging rule are fixed before validation/test inference.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from common import ROOT, device_for, load_config, save_json, source_manifest
from dataset import FrozenViews
from evaluate import _plot_confusion
from metrics import classification_metrics
from model import build_model
from train import _loader, infer


MODEL = "pooled_plus_brep_view_geometry"


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_model(path: Path, dataset: FrozenViews, classes: list[str], device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload.get("architecture") != MODEL
            or payload.get("classes") != classes
            or payload.get("manifest_sha256") != dataset.metadata["manifest_sha256"]
            or payload.get("source_checkpoint") != dataset.metadata["checkpoint"]
            or payload.get("feature_format") != dataset.metadata["format"]):
        raise ValueError(f"Checkpoint does not match the frozen TMCAD features: {path}")
    model = build_model(MODEL, len(classes), payload["settings"])
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval(), int(payload["epoch"])


def _cases(dataset: FrozenViews, probabilities: np.ndarray, truth: np.ndarray,
           config: dict) -> list[dict]:
    manifest, _, _ = source_manifest(config)
    root = Path(manifest["dataset_root"])
    classes = dataset.metadata["classes"]
    predicted = probabilities.argmax(1)
    confidence = probabilities.max(1)
    rows = []
    for class_id, class_name in enumerate(classes):
        own = np.flatnonzero(truth == class_id)
        for kind, candidates in (("correct", own[predicted[own] == class_id]),
                                 ("error", own[predicted[own] != class_id])):
            if not len(candidates):
                continue
            local = sorted(candidates, key=lambda i: (-confidence[i],
                dataset.metadata["uids"][dataset.indices[i]]))[0]
            global_index = dataset.indices[local]
            rows.append({
                "case": kind, "uid": dataset.metadata["uids"][global_index],
                "step_path": str(root / dataset.metadata["source_relpaths"][global_index]),
                "true_class": class_name,
                "predicted_class": classes[int(predicted[local])],
                "confidence": float(confidence[local]),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/repeated_seeds")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/geometry_seed_ensemble")
    parser.add_argument("--seeds", nargs="+", type=int, default=[97, 113, 127])
    args = parser.parse_args()
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Provide at least two distinct pre-specified seeds")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing ensemble results: {output}")
    source = args.source.resolve()
    config = load_config(args.config)
    settings = config["classifier"]
    device = device_for(settings["device"])
    validation = FrozenViews(config, "val", include_alternative=True,
                             include_geometry=True, include_topology=False)
    test = FrozenViews(config, "test", include_alternative=True,
                       include_geometry=True, include_topology=False)
    classes = test.metadata["classes"]
    for seed in args.seeds:
        path = source / f"seed_{seed}" / MODEL / "best.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing pre-trained seed checkpoint: {path}")

    probabilities: dict[str, list[np.ndarray]] = {"val": [], "test": []}
    truth: dict[str, np.ndarray] = {}
    members: list[dict] = []
    for seed in args.seeds:
        path = source / f"seed_{seed}" / MODEL / "best.pt"
        model, epoch = _load_model(path, validation, classes, device)
        row = {"model": f"seed_{seed}", "selected_epoch": epoch}
        for split, dataset in (("val", validation), ("test", test)):
            loader = _loader(dataset, settings, False, seed)
            prob, labels, indices, _ = infer(model, loader, device)
            if indices.tolist() != dataset.indices:
                raise ValueError(f"{split} order changed for seed {seed}")
            if split in truth and not np.array_equal(truth[split], labels):
                raise ValueError(f"{split} labels changed between seed models")
            truth[split] = labels
            probabilities[split].append(prob)
            metric = classification_metrics(prob, labels, classes)
            row[f"{split}_accuracy"] = metric["accuracy"]
            row[f"{split}_macro_f1"] = metric["macro_f1"]
            row[f"{split}_top3_accuracy"] = metric["top_3_accuracy"]
        members.append(row)
        print(f"Loaded frozen geometry head seed={seed} epoch={epoch}", flush=True)
        del model

    # Fixed equal-weight probability average; no fit, validation search, or
    # test-dependent weighting is performed.
    ensemble_prob = {split: np.mean(np.stack(items, axis=0), axis=0)
                     for split, items in probabilities.items()}
    validation_metrics = classification_metrics(ensemble_prob["val"], truth["val"], classes)
    test_metrics = classification_metrics(ensemble_prob["test"], truth["test"], classes)
    output.mkdir(parents=True)
    comparison = [*members, {
        "model": "equal_probability_ensemble", "selected_epoch": "",
        "val_accuracy": validation_metrics["accuracy"],
        "val_macro_f1": validation_metrics["macro_f1"],
        "val_top3_accuracy": validation_metrics["top_3_accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_top3_accuracy": test_metrics["top_3_accuracy"],
    }]
    _write_csv(output / "comparison.csv", comparison)
    save_json(output / "evaluation.json", {
        "model": MODEL, "seeds": args.seeds, "members": members,
        "validation": validation_metrics, "test": test_metrics,
        "rule": "Equal arithmetic average of class probabilities; no trained ensemble weights",
        "selection": "Checkpoint selection was validation-only within each seed; no test-based architecture or weight choice",
    })
    _plot_confusion(test_metrics["confusion_matrix"], classes, output / "confusion_matrix.png")
    _write_csv(output / "per_class_test.csv", [
        {"class": name, **test_metrics["per_class"][name]} for name in classes
    ])
    _write_csv(output / "qualitative_cases.csv", _cases(test, ensemble_prob["test"], truth["test"], config))
    predictions = []
    for local, global_index in enumerate(test.indices):
        order = np.argsort(ensemble_prob["test"][local])[::-1]
        predictions.append({
            "uid": test.metadata["uids"][global_index],
            "true_class": classes[int(truth["test"][local])],
            "predicted_class": classes[int(order[0])],
            "confidence": float(ensemble_prob["test"][local, order[0]]),
            "top3_classes": "|".join(classes[int(index)] for index in order[:3]),
        })
    _write_csv(output / "test_predictions.csv", predictions)
    print(f"Equal-weight ensemble: val_macro_F1={validation_metrics['macro_f1']:.4f} "
          f"test_macro_F1={test_metrics['macro_f1']:.4f}", flush=True)
    print(f"Saved ensemble results: {output}", flush=True)


if __name__ == "__main__":
    main()
