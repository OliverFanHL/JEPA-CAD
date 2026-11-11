"""Audit coupling/pulley errors without changing the frozen model or labels.

Validation parts are compared against TRAINING parts in several existing
feature spaces. STEP contact sheets permit human label/shape review. Nearest
neighbors are diagnostics, not pseudo-labels or training examples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from common import ROOT, device_for, file_sha256, load_config, save_json, source_manifest
from dataset import FrozenViews
from mesh_features import _bounds, _step_shape, _triangles
from model import build_model
from train import _device_batch, _loader


MODEL_NAME = "pooled_plus_brep_view_geometry_mesh"


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def _load_model(config: dict, checkpoint_path: Path, classes: list[str],
                dataset: FrozenViews, device: torch.device):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (saved.get("architecture") != MODEL_NAME
            or saved.get("classes") != classes
            or saved.get("manifest_sha256") != dataset.metadata["manifest_sha256"]
            or saved.get("source_checkpoint") != dataset.metadata["checkpoint"]
            or saved.get("feature_format") != dataset.metadata["format"]
            or saved.get("mesh_sha256") != file_sha256(dataset.mesh_path)):
        raise ValueError("The classifier checkpoint does not match these frozen caches/splits")
    model = build_model(MODEL_NAME, len(classes), saved["settings"])
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def _infer(model, dataset: FrozenViews, settings: dict,
           device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the 256-D penultimate supervised features via a temporary hook."""
    loader = _loader(dataset, settings, False, 0)
    all_logits, all_hidden, all_labels, all_indices = [], [], [], []
    captured = []

    def capture(_layer, inputs):
        captured.append(inputs[0].detach().float().cpu().numpy())

    hook = model.head[-1].register_forward_pre_hook(capture)
    try:
        for original in loader:
            batch = _device_batch(original, device)
            logits, _ = model(batch)
            if len(captured) != 1:
                raise RuntimeError("Classifier readout hook did not fire exactly once")
            all_logits.append(logits.detach().float().cpu().numpy())
            all_hidden.append(captured.pop())
            all_labels.append(batch["label"].cpu().numpy())
            all_indices.append(original["index"].numpy())
    finally:
        hook.remove()
    if np.concatenate(all_indices).tolist() != dataset.indices:
        raise ValueError("DataLoader altered the cached sample order")
    return (np.concatenate(all_logits), np.concatenate(all_hidden),
            np.concatenate(all_labels))


def _cached_vectors(dataset: FrozenViews) -> dict[str, np.ndarray]:
    dataset._open()
    indices = dataset.indices
    pooled = np.asarray(dataset._pooled[indices], dtype=np.float32)
    geometry = np.asarray(dataset._geometry[indices], dtype=np.float32)
    mesh = np.asarray(dataset._mesh[indices], dtype=np.float32)
    # Both blocks are train-standardized already. Divide by sqrt(width) so
    # neither block dominates solely because it has more descriptor fields.
    physical = np.concatenate((geometry / np.sqrt(geometry.shape[1]),
                               mesh / np.sqrt(mesh.shape[1])), axis=1)
    result = {"frozen_step": pooled, "geometry": geometry,
              "surface_profile": mesh, "physical_combined": physical}
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError("Nonfinite values in frozen feature caches")
    return result


def _distance(query: np.ndarray, train: np.ndarray, metric: str) -> np.ndarray:
    if metric == "cosine":
        query = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
        train = train / np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-12)
        return np.maximum(0.0, 1.0 - query @ train.T)
    if metric == "euclidean":
        from scipy.spatial.distance import cdist
        return cdist(query, train, metric="euclidean")
    raise ValueError(metric)


def _sample_path(manifest: dict, dataset: FrozenViews, local: int) -> Path:
    global_index = dataset.indices[local]
    root = Path(manifest["dataset_root"]).resolve()
    path = (root / dataset.metadata["source_relpaths"][global_index]).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"STEP path escaped TMCAD root: {path}")
    return path


def _render_triangles(path: Path, uid: str, relative_deflection: float,
                      triangle_cap: int) -> np.ndarray:
    shape = _step_shape(path)
    bounds = _bounds(shape)
    diagonal = float(np.linalg.norm(bounds[3:] - bounds[:3]))
    digest = hashlib.sha256(uid.encode("utf-8")).hexdigest()
    triangles, _ = _triangles(shape, max(diagonal * relative_deflection, 1e-4),
                             triangle_cap, digest)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not np.isfinite(triangles).all():
        raise ValueError(f"Invalid rendered triangles: {path}")
    center = (bounds[:3] + bounds[3:]) / 2
    scale = float(np.max(bounds[3:] - bounds[:3]))
    if scale <= 0:
        raise ValueError(f"Degenerate STEP bounds: {path}")
    return (triangles - center) / scale


def _contact_sheet(path: Path, parts: list[tuple[str, str, Path, str]],
                   relative_deflection: float, triangle_cap: int,
                   cache: dict[str, np.ndarray]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    views = ((0, 1, 2, "XY"), (0, 2, 1, "XZ"), (1, 2, 0, "YZ"))
    figure, axes = plt.subplots(3, 3, figsize=(12, 12))
    try:
        for row, (caption, uid, step_path, class_name) in enumerate(parts):
            if uid not in cache:
                cache[uid] = _render_triangles(step_path, uid, relative_deflection, triangle_cap)
            triangles = cache[uid]
            face_normals = np.cross(triangles[:, 1] - triangles[:, 0],
                                    triangles[:, 2] - triangles[:, 0])
            face_normals /= np.maximum(np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-12)
            for column, (axis_x, axis_y, depth, title) in enumerate(views):
                axis = axes[row, column]
                order = np.argsort(triangles[:, :, depth].mean(axis=1))
                polygons = triangles[order][:, :, [axis_x, axis_y]]
                shade = 0.55 + 0.4 * np.abs(face_normals[order, depth])
                colors = np.stack((0.34 + 0.32 * shade,
                                   0.48 + 0.27 * shade,
                                   0.63 + 0.22 * shade,
                                   np.ones_like(shade)), axis=1)
                axis.add_collection(PolyCollection(polygons, facecolors=colors,
                                                   edgecolors="none", closed=True))
                axis.set_xlim(-0.55, 0.55)
                axis.set_ylim(-0.55, 0.55)
                axis.set_aspect("equal")
                axis.axis("off")
                axis.set_title(f"{caption} · {title}\n{uid} [{class_name}]", fontsize=9)
        figure.suptitle("Three orthographic STEP views; each part scaled uniformly",
                        fontsize=12)
        figure.tight_layout()
        figure.savefig(path, dpi=125)
    finally:
        plt.close(figure)


def _pick_cases(rows: list[dict], error_limit: int,
                correct_limit: int) -> list[dict]:
    errors = [item for item in rows if item["predicted_class"] != item["true_class"]]
    correct = [item for item in rows if item["predicted_class"] == item["true_class"]]
    # Examine confident mistakes and low-confidence borderline mistakes.
    high_count = (error_limit + 1) // 2
    low_count = error_limit - high_count
    high = sorted(errors, key=lambda item: (-item["confidence"], item["uid"]))[:high_count]
    chosen = {item["uid"] for item in high}
    low = [item for item in sorted(errors, key=lambda item: (item["confidence"], item["uid"]))
           if item["uid"] not in chosen][:low_count]
    # Correct prototypes and borderline correct samples provide context.
    high_correct_count = (correct_limit + 1) // 2
    high_correct = sorted(correct, key=lambda item: (-item["confidence"], item["uid"]))[:high_correct_count]
    chosen_correct = {item["uid"] for item in high_correct}
    low_correct = [item for item in sorted(correct, key=lambda item: (item["confidence"], item["uid"]))
                   if item["uid"] not in chosen_correct][:correct_limit - high_correct_count]
    return high + low + high_correct + low_correct


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "outputs/default" / MODEL_NAME / "best.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/hard_class_audit")
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite an existing audit: {destination}")
    config = load_config(args.config)
    audit = config["hard_class_audit"]
    focus_classes = list(audit["focus_classes"])
    k = int(audit["nearest_train_k"])
    error_limit = int(audit["error_cases_per_class"])
    correct_limit = int(audit["correct_cases_per_class"])
    triangle_cap = int(audit["render_triangle_cap"])
    if (not focus_classes or len(focus_classes) != len(set(focus_classes))
            or not 1 <= k <= 20 or not 0 <= error_limit <= 30
            or not 0 <= correct_limit <= 20 or error_limit + correct_limit == 0
            or not 200 <= triangle_cap <= 20000):
        raise ValueError("Invalid hard-class audit settings")

    train = FrozenViews(config, "train", True, True, True)
    validation = FrozenViews(config, "val", True, True, True)
    classes = train.metadata["classes"]
    if any(name not in classes for name in focus_classes):
        raise ValueError(f"Unknown focus class in {focus_classes}")
    manifest, _, manifest_hash = source_manifest(config)
    if manifest_hash != train.metadata["manifest_sha256"]:
        raise ValueError("Prepared manifest changed after frozen extraction")
    device = device_for(config["classifier"]["device"])
    model = _load_model(config, args.checkpoint.resolve(), classes, validation, device)
    settings = config["classifier"]
    _, train_hidden, train_labels = _infer(model, train, settings, device)
    val_logits, val_hidden, val_labels = _infer(model, validation, settings, device)
    probability = _softmax(val_logits)
    predicted = probability.argmax(axis=1)
    focus_ids = {classes.index(name) for name in focus_classes}
    focus_local = np.flatnonzero(np.isin(val_labels, list(focus_ids)))
    if not len(focus_local):
        raise ValueError("No focus-class validation samples")

    train_vectors = _cached_vectors(train)
    val_vectors = _cached_vectors(validation)
    train_vectors["supervised_readout"] = train_hidden
    val_vectors["supervised_readout"] = val_hidden
    metrics = {"frozen_step": "cosine", "geometry": "euclidean",
               "surface_profile": "euclidean", "physical_combined": "euclidean",
               "supervised_readout": "cosine"}
    # Only focus-class queries are compared; the neighbor pool remains the
    # full training split, never validation/test parts.
    distances = {
        name: _distance(val_vectors[name][focus_local], train_vectors[name], metric)
        for name, metric in metrics.items()
    }
    # Equal-size reference pool is a diagnostic for training-class density.
    # It discards majority-class examples and changes no fitted parameters.
    rng = np.random.default_rng(int(config["seed"]))
    balanced_count = min(int(np.count_nonzero(train_labels == label))
                         for label in range(len(classes)))
    balanced_train = np.sort(np.concatenate([
        rng.choice(np.flatnonzero(train_labels == label), balanced_count,
                   replace=False)
        for label in range(len(classes))
    ]))
    metadata = validation.metadata
    rows = []
    for position, local in enumerate(focus_local):
        true_id = int(val_labels[local])
        predicted_id = int(predicted[local])
        sorted_classes = np.argsort(probability[local])[::-1]
        rival_id = predicted_id if predicted_id != true_id else int(sorted_classes[1])
        uid = metadata["uids"][validation.indices[int(local)]]
        row = {
            "uid": uid, "split": "val", "true_class": classes[true_id],
            "predicted_class": classes[predicted_id], "rival_class": classes[rival_id],
            "confidence": float(probability[local, predicted_id]),
            "true_class_probability": float(probability[local, true_id]),
            "rival_probability": float(probability[local, rival_id]),
            "step_path": str(_sample_path(manifest, validation, int(local))),
            "local_index": int(local),
        }
        for name, matrix in distances.items():
            candidate_order = np.argsort(matrix[position])[:k]
            nearest = int(candidate_order[0])
            balanced_nearest = int(balanced_train[
                np.argmin(matrix[position, balanced_train])])
            train_global = train.indices[nearest]
            row[f"{name}_nearest_uid"] = train.metadata["uids"][train_global]
            row[f"{name}_nearest_class"] = classes[int(train_labels[nearest])]
            row[f"{name}_nearest_distance"] = float(matrix[position, nearest])
            row[f"{name}_balanced_nearest_class"] = classes[int(train_labels[balanced_nearest])]
            row[f"{name}_top{k}_true_fraction"] = float(np.mean(train_labels[candidate_order] == true_id))
            same = np.flatnonzero(train_labels == true_id)
            rival = np.flatnonzero(train_labels == rival_id)
            same_index = int(same[np.argmin(matrix[position, same])])
            rival_index = int(rival[np.argmin(matrix[position, rival])])
            row[f"{name}_nearest_true_distance"] = float(matrix[position, same_index])
            row[f"{name}_nearest_rival_distance"] = float(matrix[position, rival_index])
            if name == "physical_combined":
                row["physical_nearest_true_uid"] = train.metadata["uids"][train.indices[same_index]]
                row["physical_nearest_rival_uid"] = train.metadata["uids"][train.indices[rival_index]]
                row["physical_nearest_true_train_local"] = same_index
                row["physical_nearest_rival_train_local"] = rival_index
        rows.append(row)

    summary = {
        "note": "Diagnostic only. A nearest-neighbor disagreement is not proof of a bad label.",
        "query_split": "val", "neighbor_split": "train",
        "checkpoint": str(args.checkpoint.resolve()),
        "focus_classes": focus_classes,
        "feature_spaces": metrics,
        "balanced_reference_pool_per_class": balanced_count,
        "class_results": {},
    }
    for class_name in focus_classes:
        own = [row for row in rows if row["true_class"] == class_name]
        errors = [row for row in own if row["predicted_class"] != class_name]
        result = {
            "train_samples": int(np.count_nonzero(train_labels == classes.index(class_name))),
            "val_samples": len(own), "val_correct": len(own) - len(errors),
            "val_recall": (len(own) - len(errors)) / len(own),
            "error_destinations": dict(Counter(row["predicted_class"] for row in errors)),
            "nearest_train_label_accuracy": {
                name: float(np.mean([row[f"{name}_nearest_class"] == class_name for row in own]))
                for name in metrics
            },
            "equal_count_reference_label_accuracy": {
                name: float(np.mean([row[f"{name}_balanced_nearest_class"] == class_name
                                     for row in own]))
                for name in metrics
            },
            "errors_where_nearest_train_label_is_correct": {
                name: int(sum(row[f"{name}_nearest_class"] == class_name for row in errors))
                for name in metrics
            },
        }
        summary["class_results"][class_name] = result

    destination.mkdir(parents=True)
    _write_csv(destination / "all_focus_cases.csv", rows, list(rows[0]))
    save_json(destination / "summary.json", summary)
    selected = []
    for class_name in focus_classes:
        selected.extend(_pick_cases([row for row in rows if row["true_class"] == class_name],
                                    error_limit, correct_limit))
    sheets = destination / "contact_sheets"
    sheets.mkdir()
    render_cache: dict[str, np.ndarray] = {}
    review_rows, failures = [], []
    for number, row in enumerate(selected, 1):
        true_local = int(row["physical_nearest_true_train_local"])
        rival_local = int(row["physical_nearest_rival_train_local"])
        neighbor_true = row["physical_nearest_true_uid"]
        neighbor_rival = row["physical_nearest_rival_uid"]
        parts = [
            ("QUERY", row["uid"], Path(row["step_path"]), row["true_class"]),
            ("nearest true-class train", neighbor_true,
             _sample_path(manifest, train, true_local), row["true_class"]),
            ("nearest rival-class train", neighbor_rival,
             _sample_path(manifest, train, rival_local), row["rival_class"]),
        ]
        image_path = sheets / f"{number:03d}_{row['true_class']}_{row['predicted_class']}.png"
        try:
            _contact_sheet(image_path, parts, float(config["mesh"]["relative_deflection"]),
                           triangle_cap, render_cache)
            render_status = "ok"
        except Exception as exc:
            render_status = f"failed: {type(exc).__name__}: {exc}"
            failures.append({"uid": row["uid"], "reason": render_status})
        review_rows.append({
            "uid": row["uid"], "true_class": row["true_class"],
            "predicted_class": row["predicted_class"],
            "confidence": row["confidence"],
            "image": str(image_path) if render_status == "ok" else "",
            "render_status": render_status,
            "query_step": str(parts[0][2]),
            "nearest_true_uid": neighbor_true, "nearest_true_step": str(parts[1][2]),
            "nearest_rival_uid": neighbor_rival, "nearest_rival_step": str(parts[2][2]),
            "review_label_status": "", "review_shape_overlap": "",
            "review_missing_local_detail": "", "review_notes": "",
        })
        print(f"Contact sheet {number}/{len(selected)}: {row['uid']} {render_status}", flush=True)
    _write_csv(destination / "manual_review.csv", review_rows, list(review_rows[0]))
    if failures:
        _write_csv(destination / "render_failures.csv", failures, ["uid", "reason"])
    summary["contact_sheets_requested"] = len(selected)
    summary["contact_sheets_rendered"] = len(selected) - len(failures)
    summary["render_failures"] = len(failures)
    save_json(destination / "summary.json", summary)
    print(f"Saved validation hard-class audit: {destination}", flush=True)


if __name__ == "__main__":
    main()
