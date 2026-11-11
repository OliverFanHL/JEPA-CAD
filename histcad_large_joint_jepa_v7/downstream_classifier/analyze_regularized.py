"""Validation-led per-class diagnosis and audited STEP failure visualizations."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

from attention_classifier import RoleTokenAttentionClassifier
from extract import load_frozen, resolve_paths
from labels import build_labels, load_taxonomy
from render_attention_cases import _render_step_robust
from run_attention import evaluate
from train_classifier import _class_weights, _device, _split_indices

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "attention_regularized_v2"


def _class_stats(metrics: dict, index: int) -> tuple[int, float, float, float, str, int]:
    matrix = metrics["confusion_matrix"]
    true_positive = matrix[index][index]
    total_true = sum(matrix[index])
    total_predicted = sum(row[index] for row in matrix)
    precision = true_positive / total_predicted if total_predicted else 0.0
    recall = true_positive / total_true if total_true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    alternatives = sorted(((n, count) for n, count in enumerate(matrix[index]) if n != index), key=lambda item: -item[1])
    rival_index, rival_count = alternatives[0]
    return total_true, precision, recall, f1, metrics["class_order"][rival_index], rival_count


def _font(size: int, bold: bool = False):
    path = Path("C:/Windows/Fonts") / ("arialbd.ttf" if bold else "arial.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def main() -> None:
    config = yaml.safe_load((ROOT / "config_regularized.yaml").read_text(encoding="utf-8"))
    paths = resolve_paths(config)
    source = load_frozen(paths["representations"], paths["checkpoint"], paths["graph_cache"])
    taxonomy = load_taxonomy(paths["taxonomy"])
    targets, mapping = build_labels(source, taxonomy)
    result = json.loads((OUTPUT / "attention_results.json").read_text(encoding="utf-8"))
    baseline = json.loads((ROOT / "outputs" / "mlp_refined_v2" / "baseline_results.json").read_text(encoding="utf-8"))["run"]
    task = "functional_form"
    classes = mapping["tasks"][task]["retained_classes"]
    counts = mapping["tasks"][task]["per_split"]
    rows = []
    for index, name in enumerate(classes):
        val = _class_stats(result["validation"][task], index)
        test = _class_stats(result["test"][task], index)
        base_val = _class_stats(baseline["validation"][task], index)
        base_test = _class_stats(baseline["test"][task], index)
        rows.append({"class": name, "parent": mapping["tasks"][task]["parent_by_class"][name],
                     "train_n": counts["train"]["counts"][name],
                     "val_n": counts["val"]["counts"][name], "test_n": counts["test"]["counts"][name],
                     "val_precision": val[1], "val_recall": val[2], "val_f1": val[3],
                     "test_precision": test[1], "test_recall": test[2], "test_f1": test[3],
                     "val_top_confusion": val[4], "val_top_confusion_n": val[5],
                     "mlp_val_recall": base_val[2], "mlp_test_recall": base_test[2]})
    quality = {name: {"count": 0, "truncated": 0, "brep_candidates": 0} for name in classes}
    for index, (metadata, structural) in enumerate(zip(source["metadata"], source["structural_metadata"], strict=True)):
        if metadata["split"] != "val" or targets[index, 0] < 0:
            continue
        item = quality[classes[int(targets[index, 0])]]
        item["count"] += 1
        item["truncated"] += int(structural["brep_truncated"])
        item["brep_candidates"] += int(structural["num_brep_candidates"])
    for row in rows:
        item = quality[row["class"]]
        row["val_step_truncated_fraction"] = item["truncated"] / item["count"]
        row["val_mean_brep_candidates"] = item["brep_candidates"] / item["count"]
    with (OUTPUT / "class_distribution_and_performance.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: row["val_recall"]))
    ordered = sorted(rows, key=lambda row: row["val_recall"])
    positions = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(10, 9), dpi=150)
    ax.barh(positions - 0.18, [row["val_recall"] for row in ordered], height=0.34,
            label="regularized attention", color="#2563eb")
    ax.barh(positions + 0.18, [row["mlp_val_recall"] for row in ordered], height=0.34,
            label="same-taxonomy joint MLP", color="#94a3b8")
    ax.set_yticks(positions, [f"{row['class']}  (train {row['train_n']:,}, val {row['val_n']})" for row in ordered])
    ax.invert_yaxis()
    ax.set(xlim=(0, 1), xlabel="Validation recall", title="Part-form performance and class support")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUTPUT / "classwise_validation_recall.png")
    plt.close(fig)

    device = _device(config["training"]["device"])
    model_config = config["model"]
    model = RoleTokenAttentionClassifier(mapping, width=model_config["token_width"],
        heads=model_config["attention_heads"], depth=model_config["attention_layers"],
        feedforward=model_config["feedforward_width"], dropout=model_config["dropout"]).to(device)
    checkpoint = torch.load(OUTPUT / "best_attention.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    splits = _split_indices(source)
    weights = _class_weights(targets, splits["train"], mapping, device)
    _, detail = evaluate(model, source, targets, splits["val"], mapping, device, 1024, weights,
                          True, details=True, functional_label_smoothing=config["training"]["functional_label_smoothing"])
    predictions = detail["predictions"][task]
    confidences = detail["form_confidence"]
    # Only validation labels determine which classes/examples are inspected.
    weak = [row["class"] for row in sorted(rows, key=lambda row: row["val_recall"])[:8]]
    cases = []
    for name in weak:
        class_index = classes.index(name)
        candidates = [local for local, index in enumerate(splits["val"].tolist())
                      if targets[index, 0].item() == class_index and predictions[local] != class_index]
        candidates.sort(key=lambda local: -confidences[local])
        for local in candidates[:3]:
            source_index = int(splits["val"][local])
            cases.append({"true_class": name, "predicted_class": classes[int(predictions[local])],
                          "confidence": float(confidences[local]), "uid": source["uids"][source_index],
                          "functional_description": source["metadata"][source_index].get("functional_type", ""),
                          "step_path": source["metadata"][source_index]["step_path"]})
    with (OUTPUT / "validation_failure_examples.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cases[0].keys())
        writer.writeheader()
        writer.writerows(cases)
    render_dir = OUTPUT / "validation_failure_renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    chosen, audit = [], {}
    for name in weak:
        for case in [case for case in cases if case["true_class"] == name]:
            try:
                destination = render_dir / f"{case['uid'].replace('/', '__')}.png"
                details = _render_step_robust(Path(case["step_path"]), destination)
                with Image.open(destination) as image:
                    image.verify()
                audit[case["uid"]] = details
                chosen.append((case, destination))
                break
            except Exception as error:
                audit[case["uid"]] = {"status": "invalid", "error": f"{type(error).__name__}: {error}"}
    canvas = Image.new("RGB", (340 * 4, 350 * 2), "white")
    draw = ImageDraw.Draw(canvas)
    body, small = _font(15, True), _font(13)
    for position, (case, path) in enumerate(chosen[:8]):
        left, top = (position % 4) * 340, (position // 4) * 350
        draw.rectangle((left + 2, top + 2, left + 338, top + 348), outline="#d97706", width=2)
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((330, 250), Image.Resampling.LANCZOS)
            canvas.paste(image, (left + (340 - image.width) // 2, top + 2))
        draw.text((left + 10, top + 255), f"{case['true_class']} → {case['predicted_class']} ({case['confidence']:.2f})", fill="#253449", font=body)
        description = case["functional_description"]
        draw.text((left + 10, top + 280), description[:45], fill="#526273", font=small)
        draw.text((left + 10, top + 305), case["uid"], fill="#526273", font=small)
    canvas.save(OUTPUT / "validation_failure_sheet.png")
    (OUTPUT / "validation_render_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("Wrote per-class report and rendered", len(chosen), "validation failures")


if __name__ == "__main__":
    main()
