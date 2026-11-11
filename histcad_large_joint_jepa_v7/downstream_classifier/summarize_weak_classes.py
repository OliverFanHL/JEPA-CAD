"""Create an auditable weak-class comparison from completed validation/test runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs/weak_class_diagnosis"
FOCUS = ("cylinder", "beam", "sleeve", "plug", "rail", "flange", "base", "ring", "bushing", "shim")


def _parent_recall(result: dict, name: str, parent_by_class: dict[str, str]) -> float:
    form = result["functional_form"]
    names = form["class_order"]
    row = form["confusion_matrix"][names.index(name)]
    return sum(count for label, count in zip(names, row, strict=True)
               if parent_by_class[label] == parent_by_class[name]) / sum(row)


def main() -> None:
    diagnostics = json.loads((OUTPUT / "diagnostics.json").read_text(encoding="utf-8"))["classes"]
    baseline_dir = ROOT / "outputs/attention_regularized_v2"
    revised_dir = ROOT / "outputs/parametric_residual_v2"
    baseline = json.loads((baseline_dir / "attention_results.json").read_text(encoding="utf-8"))
    revised = json.loads((revised_dir / "attention_results.json").read_text(encoding="utf-8"))
    mapping = json.loads((baseline_dir / "label_mappings.json").read_text(encoding="utf-8"))
    parents = mapping["tasks"]["functional_form"]["parent_by_class"]
    rows = []
    for name in FOCUS:
        item = diagnostics[name]
        row = {"class": name, "parent": parents[name],
               "train_n": item["train_count"], "val_n": item["val_count"],
               "joint_1nn_val_accuracy": item["joint"]["val_1nn_accuracy"],
               "parametric_1nn_val_accuracy": item["parametric"]["val_1nn_accuracy"],
               "step_1nn_val_accuracy": item["step"]["val_1nn_accuracy"],
               "joint_centroid_norm": item["joint"]["train_centroid_norm"],
               "high_similarity_conflicting_label_fraction": item["joint"]["val_high_similarity_conflict_fraction"],
               "train_step_truncated_fraction": item["step_truncated_fraction"]}
        for key, run in (("baseline", baseline), ("param_residual", revised)):
            for split in ("train", "validation", "test"):
                form = run[split]["functional_form"]
                short = "val" if split == "validation" else split
                row[f"{key}_{short}_recall"] = form["per_class_recall"][name]
                row[f"{key}_{short}_parent_recall"] = _parent_recall(run[split], name, parents)
        rows.append(row)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Saved", OUTPUT / "comparison.csv")


if __name__ == "__main__":
    main()
