"""Train-defined, abstaining labels from annotations and structured CAD metadata."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

import torch

TASKS = ("functional_form", "sketch_composition", "constraint_regime", "modeling_sequence")


def load_taxonomy(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if "base_taxonomy" in value:
        base_path = path.parent / value["base_taxonomy"]
        base = json.loads(base_path.read_text(encoding="utf-8"))
        base.update({key: item for key, item in value.items() if key != "base_taxonomy"})
        return base
    return value


def raw_labels(source: dict, taxonomy: dict) -> dict[str, list[str | None]]:
    patterns = [(item["name"], re.compile(item["pattern"], re.I)) for item in taxonomy["functional_form"]]
    head_split = re.compile(taxonomy["functional_head_split"], re.I)
    geometric = set(taxonomy["constraint_regime"]["geometric_constraint_types"])
    dimensional = set(taxonomy["constraint_regime"]["dimensional_constraint_types"])
    output: dict[str, list[str | None]] = {task: [] for task in TASKS}
    for item, structural in zip(source["metadata"], source["structural_metadata"], strict=True):
        text = (item.get("functional_type") or "").strip()
        head = head_split.split(text, maxsplit=1)[0]
        matches = [(match.end(), name) for name, pattern in patterns for match in pattern.finditer(head)]
        # The final noun in a compound ("shim plate", "spacer ring") usually
        # names its primary form; alternatives joined by "or" remain uncertain.
        if taxonomy.get("functional_selection_rule") == "prefer_specific_then_fallback":
            # "block extrusion" is a block, not a profile; "pins extrusion"
            # is pin, not profile. Generic form/process words remain a fallback
            # only when the phrase has no more specific supported noun.
            generic = set(taxonomy.get("generic_fallback_classes", ()))
            specific = [candidate for candidate in matches if candidate[1] not in generic]
            matches = specific or matches
        output["functional_form"].append(max(matches)[1] if matches and not re.search(r"\bor\b", head, re.I) else None)

        primitives = structural["primitive_counts"]
        line, circle = primitives.get("line", 0) > 0, primitives.get("circle", 0) > 0
        other_curve = any(primitives.get(name, 0) > 0 for name in ("arc", "ellipse", "elliptical_arc", "nurbs"))
        sketch = (
            "contains_arc_or_spline" if other_curve else
            "mixed_line_circle" if line and circle else
            "line_only" if line else "circle_only" if circle else None
        )
        output["sketch_composition"].append(sketch)

        constraints = structural["constraint_counts"]
        has_geo = any(constraints.get(name, 0) > 0 for name in geometric)
        has_dim = any(constraints.get(name, 0) > 0 for name in dimensional)
        regime = (
            "mixed_geometric_dimensional" if has_geo and has_dim else
            "geometric_only" if has_geo else
            "dimensional_only" if has_dim else "no_constraints"
        )
        output["constraint_regime"].append(regime)

        signature = structural.get("operation_signature", "")
        operations = [token.rsplit(":", 1)[-1] for token in signature.split(">") if token]
        if not operations or set(operations) - {"NewBody", "Join", "Cut", "Intersect"}:
            sequence = None
        elif "Intersect" in operations:
            sequence = "intersection"
        elif "Join" in operations and "Cut" in operations:
            sequence = "mixed_join_cut"
        elif "Join" in operations:
            sequence = "join_no_cut"
        elif "Cut" in operations:
            sequence = "cut_no_join"
        elif len(operations) > 1:
            sequence = "multiple_new_bodies"
        else:
            sequence = "single_new_body"
        output["modeling_sequence"].append(sequence)
    return output


def build_labels(source: dict, taxonomy: dict) -> tuple[torch.Tensor, dict]:
    """Select supported classes using training rows alone; never fold rare into 'other'."""
    raw = raw_labels(source, taxonomy)
    splits = [item["split"] for item in source["metadata"]]
    configured = {
        "functional_form": [item["name"] for item in taxonomy["functional_form"]],
        **{task: taxonomy[task]["classes"] for task in TASKS[1:]},
    }
    targets = torch.full((len(splits), len(TASKS)), -1, dtype=torch.long)
    report = {"taxonomy_version": taxonomy["version"], "minimum_training_samples_per_class": taxonomy["minimum_training_samples_per_class"], "tasks": {}}
    for task_index, task in enumerate(TASKS):
        counts = Counter(value for value, split in zip(raw[task], splits, strict=True) if split == "train" and value)
        retained = [name for name in configured[task] if counts[name] >= taxonomy["minimum_training_samples_per_class"]]
        if len(retained) < 2:
            raise ValueError(f"Task {task} has fewer than two sufficiently supported classes")
        lookup = {name: index for index, name in enumerate(retained)}
        for index, value in enumerate(raw[task]):
            targets[index, task_index] = lookup.get(value, -1)
        per_split = {}
        for split in ("train", "val", "test"):
            indices = [i for i, value in enumerate(splits) if value == split]
            split_counts = Counter(raw[task][i] for i in indices)
            per_split[split] = {
                "total": len(indices),
                "labelled": sum(split_counts[name] for name in retained),
                "coverage": sum(split_counts[name] for name in retained) / max(1, len(indices)),
                "counts": {name: split_counts[name] for name in retained},
            }
        report["tasks"][task] = {
            "source": "Functional Type annotation head noun" if task == "functional_form" else "JSON-derived structural metadata",
            "retained_classes": retained,
            "dropped_rare_classes": {name: counts[name] for name in configured[task] if name not in retained},
            "parent_by_class": {item["name"]: item["parent"] for item in taxonomy["functional_form"] if item["name"] in retained} if task == "functional_form" else {name: taxonomy[task]["parent"] for name in retained},
            "per_split": per_split,
        }
    return targets, report


def save_labels(output_dir: Path, source: dict, targets: torch.Tensor, report: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "label_mappings.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output_dir / "labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("uid", "split", *TASKS))
        classes = [report["tasks"][task]["retained_classes"] for task in TASKS]
        for uid, meta, row in zip(source["uids"], source["metadata"], targets.tolist(), strict=True):
            writer.writerow((uid, meta["split"], *(classes[j][label] if label >= 0 else "" for j, label in enumerate(row))))
