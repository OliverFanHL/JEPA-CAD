from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .parametric_graph import build_parametric_graph
from .schema import load_history, step_path_from_json, uid_from_json_path
from .step_graph import (
    DIMENSION_TARGET_NAMES,
    ENTITY_RE,
    INSTANCE_MARKERS,
    dimension_target,
    parse_step_graph,
    scan_step_entity_types,
)

PARAMETRIC_NODE_KEYS = (
    "node_type",
    "subtype",
    "aux_type",
    "numeric",
    "numeric_mask",
    "feature_index",
)
PARAMETRIC_EDGE_KEYS = ("edge_type",)
STEP_NODE_KEYS = ("node_type", "node_role", "numeric")
STEP_EDGE_KEYS = ("edge_type", "edge_weight")
QUANTILE_NAMES = ("min", "p25", "p50", "p75", "p90", "p95", "p99", "max")
QUANTILES = torch.tensor((0.0, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0))


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reason: str = ""
    detail: str = ""


def deterministic_split(uid: str) -> str:
    value = int(hashlib.sha1(uid.encode()).hexdigest()[:8], 16) % 10_000
    return "train" if value < 9_000 else "val" if value < 9_500 else "test"


def _annotations(path: Path) -> dict[str, str]:
    csv.field_size_limit(2_147_483_647)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            row["uid"]: row.get("Functional Type", "").strip() for row in csv.DictReader(handle)
        }


def discover_entries(dataset_root: str | Path) -> list[dict[str, Any]]:
    dataset_root = Path(dataset_root)
    annotation_path = dataset_root / "annotations" / "histcad_annotations.csv"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
    labels = _annotations(annotation_path)
    paths = sorted((dataset_root / "JSON").rglob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No JSON histories found below {dataset_root / 'JSON'}")
    entries = []
    for path in paths:
        uid = uid_from_json_path(path)
        entries.append(
            {
                "uid": uid,
                "source_path": str(path.resolve()),
                "step_path": str(step_path_from_json(path).resolve()),
                "split": deterministic_split(uid),
                "partition": "Fusion360" if uid.startswith("0100/") else "DeepCAD",
                "functional_type": labels.get(uid, ""),
            }
        )
    return entries


def _validate_entry(entry: dict[str, Any]) -> ValidationResult:
    json_path, step_path = Path(entry["source_path"]), Path(entry["step_path"])
    if not json_path.is_file():
        return ValidationResult(False, "missing_json", str(json_path))
    if not step_path.is_file():
        return ValidationResult(False, "missing_step", str(step_path))
    if json_path.stat().st_size == 0:
        return ValidationResult(False, "empty_json_file")
    if step_path.stat().st_size == 0:
        return ValidationResult(False, "empty_step_file")
    try:
        history = load_history(json_path)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        return ValidationResult(False, "invalid_json", str(error)[:300])
    if not history:
        return ValidationResult(False, "empty_history")
    for index, feature in enumerate(history):
        if not isinstance(feature.get("sketch", {}), dict):
            return ValidationResult(False, "invalid_sketch", f"feature {index}")
        if not isinstance(feature.get("constraints", {}), dict):
            return ValidationResult(False, "invalid_constraints", f"feature {index}")
    try:
        text = step_path.read_text(encoding="latin-1", errors="ignore")
    except OSError as error:
        return ValidationResult(False, "unreadable_step", str(error)[:300])
    records = ENTITY_RE.findall(text)
    if not records:
        return ValidationResult(False, "no_step_entities")
    if not any(any(marker in name for marker in INSTANCE_MARKERS) for _, name, _ in records):
        return ValidationResult(False, "no_brep_entities")
    return ValidationResult(True)


def validate_entries(
    entries: list[dict[str, Any]], workers: int, monitor_every_samples: int
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    seen: set[str] = set()
    unique_entries, excluded = [], []
    for entry in entries:
        if entry["uid"] in seen:
            excluded.append({**entry, "reason": "duplicate_uid", "detail": ""})
        else:
            seen.add(entry["uid"])
            unique_entries.append(entry)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = pool.map(_validate_entry, unique_entries)
        retained = []
        for number, (entry, result) in enumerate(
            zip(unique_entries, results, strict=True), start=1
        ):
            if result.valid:
                retained.append(entry)
            else:
                excluded.append({**entry, "reason": result.reason, "detail": result.detail})
            if number % monitor_every_samples == 0 or number == len(unique_entries):
                print(f"Dataset validation {number:,}/{len(unique_entries):,}")
    return retained, excluded


def _finite_graph(graph: dict[str, Any], numeric_keys: tuple[str, ...]) -> bool:
    return all(torch.isfinite(graph[key]).all().item() for key in numeric_keys)


def _write_audit(
    report_dir: Path,
    discovered: int,
    retained: list[dict[str, Any]],
    excluded: list[dict[str, str]],
) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    reasons = Counter(item["reason"] for item in excluded)
    split_counts = {
        split: sum(item["split"] == split for item in retained)
        for split in ("train", "val", "test")
    }
    report = {
        "discovered_samples": discovered,
        "retained_samples": len(retained),
        "excluded_samples": len(excluded),
        "retained_by_split": split_counts,
        "excluded_by_reason": dict(sorted(reasons.items())),
        "missing_functional_labels_retained": sum(
            not bool(item["functional_type"]) for item in retained
        ),
        "functional_labels_used_for_training": False,
    }
    (report_dir / "dataset_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fields = ("uid", "source_path", "step_path", "split", "partition", "functional_type")
    with (report_dir / "retained_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: item.get(field, "") for field in fields} for item in retained)
    excluded_fields = (*fields, "reason", "detail")
    with (report_dir / "excluded_samples.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=excluded_fields)
        writer.writeheader()
        writer.writerows(
            {field: item.get(field, "") for field in excluded_fields} for item in excluded
        )
    print(
        f"Dataset audit: discovered={discovered:,} retained={len(retained):,} excluded={len(excluded):,} "
        f"train={split_counts['train']:,} val={split_counts['val']:,} test={split_counts['test']:,}"
    )
    if reasons:
        print(
            "Exclusions: "
            + ", ".join(f"{name}={count:,}" for name, count in sorted(reasons.items()))
        )
    return report


def _distribution(values: list[int] | torch.Tensor) -> dict[str, float]:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    result = torch.quantile(tensor, QUANTILES)
    return {
        name: round(float(value), 3) for name, value in zip(QUANTILE_NAMES, result, strict=True)
    }


def _bounded_histogram_distribution(
    graphs: list[dict[str, Any]], width: int, key: str, use_mask: bool
) -> tuple[dict[str, float], float]:
    """Compute exact bounded integer quantiles without one huge per-node tensor."""
    histogram = torch.zeros(width + 1, dtype=torch.long)
    for graph in graphs:
        tensor = graph[key]
        counts = tensor.sum(-1) if use_mask else (tensor != 0).sum(-1)
        histogram += torch.bincount(counts.to(torch.long), minlength=width + 1)
    cumulative = histogram.cumsum(0)
    total = int(cumulative[-1])

    def value_at(rank: int) -> int:
        return int(torch.searchsorted(cumulative, torch.tensor(rank + 1)).item())

    quantiles = []
    for quantile in QUANTILES.tolist():
        position = quantile * (total - 1)
        lower, upper = math.floor(position), math.ceil(position)
        fraction = position - lower
        value = value_at(lower) * (1.0 - fraction) + value_at(upper) * fraction
        quantiles.append(round(value, 3))
    mean = float((histogram * torch.arange(width + 1)).sum() / total)
    return dict(zip(QUANTILE_NAMES, quantiles, strict=True)), mean


def _representation_audit(
    parametric_graphs: list[dict[str, Any]],
    step_graphs: list[dict[str, Any]],
    max_step_instances: int,
) -> dict[str, Any]:
    parametric_nodes = [len(graph["node_type"]) for graph in parametric_graphs]
    parametric_edges = [graph["edge_index"].shape[1] for graph in parametric_graphs]
    step_nodes = [len(graph["node_type"]) for graph in step_graphs]
    step_edges = [graph["edge_index"].shape[1] for graph in step_graphs]
    step_types = [graph["num_step_types"] for graph in step_graphs]
    step_candidates = [graph["num_brep_candidates"] for graph in step_graphs]
    step_retained = [graph["num_brep_instances"] for graph in step_graphs]
    feature_counts = [graph["metadata"]["num_features"] for graph in parametric_graphs]
    primitive_counts = [
        sum(graph["metadata"]["primitive_counts"].values()) for graph in parametric_graphs
    ]
    constraint_counts = [
        sum(graph["metadata"]["constraint_counts"].values()) for graph in parametric_graphs
    ]
    parametric_active, parametric_active_mean = _bounded_histogram_distribution(
        parametric_graphs, 24, "numeric_mask", True
    )
    step_nonzero, step_nonzero_mean = _bounded_histogram_distribution(
        step_graphs, 12, "numeric", False
    )
    truncated = sum(graph["brep_truncated"] for graph in step_graphs)
    return {
        "samples": len(parametric_graphs),
        "graph_batching_uses_padding": False,
        "parametric_numeric_width": 24,
        "step_numeric_width": 12,
        "parametric_nodes": _distribution(parametric_nodes),
        "parametric_edges": _distribution(parametric_edges),
        "feature_operations": _distribution(feature_counts),
        "sketch_primitives": _distribution(primitive_counts),
        "geometric_and_dimensional_constraints": _distribution(constraint_counts),
        "parametric_active_numeric_fields_per_node": parametric_active,
        "parametric_numeric_field_utilization": round(parametric_active_mean / 24, 6),
        "step_nodes_after_selection": _distribution(step_nodes),
        "step_edges_after_selection": _distribution(step_edges),
        "step_entity_types_per_part": _distribution(step_types),
        "step_brep_candidates_before_selection": _distribution(step_candidates),
        "step_brep_instances_retained": _distribution(step_retained),
        "step_numeric_nonzero_fields_per_node": step_nonzero,
        "step_numeric_nonzero_fraction": round(step_nonzero_mean / 12, 6),
        "max_step_instances": max_step_instances,
        "step_parts_truncated": truncated,
        "step_parts_truncated_fraction": round(truncated / len(step_graphs), 6),
        "branch_vector_padding": {
            "parametric": False,
            "step": False,
            "explanation": (
                "Branch vectors concatenate learned role-aware summaries and a learned global token; "
                "they are not fixed raw-feature arrays padded with zeros."
            ),
        },
    }


def pack_graphs(
    graphs: list[dict[str, Any]], node_keys: tuple[str, ...], edge_keys: tuple[str, ...]
) -> dict[str, torch.Tensor]:
    node_pointer, edge_pointer = [0], [0]
    for graph in graphs:
        node_pointer.append(node_pointer[-1] + len(graph[node_keys[0]]))
        edge_pointer.append(edge_pointer[-1] + graph["edge_index"].shape[1])
    packed = {key: torch.cat([graph[key] for graph in graphs]) for key in (*node_keys, *edge_keys)}
    packed["edge_index"] = torch.cat([graph["edge_index"] for graph in graphs], dim=1)
    packed["node_pointer"] = torch.tensor(node_pointer)
    packed["edge_pointer"] = torch.tensor(edge_pointer)
    return packed


def unpack_graph(
    packed: dict[str, torch.Tensor],
    index: int,
    node_keys: tuple[str, ...],
    edge_keys: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    node_start, node_stop = packed["node_pointer"][index : index + 2].tolist()
    edge_start, edge_stop = packed["edge_pointer"][index : index + 2].tolist()
    graph = {key: packed[key][node_start:node_stop] for key in node_keys}
    graph.update({key: packed[key][edge_start:edge_stop] for key in edge_keys})
    graph["edge_index"] = packed["edge_index"][:, edge_start:edge_stop]
    graph["global_index"] = torch.tensor(0)
    return graph


def build_cache(
    dataset_root: str | Path,
    output_path: str | Path,
    max_step_instances: int = 128,
    max_parametric_nodes: int | None = None,
    validation_workers: int = 8,
    monitor_every_samples: int = 5_000,
) -> dict[str, Any]:
    """Audit all raw pairs and build a cache from every usable HistCAD sample."""
    dataset_root, output_path = Path(dataset_root), Path(output_path)
    discovered_entries = discover_entries(dataset_root)
    entries, excluded = validate_entries(
        discovered_entries, validation_workers, monitor_every_samples
    )
    entity_types = scan_step_entity_types(
        [Path(entry["step_path"]) for entry in entries], monitor_every_samples
    )
    entity_to_id = {name: index for index, name in enumerate(entity_types)}
    valid_entries, parametric_graphs, step_graphs, targets, structural = [], [], [], [], []
    for number, entry in enumerate(entries, start=1):
        try:
            history = load_history(entry["source_path"])
            parametric = build_parametric_graph(history, entry["source_path"], max_parametric_nodes)
            step = parse_step_graph(entry["step_path"], entity_to_id, max_step_instances)
            if not _finite_graph(parametric.as_dict(), ("numeric",)):
                raise ValueError("non-finite parametric numeric values")
            if not _finite_graph(step, ("numeric", "edge_weight", "bbox_mm")):
                raise ValueError("non-finite STEP numeric values")
            if step["num_brep_instances"] < 1:
                raise ValueError("STEP graph contains no usable B-Rep instance")
            target = dimension_target(history, parametric.metadata, step["bbox_mm"])
            if not torch.isfinite(target).all():
                raise ValueError("non-finite dimension target")
        except (OSError, UnicodeError, ValueError, RuntimeError, IndexError, KeyError) as error:
            excluded.append({**entry, "reason": "graph_build_error", "detail": str(error)[:300]})
            continue
        valid_entries.append(entry)
        parametric_graphs.append(parametric.as_dict())
        step_graphs.append(step)
        targets.append(target)
        structural.append(
            {
                **parametric.metadata,
                "num_step_entities": step["num_step_entities"],
                "num_step_types": step["num_step_types"],
                "num_brep_candidates": step["num_brep_candidates"],
                "num_brep_instances": step["num_brep_instances"],
                "brep_truncated": step["brep_truncated"],
            }
        )
        if number % monitor_every_samples == 0 or number == len(entries):
            print(f"Preprocessing {number:,}/{len(entries):,} candidate samples")

    split_order = {"train": 0, "val": 1, "test": 2}
    order = sorted(
        range(len(valid_entries)),
        key=lambda index: (split_order[valid_entries[index]["split"]], valid_entries[index]["uid"]),
    )
    valid_entries = [valid_entries[index] for index in order]
    parametric_graphs = [parametric_graphs[index] for index in order]
    step_graphs = [step_graphs[index] for index in order]
    targets = [targets[index] for index in order]
    structural = [structural[index] for index in order]
    audit = _write_audit(output_path.parent, len(discovered_entries), valid_entries, excluded)
    if not valid_entries:
        raise RuntimeError("Dataset audit retained no usable samples")
    representation_audit = _representation_audit(parametric_graphs, step_graphs, max_step_instances)
    (output_path.parent / "representation_audit.json").write_text(
        json.dumps(representation_audit, indent=2), encoding="utf-8"
    )
    result = {
        "format_version": 7,
        "uids": [entry["uid"] for entry in valid_entries],
        "metadata": valid_entries,
        "structural_metadata": structural,
        "split_sizes": audit["retained_by_split"],
        "step_entity_types": entity_types,
        "parametric_graphs": pack_graphs(
            parametric_graphs, PARAMETRIC_NODE_KEYS, PARAMETRIC_EDGE_KEYS
        ),
        "step_graphs": pack_graphs(step_graphs, STEP_NODE_KEYS, STEP_EDGE_KEYS),
        "dimension_targets": torch.stack(targets),
        "dimension_target_names": tuple(DIMENSION_TARGET_NAMES),
        "dataset_audit": audit,
        "representation_audit": representation_audit,
        "provenance": {
            "dataset_root": str(dataset_root.resolve()),
            "split": "sha1(uid), 90/5/5 after validity filtering",
            "all_valid_training_samples_used": True,
            "functional_labels_used_for_training": False,
            "max_step_instances": max_step_instances,
            "max_parametric_nodes": max_parametric_nodes,
            "graph_batching_uses_padding": False,
            "step_selection": (
                "All entity types and type references; prioritized B-Rep instance nodes up to the cap"
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)
    print(f"Saved paired graph cache: {output_path.resolve()}")
    return result
