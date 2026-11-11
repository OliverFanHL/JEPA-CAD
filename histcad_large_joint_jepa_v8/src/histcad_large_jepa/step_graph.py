from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from .schema import constraint_instances, iter_numbers, iter_strings, parse_dimension

ENTITY_RE = re.compile(r"#(\d+)\s*=\s*([A-Z][A-Z0-9_]*)\s*\((.*?)\);", re.DOTALL)
REFERENCE_RE = re.compile(r"#(\d+)")
NUMBER_RE = re.compile(r"(?<![#A-Za-z_])[-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?")
SURFACE_MARKERS = ("SURFACE", "PLANE", "CYLIND", "CONICAL", "TOROIDAL", "SPHERICAL")
CURVE_MARKERS = ("CURVE", "LINE", "CIRCLE", "ELLIPSE", "B_SPLINE")
TOPOLOGY_MARKERS = ("SOLID", "SHELL", "FACE", "LOOP", "BOUND", "EDGE", "VERTEX")
INSTANCE_MARKERS = TOPOLOGY_MARKERS + SURFACE_MARKERS + CURVE_MARKERS
STEP_NUMERIC_DIM = 12
DIMENSION_TARGET_NAMES = (
    "log_scale_mm",
    "log_bbox_x_mm",
    "log_bbox_y_mm",
    "log_bbox_z_mm",
    "log_bbox_x_over_y",
    "log_bbox_x_over_z",
    "log_bbox_y_over_z",
    "log_dimensional_constraint_count",
    "log_mean_dimensional_constraint_mm",
    "log_max_dimensional_constraint_mm",
    "mean_dimensional_constraint_over_scale",
    "max_dimensional_constraint_over_scale",
    "log_feature_size_count",
    "log_mean_feature_size_mm",
    "log_max_feature_size_mm",
    "mean_feature_size_over_scale",
    "max_feature_size_over_scale",
    "log_angle_constraint_count",
    "mean_absolute_angle_over_180",
    "log_feature_count",
    "log_primitive_count",
    "log_constraint_count",
)


def scan_step_entity_types(paths: list[Path], progress_every: int = 5_000) -> tuple[str, ...]:
    """Build an exact vocabulary; unlike v4, distinct STEP names never share a hash bucket."""
    names: set[str] = set()
    for number, path in enumerate(paths, start=1):
        text = path.read_text(encoding="latin-1", errors="ignore")
        names.update(name for _, name, _ in ENTITY_RE.findall(text))
        if number % progress_every == 0 or number == len(paths):
            print(f"STEP vocabulary scan {number:,}/{len(paths):,}")
    return ("PAD", *sorted(names))


def _signed_log(value: float) -> float:
    return math.copysign(math.log1p(abs(value)), value)


def _numeric(name: str, values: list[float], count: int, instance: bool) -> list[float]:
    tensor = torch.tensor(values) if values else torch.zeros(1)
    return [
        math.log1p(count),
        math.log1p(len(values)),
        _signed_log(float(tensor.mean())),
        math.log1p(float(tensor.abs().mean())),
        math.log1p(float(tensor.std())) if len(tensor) > 1 else 0.0,
        math.log1p(float(tensor.abs().max())),
        float(any(marker in name for marker in SURFACE_MARKERS)),
        float(any(marker in name for marker in CURVE_MARKERS)),
        float(any(marker in name for marker in TOPOLOGY_MARKERS)),
        len(name) / 40.0,
        float(instance),
        math.log1p(sum(abs(item) for item in values)),
    ]


def parse_step_graph(
    path: str | Path, entity_to_id: dict[str, int], max_instances: int = 128
) -> dict[str, Any]:
    text = Path(path).read_text(encoding="latin-1", errors="ignore")
    records = ENTITY_RE.findall(text)
    id_to_type = {identifier: name for identifier, name, _ in records}
    body_by_id = {identifier: body for identifier, _, body in records}
    type_counts = Counter(name for _, name, _ in records)
    grouped_numbers: dict[str, list[float]] = defaultdict(list)
    type_references: Counter[tuple[str, str]] = Counter()
    cartesian_points: list[list[float]] = []
    for _, name, body in records:
        values = [float(value) for value in NUMBER_RE.findall(body)]
        grouped_numbers[name].extend(values)
        if name == "CARTESIAN_POINT" and len(values) >= 2:
            cartesian_points.append(values[-3:] if len(values) >= 3 else [*values[-2:], 0.0])
        for target in REFERENCE_RE.findall(body):
            if target in id_to_type:
                type_references[(name, id_to_type[target])] += 1

    names = sorted(type_counts)
    type_local = {name: number + 1 for number, name in enumerate(names)}
    node_type, node_role = [0], [0]
    numeric = [[0.0] * STEP_NUMERIC_DIM]
    for name in names:
        node_type.append(entity_to_id[name])
        node_role.append(1)
        numeric.append(_numeric(name, grouped_numbers[name], type_counts[name], False))

    candidates = [
        record for record in records if any(marker in record[1] for marker in INSTANCE_MARKERS)
    ]
    candidates.sort(
        key=lambda record: next(
            (priority for priority, marker in enumerate(INSTANCE_MARKERS) if marker in record[1]),
            len(INSTANCE_MARKERS),
        )
    )
    selected = candidates[:max_instances]
    instance_local: dict[str, int] = {}
    for identifier, name, body in selected:
        instance_local[identifier] = len(node_type)
        values = [float(value) for value in NUMBER_RE.findall(body)]
        node_type.append(entity_to_id[name])
        node_role.append(2)
        numeric.append(_numeric(name, values, 1, True))

    edges: list[tuple[int, int, int, float]] = [
        (index, index, 0, 1.0) for index in range(len(node_type))
    ]
    for index in range(1, len(names) + 1):
        edges.extend(((0, index, 1, 1.0), (index, 0, 2, 1.0)))
    for (source, target), count in type_references.items():
        left, right, weight = type_local[source], type_local[target], math.log1p(count)
        edges.extend(((left, right, 3, weight), (right, left, 4, weight)))
    for identifier, name, _ in selected:
        instance, aggregate = instance_local[identifier], type_local[name]
        edges.extend(((aggregate, instance, 5, 1.0), (instance, aggregate, 6, 1.0)))
        for target in REFERENCE_RE.findall(body_by_id[identifier]):
            if target in instance_local:
                other = instance_local[target]
                edges.extend(((instance, other, 7, 1.0), (other, instance, 8, 1.0)))
    points = torch.tensor(cartesian_points).float() if cartesian_points else torch.zeros((0, 3))
    bbox = points.max(0).values - points.min(0).values if len(points) else torch.zeros(3)
    return {
        "node_type": torch.tensor(node_type),
        "node_role": torch.tensor(node_role),
        "numeric": torch.tensor(numeric, dtype=torch.float32),
        "edge_index": torch.tensor([(left, right) for left, right, _, _ in edges]).T.long(),
        "edge_type": torch.tensor([kind for _, _, kind, _ in edges]),
        "edge_weight": torch.tensor([weight for _, _, _, weight in edges], dtype=torch.float32),
        "global_index": torch.tensor(0),
        "bbox_mm": bbox,
        "num_step_entities": len(records),
        "num_step_types": len(names),
        "num_brep_candidates": len(candidates),
        "num_brep_instances": len(selected),
        "brep_truncated": len(candidates) > len(selected),
    }


def dimension_target(
    history: list[dict[str, Any]], metadata: dict[str, Any], bbox: torch.Tensor
) -> torch.Tensor:
    def parsed(value: Any) -> list[float]:
        result = [
            item for text in iter_strings(value) if (item := parse_dimension(text)) is not None
        ]
        result.extend(iter_numbers(value))
        return [abs(float(item)) for item in result if math.isfinite(float(item))]

    dimensional_types = {"Diameter", "Radius", "MajorRadius", "MinorRadius", "Distance", "Length"}
    dimensional: list[float] = []
    sizes: list[float] = []
    angles: list[float] = []
    for feature in history:
        for key in ("towards", "opposite", "pitch", "radius", "dist"):
            sizes.extend(parsed(feature.get(key)))
        for primitive in feature.get("sketch", {}).values():
            if isinstance(primitive, dict):
                for key in ("radius", "major", "minor"):
                    sizes.extend(parsed(primitive.get(key)))
        for kind, records in feature.get("constraints", {}).items():
            for record in constraint_instances(records):
                values = parsed(record)
                if kind in dimensional_types:
                    dimensional.extend(values[-1:])
                elif kind == "Angle":
                    angles.extend(values[-1:])
    scale = max(float(metadata["scale_mm"]), 1e-6)
    box = bbox.clamp_min(0).tolist()

    def summary(values: list[float]) -> list[float]:
        if not values:
            return [0.0] * 5
        mean, maximum = sum(values) / len(values), max(values)
        return [
            math.log1p(len(values)),
            math.log1p(mean),
            math.log1p(maximum),
            mean / scale,
            maximum / scale,
        ]

    def ratio(left: float, right: float) -> float:
        return math.log((left + 1e-3) / (right + 1e-3))

    values = [
        math.log1p(scale),
        *(math.log1p(item) for item in box),
        ratio(box[0], box[1]),
        ratio(box[0], box[2]),
        ratio(box[1], box[2]),
        *summary(dimensional),
        *summary(sizes),
        math.log1p(len(angles)),
        sum(angles) / max(1, len(angles)) / 180.0,
        math.log1p(metadata["num_features"]),
        math.log1p(sum(metadata["primitive_counts"].values())),
        math.log1p(sum(metadata["constraint_counts"].values())),
    ]
    if len(values) != len(DIMENSION_TARGET_NAMES):
        raise RuntimeError("Dimension-target schema mismatch")
    return torch.tensor(values, dtype=torch.float32)


def collate_step_graphs(graphs: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    node_types, roles, numeric, edges, edge_types, edge_weights = [], [], [], [], [], []
    graph_index, global_indices = [], []
    offset = 0
    for graph_number, graph in enumerate(graphs):
        count = len(graph["node_type"])
        node_types.append(graph["node_type"])
        roles.append(graph["node_role"])
        numeric.append(graph["numeric"])
        edges.append(graph["edge_index"] + offset)
        edge_types.append(graph["edge_type"])
        edge_weights.append(graph["edge_weight"])
        graph_index.append(torch.full((count,), graph_number, dtype=torch.long))
        global_indices.append(offset)
        offset += count
    return {
        "node_type": torch.cat(node_types),
        "node_role": torch.cat(roles),
        "numeric": torch.cat(numeric),
        "edge_index": torch.cat(edges, dim=1),
        "edge_type": torch.cat(edge_types),
        "edge_weight": torch.cat(edge_weights),
        "graph_index": torch.cat(graph_index),
        "global_index": torch.tensor(global_indices),
    }
