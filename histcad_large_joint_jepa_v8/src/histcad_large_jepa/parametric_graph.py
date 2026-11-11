from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch

from .schema import (
    constraint_instances,
    infer_feature_type,
    iter_numbers,
    iter_strings,
    parse_dimension,
    primitive_type,
    uid_from_json_path,
)
from .vocab import (
    AUX_TO_ID,
    NODE_TO_ID,
    PARAMETRIC_RELATION_TO_ID,
    SUBTYPE_TO_ID,
    lookup,
)

PARAMETRIC_NUMERIC_DIM = 24


@dataclass
class ParametricGraph:
    uid: str
    node_type: torch.Tensor
    subtype: torch.Tensor
    aux_type: torch.Tensor
    numeric: torch.Tensor
    numeric_mask: torch.Tensor
    feature_index: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    metadata: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return vars(self)


def _put(values: list[float], mask: list[bool], index: int, value: float | None) -> None:
    if value is not None and math.isfinite(float(value)):
        values[index], mask[index] = float(value), True


def _numbers(value: Any) -> list[float]:
    return (
        [float(item) for item in value if isinstance(item, (int, float))]
        if isinstance(value, list)
        else []
    )


def _sample_scale(history: list[dict[str, Any]]) -> float:
    magnitudes: list[float] = []
    for feature in history:
        system = feature.get("coordinate_system", {})
        magnitudes.extend(abs(x) for x in iter_numbers(system.get("Translation Vector", [])))
        for entity in feature.get("sketch", {}).values():
            if isinstance(entity, dict):
                for key, value in entity.items():
                    if key not in {
                        "angle",
                        "degree",
                        "periodic",
                        "large_arc",
                        "sweep",
                        "knots",
                        "weights",
                    }:
                        magnitudes.extend(abs(x) for x in iter_numbers(value))
        for key in ("towards", "opposite", "pitch", "radius", "dist", "near_points"):
            magnitudes.extend(abs(x) for x in iter_numbers(feature.get(key, [])))
    return max(1.0, max((x for x in magnitudes if math.isfinite(x)), default=1.0))


def _primitive_features(
    kind: str, entity: dict[str, Any], scale: float
) -> tuple[list[float], list[bool]]:
    values, mask = [0.0] * PARAMETRIC_NUMERIC_DIM, [False] * PARAMETRIC_NUMERIC_DIM
    if kind == "line":
        data = _numbers(entity.get("start"))[:2] + _numbers(entity.get("end"))[:2]
        for index, item in enumerate(data):
            _put(values, mask, index, item / scale)
    elif kind == "circle":
        for index, item in enumerate(_numbers(entity.get("center"))[:2]):
            _put(values, mask, index, item / scale)
        radius = parse_dimension(entity.get("radius"))
        _put(values, mask, 2, radius / scale if radius is not None else None)
    elif kind == "arc":
        data = (
            _numbers(entity.get("start"))[:2]
            + _numbers(entity.get("middle"))[:2]
            + _numbers(entity.get("end"))[:2]
        )
        for index, item in enumerate(data):
            _put(values, mask, index, item / scale)
    elif kind in {"ellipse", "elliptical_arc"}:
        data = (
            _numbers(entity.get("center"))[:2]
            if kind == "ellipse"
            else _numbers(entity.get("start"))[:2] + _numbers(entity.get("end"))[:2]
        )
        for index, item in enumerate(data):
            _put(values, mask, index, item / scale)
        offset = len(data)
        for key in ("major", "minor"):
            item = parse_dimension(entity.get(key))
            _put(values, mask, offset, item / scale if item is not None else None)
            offset += 1
        angle = parse_dimension(entity.get("angle"))
        if angle is not None:
            _put(values, mask, offset, math.sin(math.radians(angle)))
            _put(values, mask, offset + 1, math.cos(math.radians(angle)))
        if kind == "elliptical_arc":
            _put(values, mask, offset + 2, float(bool(entity.get("large_arc"))))
            _put(values, mask, offset + 3, float(bool(entity.get("sweep"))))
    elif kind == "nurbs":
        _put(values, mask, 0, float(entity.get("degree", 0)) / 5.0)
        _put(values, mask, 1, float(bool(entity.get("periodic"))))
        _put(values, mask, 2, len(entity.get("controls", [])) / 32.0)
        weights, knots = (
            list(iter_numbers(entity.get("weights", []))),
            list(iter_numbers(entity.get("knots", []))),
        )
        if weights:
            _put(values, mask, 3, sum(weights) / len(weights))
        if knots:
            _put(values, mask, 4, min(knots))
            _put(values, mask, 5, max(knots))
    return values, mask


def _feature_features(
    feature: dict[str, Any], kind: str, scale: float
) -> tuple[list[float], list[bool]]:
    values, mask = [0.0] * PARAMETRIC_NUMERIC_DIM, [False] * PARAMETRIC_NUMERIC_DIM
    system = feature.get("coordinate_system", {})
    for index, item in enumerate(_numbers(system.get("Translation Vector"))[:3]):
        _put(values, mask, index, item / scale)
    for index, angle in enumerate(_numbers(system.get("Euler Angles"))[:3]):
        _put(values, mask, 3 + 2 * index, math.sin(math.radians(angle)))
        _put(values, mask, 4 + 2 * index, math.cos(math.radians(angle)))
    if kind == "Extrude":
        for index, key in enumerate(("towards", "opposite"), start=9):
            item = parse_dimension(feature.get(key))
            _put(values, mask, index, item / scale if item is not None else None)
    elif kind == "Revolve":
        for index, key in enumerate(("start", "end")):
            angle = parse_dimension(feature.get(key))
            if angle is not None:
                _put(values, mask, 9 + 2 * index, math.sin(math.radians(angle)))
                _put(values, mask, 10 + 2 * index, math.cos(math.radians(angle)))
        axis = feature.get("axis", [[], []])
        if isinstance(axis, list) and len(axis) >= 2:
            for index, item in enumerate(_numbers(axis[0])[:3]):
                _put(values, mask, 13 + index, item / scale)
            for index, item in enumerate(_numbers(axis[1])[:3]):
                _put(values, mask, 16 + index, item)
    elif kind == "HelixSweep":
        pitch = parse_dimension(feature.get("pitch"))
        _put(values, mask, 9, pitch / scale if pitch is not None else None)
        _put(values, mask, 10, parse_dimension(feature.get("turns")))
    elif kind == "Fillet":
        radii = [item / scale for item in iter_numbers(feature.get("radius", []))]
        if radii:
            _put(values, mask, 9, sum(radii) / len(radii))
            _put(values, mask, 10, max(radii))
    elif kind == "Chamfer":
        distance = parse_dimension(feature.get("dist"))
        _put(values, mask, 12, distance / scale if distance is not None else None)
        angle = parse_dimension(feature.get("angle"))
        if angle is not None:
            _put(values, mask, 13, math.sin(math.radians(angle)))
            _put(values, mask, 14, math.cos(math.radians(angle)))
    return values, mask


def _constraint_features(
    kind: str, record: Any, scale: float
) -> tuple[list[float], list[bool], str]:
    values, mask = [0.0] * PARAMETRIC_NUMERIC_DIM, [False] * PARAMETRIC_NUMERIC_DIM
    direction = str(record.get("direction", "NONE")) if isinstance(record, dict) else "NONE"
    parsed = [item for text in iter_strings(record) if (item := parse_dimension(text)) is not None]
    parsed.extend(iter_numbers(record))
    if kind == "Angle" and parsed:
        _put(values, mask, 0, math.sin(math.radians(parsed[-1])))
        _put(values, mask, 1, math.cos(math.radians(parsed[-1])))
    elif (
        kind in {"Diameter", "Radius", "MajorRadius", "MinorRadius", "Distance", "Length"}
        and parsed
    ):
        _put(values, mask, 0, parsed[-1] / scale)
    return values, mask, direction


def build_parametric_graph(
    history: list[dict[str, Any]], source_path: str | Path, max_nodes: int | None = None
) -> ParametricGraph:
    scale = _sample_scale(history)
    node_type: list[int] = []
    subtype: list[int] = []
    aux_type: list[int] = []
    numeric: list[list[float]] = []
    numeric_mask: list[list[bool]] = []
    feature_index: list[int] = []
    edges: list[tuple[int, int, int]] = []

    def node(kind: str, sub: str, aux: str, values=None, mask=None, feature=-1) -> int:
        index = len(node_type)
        node_type.append(NODE_TO_ID[kind])
        subtype.append(lookup(SUBTYPE_TO_ID, sub, "UNKNOWN"))
        aux_type.append(lookup(AUX_TO_ID, aux, "NONE"))
        numeric.append(values if values is not None else [0.0] * PARAMETRIC_NUMERIC_DIM)
        numeric_mask.append(mask if mask is not None else [False] * PARAMETRIC_NUMERIC_DIM)
        feature_index.append(feature)
        return index

    global_id = node("GLOBAL", "GLOBAL", "NONE")
    feature_ids: list[int] = []
    feature_types: list[str] = []
    primitive_counts: dict[str, int] = {}
    constraint_counts: dict[str, int] = {}
    relation = PARAMETRIC_RELATION_TO_ID
    for feature_number, feature in enumerate(history):
        feature_type = infer_feature_type(feature)
        feature_types.append(feature_type)
        values, mask = _feature_features(feature, feature_type, scale)
        feature_id = node(
            "FEATURE",
            feature_type,
            str(feature.get("operation", "NONE")),
            values,
            mask,
            feature_number,
        )
        feature_ids.append(feature_id)
        entities: dict[str, int] = {}
        for name, entity in feature.get("sketch", {}).items():
            kind = primitive_type(name)
            primitive_counts[kind] = primitive_counts.get(kind, 0) + 1
            values, mask = _primitive_features(kind, entity, scale)
            primitive_id = node("PRIMITIVE", kind, "NONE", values, mask, feature_number)
            entities[name] = primitive_id
            edges.extend(
                (
                    (feature_id, primitive_id, relation["FEATURE_TO_MEMBER"]),
                    (primitive_id, feature_id, relation["MEMBER_TO_FEATURE"]),
                )
            )
            if kind == "nurbs" and isinstance(entity, dict):
                controls, weights = entity.get("controls", []), entity.get("weights", [])
                for number, control in enumerate(controls):
                    child_values, child_mask = (
                        [0.0] * PARAMETRIC_NUMERIC_DIM,
                        [False] * PARAMETRIC_NUMERIC_DIM,
                    )
                    for index, item in enumerate(_numbers(control)[:2]):
                        _put(child_values, child_mask, index, item / scale)
                    _put(
                        child_values,
                        child_mask,
                        2,
                        parse_dimension(weights[number] if number < len(weights) else 1.0),
                    )
                    _put(child_values, child_mask, 3, number / max(1, len(controls) - 1))
                    child = node(
                        "CONTROL_POINT",
                        "control_point",
                        "NONE",
                        child_values,
                        child_mask,
                        feature_number,
                    )
                    edges.extend(
                        (
                            (primitive_id, child, relation["PRIMITIVE_TO_PARAMETER"]),
                            (child, primitive_id, relation["PARAMETER_TO_PRIMITIVE"]),
                        )
                    )
                knots = entity.get("knots", [])
                for number, knot in enumerate(knots):
                    child_values, child_mask = (
                        [0.0] * PARAMETRIC_NUMERIC_DIM,
                        [False] * PARAMETRIC_NUMERIC_DIM,
                    )
                    _put(child_values, child_mask, 0, parse_dimension(knot))
                    _put(child_values, child_mask, 1, number / max(1, len(knots) - 1))
                    child = node("KNOT", "knot", "NONE", child_values, child_mask, feature_number)
                    edges.extend(
                        (
                            (primitive_id, child, relation["PRIMITIVE_TO_PARAMETER"]),
                            (child, primitive_id, relation["PARAMETER_TO_PRIMITIVE"]),
                        )
                    )
        for constraint_type, records in feature.get("constraints", {}).items():
            for record in constraint_instances(records):
                constraint_counts[constraint_type] = constraint_counts.get(constraint_type, 0) + 1
                values, mask, direction = _constraint_features(constraint_type, record, scale)
                constraint_id = node(
                    "CONSTRAINT", constraint_type, direction, values, mask, feature_number
                )
                edges.extend(
                    (
                        (feature_id, constraint_id, relation["FEATURE_TO_MEMBER"]),
                        (constraint_id, feature_id, relation["MEMBER_TO_FEATURE"]),
                    )
                )
                referenced: set[tuple[int, str]] = set()
                for text in iter_strings(record):
                    parts = text.split(".", 1)
                    if parts[0] in entities:
                        role = parts[1].lower() if len(parts) == 2 else "entity"
                        referenced.add(
                            (
                                entities[parts[0]],
                                role if role in {"start", "end", "middle", "center"} else "entity",
                            )
                        )
                for primitive_id, role in referenced:
                    forward, backward = (
                        f"CONSTRAINT_TO_{role.upper()}",
                        f"{role.upper()}_TO_CONSTRAINT",
                    )
                    edges.extend(
                        (
                            (constraint_id, primitive_id, relation[forward]),
                            (primitive_id, constraint_id, relation[backward]),
                        )
                    )
        radii = list(iter_numbers(feature.get("radius", [])))
        near_points = feature.get("near_points", [])
        for number, point in enumerate(near_points if isinstance(near_points, list) else []):
            values, mask = [0.0] * PARAMETRIC_NUMERIC_DIM, [False] * PARAMETRIC_NUMERIC_DIM
            for index, item in enumerate(_numbers(point)[:3]):
                _put(values, mask, index, item / scale)
            if radii:
                _put(values, mask, 3, radii[min(number, len(radii) - 1)] / scale)
            boundary = node("BOUNDARY_REF", "near_point", "NONE", values, mask, feature_number)
            edges.extend(
                (
                    (feature_id, boundary, relation["FEATURE_TO_MEMBER"]),
                    (boundary, feature_id, relation["MEMBER_TO_FEATURE"]),
                )
            )

    count = len(node_type)
    if max_nodes is not None and count > max_nodes:
        raise ValueError(f"Parametric graph has {count} nodes; configured limit is {max_nodes}")
    for index in range(count):
        edges.append((index, index, relation["SELF"]))
    for index in range(1, count):
        edges.extend(
            (
                (global_id, index, relation["GLOBAL_TO_NODE"]),
                (index, global_id, relation["NODE_TO_GLOBAL"]),
            )
        )
    for left, right in pairwise(feature_ids):
        edges.extend(
            ((left, right, relation["NEXT_FEATURE"]), (right, left, relation["PREVIOUS_FEATURE"]))
        )
    global_values, global_mask = numeric[0], numeric_mask[0]
    for index, (value, denominator) in enumerate(
        (
            (len(history), 16),
            (count, 256),
            (sum(primitive_counts.values()), 128),
            (sum(constraint_counts.values()), 128),
        )
    ):
        _put(global_values, global_mask, index, value / denominator)
    edge_index = torch.tensor([(source, target) for source, target, _ in edges], dtype=torch.long).T
    metadata = {
        "source_path": str(Path(source_path).resolve()),
        "scale_mm": scale,
        "feature_types": feature_types,
        "operation_signature": ">".join(
            f"{infer_feature_type(item)}:{item.get('operation', 'NONE')}" for item in history
        ),
        "primitive_counts": primitive_counts,
        "constraint_counts": constraint_counts,
        "num_features": len(history),
        "num_nodes": count,
    }
    return ParametricGraph(
        uid_from_json_path(source_path),
        torch.tensor(node_type),
        torch.tensor(subtype),
        torch.tensor(aux_type),
        torch.tensor(numeric, dtype=torch.float32),
        torch.tensor(numeric_mask, dtype=torch.bool),
        torch.tensor(feature_index),
        edge_index,
        torch.tensor([kind for _, _, kind in edges]),
        metadata,
    )


def collate_parametric_graphs(graphs: list[dict[str, Any]]) -> dict[str, Any]:
    node_keys = ("node_type", "subtype", "aux_type", "numeric", "numeric_mask", "feature_index")
    result: dict[str, Any] = {key: [] for key in node_keys}
    result.update(
        edge_index=[], edge_type=[], graph_index=[], global_index=[], uids=[], metadata=[]
    )
    offset = 0
    for graph_number, graph in enumerate(graphs):
        count = len(graph["node_type"])
        for key in node_keys:
            result[key].append(graph[key])
        result["edge_index"].append(graph["edge_index"] + offset)
        result["edge_type"].append(graph["edge_type"])
        result["graph_index"].append(torch.full((count,), graph_number, dtype=torch.long))
        result["global_index"].append(offset)
        result["uids"].append(graph["uid"])
        result["metadata"].append(graph["metadata"])
        offset += count
    for key in node_keys:
        result[key] = torch.cat(result[key])
    result["edge_index"] = torch.cat(result["edge_index"], dim=1)
    result["edge_type"] = torch.cat(result["edge_type"])
    result["graph_index"] = torch.cat(result["graph_index"])
    result["global_index"] = torch.tensor(result["global_index"])
    return result
