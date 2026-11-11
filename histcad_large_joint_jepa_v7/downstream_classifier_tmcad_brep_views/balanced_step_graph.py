"""A complementary, geometry-balanced 256-instance STEP graph for frozen v7.

The node/edge/numeric schema is identical to the original v7 parser. Only the
choice of instance nodes differs. Full-file type aggregates and typed-reference
edges remain unchanged; no model parameter is updated.
"""

from __future__ import annotations

import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from step_statements import STEP_MATCHER

V7_ROOT = Path(__file__).resolve().parent.parent
if str(V7_ROOT) not in sys.path:
    sys.path.insert(0, str(V7_ROOT))
from src.histcad_large_jepa.step_graph import (
    CURVE_MARKERS, INSTANCE_MARKERS, NUMBER_RE, REFERENCE_RE,
    STEP_NUMERIC_DIM, SURFACE_MARKERS, _numeric,
)


def _category(name: str) -> str:
    if any(marker in name for marker in SURFACE_MARKERS):
        return "surface"
    if not name.startswith("EDGE_") and any(marker in name for marker in CURVE_MARKERS):
        return "curve"
    return "topology"


def _priority(name: str) -> int:
    return next((i for i, marker in enumerate(INSTANCE_MARKERS) if marker in name),
                len(INSTANCE_MARKERS))


def _diverse(records: list[tuple[str, str, str]], limit: int) -> list[tuple[str, str, str]]:
    """Deterministic type round-robin; retain original order within each type."""
    buckets: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for record in records:
        buckets[record[1]].append(record)
    positions = {name: 0 for name in buckets}
    selected = []
    while len(selected) < limit:
        added = False
        for name in sorted(buckets):
            position = positions[name]
            if position < len(buckets[name]):
                selected.append(buckets[name][position])
                positions[name] += 1
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
    return selected


def _select_instances(records: list[tuple[str, str, str]], cap: int):
    candidates = [record for record in records
                  if any(marker in record[1] for marker in INSTANCE_MARKERS)]
    groups = {kind: [record for record in candidates if _category(record[1]) == kind]
              for kind in ("topology", "surface", "curve")}
    surface = _diverse(groups["surface"], cap // 4)
    curve = _diverse(groups["curve"], cap // 4)
    geometry_ids = {record[0] for record in surface + curve}
    # Prefer topology nodes that directly reference selected surfaces/curves.
    topology = sorted(groups["topology"], key=lambda record: (
        -sum(ref in geometry_ids for ref in REFERENCE_RE.findall(record[2])),
        _priority(record[1]),
        int(record[0]),
    ))[:cap // 2]
    selected = topology + surface + curve
    selected_ids = {record[0] for record in selected}
    if len(selected) < cap:
        remaining = sorted((record for record in candidates if record[0] not in selected_ids),
                           key=lambda record: (_priority(record[1]), int(record[0])))
        selected.extend(remaining[:cap - len(selected)])
    if len(selected) > cap:
        raise RuntimeError("Balanced STEP selection exceeded its instance cap")
    original_order = {record[0]: index for index, record in enumerate(records)}
    selected.sort(key=lambda record: (_priority(record[1]), original_order[record[0]]))
    category_counts = dict(Counter(_category(name) for _, name, _ in selected))
    return candidates, selected, category_counts


def parse_balanced_step_graph(path: str | Path, vocabulary: dict[str, int],
                              max_instances: int = 256) -> dict:
    if max_instances < 4:
        raise ValueError("The balanced graph needs at least four instance slots")
    text = Path(path).read_text(encoding="latin-1", errors="ignore")
    records = STEP_MATCHER.findall(text)
    if not records:
        raise ValueError(f"No parsable STEP DATA entities: {path}")
    id_to_type = {identifier: name for identifier, name, _ in records}
    body_by_id = {identifier: body for identifier, _, body in records}
    counts = Counter(name for _, name, _ in records)
    grouped_numbers: dict[str, list[float]] = defaultdict(list)
    type_references: Counter[tuple[str, str]] = Counter()
    for _, name, body in records:
        grouped_numbers[name].extend(float(value) for value in NUMBER_RE.findall(body))
        for reference in REFERENCE_RE.findall(body):
            if reference in id_to_type:
                type_references[(name, id_to_type[reference])] += 1
    names = sorted(counts)
    type_local = {name: index + 1 for index, name in enumerate(names)}
    node_type, node_role = [0], [0]
    numeric = [[0.0] * STEP_NUMERIC_DIM]
    for name in names:
        node_type.append(vocabulary.get(name, 0))
        node_role.append(1)
        numeric.append(_numeric(name, grouped_numbers[name], counts[name], False))

    candidates, selected, selected_counts = _select_instances(records, max_instances)
    instance_local = {}
    for identifier, name, body in selected:
        instance_local[identifier] = len(node_type)
        numbers = [float(value) for value in NUMBER_RE.findall(body)]
        node_type.append(vocabulary.get(name, 0))
        node_role.append(2)
        numeric.append(_numeric(name, numbers, 1, True))
    edges = [(index, index, 0, 1.0) for index in range(len(node_type))]
    for index in range(1, len(names) + 1):
        edges.extend(((0, index, 1, 1.0), (index, 0, 2, 1.0)))
    for (source, target), count in type_references.items():
        left, right, weight = type_local[source], type_local[target], math.log1p(count)
        edges.extend(((left, right, 3, weight), (right, left, 4, weight)))
    for identifier, name, _ in selected:
        instance, aggregate = instance_local[identifier], type_local[name]
        edges.extend(((aggregate, instance, 5, 1.0), (instance, aggregate, 6, 1.0)))
        for reference in REFERENCE_RE.findall(body_by_id[identifier]):
            if reference in instance_local:
                other = instance_local[reference]
                edges.extend(((instance, other, 7, 1.0), (other, instance, 8, 1.0)))
    graph = {
        "node_type": torch.tensor(node_type),
        "node_role": torch.tensor(node_role),
        "numeric": torch.tensor(numeric, dtype=torch.float32),
        "edge_index": torch.tensor([(left, right) for left, right, _, _ in edges]).T.long(),
        "edge_type": torch.tensor([kind for _, _, kind, _ in edges]),
        "edge_weight": torch.tensor([weight for _, _, _, weight in edges], dtype=torch.float32),
        "num_step_entities": len(records),
        "num_step_types": len(names),
        "num_brep_candidates": len(candidates),
        "num_brep_instances": len(selected),
        "selected_category_counts": selected_counts,
    }
    if not torch.isfinite(graph["numeric"]).all() or not torch.isfinite(graph["edge_weight"]).all():
        raise ValueError(f"Non-finite balanced STEP graph: {path}")
    return graph
