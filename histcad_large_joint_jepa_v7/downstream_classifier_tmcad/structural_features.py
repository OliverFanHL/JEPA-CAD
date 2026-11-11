"""Train-only-scaled B-Rep summaries from the existing cached STEP graphs.

These descriptors are a controlled test of whether role pooling discards useful
typed/topological information. They do not require re-parsing STEP files or
changing the frozen JEPA encoder. Labels are never used to select features.
"""

from __future__ import annotations

import json
import math
from collections import Counter

import torch
from common import read_pretrained_metadata, sha256_file


BREP_MARKERS = (
    "SOLID", "SHELL", "FACE", "LOOP", "BOUND", "EDGE", "VERTEX",
    "SURFACE", "PLANE", "CYLIND", "CONICAL", "TOROIDAL", "SPHERICAL",
    "CURVE", "LINE", "CIRCLE", "ELLIPSE", "B_SPLINE",
)
NON_GEOMETRIC_MARKERS = (
    "STYLE", "FONT", "ANNOTATION", "DRAUGHT", "PRESENTATION",
    "COLOUR", "COLOR", "MEASURE", "UNIT", "SHAPE_REPRESENTATION",
    "TESSELLATED", "TRIANGULATED",
)
GRAPH_STAT_NAMES = (
    "log_step_entities", "log_step_types", "log_brep_candidates",
    "log_brep_instances", "retained_brep_fraction",
)
EDGE_TYPES = (3, 4, 5, 6, 7, 8)


def _brep_name(name: str) -> bool:
    return (any(marker in name for marker in BREP_MARKERS)
            and not any(marker in name for marker in NON_GEOMETRIC_MARKERS))


def _graph_descriptors(graph: dict, sample: dict) -> tuple[dict[int, float], list[float]]:
    roles = graph["node_role"].long()
    types = graph["node_type"].long()
    numeric = graph["numeric"].float()
    type_nodes = roles == 1
    counts = torch.expm1(numeric[type_nodes, 0]).clamp_min(0)
    by_type: dict[int, float] = {}
    for type_id, count in zip(types[type_nodes].tolist(), counts.tolist(), strict=True):
        by_type[type_id] = by_type.get(type_id, 0.0) + float(count)

    edges = torch.bincount(graph["edge_type"].long(), minlength=max(EDGE_TYPES) + 1)
    candidates = int(sample["num_brep_candidates"])
    retained = int(sample["num_brep_instances"])
    stats = [
        math.log1p(float(sample["num_step_entities"])),
        math.log1p(float(sample["num_step_types"])),
        math.log1p(float(candidates)),
        math.log1p(float(retained)),
        retained / max(candidates, 1),
        *(torch.log1p(edges[list(EDGE_TYPES)].float()).tolist()),
    ]
    return by_type, stats


def load_structural_features(config: dict, features: dict) -> tuple[torch.Tensor, list[str]]:
    """Return standardized descriptors; build once from the already prepared cache."""
    processed = config["paths"]["processed_dir"]
    manifest_path = processed / "manifest.json"
    manifest_hash = sha256_file(manifest_path)
    settings = config.get("structural", {})
    known_limit = int(settings.get("known_brep_types", 64))
    unknown_limit = int(settings.get("unknown_brep_types", 32))
    if known_limit < 0 or unknown_limit < 0:
        raise ValueError("B-Rep feature limits must be nonnegative")
    cache_path = processed / f"structural_features_v2_k{known_limit}_u{unknown_limit}.pt"
    if cache_path.is_file():
        saved = torch.load(cache_path, map_location="cpu", weights_only=True)
        if (saved.get("format") == "tmcad_step_structural_v2"
                and saved.get("manifest_sha256") == manifest_hash
                and saved.get("uids") == features["uids"]):
            return saved["values"], saved["names"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest["samples"]
    if [item["uid"] for item in samples] != features["uids"]:
        raise ValueError("Structural features and frozen embeddings have different sample order")
    _, vocabulary = read_pretrained_metadata(config)
    graph_rows: list[tuple[dict[int, float], list[float]]] = []
    known_frequency: Counter[int] = Counter()
    unknown_frequency: Counter[str] = Counter()
    for index, sample in enumerate(samples):
        payload = torch.load(processed / sample["graph_relpath"],
                             map_location="cpu", weights_only=True)
        by_type, stats = _graph_descriptors(payload["graph"], sample)
        graph_rows.append((by_type, stats))
        if sample["split"] == "train":
            known_frequency.update({
                type_id: count for type_id, count in by_type.items()
                if 0 < type_id < len(vocabulary) and _brep_name(vocabulary[type_id])
            })
            unknown_frequency.update({
                name: 1 for name in sample.get("unknown_types", []) if _brep_name(name)
            })
        if (index + 1) % 1000 == 0 or index + 1 == len(samples):
            print(f"Structural cache {index + 1:,}/{len(samples):,}", flush=True)

    known_ids = [item for item, _ in sorted(known_frequency.items(),
                                            key=lambda entry: (-entry[1], vocabulary[entry[0]]))[:known_limit]]
    unknown_names = [item for item, _ in sorted(unknown_frequency.items(),
                                                key=lambda entry: (-entry[1], entry[0]))[:unknown_limit]]
    names = ([f"known:{vocabulary[type_id]}" for type_id in known_ids]
             + [f"unseen:{name}:present" for name in unknown_names]
             + list(GRAPH_STAT_NAMES)
             + [f"log_reference_edges_{kind}" for kind in EDGE_TYPES])
    raw = torch.zeros((len(samples), len(names)), dtype=torch.float32)
    for row, (sample, (by_type, stats)) in enumerate(zip(samples, graph_rows, strict=True)):
        known = [math.log1p(by_type.get(type_id, 0.0)) for type_id in known_ids]
        unknown = [float(name in sample.get("unknown_types", [])) for name in unknown_names]
        raw[row] = torch.tensor([*known, *unknown, *stats], dtype=torch.float32)
    train_rows = torch.tensor([i for i, sample in enumerate(samples) if sample["split"] == "train"],
                              dtype=torch.long)
    if not len(train_rows):
        raise ValueError("Cannot standardize descriptors without training samples")
    mean = raw[train_rows].mean(0)
    scale = raw[train_rows].std(0).clamp_min(1e-3)
    values = ((raw - mean) / scale).clamp(-5, 5)
    if not torch.isfinite(values).all():
        raise ValueError("Non-finite structural features")
    torch.save({"format": "tmcad_step_structural_v2", "manifest_sha256": manifest_hash,
                "uids": features["uids"], "names": names, "values": values,
                "train_mean": mean, "train_scale": scale,
                "note": "B-Rep type counts, OOV type presence, and graph connectivity; no labels or test-set scaling."},
               cache_path)
    print(f"Saved {len(names)} structural features: {cache_path}", flush=True)
    return values, names
