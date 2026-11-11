"""Compact full-file B-Rep adjacency summaries, without class-dependent rules."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from common import save_json, source_manifest
from step_statements import PARSER_VERSION, STEP_MATCHER


FORMAT = "tmcad_full_step_topology_v1"
PAIR_TYPES = (
    "plane_plane", "plane_cylinder", "cylinder_cylinder",
    "cylinder_cone", "cylinder_torus", "other_same", "other_mixed",
)
CURVE_TYPES = ("circle", "line", "ellipse", "bspline", "other")
FEATURE_NAMES = (
    "log_advanced_faces", "bounded_face_fraction", "log_edge_curves",
    "observed_edge_fraction", "shared_edge_fraction", "boundary_edge_fraction",
    "nonmanifold_edge_fraction", "log_face_components", "largest_component_fraction",
    "log_median_face_degree", "log_q90_face_degree",
    *(f"shared_surface_pair_{name}_fraction" for name in PAIR_TYPES),
    *(f"shared_curve_{name}_fraction" for name in CURVE_TYPES),
)
_QUOTED = re.compile(r"'(?:[^']|'')*'", re.DOTALL)
_REFERENCE = re.compile(r"#(\d+)")


def _refs(body: str) -> list[str]:
    return _REFERENCE.findall(_QUOTED.sub("", body))


def _surface(name: str) -> str:
    if "PLANE" in name:
        return "plane"
    if "CYLIND" in name:
        return "cylinder"
    if "CONICAL" in name or "CONE" in name:
        return "cone"
    if "TOROID" in name or "TORUS" in name:
        return "torus"
    return "other"


def _curve(name: str) -> str:
    if "CIRCLE" in name:
        return "circle"
    if name == "LINE":
        return "line"
    if "ELLIPSE" in name:
        return "ellipse"
    if "B_SPLINE" in name:
        return "bspline"
    return "other"


def _pair(left: str, right: str) -> str:
    pair = frozenset((left, right))
    if pair == {"plane"}:
        return "plane_plane"
    if pair == {"plane", "cylinder"}:
        return "plane_cylinder"
    if pair == {"cylinder"}:
        return "cylinder_cylinder"
    if pair == {"cylinder", "cone"}:
        return "cylinder_cone"
    if pair == {"cylinder", "torus"}:
        return "cylinder_torus"
    return "other_same" if left == right else "other_mixed"


def describe_topology(path: Path) -> tuple[np.ndarray, bool]:
    records = STEP_MATCHER.findall(path.read_text(encoding="latin-1", errors="ignore"))
    if not records:
        raise ValueError(f"No parseable STEP DATA entities: {path}")
    entities = {identifier: (name, _refs(body)) for identifier, name, body in records}
    faces = [identifier for identifier, (name, _) in entities.items() if name == "ADVANCED_FACE"]
    total_edge_curves = sum(name == "EDGE_CURVE" for name, _ in entities.values())
    edge_to_faces: dict[str, set[str]] = defaultdict(set)
    face_surface: dict[str, str] = {}
    face_edges: dict[str, set[str]] = {}
    for face in faces:
        _, references = entities[face]
        bounds: list[str] = []
        surface = "other"
        for reference in references:
            name = entities.get(reference, ("", []))[0]
            if name in {"FACE_BOUND", "FACE_OUTER_BOUND"}:
                bounds.append(reference)
            elif ("SURFACE" in name or name == "PLANE"):
                surface = _surface(name)
        face_surface[face] = surface
        edges: set[str] = set()
        for bound in bounds:
            for loop_id in entities[bound][1]:
                if entities.get(loop_id, ("", []))[0] != "EDGE_LOOP":
                    continue
                for oriented_id in entities[loop_id][1]:
                    oriented = entities.get(oriented_id)
                    if oriented is None or oriented[0] != "ORIENTED_EDGE":
                        continue
                    for edge_id in oriented[1]:
                        if entities.get(edge_id, ("", []))[0] == "EDGE_CURVE":
                            edges.add(edge_id)
        face_edges[face] = edges
        for edge in edges:
            edge_to_faces[edge].add(face)
    shared = [edge for edge, adjacent in edge_to_faces.items() if len(adjacent) == 2]
    boundary = sum(len(adjacent) == 1 for adjacent in edge_to_faces.values())
    nonmanifold = sum(len(adjacent) > 2 for adjacent in edge_to_faces.values())
    pair_counts: Counter[str] = Counter()
    curve_counts: Counter[str] = Counter()
    parent = {face: face for face in faces}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for edge in shared:
        first, second = sorted(edge_to_faces[edge])
        parent[find(first)] = find(second)
        pair_counts[_pair(face_surface[first], face_surface[second])] += 1
        # EDGE_CURVE references two vertices and its underlying curve.
        curve_name = ""
        for reference in entities[edge][1]:
            target = entities.get(reference)
            if target and target[0] not in {"VERTEX_POINT", "VERTEX_LOOP"}:
                curve_name = target[0]
        curve_counts[_curve(curve_name)] += 1
    component_sizes = Counter(find(face) for face in faces)
    degrees = np.asarray([len(face_edges[face]) for face in faces], dtype=np.float32)
    observed = len(edge_to_faces)
    denominator = max(observed, 1)
    shared_denominator = max(len(shared), 1)
    vector = np.asarray([
        math.log1p(len(faces)),
        sum(bool(face_edges[face]) for face in faces) / max(len(faces), 1),
        math.log1p(total_edge_curves),
        observed / max(total_edge_curves, 1),
        len(shared) / denominator,
        boundary / denominator,
        nonmanifold / denominator,
        math.log1p(len(component_sizes)),
        max(component_sizes.values(), default=0) / max(len(faces), 1),
        math.log1p(float(np.median(degrees))) if len(degrees) else 0.0,
        math.log1p(float(np.quantile(degrees, 0.9))) if len(degrees) else 0.0,
        *(pair_counts[name] / shared_denominator for name in PAIR_TYPES),
        *(curve_counts[name] / shared_denominator for name in CURVE_TYPES),
    ], dtype=np.float32)
    if vector.shape != (len(FEATURE_NAMES),) or not np.isfinite(vector).all():
        raise ValueError(f"Invalid B-Rep topology descriptors: {path}")
    return vector, bool(faces and observed)


def topology_cache(config: dict) -> tuple[Path, dict]:
    manifest, manifest_path, manifest_hash = source_manifest(config)
    path = config["paths"]["topology_cache"] / "topology.npy"
    return path, {
        "format": FORMAT, "manifest_sha256": manifest_hash,
        "manifest_path": str(manifest_path), "parser_version": PARSER_VERSION,
        "feature_names": list(FEATURE_NAMES), "sample_count": len(manifest["samples"]),
        "uids": [sample["uid"] for sample in manifest["samples"]],
    }


def extract_topology(config: dict) -> Path:
    manifest, _, _ = source_manifest(config)
    path, expected = topology_cache(config)
    metadata_path = path.with_name("topology_metadata.json")
    if path.is_file() and metadata_path.is_file():
        saved = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (all(saved.get(key) == value for key, value in expected.items())
                and np.load(path, mmap_mode="r").shape ==
                (expected["sample_count"], len(FEATURE_NAMES))):
            print(f"Reusing full-file topology descriptors: {path}", flush=True)
            return path
    samples = manifest["samples"]
    source_root = Path(manifest["dataset_root"])
    raw = np.empty((len(samples), len(FEATURE_NAMES)), dtype=np.float32)
    by_hash: dict[str, tuple[np.ndarray, bool]] = {}
    parts_with_links = 0
    for index, sample in enumerate(samples):
        digest = sample["sha256"]
        if digest not in by_hash:
            by_hash[digest] = describe_topology(source_root / sample["source_relpath"])
        raw[index], has_links = by_hash[digest]
        parts_with_links += has_links
        if (index + 1) % 250 == 0 or index + 1 == len(samples):
            print(f"Full-file topology {index + 1:,}/{len(samples):,}", flush=True)
    train = np.asarray([index for index, sample in enumerate(samples)
                        if sample["split"] == "train"], dtype=np.int64)
    if not len(train):
        raise ValueError("Cannot standardize topology without training samples")
    mean = raw[train].mean(0, dtype=np.float64)
    std = raw[train].std(0, dtype=np.float64)
    scaled = np.clip((raw - mean) / np.maximum(std, 1e-3), -5, 5).astype(np.float32)
    if not np.isfinite(scaled).all():
        raise ValueError("Non-finite standardized topology descriptors")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("topology.partial.npy")
    np.save(temporary, scaled)
    temporary.replace(path)
    save_json(metadata_path, {
        **expected, "train_mean": mean.tolist(), "train_scale": np.maximum(std, 1e-3).tolist(),
        "train_nonconstant_features": int(np.count_nonzero(std > 1e-3)),
        "parts_with_face_edge_links": parts_with_links,
        "description": "Full-file face-edge-surface adjacency; train-only scaling and no labels as features.",
    })
    print(f"Saved {len(FEATURE_NAMES)} topology descriptors: {path}", flush=True)
    return path
