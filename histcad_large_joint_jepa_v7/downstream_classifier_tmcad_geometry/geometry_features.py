"""Uncapped, unit-invariant STEP geometry summaries; labels are never features."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
from common import save_json, source_manifest
from step_statements import PARSER_VERSION, STEP_MATCHER


FORMAT = "tmcad_full_step_geometry_v1"
FAMILIES = ("circle", "cylinder", "torus_major", "torus_minor", "sphere")
FEATURE_NAMES = (
    "log_cartesian_point_count", "bbox_long_over_mid", "bbox_long_over_short",
    "bbox_mid_over_short", "bbox_span_long_over_diagonal",
    "bbox_span_mid_over_diagonal", "bbox_span_short_over_diagonal",
    *(f"{family}_{stat}" for family in FAMILIES for stat in
      ("log_count", "radius_q10_over_diagonal", "radius_q50_over_diagonal",
       "radius_q90_over_diagonal", "log_radius_q90_over_q10", "unique_radius_fraction")),
)
_STRING = re.compile(r"'(?:[^']|'')*'", re.DOTALL)
_REFERENCE = re.compile(r"#\d+")
_NUMBER = re.compile(r"(?<![A-Za-z0-9_#])[-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?")


def _numbers(body: str) -> list[float]:
    # Strip references *before* reading numbers. The v7 generic numeric
    # summary can otherwise match a suffix inside a multi-digit #reference.
    clean = _REFERENCE.sub("", _STRING.sub("", body))
    return [float(match.group()) for match in _NUMBER.finditer(clean)
            if math.isfinite(float(match.group()))]


def _radius_summary(values: list[float], scale: float) -> list[float]:
    radii = np.asarray([value for value in values if 0 < value < 1e30], dtype=np.float64)
    if not len(radii):
        return [0.0] * 6
    q10, q50, q90 = np.quantile(radii, (0.1, 0.5, 0.9))
    rounded = np.round(radii / max(q50, 1e-12), 3)
    return [
        math.log1p(len(radii)),
        float(np.log1p(q10 / scale)),
        float(np.log1p(q50 / scale)),
        float(np.log1p(q90 / scale)),
        float(np.log1p(q90 / max(q10, 1e-12))),
        float(len(np.unique(rounded)) / len(radii)),
    ]


def describe_step(path: Path) -> np.ndarray:
    records = STEP_MATCHER.findall(path.read_text(encoding="latin-1", errors="ignore"))
    if not records:
        raise ValueError(f"No parseable STEP DATA entities: {path}")
    points: list[list[float]] = []
    radii: dict[str, list[float]] = {family: [] for family in FAMILIES}
    for _, entity, body in records:
        if entity not in {"CARTESIAN_POINT", "CIRCLE", "CYLINDRICAL_SURFACE",
                          "TOROIDAL_SURFACE", "SPHERICAL_SURFACE"}:
            continue
        values = _numbers(body)
        if entity == "CARTESIAN_POINT" and len(values) >= 3:
            points.append(values[-3:])
        elif entity == "TOROIDAL_SURFACE" and len(values) >= 2:
            radii["torus_major"].append(abs(values[-2]))
            radii["torus_minor"].append(abs(values[-1]))
        elif values:
            family = {"CIRCLE": "circle", "CYLINDRICAL_SURFACE": "cylinder",
                      "SPHERICAL_SURFACE": "sphere"}.get(entity)
            if family is not None:
                radii[family].append(abs(values[-1]))
    if points:
        coordinates = np.asarray(points, dtype=np.float64)
        spans = np.sort(np.ptp(coordinates, axis=0))[::-1]
    else:
        spans = np.zeros(3, dtype=np.float64)
    diagonal = float(np.linalg.norm(spans))
    # A radius-only STEP file still receives dimensionless radius statistics.
    fallback = max((value for group in radii.values() for value in group), default=1.0)
    scale = diagonal if diagonal > 1e-12 else max(2 * fallback, 1e-12)
    positive = spans[spans > 1e-12]
    short = max(float(spans[2]), 1e-12)
    mid = max(float(spans[1]), 1e-12)
    shape = [
        math.log1p(len(points)),
        math.log1p(float(spans[0]) / mid) if len(positive) else 0.0,
        math.log1p(float(spans[0]) / short) if len(positive) else 0.0,
        math.log1p(float(spans[1]) / short) if len(positive) else 0.0,
        *(float(span / scale) for span in spans),
    ]
    vector = np.asarray([*shape, *(
        item for family in FAMILIES for item in _radius_summary(radii[family], scale)
    )], dtype=np.float32)
    if vector.shape != (len(FEATURE_NAMES),) or not np.isfinite(vector).all():
        raise ValueError(f"Invalid full-file geometry descriptors: {path}")
    return vector


def geometry_cache(config: dict) -> tuple[Path, dict]:
    manifest, manifest_path, manifest_hash = source_manifest(config)
    root = config["paths"]["geometry_cache"]
    return root / "geometry.npy", {
        "format": FORMAT, "manifest_sha256": manifest_hash,
        "manifest_path": str(manifest_path), "parser_version": PARSER_VERSION,
        "feature_names": list(FEATURE_NAMES), "sample_count": len(manifest["samples"]),
        "uids": [sample["uid"] for sample in manifest["samples"]],
    }


def extract_geometry(config: dict) -> Path:
    manifest, _, _ = source_manifest(config)
    target, expected = geometry_cache(config)
    metadata_path = target.with_name("geometry_metadata.json")
    if target.is_file() and metadata_path.is_file():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (all(existing.get(key) == value for key, value in expected.items())
                and np.load(target, mmap_mode="r").shape ==
                (expected["sample_count"], len(FEATURE_NAMES))):
            print(f"Reusing full-file geometry descriptors: {target}", flush=True)
            return target
    samples = manifest["samples"]
    source_root = Path(manifest["dataset_root"])
    raw = np.empty((len(samples), len(FEATURE_NAMES)), dtype=np.float32)
    by_hash: dict[str, np.ndarray] = {}
    for index, sample in enumerate(samples):
        digest = sample["sha256"]
        if digest not in by_hash:
            by_hash[digest] = describe_step(source_root / sample["source_relpath"])
        raw[index] = by_hash[digest]
        if (index + 1) % 250 == 0 or index + 1 == len(samples):
            print(f"Full-file geometry {index + 1:,}/{len(samples):,}", flush=True)
    train = np.asarray([index for index, sample in enumerate(samples)
                        if sample["split"] == "train"], dtype=np.int64)
    if not len(train):
        raise ValueError("Cannot standardize geometry without training samples")
    mean = raw[train].mean(0, dtype=np.float64)
    std = raw[train].std(0, dtype=np.float64)
    scaled = np.clip((raw - mean) / np.maximum(std, 1e-3), -5, 5).astype(np.float32)
    if not np.isfinite(scaled).all():
        raise ValueError("Non-finite standardized geometry descriptors")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name("geometry.partial.npy")
    np.save(temporary, scaled)
    temporary.replace(target)
    save_json(metadata_path, {
        **expected, "train_mean": mean.tolist(), "train_scale": np.maximum(std, 1e-3).tolist(),
        "train_nonconstant_features": int(np.count_nonzero(std > 1e-3)),
        "coverage": {
            "parts_with_3d_points": int(np.count_nonzero(raw[:, 0] > 0)),
            **{f"parts_with_{family}": int(np.count_nonzero(raw[:, 7 + 6 * offset] > 0))
               for offset, family in enumerate(FAMILIES)},
        },
        "description": "Full-file bbox aspect ratios and typed radius distributions; train-only scaling, no labels or class names as features.",
    })
    print(f"Saved {len(FEATURE_NAMES)} geometric descriptors: {target}", flush=True)
    return target
