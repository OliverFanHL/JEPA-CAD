"""Area-weighted local surface profile from actual STEP solids.

Unlike the frozen STEP parser's 256 selected instances or full-file radius
quantiles, this view retains where surface area lies along and around a part's
principal axis. The learned classifier remains a linear residual on a fixed
descriptor: no new 3D encoder or JEPA fine-tuning is introduced.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from common import save_json, source_manifest


FORMAT = "tmcad_occ_surface_profile_v2"
AXIAL_BINS = 8
RADIAL_BINS = 8
NORMAL_BINS = 8
FEATURE_NAMES = (
    "log_triangle_count", "log_area_over_bbox_diagonal_squared",
    "log_principal_extent_ratio_1", "log_principal_extent_ratio_2",
    *(f"surface_area_axial_{a}_radial_{r}" for a in range(AXIAL_BINS)
      for r in range(RADIAL_BINS)),
    *(f"surface_normal_axis_bin_{i}" for i in range(NORMAL_BINS)),
    *(f"radial_spread_axial_bin_{i}" for i in range(AXIAL_BINS)),
)


def _step_shape(path: Path):
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(path))
    if status != IFSelect_RetDone or reader.TransferRoots() <= 0:
        raise ValueError(f"OpenCASCADE could not transfer STEP roots: {path}")
    shape = reader.OneShape()
    if shape.IsNull():
        raise ValueError(f"Null OpenCASCADE shape: {path}")
    return shape


def _bounds(shape) -> np.ndarray:
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, box)
    if box.IsVoid():
        raise ValueError("Shape has no finite bounding box")
    if hasattr(box, "GetXMin"):
        values = [box.GetXMin(), box.GetYMin(), box.GetZMin(),
                  box.GetXMax(), box.GetYMax(), box.GetZMax()]
    else:
        values = box.Get()
    bounds = np.asarray(values, dtype=np.float64)
    if bounds.shape != (6,) or not np.isfinite(bounds).all():
        raise ValueError("Invalid OpenCASCADE shape bounds")
    return bounds


def _triangles(shape, deflection: float, max_triangles: int,
               digest: str) -> tuple[np.ndarray, int]:
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    BRepMesh_IncrementalMesh(shape, deflection, False, 0.35, True).Perform()
    # Count face triangles first, then materialize only a deterministic
    # uniform subset. Dense gears can otherwise exhaust memory before the
    # downstream triangle cap is applied.
    faces = []
    triangle_count = 0
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        if triangulation is not None:
            count = triangulation.NbTriangles()
            if count:
                faces.append((triangulation, location.Transformation(),
                              face.Orientation() == TopAbs_REVERSED, triangle_count, count))
                triangle_count += count
        explorer.Next()
    if triangle_count == 0:
        raise ValueError("STEP shape produced no surface triangles")
    if triangle_count > max_triangles:
        rng = np.random.default_rng(int(digest[:16], 16))
        selected = np.sort(rng.choice(triangle_count, size=max_triangles, replace=False))
    else:
        selected = np.arange(triangle_count)
    output: list[list[list[float]]] = []
    for triangulation, transform, reverse, offset, count in faces:
        first = int(np.searchsorted(selected, offset, side="left"))
        stop = int(np.searchsorted(selected, offset + count, side="left"))
        for global_index in selected[first:stop]:
            index = int(global_index - offset + 1)
            node_ids = list(triangulation.Triangle(index).Get())
            if reverse:
                node_ids[1], node_ids[2] = node_ids[2], node_ids[1]
            points = []
            for node_id in node_ids:
                point = triangulation.Node(node_id).Transformed(transform)
                points.append([point.X(), point.Y(), point.Z()])
            output.append(points)
    return np.asarray(output, dtype=np.float64), triangle_count


def describe_step_mesh(path: Path, digest: str, relative_deflection: float,
                       max_triangles: int) -> np.ndarray:
    shape = _step_shape(path)
    bounds = _bounds(shape)
    diagonal = float(np.linalg.norm(bounds[3:] - bounds[:3]))
    if not math.isfinite(diagonal) or diagonal <= 1e-9:
        raise ValueError("Degenerate STEP solid bounds")
    triangles, original_count = _triangles(
        shape, max(diagonal * relative_deflection, 1e-4), max_triangles, digest
    )
    first = triangles[:, 1] - triangles[:, 0]
    second = triangles[:, 2] - triangles[:, 0]
    cross = np.cross(first, second)
    double_area = np.linalg.norm(cross, axis=1)
    valid = np.isfinite(double_area) & (double_area > 1e-16)
    if valid.sum() < 8:
        raise ValueError("Too few nondegenerate surface triangles")
    triangles, cross, double_area = triangles[valid], cross[valid], double_area[valid]
    area = double_area / 2
    area *= original_count / len(triangles) if original_count > max_triangles else 1.0
    area_total = float(area.sum())
    if not math.isfinite(area_total) or area_total <= 0:
        raise ValueError("Invalid triangulated surface area")
    centroids = triangles.mean(axis=1)
    normals = cross / double_area[:, None]
    center = np.average(centroids, axis=0, weights=area)
    centered = centroids - center
    covariance = (centered * area[:, None]).T @ centered / area_total
    eigenvalues, directions = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1e-12)[::-1]
    # A disc's axis is its shortest PCA direction; a rod's is its longest.
    disc_contrast = eigenvalues[1] / eigenvalues[2]
    rod_contrast = eigenvalues[0] / eigenvalues[1]
    axis = directions[:, 0] if disc_contrast > rod_contrast else directions[:, 2]
    axial = centered @ axis
    radial = np.linalg.norm(centered - axial[:, None] * axis[None, :], axis=1)
    axial_scale = max(float(np.max(np.abs(axial))), 1e-12)
    radial_scale = max(float(np.max(radial)), 1e-12)
    axial_unit = np.clip(axial / axial_scale, -1.0, 1.0)
    radial_unit = np.clip(radial / radial_scale, 0.0, 1.0)
    profile = np.histogram2d(
        axial_unit, radial_unit, bins=(AXIAL_BINS, RADIAL_BINS),
        range=((-1, 1), (0, 1)), weights=area,
    )[0]
    # An unlabeled CAD part has no intrinsic positive axial direction.
    profile = (profile + profile[::-1]) / 2
    profile /= area_total
    normal_alignment = np.abs(normals @ axis)
    normal_hist = np.histogram(normal_alignment, bins=NORMAL_BINS,
                               range=(0, 1), weights=area)[0] / area_total
    axial_index = np.clip(((axial_unit + 1) * AXIAL_BINS / 2).astype(int), 0, AXIAL_BINS - 1)
    spread = np.zeros(AXIAL_BINS, dtype=np.float64)
    for bin_index in range(AXIAL_BINS):
        keep = axial_index == bin_index
        if keep.any():
            mean_radius = np.average(radial_unit[keep], weights=area[keep])
            spread[bin_index] = math.sqrt(float(np.average(
                (radial_unit[keep] - mean_radius) ** 2, weights=area[keep]
            )))
    spread = (spread + spread[::-1]) / 2
    # Near-isotropic shapes have no stable principal axis. Do not let an
    # arbitrary eigensolver orientation create a spurious local signature.
    if max(disc_contrast, rod_contrast) < 1.10:
        profile.fill(0)
        normal_hist.fill(0)
        spread.fill(0)
    vector = np.asarray([
        math.log1p(original_count),
        math.log1p(area_total / (diagonal * diagonal)),
        math.log1p(eigenvalues[0] / eigenvalues[1]),
        math.log1p(eigenvalues[1] / eigenvalues[2]),
        *profile.flatten(), *normal_hist, *spread,
    ], dtype=np.float32)
    if vector.shape != (len(FEATURE_NAMES),) or not np.isfinite(vector).all():
        raise ValueError("Non-finite surface profile")
    return vector


def mesh_cache(config: dict) -> tuple[Path, dict]:
    manifest, manifest_path, manifest_hash = source_manifest(config)
    settings = config["mesh"]
    path = config["paths"]["mesh_cache"] / "mesh_profile.npy"
    return path, {
        "format": FORMAT, "manifest_sha256": manifest_hash,
        "manifest_path": str(manifest_path),
        "feature_names": list(FEATURE_NAMES), "sample_count": len(manifest["samples"]),
        "uids": [sample["uid"] for sample in manifest["samples"]],
        "relative_deflection": float(settings["relative_deflection"]),
        "max_triangles_per_part": int(settings["max_triangles_per_part"]),
    }


def extract_mesh_profiles(config: dict) -> Path:
    try:
        import OCP  # noqa: F401
    except ImportError as error:
        raise ImportError("Install cadquery-ocp to process actual STEP surfaces") from error
    manifest, _, _ = source_manifest(config)
    path, expected = mesh_cache(config)
    metadata_path = path.with_name("mesh_profile_metadata.json")
    if path.is_file() and metadata_path.is_file():
        saved = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (all(saved.get(key) == value for key, value in expected.items())
                and np.load(path, mmap_mode="r").shape ==
                (expected["sample_count"], len(FEATURE_NAMES))):
            print(f"Reusing cached STEP surface profiles: {path}", flush=True)
            return path
    settings = config["mesh"]
    relative_deflection = float(settings["relative_deflection"])
    max_triangles = int(settings["max_triangles_per_part"])
    if not (0 < relative_deflection <= 0.05 and max_triangles >= 1000):
        raise ValueError("Invalid mesh deflection or triangle cap")
    samples = manifest["samples"]
    root = Path(manifest["dataset_root"])
    raw = np.zeros((len(samples), len(FEATURE_NAMES)), dtype=np.float32)
    valid = np.zeros(len(samples), dtype=bool)
    by_hash: dict[str, tuple[np.ndarray | None, str | None]] = {}
    errors: list[dict] = []
    failed_by_class: Counter[str] = Counter()
    for index, sample in enumerate(samples):
        digest = sample["sha256"]
        if digest not in by_hash:
            try:
                feature = describe_step_mesh(root / sample["source_relpath"], digest,
                                             relative_deflection, max_triangles)
                by_hash[digest] = (feature, None)
            except (RuntimeError, ValueError, OSError) as error:
                by_hash[digest] = (None, f"{type(error).__name__}: {error}")
        feature, message = by_hash[digest]
        if feature is not None:
            raw[index], valid[index] = feature, True
        else:
            failed_by_class[sample["label"]] += 1
            if len(errors) < 100:
                errors.append({"uid": sample["uid"], "reason": message})
        if (index + 1) % int(settings["progress_every"]) == 0 or index + 1 == len(samples):
            print(f"STEP surface profile {index + 1:,}/{len(samples):,} "
                  f"valid={int(valid[:index + 1].sum()):,} failed={index + 1 - int(valid[:index + 1].sum()):,}",
                  flush=True)
    failure_fraction = 1 - float(valid.mean())
    if failure_fraction > float(settings["max_failure_fraction"]):
        raise RuntimeError(f"OpenCASCADE mesh failure fraction {failure_fraction:.3%} "
                           "exceeds configured limit; inspect STEP files/settings before training")
    train = np.asarray([index for index, sample in enumerate(samples)
                        if sample["split"] == "train" and valid[index]], dtype=np.int64)
    if not len(train):
        raise ValueError("No valid training meshes")
    mean = raw[train].mean(0, dtype=np.float64)
    std = raw[train].std(0, dtype=np.float64)
    scaled = np.clip((raw - mean) / np.maximum(std, 1e-3), -5, 5).astype(np.float32)
    # Invalid meshes contribute exactly zero to the bias-free downstream head;
    # both arms keep the original split and every sample.
    scaled[~valid] = 0
    if not np.isfinite(scaled).all():
        raise ValueError("Non-finite standardized surface profiles")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("mesh_profile.partial.npy")
    np.save(temporary, scaled)
    temporary.replace(path)
    save_json(metadata_path, {
        **expected, "valid": int(valid.sum()), "failed": int((~valid).sum()),
        "failure_fraction": failure_fraction, "failed_by_class": dict(failed_by_class),
        "sample_failures": errors, "train_nonconstant_features": int(np.count_nonzero(std > 1e-3)),
        "train_mean": mean.tolist(), "train_scale": np.maximum(std, 1e-3).tolist(),
        "description": "Actual surface-triangle radial/axial area profile; train-only scaling; no labels as inputs.",
    })
    print(f"Saved {len(FEATURE_NAMES)}-D STEP surface profiles: {path}", flush=True)
    return path
