"""Bounded, two-view rendering of actual STEP topology for qualitative review."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _shape(path: Path):
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader

    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone or reader.TransferRoots() <= 0:
        raise ValueError(f"STEP roots could not be transferred: {path}")
    shape = reader.OneShape()
    if shape.IsNull():
        raise ValueError(f"STEP produced a null shape: {path}")
    return shape


def _bounds(shape) -> tuple[np.ndarray, np.ndarray]:
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, box)
    if box.IsVoid():
        raise ValueError("STEP shape has no finite bounding box")
    if hasattr(box, "GetXMin"):
        values = [box.GetXMin(), box.GetYMin(), box.GetZMin(),
                  box.GetXMax(), box.GetYMax(), box.GetZMax()]
    else:
        values = box.Get()
    bounds = np.asarray(values, dtype=np.float64)
    if bounds.shape != (6,) or not np.isfinite(bounds).all():
        raise ValueError("STEP shape has invalid bounds")
    lower, upper = bounds[:3], bounds[3:]
    if float(np.linalg.norm(upper - lower)) <= 1e-9:
        raise ValueError("STEP shape has degenerate bounds")
    return lower, upper


def _triangles(shape, diagonal: float, limit: int) -> tuple[np.ndarray, int]:
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    BRepMesh_IncrementalMesh(
        shape, max(diagonal * 0.004, 1e-4), False, 0.35, True,
    ).Perform()
    faces = []
    total = 0
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        if triangulation is not None and triangulation.NbTriangles() > 0:
            count = triangulation.NbTriangles()
            faces.append((triangulation, location.Transformation(),
                          face.Orientation() == TopAbs_REVERSED, total, count))
            total += count
        explorer.Next()
    if total == 0:
        raise ValueError("STEP shape yielded no surface triangles")
    chosen = (np.arange(total, dtype=np.int64) if total <= limit else
              np.linspace(0, total - 1, num=limit, dtype=np.int64))
    output = np.empty((len(chosen), 3, 3), dtype=np.float64)
    for triangulation, transform, reversed_face, offset, count in faces:
        first = int(np.searchsorted(chosen, offset, side="left"))
        stop = int(np.searchsorted(chosen, offset + count, side="left"))
        for position in range(first, stop):
            local = int(chosen[position] - offset + 1)
            node_ids = list(triangulation.Triangle(local).Get())
            if reversed_face:
                node_ids[1], node_ids[2] = node_ids[2], node_ids[1]
            for corner, node_id in enumerate(node_ids):
                point = triangulation.Node(node_id).Transformed(transform)
                output[position, corner] = [point.X(), point.Y(), point.Z()]
    if not np.isfinite(output).all():
        raise ValueError("STEP tessellation contains non-finite coordinates")
    return output, total


def _edges(shape, diagonal: float, limit: int) -> tuple[list[np.ndarray], int]:
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GCPnts import GCPnts_QuasiUniformDeflection
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    lines = []
    total = 0
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while explorer.More():
        total += 1
        if len(lines) < limit:
            try:
                curve = BRepAdaptor_Curve(TopoDS.Edge(explorer.Current()))
                sample = GCPnts_QuasiUniformDeflection(
                    curve, max(diagonal * 0.003, 1e-4),
                )
                if sample.IsDone() and sample.NbPoints() >= 2:
                    count = sample.NbPoints()
                    positions = (np.arange(1, count + 1) if count <= 200 else
                                 np.unique(np.linspace(1, count, 200, dtype=int)))
                    points = np.asarray([
                        [sample.Value(int(i)).X(), sample.Value(int(i)).Y(),
                         sample.Value(int(i)).Z()] for i in positions
                    ], dtype=np.float64)
                    if np.isfinite(points).all():
                        lines.append(points)
            except (RuntimeError, ValueError):
                pass
        explorer.Next()
    return lines, total


def render_step(path: Path, destination: Path, image_size: int,
                max_triangles: int, max_edges: int) -> dict:
    """Render one imported solid from two opposite isometric directions.

    Triangle/edge limits affect only the PNG, never model inputs or metrics.
    Both views share exact STEP bounds and scale; no voxel or proxy is used.
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from PIL import Image

    if not path.is_file():
        raise FileNotFoundError(path)
    shape = _shape(path)
    lower, upper = _bounds(shape)
    diagonal = float(np.linalg.norm(upper - lower))
    triangles, original_triangle_count = _triangles(shape, diagonal, max_triangles)
    edges, original_edge_count = _edges(shape, diagonal, max_edges)
    normals = np.cross(triangles[:, 1] - triangles[:, 0],
                       triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1e-12)
    light = np.asarray([0.35, -0.45, 0.82])
    shade = np.clip(0.52 + 0.42 * np.abs(normals @ light), 0, 1)
    colors = np.clip(np.asarray([0.23, 0.55, 0.91])[None, :] * shade[:, None]
                     + 0.12, 0, 1)
    center = (lower + upper) / 2
    radius = max(float(np.max(upper - lower)) * 0.58, 1e-6)
    destination.parent.mkdir(parents=True, exist_ok=True)
    views = []
    for index, (elevation, azimuth) in enumerate(((24, -48), (24, 132))):
        target = destination.with_name(f"{destination.stem}_view{index + 1}.png")
        figure = plt.figure(figsize=(image_size / 100, image_size / 100), dpi=100)
        try:
            axis = figure.add_subplot(111, projection="3d")
            axis.set_proj_type("ortho")
            axis.add_collection3d(Poly3DCollection(
                triangles, facecolors=colors, edgecolors="none",
            ))
            # STEP curves have different point counts. Plot each separately:
            # Line3DCollection attempts a rectangular conversion and fails
            # on ragged edge samples for real TMCAD parts.
            for edge in edges:
                axis.plot(edge[:, 0], edge[:, 1], edge[:, 2],
                          color="#18324d", linewidth=0.4)
            axis.set_xlim(center[0] - radius, center[0] + radius)
            axis.set_ylim(center[1] - radius, center[1] + radius)
            axis.set_zlim(center[2] - radius, center[2] + radius)
            axis.set_box_aspect((1, 1, 1))
            axis.view_init(elev=elevation, azim=azimuth)
            axis.set_axis_off()
            figure.patch.set_facecolor("white")
            figure.savefig(target, dpi=100, facecolor="white")
        finally:
            plt.close(figure)
        with Image.open(target) as check:
            check.verify()
        views.append(str(target))
    return {
        "status": "valid", "step_path": str(path), "views": views,
        "bounds": [float(x) for x in np.concatenate((lower, upper))],
        "triangles_in_step_tessellation": int(original_triangle_count),
        "triangles_rendered": int(len(triangles)),
        "topological_edges": int(original_edge_count),
        "edges_rendered": int(len(edges)),
        "view_angles": [[24, -48], [24, 132]],
    }
