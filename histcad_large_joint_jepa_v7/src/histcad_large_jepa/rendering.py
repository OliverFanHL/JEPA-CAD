from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np


def _step_shape(path: Path):
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(path))
    if status != IFSelect_RetDone:
        raise ValueError(f"OpenCASCADE could not read STEP file: status={status}")
    transferred = reader.TransferRoots()
    if transferred <= 0:
        raise ValueError("STEP file contained no transferable roots")
    shape = reader.OneShape()
    if shape.IsNull():
        raise ValueError("STEP transfer produced a null shape")
    return shape


def _shape_bounds(shape) -> tuple[float, float, float, float, float, float]:
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    bounds = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, bounds)
    if bounds.IsVoid():
        raise ValueError("STEP shape has no finite bounding box")
    # OCP releases expose either coordinate accessors or a six-value Get().
    if hasattr(bounds, "GetXMin"):
        values = (
            float(bounds.GetXMin()),
            float(bounds.GetYMin()),
            float(bounds.GetZMin()),
            float(bounds.GetXMax()),
            float(bounds.GetYMax()),
            float(bounds.GetZMax()),
        )
    else:
        values = tuple(float(value) for value in bounds.Get())
    if not all(math.isfinite(value) for value in values):
        raise ValueError("STEP shape bounding box is not finite")
    return values


def _triangles(shape, diagonal: float) -> np.ndarray:
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    BRepMesh_IncrementalMesh(shape, max(diagonal * 0.003, 1e-4), False, 0.30, True).Perform()
    output: list[list[list[float]]] = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        if triangulation is not None:
            transform = location.Transformation()
            reverse = face.Orientation() == TopAbs_REVERSED
            for triangle_index in range(1, triangulation.NbTriangles() + 1):
                node_indices = list(triangulation.Triangle(triangle_index).Get())
                if reverse:
                    node_indices[1], node_indices[2] = node_indices[2], node_indices[1]
                points = []
                for node_index in node_indices:
                    point = triangulation.Node(node_index).Transformed(transform)
                    points.append([point.X(), point.Y(), point.Z()])
                output.append(points)
        explorer.Next()
    if not output:
        raise ValueError("STEP shape produced no renderable triangles")
    return np.asarray(output, dtype=np.float64)


def _edges(shape, diagonal: float) -> list[np.ndarray]:
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GCPnts import GCPnts_QuasiUniformDeflection
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    output = []
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while explorer.More():
        try:
            curve = BRepAdaptor_Curve(TopoDS.Edge(explorer.Current()))
            sampler = GCPnts_QuasiUniformDeflection(curve, max(diagonal * 0.002, 1e-4))
            if sampler.IsDone() and sampler.NbPoints() >= 2:
                points = []
                for point_index in range(1, sampler.NbPoints() + 1):
                    point = sampler.Value(point_index)
                    points.append([point.X(), point.Y(), point.Z()])
                output.append(np.asarray(points, dtype=np.float64))
        except (RuntimeError, ValueError):
            pass
        explorer.Next()
    return output


def render_step(path: Path, output_path: Path, image_size: int = 320) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    shape = _step_shape(path)
    bounds = _shape_bounds(shape)
    lower = np.asarray(bounds[:3])
    upper = np.asarray(bounds[3:])
    diagonal = float(np.linalg.norm(upper - lower))
    if diagonal <= 1e-9:
        raise ValueError("STEP shape has a degenerate bounding box")
    triangles = _triangles(shape, diagonal)
    edges = _edges(shape, diagonal)

    vectors_a = triangles[:, 1] - triangles[:, 0]
    vectors_b = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(vectors_a, vectors_b)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1e-12)
    light = np.asarray([0.35, -0.45, 0.82])
    shade = np.clip(0.52 + 0.42 * np.abs(normals @ light), 0.0, 1.0)
    base = np.asarray([0.23, 0.55, 0.91])
    colors = np.clip(base[None, :] * shade[:, None] + 0.12, 0.0, 1.0)

    figure = plt.figure(figsize=(image_size / 100, image_size / 100), dpi=100)
    axis = figure.add_subplot(111, projection="3d")
    axis.set_proj_type("ortho")
    axis.add_collection3d(Poly3DCollection(triangles, facecolors=colors, edgecolors="none"))
    if edges:
        axis.add_collection3d(Line3DCollection(edges, colors="#18324d", linewidths=0.45))
    center = (lower + upper) / 2
    radius = max(upper - lower) / 2 * 1.08
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1), zoom=1.35)
    axis.view_init(elev=24, azim=-48)
    axis.set_axis_off()
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=100, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(figure)
    return {
        "step_path": str(path),
        "render_path": str(output_path),
        "triangles": int(len(triangles)),
        "topological_edges_rendered": len(edges),
        "bounds": list(bounds),
        "status": "valid",
    }
