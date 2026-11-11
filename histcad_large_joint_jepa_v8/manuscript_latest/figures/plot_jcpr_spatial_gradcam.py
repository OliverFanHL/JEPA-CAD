"""Publication PDF with verified JCPR graph-to-STEP-face attribution."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / ".plot_deps"))
import scienceplots  # noqa: F401
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED, TopAbs_VERTEX
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS

HERE = Path(__file__).resolve().parent


def mesh(path: Path, step_faces: dict):
    reader = STEPControl_Reader()
    reader.ReadFile(str(path))
    reader.TransferRoots()
    shape = reader.OneShape()
    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, box)
    lower = np.array([box.GetXMin(), box.GetYMin(), box.GetZMin()])
    upper = np.array([box.GetXMax(), box.GetYMax(), box.GetZMax()])
    BRepMesh_IncrementalMesh(shape, max(np.linalg.norm(upper - lower) * .004, 1e-4),
                             False, .35, True).Perform()
    signatures = {}
    counts = Counter(frozenset(tuple(v) for v in vertices)
                     for vertices in step_faces.values())
    for face_id, vertices in step_faces.items():
        key = frozenset(tuple(v) for v in vertices)
        if counts[key] == 1 and len(key) > 1:
            signatures[key] = face_id
    triangles, identifiers = [], []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    matched = 0
    while explorer.More():
        face = TopoDS.Face(explorer.Current())
        points = set()
        vex = TopExp_Explorer(face, TopAbs_VERTEX)
        while vex.More():
            p = BRep_Tool.Pnt_s(TopoDS.Vertex(vex.Current()))
            points.add(tuple(round(x, 4) for x in (p.X(), p.Y(), p.Z())))
            vex.Next()
        face_id = signatures.get(frozenset(points))
        matched += face_id is not None
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None and tri.NbTriangles():
            count = tri.NbTriangles()
            sample = np.unique(np.linspace(1, count, min(count, 350), dtype=int))
            transform = loc.Transformation()
            for i in sample:
                corners = list(tri.Triangle(int(i)).Get())
                if face.Orientation() == TopAbs_REVERSED:
                    corners[1], corners[2] = corners[2], corners[1]
                vertices = []
                for corner in corners:
                    p = tri.Node(corner).Transformed(transform)
                    vertices.append([p.X(), p.Y(), p.Z()])
                triangles.append(vertices)
                identifiers.append(face_id)
        explorer.Next()
    return np.asarray(triangles), identifiers, lower, upper, matched


def main():
    data = json.loads((HERE / "jcpr_spatial_gradcam.json").read_text(encoding="utf-8"))
    with plt.style.context(["science", "no-latex"]):
        plt.rcParams.update({"font.family": "Arial", "font.size": 9,
                             "axes.titlesize": 9, "pdf.fonttype": 42})
        figure = plt.figure(figsize=(7.15, 7.00))
        grid = figure.add_gridspec(3, 3, left=.045, right=.91, top=.96,
                                   bottom=.06, wspace=.02, hspace=.01)
        cmap = LinearSegmentedColormap.from_list(
            "jcpr", ["#d9dfdd", "#94bac0", "#457e89", "#194c5b"])
        rows = []
        for i, case in enumerate(data["cases"]):
            triangles, face_ids, lower, upper, matched = mesh(
                Path(case["step_path"]), case["face_vertices"])
            if matched / len(case["face_vertices"]) < .7:
                raise ValueError(f"Insufficient verified face correspondence: {case['uid']}")
            rows.append({"uid": case["uid"], "matched_faces": matched,
                         "total_faces": len(case["face_vertices"])})
            center = (lower + upper) / 2
            radius = max(float(np.max(upper - lower)) * .58, 1e-6)
            for j in range(3):
                axis = figure.add_subplot(grid[i, j], projection="3d")
                if j == 0:
                    values = np.zeros(len(face_ids))
                else:
                    name = case["true_class"] if j == 1 else case["control_prediction"]
                    values = np.array([case["gradcam"][name].get(k, 0.0)
                                       if k is not None else 0.0 for k in face_ids])
                normals = np.cross(triangles[:, 1] - triangles[:, 0],
                                   triangles[:, 2] - triangles[:, 0])
                normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1e-12)
                shade = .76 + .22 * np.abs(normals @ np.array([.35, -.45, .82]))
                colors = cmap(values)
                colors[:, :3] *= shade[:, None]
                if j == 0:
                    colors[:, :3] = np.array([.69, .75, .75])[None] * shade[:, None]
                axis.add_collection3d(Poly3DCollection(
                    triangles, facecolors=colors, edgecolors="none", rasterized=True))
                axis.set(xlim=(center[0]-radius, center[0]+radius),
                         ylim=(center[1]-radius, center[1]+radius),
                         zlim=(center[2]-radius, center[2]+radius))
                axis.set_box_aspect((1, 1, 1), zoom=1.10)
                axis.set_proj_type("ortho")
                axis.view_init(elev=24, azim=-48)
                axis.set_axis_off()
                if j == 0:
                    axis.text2D(-.02, .9, "abc"[i], transform=axis.transAxes,
                                fontsize=10, fontweight="bold", color="#173c49")
        cax = figure.add_axes([.925, .17, .017, .66])
        cb = figure.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cmap), cax=cax)
        cb.set_ticks([0, .5, 1])
        cb.set_ticklabels(["0", "0.5", "1"])
        cb.set_label("Relative response", fontsize=8, labelpad=5)
        destination = HERE / "jcpr_spatial_gradcam.pdf"
        figure.savefig(destination, dpi=450, bbox_inches="tight", pad_inches=.03)
        figure.savefig(HERE / "qa_previews/jcpr_spatial_gradcam.png", dpi=220,
                       bbox_inches="tight", pad_inches=.03)
        plt.close(figure)
        (HERE / "jcpr_spatial_gradcam_coverage.json").write_text(
            json.dumps(rows, indent=2), encoding="utf-8")
        print(rows)
        print(destination)


if __name__ == "__main__":
    main()
