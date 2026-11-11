"""Check exact STEP face-to-rendered-face correspondence via boundary vertices."""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

from OCP.BRep import BRep_Tool
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_FACE, TopAbs_VERTEX
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "histcad_large_joint_jepa_v7"))
from src.histcad_large_jepa.step_graph import REFERENCE_RE, NUMBER_RE  # noqa: E402
sys.path.insert(0, str(ROOT / "histcad_large_joint_jepa_v7" / "downstream_classifier_tmcad"))
from step_statements import STEP_MATCHER  # noqa: E402

FOLLOW = {"FACE_BOUND", "FACE_OUTER_BOUND", "EDGE_LOOP", "ORIENTED_EDGE",
          "EDGE_CURVE", "VERTEX_POINT", "CARTESIAN_POINT"}


def step_faces(path: Path):
    records = STEP_MATCHER.findall(path.read_text(encoding="latin-1", errors="ignore"))
    entities = {key: (kind, body) for key, kind, body in records}

    def vertices(key, seen):
        if key in seen or key not in entities:
            return set()
        seen.add(key)
        kind, body = entities[key]
        if kind == "CARTESIAN_POINT":
            values = [float(x) for x in NUMBER_RE.findall(body)]
            return {tuple(round(x, 4) for x in values[-3:])} if len(values) >= 3 else set()
        found = set()
        for ref in REFERENCE_RE.findall(body):
            if ref in entities and entities[ref][0] in FOLLOW:
                found |= vertices(ref, seen)
        return found

    return {key: vertices(key, set()) for key, (kind, _) in entities.items()
            if kind == "ADVANCED_FACE"}


def rendered_faces(path: Path):
    reader = STEPControl_Reader()
    reader.ReadFile(str(path))
    reader.TransferRoots()
    shape = reader.OneShape()
    output = []
    faces = TopExp_Explorer(shape, TopAbs_FACE)
    while faces.More():
        points = set()
        vertices = TopExp_Explorer(TopoDS.Face(faces.Current()), TopAbs_VERTEX)
        while vertices.More():
            p = BRep_Tool.Pnt_s(TopoDS.Vertex(vertices.Current()))
            points.add(tuple(round(x, 4) for x in (p.X(), p.Y(), p.Z())))
            vertices.Next()
        output.append(points)
        faces.Next()
    return output


if __name__ == "__main__":
    for uid in sys.argv[1:]:
        path = ROOT / "TMCAD" / uid
        source, target = step_faces(path), rendered_faces(path)
        signatures = Counter(frozenset(v) for v in source.values())
        matched = [i for i, vertices in enumerate(target)
                   if len(vertices) > 1 and signatures[frozenset(vertices)] == 1]
        print(uid, "STEP faces", len(source), "rendered faces", len(target),
              "unique exact matches", len(matched), "empty", sum(not x for x in source.values()))
