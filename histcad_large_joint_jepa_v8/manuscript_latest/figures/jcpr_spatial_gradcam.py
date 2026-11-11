"""Project JCPR graph Grad-CAM onto matched faces of the original STEP solid.

The target is the saved JCPR STEP control's class logit. Face identification is
verified by exact (0.0001 model-unit) STEP boundary-vertex correspondence to
the OpenCascade tessellation. Ambiguous faces remain uncolored.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V7 = ROOT / "histcad_large_joint_jepa_v7"
V8 = ROOT / "histcad_large_joint_jepa_v8"
sys.path.insert(0, str(V7))
sys.path.insert(0, str(V8 / "downstream_classifier_tmcad_final"))
from src.histcad_large_jepa.model import StepEncoder  # noqa: E402
from src.histcad_large_jepa.step_graph import (  # noqa: E402
    INSTANCE_MARKERS, collate_step_graphs,
)
from data import TMCADSource, load_config  # noqa: E402
from probe_step_face_mapping import step_faces  # noqa: E402
from step_statements import STEP_MATCHER  # noqa: E402

CASES = ("bearing/55.stp", "pulley/423.stp", "coupling/257.stp")


def selected_nodes(path: Path, graph: dict) -> dict[str, int]:
    records = STEP_MATCHER.findall(path.read_text(encoding="latin-1", errors="ignore"))
    types = sorted({kind for _, kind, _ in records})
    candidates = [r for r in records if any(x in r[1] for x in INSTANCE_MARKERS)]
    candidates.sort(key=lambda r: next((i for i, x in enumerate(INSTANCE_MARKERS)
                                        if x in r[1]), len(INSTANCE_MARKERS)))
    selected = candidates[:int((graph["node_role"] == 2).sum())]
    assert len(types) + 1 + len(selected) == len(graph["node_type"])
    return {key: 1 + len(types) + i for i, (key, kind, _) in enumerate(selected)
            if kind == "ADVANCED_FACE"}


def load_encoder():
    checkpoint = torch.load(V7 / "outputs/full_dataset_4096/best.pt", map_location="cpu",
                            mmap=True, weights_only=False)
    settings = checkpoint["model_config"]
    encoder = StepEncoder(
        len(checkpoint["step_entity_types"]), settings["step_node_dim"],
        settings["step_output_dim"], settings["step_depth"],
        settings["attention_heads"], settings["step_pooling_slots"],
        settings["dropout"],
    )
    encoder.load_state_dict({k.removeprefix("step_encoder."): v
                             for k, v in checkpoint["model"].items()
                             if k.startswith("step_encoder.")}, strict=True)
    encoder.eval().requires_grad_(False)
    return encoder


def main():
    torch.set_num_threads(4)
    encoder = load_encoder()
    source = TMCADSource(load_config(V8 / "downstream_classifier_tmcad_final/config.yaml"))
    classifier = source.baseline(torch.device("cpu"))
    predictions = {r["uid"]: r for r in csv.DictReader((
        V8 / "downstream_classifier_tmcad_final/outputs/default/test_predictions.csv"
    ).open(newline="", encoding="utf-8"))}
    output = []
    for uid in CASES:
        idx = next(i for i, row in enumerate(source.manifest["samples"]) if row["uid"] == uid)
        sample = source.manifest["samples"][idx]
        path = Path(predictions[uid]["step_path"])
        payload = torch.load(V7 / "downstream_classifier_tmcad/data/processed" /
                             sample["graph_relpath"], weights_only=True)
        graph = payload["graph"]
        node_by_face = selected_nodes(path, graph)
        batch_graph = collate_step_graphs([graph])
        holder = {}

        def hook(_module, _inputs, values):
            tracked = values.detach().requires_grad_(True)
            holder["values"] = tracked
            return tracked

        hook_handle = encoder.node_norm.register_forward_hook(hook)
        embedding = F.normalize(encoder(batch_graph).float(), dim=-1)
        hook_handle.remove()
        cached = torch.tensor(np.array(source.pooled[idx], dtype=np.float32))[None]
        embedding_error = float((embedding - cached).abs().max())
        if embedding_error > 0.002:
            raise AssertionError(f"JCPR cached embedding mismatch for {uid}: {embedding_error}")
        inputs = {
            "pooled": embedding,
            "alternative": torch.tensor(np.array(source.alternative[idx], dtype=np.float32))[None],
            "geometry": torch.tensor(np.array(source.geometry[idx], dtype=np.float32))[None],
            "mesh": torch.tensor(np.array(source.mesh[idx], dtype=np.float32))[None],
        }
        logits, _ = classifier(inputs)
        true_name = predictions[uid]["true_class"]
        competitor_name = predictions[uid]["baseline_prediction"]
        assert true_name != competitor_name
        assert source.classes[int(logits.argmax())] == competitor_name
        saliencies = {}
        for name in (true_name, competitor_name):
            target = source.classes.index(name)
            gradients = torch.autograd.grad(logits[0, target], holder["values"],
                                            retain_graph=True)[0]
            # Node-wise gradient times activation (HiResCAM) retains the
            # graph-specific gradient instead of averaging across dissimilar
            # node roles, which can cancel the face response entirely.
            cams = F.relu((holder["values"] * gradients).sum(dim=1)).detach().numpy()
            raw = {face: float(cams[node]) for face, node in node_by_face.items()}
            scale = max(raw.values(), default=0.0)
            saliencies[name] = {face: value / scale if scale > 0 else 0.0
                                for face, value in raw.items()}
        face_vertices = step_faces(path)
        output.append({
            "uid": uid, "step_path": str(path), "true_class": true_name,
            "control_prediction": competitor_name,
            "residual_prediction": predictions[uid]["final_prediction"],
            "control_margin": float(logits[0, source.classes.index(true_name)]
                                    - logits[0, source.classes.index(competitor_name)]),
            "cached_embedding_max_absolute_error": embedding_error,
            "selected_face_nodes": len(node_by_face),
            "step_face_count": len(face_vertices),
            "face_vertices": {k: sorted(v) for k, v in face_vertices.items()},
            "gradcam": saliencies,
        })
        print(uid, true_name, competitor_name,
              "face nodes", len(node_by_face), "embedding error", embedding_error,
              flush=True)
    target = HERE / "jcpr_spatial_gradcam.json"
    target.write_text(json.dumps({
        "method": "HiResCAM (positive gradient times activation) on final JCPR STEP node layer",
        "objective": "saved STEP control class logit",
        "face_mapping": "STEP and OpenCascade boundary vertices, rounded to 0.0001 model units",
        "sources": {
            "encoder_checkpoint": str(V7 / "outputs/full_dataset_4096/best.pt"),
            "classifier_checkpoint": str(V7 / "downstream_classifier_tmcad_mesh/outputs/default/pooled_plus_brep_view_geometry_mesh/best.pt"),
            "tmcad_graph_cache": str(V7 / "downstream_classifier_tmcad/data/processed"),
        }, "cases": output,
    }, indent=2), encoding="utf-8")
    print(target, flush=True)


if __name__ == "__main__":
    main()
