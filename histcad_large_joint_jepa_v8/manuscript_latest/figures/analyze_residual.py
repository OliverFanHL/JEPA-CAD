"""Audit the final TMCAD physical residual from immutable test inputs."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CLASSIFIER = ROOT / "downstream_classifier_tmcad_final"
sys.path.insert(0, str(CLASSIFIER))
from model import ResidualAdapter  # noqa: E402


def main() -> None:
    inputs = CLASSIFIER / "data/processed/test_frozen_inputs.npz"
    checkpoint = CLASSIFIER / "outputs/default/physical_residual/best.pt"
    predictions = CLASSIFIER / "outputs/default/test_predictions.csv"
    with np.load(inputs) as source:
        data = {key: source[key].copy() for key in source.files}
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = ResidualAdapter(len(saved["classes"]), saved["settings"])
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    base = data["baseline_logits"]

    def infer(geometry: np.ndarray, mesh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with torch.inference_mode():
            logits, residual = model(torch.from_numpy(base),
                                     torch.from_numpy(geometry),
                                     torch.from_numpy(mesh))
        return logits.numpy(), residual.numpy()

    full, residual = infer(data["geometry"], data["mesh"])
    # The train-split mean is the reference for each held-out feature group.
    with np.load(CLASSIFIER / "data/processed/train_frozen_inputs.npz") as train:
        geometry_mean = train["geometry"].mean(axis=0)
        mesh_mean = train["mesh"].mean(axis=0)
    geometry_masked, _ = infer(np.broadcast_to(geometry_mean, data["geometry"].shape).copy(),
                               data["mesh"])
    mesh_masked, _ = infer(data["geometry"],
                           np.broadcast_to(mesh_mean, data["mesh"].shape).copy())
    both_masked, _ = infer(np.broadcast_to(geometry_mean, data["geometry"].shape).copy(),
                           np.broadcast_to(mesh_mean, data["mesh"].shape).copy())
    rows = list(csv.DictReader(predictions.open(newline="", encoding="utf-8")))
    assert len(rows) == len(base)
    class_to_index = {name: i for i, name in enumerate(saved["classes"])}
    assert all(int(data["label"][i]) == class_to_index[row["true_class"]]
               and base[i].argmax() == class_to_index[row["baseline_prediction"]]
               and full[i].argmax() == class_to_index[row["final_prediction"]]
               for i, row in enumerate(rows))

    def metrics(logits: np.ndarray) -> dict:
        pred = logits.argmax(axis=1)
        f1 = []
        for c in range(len(saved["classes"])):
            tp = np.sum((pred == c) & (data["label"] == c))
            fp = np.sum((pred == c) & (data["label"] != c))
            fn = np.sum((pred != c) & (data["label"] == c))
            f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
        return {"accuracy": float(np.mean(pred == data["label"])),
                "macro_f1": float(np.mean(f1)),
                "prediction_changed": int(np.sum(pred != full.argmax(axis=1)))}

    masks = {"baseline": base, "full": full, "geometry_masked": geometry_masked,
             "surface_masked": mesh_masked, "both_masked": both_masked}
    score = {key: metrics(value) for key, value in masks.items()}
    truth = data["label"].astype(int)
    other_logits = full.copy()
    other_logits[np.arange(len(truth)), truth] = -np.inf
    competitor = other_logits.argmax(axis=1)
    full_margin = full[np.arange(len(truth)), truth] - full[np.arange(len(truth)), competitor]
    outcome = full.argmax(axis=1) == truth
    effects = {}
    for group, masked in (("geometry", geometry_masked), ("surface_profile", mesh_masked)):
        masked_margin = masked[np.arange(len(truth)), truth] - masked[np.arange(len(truth)), competitor]
        effect = full_margin - masked_margin
        effects[group] = {}
        for name, selected_outcome in (("correct", outcome), ("incorrect", ~outcome)):
            values = effect[selected_outcome]
            effects[group][name] = {"n": int(len(values)),
                "median": float(np.median(values)),
                "q25": float(np.quantile(values, .25)),
                "q75": float(np.quantile(values, .75)),
                "mean": float(np.mean(values)),
                "positive_fraction": float(np.mean(values > 0))}
    selected = ["coupling/337.stp", "pulley/530.stp", "coupling/257.stp"]
    case_output = []
    for uid in selected:
        i = next(j for j, row in enumerate(rows) if row["uid"] == uid)
        true = int(data["label"][i])
        competitor = int(np.argmax(np.where(np.arange(len(saved["classes"])) == true,
                                              -np.inf, base[i])))
        margin = lambda arr: float(arr[i, true] - arr[i, competitor])
        case_output.append({"uid": uid, "truth": rows[i]["true_class"],
                            "competitor": saved["classes"][competitor],
                            "baseline_margin": margin(base),
                            "residual_margin": float(residual[i, true] - residual[i, competitor]),
                            "full_margin": margin(full),
                            "geometry_masked_margin": margin(geometry_masked),
                            "surface_masked_margin": margin(mesh_masked)})
    destination = HERE / "residual_audit.json"
    destination.write_text(json.dumps({"sources": {"inputs": str(inputs),
                    "checkpoint": str(checkpoint), "predictions": str(predictions)},
                    "metrics": score, "group_margin_effects": effects,
                    "cases": case_output}, indent=2), encoding="utf-8")
    print(destination.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
