"""Integrated-gradient decision audit for saved JCPR TMCAD classifiers."""
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
from data import TMCADSource, load_config  # noqa: E402
from model import ResidualAdapter  # noqa: E402

GROUPS = ("Typed entities", "B-rep instances", "Connectivity", "Global STEP",
          "Alternate view", "Geometry", "Surface profile")
CASES = ("coupling/337.stp", "pulley/530.stp", "coupling/257.stp")


def main() -> None:
    torch.set_num_threads(4)
    config = load_config(CLASSIFIER / "config.yaml")
    source = TMCADSource(config)
    control = source.baseline(torch.device("cpu"))
    saved = torch.load(CLASSIFIER / "outputs/default/physical_residual/best.pt",
                       map_location="cpu", weights_only=False)
    residual = ResidualAdapter(len(source.classes), saved["settings"])
    residual.load_state_dict(saved["state_dict"], strict=True)
    residual.eval().requires_grad_(False)

    train = source.indices["train"]
    arrays = {key: getattr(source, key) for key in ("pooled", "alternative", "geometry", "mesh")}
    means = {key: np.asarray(arrays[key][train], dtype=np.float32).mean(axis=0)
             for key in arrays}
    predictions = list(csv.DictReader((CLASSIFIER / "outputs/default/test_predictions.csv")
                                      .open(newline="", encoding="utf-8")))
    test_indices = source.indices["test"]
    assert len(predictions) == len(test_indices)
    results = []
    for uid in CASES:
        local = next(i for i, row in enumerate(predictions) if row["uid"] == uid)
        global_index = int(test_indices[local])
        assert source.manifest["samples"][global_index]["uid"] == uid
        true = int(source.labels[global_index])
        target = {key: torch.tensor(np.array(arrays[key][global_index], dtype=np.float32))
                  for key in arrays}
        reference = {key: torch.tensor(means[key]) for key in arrays}

        def forward(inputs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            base, _ = control(inputs)
            final, _ = residual(base, inputs["geometry"], inputs["mesh"])
            return base, final

        with torch.no_grad():
            base_logits, final_logits = forward({key: val[None] for key, val in target.items()})
            competitor = int(torch.argmax(torch.where(
                torch.arange(len(source.classes)) == true,
                torch.tensor(-torch.inf), base_logits[0])).item())
            assert source.classes[base_logits.argmax(dim=1).item()] == predictions[local]["baseline_prediction"]
            assert source.classes[final_logits.argmax(dim=1).item()] == predictions[local]["final_prediction"]
            original = [float(logits[0, true] - logits[0, competitor])
                        for logits in (base_logits, final_logits)]
            reference_logits = forward({key: val[None] for key, val in reference.items()})
            reference_margin = [float(logits[0, true] - logits[0, competitor])
                                for logits in reference_logits]

        steps = 96
        alphas = ((torch.arange(steps, dtype=torch.float32) + .5) / steps)
        interpolated = {
            key: (reference[key][None] + alphas.reshape((-1,) + (1,) * target[key].ndim)
                  * (target[key] - reference[key])[None]).detach().requires_grad_(True)
            for key in arrays
        }
        base, final = forward(interpolated)
        ordered = tuple(interpolated.values())
        base_margin = base[:, true] - base[:, competitor]
        final_margin = final[:, true] - final[:, competitor]
        grads = [
            torch.autograd.grad(base_margin.sum(), ordered, retain_graph=True),
            torch.autograd.grad(final_margin.sum(), ordered),
        ]
        attributions = []
        for model_grad in grads:
            parts = []
            for key, grad in zip(arrays, model_grad):
                parts.append(((target[key] - reference[key]) * grad.mean(dim=0)).detach().numpy())
            pooled, alternative, geometry, surface = parts
            group = np.array([pooled[i*512:(i+1)*512].sum() for i in range(4)] +
                             [alternative.sum(), geometry.sum(), surface.sum()], dtype=float)
            attributions.append(group)
        entry = {
            "uid": uid, "true_class": source.classes[true],
            "competitor": source.classes[competitor],
            "control_prediction": predictions[local]["baseline_prediction"],
            "residual_prediction": predictions[local]["final_prediction"],
            "control_margin": original[0], "residual_margin": original[1],
            "reference_control_margin": reference_margin[0],
            "reference_residual_margin": reference_margin[1],
            "control_group_attribution": attributions[0].tolist(),
            "residual_group_attribution": attributions[1].tolist(),
            "control_completeness_error": float(attributions[0].sum() - (original[0] - reference_margin[0])),
            "residual_completeness_error": float(attributions[1].sum() - (original[1] - reference_margin[1])),
        }
        results.append(entry)
        print(uid, "margin", original, "IG sums", [float(x.sum()) for x in attributions],
              "errors", entry["control_completeness_error"], entry["residual_completeness_error"], flush=True)
    destination = HERE / "gradient_attribution.json"
    destination.write_text(json.dumps({
        "method": "midpoint integrated gradients",
        "steps": steps,
        "baseline": "per-feature training-set mean",
        "class_pair": "true class versus strongest non-true control class",
        "groups": GROUPS,
        "sources": {"config": str(CLASSIFIER / "config.yaml"),
                    "control_checkpoint": str(config["paths"]["baseline_checkpoint"]),
                    "residual_checkpoint": str(CLASSIFIER / "outputs/default/physical_residual/best.pt"),
                    "predictions": str(CLASSIFIER / "outputs/default/test_predictions.csv")},
        "cases": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
