"""Train/validation-only diagnosis of weak form labels in frozen v7 space."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from extract import load_frozen
from labels import build_labels, load_taxonomy

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "weak_class_diagnosis"
CLASSES = ("cylinder", "beam", "sleeve", "plug", "rail", "flange", "base", "ring", "bushing", "shim")
DESCRIPTORS = {
    "hollow_or_bore": r"\b(?:hollow|bore[ds]?|tubular|inner diameter)\b",
    "solid": r"\bsolid\b",
    "flange": r"\bflang(?:e|ed|es)\b",
    "stepped": r"\bstepped?\b",
    "slotted": r"\bslots?|slotted\b",
    "threaded": r"\bthreads?|threaded\b",
    "hole": r"\bholes?|perforat(?:ed|ion)\b",
    "rectangular": r"\brectangular\b",
    "round": r"\b(?:round|circular|cylindrical)\b",
    "mounting": r"\bmount(?:ing|ed)?\b",
    "thin": r"\bthin\b",
    "long": r"\b(?:long|elongated)\b",
}


def main() -> None:
    torch.set_num_threads(6)
    source = load_frozen(
        ROOT.parent / "outputs/full_dataset_4096/representations.pt",
        ROOT.parent / "outputs/full_dataset_4096/best.pt",
        ROOT.parent / "data/processed/full_paired_graphs_v7.pt",
    )
    targets, report = build_labels(source, load_taxonomy(ROOT / "taxonomy_refinement_v2.json"))
    names = report["tasks"]["functional_form"]["retained_classes"]
    split = np.asarray([m["split"] for m in source["metadata"]])
    labels = targets[:, 0].numpy()
    chosen = [names.index(name) for name in CLASSES]
    generator = np.random.default_rng(97)
    train_by_class = {j: np.flatnonzero((split == "train") & (labels == j)) for j in range(len(names))}
    val_by_class = {j: np.flatnonzero((split == "val") & (labels == j)) for j in chosen}
    references = np.concatenate([generator.choice(rows, min(250, len(rows)), replace=False)
                                for rows in train_by_class.values()])
    questions = np.concatenate(list(val_by_class.values()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = {"protocol": "Train references and validation queries only; 250 train references/class, seed 97.",
              "classes": {name: {} for name in CLASSES}}
    for j in chosen:
        name = names[j]
        rows = train_by_class[j]
        text = [source["metadata"][int(i)].get("functional_type", "") for i in rows]
        counts = Counter({key: sum(bool(re.search(pattern, t, re.I)) for t in text)
                          for key, pattern in DESCRIPTORS.items()})
        shape_count = Counter(re.sub(r"\b\d+(?:\.\d+)?\b", "#", t.lower()) for t in text)
        output["classes"][name]["text_descriptor_fraction"] = {
            key: round(value / len(rows), 4) for key, value in counts.items()}
        output["classes"][name]["distinct_normalized_descriptions_fraction"] = round(len(shape_count) / len(rows), 4)
        output["classes"][name]["train_count"] = int(len(rows))
        output["classes"][name]["val_count"] = int(len(val_by_class[j]))
        complexity = [source["structural_metadata"][int(i)] for i in rows]
        output["classes"][name]["feature_count_median_iqr"] = np.quantile(
            [m["num_features"] for m in complexity], [.25, .5, .75]).tolist()
        output["classes"][name]["step_truncated_fraction"] = round(np.mean(
            [m["brep_truncated"] for m in complexity]), 4)

    for branch in ("joint", "parametric", "step"):
        vectors = source["embeddings"][branch]
        reference_matrix = vectors[references].float().to(device)
        reference_matrix = torch.nn.functional.normalize(reference_matrix, dim=1)
        reference_labels = torch.as_tensor(labels[references], device=device)
        query_matrix = torch.nn.functional.normalize(vectors[questions].float().to(device), dim=1)
        query_labels = labels[questions]
        sims = query_matrix @ reference_matrix.T
        top = sims.topk(5, dim=1).indices
        pred = reference_labels[top].cpu().numpy()
        top_similarity = sims.gather(1, top[:, :1]).squeeze(1).cpu().numpy()
        centroids = []
        cohesion = []
        for j in range(len(names)):
            rows = train_by_class[j]
            average = torch.zeros(vectors.shape[1], device=device)
            for start in range(0, len(rows), 1024):
                average += vectors[rows[start:start + 1024]].float().to(device).sum(0)
            average /= len(rows)
            cohesion.append(float(average.norm().item()))
            centroids.append(torch.nn.functional.normalize(average, dim=0))
        centroid_matrix = torch.stack(centroids)
        centroid_sim = (query_matrix @ centroid_matrix.T).cpu().numpy()
        centroid_pred = centroid_sim.argmax(1)
        for j in chosen:
            name = names[j]
            mask = query_labels == j
            own = centroid_sim[mask, j]
            rivals = centroid_sim[mask].copy()
            rivals[:, j] = -np.inf
            rival_id = rivals.argmax(1)
            branch_out = {
                "train_centroid_norm": round(cohesion[j], 4),
                "val_1nn_accuracy": round(float(np.mean(pred[mask, 0] == j)), 4),
                "val_1nn_similarity_median": round(float(np.median(top_similarity[mask])), 4),
                "val_high_similarity_conflict_fraction": round(float(np.mean(
                    (top_similarity[mask] > .98) & (pred[mask, 0] != j))), 4),
                "val_top5_same_label_fraction": round(float(np.mean(pred[mask] == j)), 4),
                "val_nearest_centroid_accuracy": round(float(np.mean(centroid_pred[mask] == j)), 4),
                "val_own_minus_nearest_rival_centroid_median": round(float(np.median(own - rivals.max(1))), 4),
                "val_1nn_top_confusions": Counter(names[int(k)] for k in pred[mask, 0] if k != j).most_common(4),
                "val_closest_rival_centroids": Counter(names[int(k)] for k in rival_id).most_common(4),
            }
            if branch == "joint":
                query_positions = np.flatnonzero(mask)
                conflicts = [int(pos) for pos in query_positions
                             if top_similarity[pos] > .98 and pred[pos, 0] != j]
                branch_out["high_similarity_conflict_examples"] = [
                    {"query_uid": source["uids"][int(questions[pos])],
                     "query_text": source["metadata"][int(questions[pos])]["functional_type"],
                     "neighbor_uid": source["uids"][int(references[int(top[pos, 0])])],
                     "neighbor_label": names[int(pred[pos, 0])],
                     "neighbor_text": source["metadata"][int(references[int(top[pos, 0])])]["functional_type"],
                     "cosine": round(float(top_similarity[pos]), 5)}
                    for pos in conflicts[:6]
                ]
            output["classes"][name][branch] = branch_out
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "diagnostics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    for name in CLASSES:
        row = output["classes"][name]
        print(f"{name:9} n={row['train_count']:5} "
              f"joint 1NN={row['joint']['val_1nn_accuracy']:.3f} "
              f"centroid={row['joint']['val_nearest_centroid_accuracy']:.3f} "
              f"cohesion={row['joint']['train_centroid_norm']:.3f} "
              f"P 1NN={row['parametric']['val_1nn_accuracy']:.3f} "
              f"S 1NN={row['step']['val_1nn_accuracy']:.3f}")


if __name__ == "__main__":
    main()
