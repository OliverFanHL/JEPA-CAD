"""Paired full/stripped text × downstream CAD-target ablation."""

from __future__ import annotations

import json

import torch
from evaluate import (
    detailed_metrics,
    dimension_reference,
    metric_inputs,
    neighbors,
    paired_interval,
)

from data import ROOT, config, payload, split_indices


def main() -> None:
    settings = config()
    source = payload(settings)
    test_indices = split_indices(source)["test"]
    targets, labels, tokens = metric_inputs(source, test_indices)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    refs = dimension_reference(targets, settings["evaluation"]["retrieval_chunk_size"], device)
    conditions = {
        "stripped_text_only": (ROOT / "outputs_clean", "semantic_only"),
        "full_text_only": (ROOT / "outputs", "semantic_only"),
        "stripped_text_plus_physical": (ROOT / "outputs_clean", "semantic_plus_physical_sizes"),
        "full_text_plus_physical": (ROOT / "outputs", "semantic_plus_physical_sizes"),
        "stripped_text_plus_dimensions": (ROOT / "outputs_clean", "semantic_plus_dimensions"),
        "full_text_plus_dimensions": (ROOT / "outputs", "semantic_plus_dimensions"),
        "stripped_text_plus_numeric": (ROOT / "outputs_clean", "semantic_plus_geometry"),
        "full_text_plus_numeric": (ROOT / "outputs", "semantic_plus_geometry"),
    }
    metrics, query_rows = {}, {}
    for name, (directory, representation) in conditions.items():
        expected_uids = [source["uids"][i] for i in test_indices.tolist()]
        if representation in ("semantic_plus_dimensions", "semantic_plus_physical_sizes"):
            embedded = torch.load(
                directory
                / (
                    "dimension_only_embeddings.pt"
                    if representation == "semantic_plus_dimensions"
                    else "physical_size_embeddings.pt"
                ),
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
            if embedded["uids"]["test"] != expected_uids:
                raise ValueError(f"UID mismatch for {name}")
            top, _ = neighbors(
                embedded["embeddings"]["test"],
                10,
                settings["evaluation"]["retrieval_chunk_size"],
                device,
            )
        else:
            neighbor_payload = torch.load(
                directory / "test_neighbors.pt", map_location="cpu", weights_only=False
            )
            if neighbor_payload["uids"] != expected_uids:
                raise ValueError(f"UID mismatch for {name}")
            top = neighbor_payload["neighbors"][representation]["indices"]
        metrics[name], query_rows[name] = detailed_metrics(top, targets, refs, labels, tokens)
    key_metrics = (
        "dimension_recall_at_10",
        "dimension_mae_at_1",
        "function_at_1_eligible",
        "function_raw_at_1_eligible",
        "function_recall_at_5_eligible",
        "text_token_jaccard_at_1",
        "history_at_1_all",
        "feature_count_at_1_all",
        "primitive_at_1_all",
        "geometry_structure_at_1",
    )
    contrasts = {
        "full_minus_stripped_text_only": ("stripped_text_only", "full_text_only"),
        "full_minus_stripped_with_physical": (
            "stripped_text_plus_physical",
            "full_text_plus_physical",
        ),
        "full_minus_stripped_with_dimensions": (
            "stripped_text_plus_dimensions",
            "full_text_plus_dimensions",
        ),
        "full_minus_stripped_with_numeric": (
            "stripped_text_plus_numeric",
            "full_text_plus_numeric",
        ),
        "numeric_minus_none_stripped_text": (
            "stripped_text_only",
            "stripped_text_plus_numeric",
        ),
        "numeric_minus_none_full_text": ("full_text_only", "full_text_plus_numeric"),
        "dimensions_minus_none_stripped_text": (
            "stripped_text_only",
            "stripped_text_plus_dimensions",
        ),
        "physical_minus_none_stripped_text": (
            "stripped_text_only",
            "stripped_text_plus_physical",
        ),
        "physical_minus_none_full_text": (
            "full_text_only",
            "full_text_plus_physical",
        ),
        "descriptor_minus_physical_stripped_text": (
            "stripped_text_plus_physical",
            "stripped_text_plus_dimensions",
        ),
        "descriptor_minus_physical_full_text": (
            "full_text_plus_physical",
            "full_text_plus_dimensions",
        ),
        "dimensions_minus_none_full_text": (
            "full_text_only",
            "full_text_plus_dimensions",
        ),
        "structure_minus_dimensions_stripped_text": (
            "stripped_text_plus_dimensions",
            "stripped_text_plus_numeric",
        ),
        "structure_minus_dimensions_full_text": (
            "full_text_plus_dimensions",
            "full_text_plus_numeric",
        ),
    }
    paired = {
        name: {
            metric: paired_interval(query_rows[left][metric], query_rows[right][metric])
            for metric in key_metrics
        }
        for name, (left, right) in contrasts.items()
    }
    interaction = {}
    for metric in key_metrics:
        effect_without_numeric = (
            query_rows["full_text_only"][metric] - query_rows["stripped_text_only"][metric]
        )
        effect_with_numeric = (
            query_rows["full_text_plus_numeric"][metric]
            - query_rows["stripped_text_plus_numeric"][metric]
        )
        zero = torch.zeros_like(effect_without_numeric)
        interaction[metric] = paired_interval(zero, effect_with_numeric - effect_without_numeric)
    independent = {
        name: json.loads((directory / "annotation_transfer.json").read_text(encoding="utf-8"))[
            "metrics"
        ][representation]
        for name, (directory, representation) in conditions.items()
        if representation not in ("semantic_plus_dimensions", "semantic_plus_physical_sizes")
    }
    output = {
        "protocol": "Same 145,899 labeled training items, architecture, seed, six-epoch budget, val/test split; only text normalization and CAD auxiliary loss vary (none, 15 physical size/ratio fields, full 22-D descriptor, or descriptor plus count/operation heads). JEPA and text teacher frozen.",
        "conditions": metrics,
        "paired_contrasts": paired,
        "interaction_full_text_by_numeric": interaction,
        "evaluation_only_annotation_transfer": independent,
        "interpretation_note": "The full text may encode dimensions in natural language; generic text embedding is not a unit-aware numeric representation. Numeric auxiliary targets are derived from CAD data and do not change the frozen JEPA encoder.",
    }
    target = ROOT / "outputs_clean" / "factorial_ablation.json"
    target.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "saved": str(target),
                "conditions": {
                    name: {
                        key: round(row[key], 4)
                        for key in (
                            "dimension_recall_at_10",
                            "dimension_mae_at_1",
                            "function_at_1_eligible",
                            "text_token_jaccard_at_1",
                        )
                    }
                    for name, row in metrics.items()
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
