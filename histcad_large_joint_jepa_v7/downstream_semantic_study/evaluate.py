"""Same-split CAD retrieval for frozen and controlled downstream representations."""

from __future__ import annotations

import json
import re
from collections import Counter

import torch
from torch.nn import functional as F

from data import config, payload, resolve, split_indices

STOPWORDS = {"a", "an", "and", "for", "of", "the", "with", "to", "in", "on", "by", "mm"}


def canonical(value: str) -> str:
    value = re.sub(r"\([^)]*\)", "", value.lower())
    value = re.sub(r"\b(?:with|and|for|of|the|a|an)\b", " ", value)
    return " ".join(sorted(set(re.findall(r"[a-z0-9]+", value))))


def words(value: str) -> set[str]:
    value = re.sub(r"\([^)]*\)", " ", value.lower())
    return {word for word in re.findall(r"[a-z]+", value) if word not in STOPWORDS}


def neighbors(
    values: torch.Tensor, k: int, chunk: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    bank = F.normalize(values.float().to(device), dim=-1)
    positions, scores = [], []
    for start in range(0, len(bank), chunk):
        stop = min(start + chunk, len(bank))
        similarity = bank[start:stop] @ bank.T
        similarity[
            torch.arange(stop - start, device=device), torch.arange(start, stop, device=device)
        ] = -torch.inf
        score, index = similarity.topk(k, dim=1)
        positions.append(index.cpu())
        scores.append(score.cpu())
    return torch.cat(positions), torch.cat(scores)


def dimension_reference(targets: torch.Tensor, chunk: int, device: torch.device) -> torch.Tensor:
    bank = targets.to(device)
    output = []
    for start in range(0, len(bank), chunk):
        stop = min(start + chunk, len(bank))
        distance = torch.cdist(bank[start:stop], bank, p=1)
        distance[
            torch.arange(stop - start, device=device), torch.arange(start, stop, device=device)
        ] = torch.inf
        output.append(distance.topk(10, dim=1, largest=False).indices.cpu())
    return torch.cat(output)


def detailed_metrics(
    top: torch.Tensor,
    targets: torch.Tensor,
    refs: torch.Tensor,
    labels: dict[str, list[str]],
    tokens: list[set[str]],
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    count = len(top)
    nearest = top[:, 0].tolist()
    dimension = (targets[top[:, 0]] - targets).abs().mean(1)
    relevant = (top[:, :, None] == refs[:, None, :]).any(-1)
    dimension_recall = relevant[:, :10].float().sum(1) / 10
    results: dict[str, torch.Tensor] = {
        "dimension_mae_at_1": dimension,
        "dimension_recall_at_10": dimension_recall,
        "dimension_close_at_1": (dimension <= 0.2).float(),
        "text_token_jaccard_at_1": torch.tensor(
            [
                len(tokens[i] & tokens[j]) / max(1, len(tokens[i] | tokens[j]))
                for i, j in enumerate(nearest)
            ]
        ),
    }
    for name, label in labels.items():
        counts = Counter(label)
        eligible = torch.tensor([bool(value) and counts[value] > 1 for value in label])
        match = torch.tensor(
            [bool(label[i]) and label[i] == label[j] for i, j in enumerate(nearest)]
        )
        results[f"{name}_at_1_all"] = match.float()
        if eligible.any():
            results[f"{name}_at_1_eligible"] = match[eligible].float()
        if name in ("function", "function_raw"):
            for k in (5, 10):
                hit = torch.tensor(
                    [
                        bool(label[i]) and any(label[i] == label[j] for j in top[i, :k].tolist())
                        for i in range(count)
                    ]
                )
                results[f"{name}_recall_at_{k}_eligible"] = hit[eligible].float()
    results["geometry_structure_at_1"] = (
        (dimension <= 0.2)
        & (results["history_at_1_all"] > 0)
        & (results["feature_count_at_1_all"] > 0)
    ).float()
    return {name: float(value.mean()) for name, value in results.items()}, results


def metric_inputs(
    source: dict, indices: torch.Tensor
) -> tuple[torch.Tensor, dict[str, list[str]], list[set[str]]]:
    metadata = [source["metadata"][i] for i in indices.tolist()]
    structural = [source["structural_metadata"][i] for i in indices.tolist()]
    targets = (source["dimension_targets"][indices] - source["dimension_target_mean"]) / source[
        "dimension_target_std"
    ]
    descriptions = [item.get("functional_type", "") for item in metadata]
    labels = {
        "function": [canonical(value) for value in descriptions],
        "function_raw": [value.strip().lower() for value in descriptions],
        "history": [item["operation_signature"] for item in structural],
        "feature_count": [str(item["num_features"]) for item in structural],
        "primitive": [str(sorted(item["primitive_counts"].items())) for item in structural],
    }
    return targets.float(), labels, [words(value) for value in descriptions]


def quality(
    values: torch.Tensor,
    targets: torch.Tensor,
    refs: torch.Tensor,
    labels: dict[str, list[str]],
    tokens: list[set[str]],
    settings: dict,
    device: torch.device,
) -> tuple[dict, torch.Tensor, torch.Tensor, dict]:
    top, scores = neighbors(values, 10, settings["retrieval_chunk_size"], device)
    row, per_query = detailed_metrics(top, targets, refs, labels, tokens)
    return row, top, scores, per_query


def fusion(base: torch.Tensor, semantic: torch.Tensor, weight: float) -> torch.Tensor:
    return F.normalize(
        torch.cat(
            (F.normalize(base.float(), dim=-1), weight * F.normalize(semantic.float(), dim=-1)),
            dim=-1,
        ),
        dim=-1,
    )


def paired_interval(left: torch.Tensor, right: torch.Tensor, seed: int = 431) -> dict[str, float]:
    delta = right.float() - left.float()
    generator = torch.Generator().manual_seed(seed)
    means = [
        delta[torch.randint(len(delta), (len(delta),), generator=generator)].mean()
        for _ in range(1000)
    ]
    bounds = torch.quantile(torch.stack(means), torch.tensor([0.025, 0.975]))
    return {
        "delta": float(delta.mean()),
        "ci95_low": float(bounds[0]),
        "ci95_high": float(bounds[1]),
    }


def main() -> None:
    settings = config()
    torch.set_num_threads(8)
    output_dir = resolve(settings["paths"]["output_dir"])
    source = payload(settings)
    split = split_indices(source)
    study = torch.load(
        output_dir / "study_embeddings.pt", map_location="cpu", mmap=True, weights_only=False
    )
    existing = torch.load(
        resolve(settings["paths"]["existing_adapter_representations"]),
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    if existing["uids"] != source["uids"]:
        raise ValueError("Existing adapter vectors are not UID-aligned")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {
        "protocol": "Frozen v7 JEPA; train-only Functional Type text teacher; same-split held-out cosine retrieval; validation-only fusion selection",
        "text_used_at_retrieval_time": False,
        "split_counts": {name: len(indices) for name, indices in split.items()},
        "validation_fusion_grid": {},
        "metrics": {},
        "paired_test_deltas": {},
        "paired_test_vs_existing_adapter": {},
    }
    selected = {}
    saved_neighbors = {}
    test_per_query = {}
    for split_name in ("val", "test"):
        indices = split[split_name]
        targets, labels, tokens = metric_inputs(source, indices)
        refs = dimension_reference(targets, settings["evaluation"]["retrieval_chunk_size"], device)
        semantic = study["embeddings"]["semantic_only"][split_name]
        geometry_semantic = study["embeddings"]["semantic_plus_geometry"][split_name]
        direct = source["embeddings"]["joint"][indices]
        old = existing["embeddings"]["adapted_joint"][indices]
        representations = {
            "direct_jepa": direct,
            "existing_geometry_adapter": old,
            "semantic_only": semantic,
            "semantic_plus_geometry": geometry_semantic,
        }
        rows, neighbors_by_name, scores_by_name, per_query = {}, {}, {}, {}
        for name, values in representations.items():
            rows[name], neighbors_by_name[name], scores_by_name[name], per_query[name] = quality(
                values, targets, refs, labels, tokens, settings["evaluation"], device
            )
            print(
                json.dumps({"split": split_name, "representation": name, "metrics": rows[name]}),
                flush=True,
            )
        for base_name, base in (("direct_jepa", direct), ("existing_geometry_adapter", old)):
            fused_name = base_name + "_plus_semantic"
            if split_name == "val":
                grid = {}
                for weight in settings["evaluation"]["fusion_weights"]:
                    vector = fusion(base, semantic, weight)
                    grid[str(weight)] = quality(
                        vector, targets, refs, labels, tokens, settings["evaluation"], device
                    )[0]
                baseline = rows[base_name]
                eligible = [
                    (metrics["text_token_jaccard_at_1"], float(weight))
                    for weight, metrics in grid.items()
                    if metrics["dimension_mae_at_1"]
                    <= baseline["dimension_mae_at_1"]
                    * (1 + settings["evaluation"]["max_dimension_mae_relative_increase"])
                    and metrics["dimension_recall_at_10"]
                    >= baseline["dimension_recall_at_10"]
                    - settings["evaluation"]["max_dimension_recall_absolute_drop"]
                ]
                selected[base_name] = max(eligible)[1] if eligible else 0.0
                result["validation_fusion_grid"][base_name] = grid
            weight = selected[base_name]
            vector = fusion(base, semantic, weight)
            (
                rows[fused_name],
                neighbors_by_name[fused_name],
                scores_by_name[fused_name],
                per_query[fused_name],
            ) = quality(vector, targets, refs, labels, tokens, settings["evaluation"], device)
            print(
                json.dumps(
                    {
                        "split": split_name,
                        "representation": fused_name,
                        "weight": weight,
                        "metrics": rows[fused_name],
                    }
                ),
                flush=True,
            )
        result["metrics"][split_name] = rows
        if split_name == "test":
            saved_neighbors = {
                name: {"indices": neighbors_by_name[name], "scores": scores_by_name[name]}
                for name in rows
            }
            test_per_query = per_query
    result["selected_fusion_weights"] = selected
    for name in result["metrics"]["test"]:
        if name == "direct_jepa":
            continue
        result["paired_test_deltas"][name] = {
            metric: paired_interval(
                test_per_query["direct_jepa"][metric], test_per_query[name][metric]
            )
            for metric in (
                "dimension_mae_at_1",
                "dimension_recall_at_10",
                "text_token_jaccard_at_1",
                "function_at_1_all",
                "function_raw_at_1_eligible",
                "geometry_structure_at_1",
            )
        }
    for name in ("semantic_only", "semantic_plus_geometry", "existing_geometry_adapter_plus_semantic"):
        result["paired_test_vs_existing_adapter"][name] = {
            metric: paired_interval(
                test_per_query["existing_geometry_adapter"][metric], test_per_query[name][metric]
            )
            for metric in (
                "dimension_mae_at_1",
                "dimension_recall_at_10",
                "text_token_jaccard_at_1",
                "function_at_1_eligible",
                "function_raw_at_1_eligible",
                "function_recall_at_5_eligible",
                "geometry_structure_at_1",
            )
        }
    torch.save(
        {"uids": study["uids"]["test"], "neighbors": saved_neighbors},
        output_dir / "test_neighbors.pt",
    )
    (output_dir / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
