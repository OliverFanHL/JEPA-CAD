from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any

import torch
from torch.nn import functional as F

STOPWORDS = {"a", "an", "and", "for", "of", "the", "with", "to", "in", "on", "by", "mm"}


def classification_metrics(logits: torch.Tensor, target: torch.Tensor, classes: int) -> dict[str, Any]:
    prediction = logits.argmax(1)
    confusion = torch.zeros((classes, classes), dtype=torch.long)
    confusion.index_put_((target.cpu(), prediction.cpu()), torch.ones(len(target), dtype=torch.long), accumulate=True)
    precision, recall, f1 = [], [], []
    for label in range(classes):
        tp = int(confusion[label, label])
        fp = int(confusion[:, label].sum()) - tp
        fn = int(confusion[label].sum()) - tp
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / max(1e-12, p + r))
    return {
        "samples": len(target),
        "accuracy": float((prediction == target).float().mean()),
        "balanced_accuracy": sum(recall) / classes,
        "macro_f1": sum(f1) / classes,
        "top_3_accuracy": float(
            (logits.topk(min(3, classes), 1).indices == target[:, None]).any(1).float().mean()
        ),
        "confusion_matrix": confusion.tolist(),
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1,
    }


def nearest_neighbors(values: torch.Tensor, k: int, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    bank = F.normalize(values.float(), dim=1)
    all_indices, all_scores = [], []
    for start in range(0, len(bank), chunk_size):
        stop = min(start + chunk_size, len(bank))
        similarity = bank[start:stop] @ bank.T
        similarity[torch.arange(stop - start), torch.arange(start, stop)] = -torch.inf
        scores, indices = similarity.topk(min(k, len(bank) - 1), dim=1)
        all_indices.append(indices.cpu())
        all_scores.append(scores.cpu())
    return torch.cat(all_indices), torch.cat(all_scores)


def categorical_retrieval(labels: list[Any], neighbors: torch.Tensor) -> dict[str, float | int]:
    counts = Counter(labels)
    eligible = [i for i, label in enumerate(labels) if label not in (None, "") and counts[label] > 1]
    result: dict[str, float | int] = {"eligible_queries": len(eligible)}
    for k in (1, 5, 10):
        k = min(k, neighbors.shape[1])
        matches = torch.tensor([
            [labels[int(j)] == labels[i] for j in neighbors[i, :k]] for i in eligible
        ], dtype=torch.float)
        result[f"hit_at_{k}"] = float(matches.any(1).float().mean()) if len(matches) else math.nan
        result[f"precision_at_{k}"] = float(matches.mean()) if len(matches) else math.nan
        reciprocal = []
        for row in matches:
            found = torch.where(row > 0)[0]
            reciprocal.append(0.0 if not len(found) else 1.0 / (int(found[0]) + 1))
        result[f"mrr_at_{k}"] = sum(reciprocal) / max(1, len(reciprocal))
    return result


def retrieval_metrics(
    values: torch.Tensor,
    labels: dict[str, list[Any]],
    targets: torch.Tensor | None,
    chunk_size: int,
) -> dict[str, Any]:
    normalized = F.normalize(values.float(), dim=1)
    neighbors, scores = nearest_neighbors(normalized, 10, chunk_size)
    result: dict[str, Any] = {
        "samples": len(values),
        "dimensions": values.shape[1],
        "labels": {name: categorical_retrieval(items, neighbors) for name, items in labels.items()},
    }
    if targets is not None:
        result["target_mae"] = {
            f"at_{k}": float((targets[neighbors[:, :k]] - targets[:, None]).abs().mean())
            for k in (1, 5, 10)
        }
    hashes = [hashlib.sha1(row.numpy().tobytes()).hexdigest() for row in normalized.cpu()]
    duplicate_counts = Counter(hashes)
    gaps = scores[:, :-1] - scores[:, 1:]
    result["duplicates_and_ties"] = {
        "samples_in_exact_embedding_duplicate_groups": sum(
            count for count in duplicate_counts.values() if count > 1
        ),
        "queries_with_top10_tie_at_1e_7": int((gaps.abs() <= 1e-7).any(1).sum()),
        "queries_with_top10_near_tie_at_1e_4": int((gaps.abs() <= 1e-4).any(1).sum()),
        "median_top1_top2_margin": float((scores[:, 0] - scores[:, 1]).median()),
        "mean_top1_top2_margin": float((scores[:, 0] - scores[:, 1]).mean()),
    }
    return result


def dimension_reference_neighbors(targets: torch.Tensor, chunk_size: int, count: int = 10) -> torch.Tensor:
    references = []
    targets = targets.float()
    for start in range(0, len(targets), chunk_size):
        stop = min(start + chunk_size, len(targets))
        distance = torch.cdist(targets[start:stop], targets, p=1)
        distance[torch.arange(stop - start), torch.arange(start, stop)] = torch.inf
        references.append(distance.topk(count, dim=1, largest=False).indices)
    return torch.cat(references)


def controlled_retrieval_metrics(
    values: torch.Tensor,
    targets: torch.Tensor,
    labels: dict[str, list[Any]],
    references: torch.Tensor,
    chunk_size: int,
) -> dict[str, Any]:
    """Metrics used by the existing downstream_retrieval_adapter study."""
    normalized = F.normalize(values.float(), dim=1)
    neighbors, scores = nearest_neighbors(normalized, 10, chunk_size)
    dimension: dict[str, float] = {}
    for k in (1, 5, 10):
        differences = (targets[neighbors[:, :k]] - targets[:, None]).abs().mean(-1)
        dimension[f"mae_at_{k}"] = float(differences.mean())
        dimension[f"best_mae_at_{k}"] = float(differences.min(1).values.mean())
    dimension_neighbors: dict[str, float | int] = {
        "reference_neighbors_per_query": references.shape[1],
        "exact_dimension_nearest_neighbor_accuracy": float((neighbors[:, 0] == references[:, 0]).float().mean()),
    }
    for k in (1, 5, 10):
        relevant = (neighbors[:, :k, None] == references[:, None, :]).any(-1)
        dimension_neighbors[f"precision_at_{k}"] = float(relevant.float().mean())
        dimension_neighbors[f"recall_at_{k}"] = float((relevant.sum(1) / references.shape[1]).float().mean())
        dimension_neighbors[f"query_success_at_{k}"] = float(relevant.any(1).float().mean())
    gaps = scores[:, :-1] - scores[:, 1:]
    hashes = [hashlib.sha1(row.numpy().tobytes()).hexdigest() for row in normalized.cpu()]
    duplicate_counts = Counter(hashes)
    return {
        "dimensions": values.shape[1],
        "dimension_retrieval": dimension,
        "dimension_neighbor_retrieval": dimension_neighbors,
        "operation_signature": categorical_retrieval(labels["operation_signature"], neighbors),
        "feature_type_sequence": categorical_retrieval(labels["feature_sequence"], neighbors),
        "final_boolean_operation": categorical_retrieval(labels["final_operation"], neighbors),
        "feature_count_bucket": categorical_retrieval(labels["feature_count_bucket"], neighbors),
        "primitive_profile": categorical_retrieval(labels["primitive_profile"], neighbors),
        "dominant_constraint_type": categorical_retrieval(labels["dominant_constraint"], neighbors),
        "functional_type_evaluation_only": categorical_retrieval(labels["functional_type"], neighbors),
        "canonical_functional_type_evaluation_only": categorical_retrieval(
            labels["canonical_functional_type"], neighbors
        ),
        "ties": {
            "samples_in_exact_duplicate_groups": sum(count for count in duplicate_counts.values() if count > 1),
            "queries_with_top10_tie_at_1e_7": int((gaps.abs() <= 1e-7).any(1).sum()),
            "queries_with_top10_gap_below_1e_4": int((gaps.abs() <= 1e-4).any(1).sum()),
            "median_top1_top2_margin": float((scores[:, 0] - scores[:, 1]).median()),
        },
    }


def _semantic_words(value: str) -> set[str]:
    value = re.sub(r"\([^)]*\)", " ", value.lower())
    return {word for word in re.findall(r"[a-z]+", value) if word not in STOPWORDS}


def semantic_study_metrics(
    values: torch.Tensor,
    targets: torch.Tensor,
    labels: dict[str, list[Any]],
    references: torch.Tensor,
    chunk_size: int,
) -> dict[str, float]:
    """Exact metric family used by downstream_semantic_study/evaluate.py."""
    neighbors, _ = nearest_neighbors(values, 10, chunk_size)
    nearest = neighbors[:, 0].tolist()
    dimension = (targets[neighbors[:, 0]] - targets).abs().mean(1)
    relevant = (neighbors[:, :, None] == references[:, None, :]).any(-1)
    tokens = [_semantic_words(str(value)) for value in labels["functional_type"]]
    rows: dict[str, torch.Tensor] = {
        "dimension_mae_at_1": dimension,
        "dimension_recall_at_10": relevant[:, :10].float().sum(1) / references.shape[1],
        "dimension_close_at_1": (dimension <= 0.2).float(),
        "text_token_jaccard_at_1": torch.tensor([
            len(tokens[i] & tokens[j]) / max(1, len(tokens[i] | tokens[j]))
            for i, j in enumerate(nearest)
        ]),
    }
    semantic_labels = {
        "function": labels["canonical_functional_type"],
        "function_raw": [str(value).strip().lower() for value in labels["functional_type"]],
        "history": labels["operation_signature"],
        "feature_count": labels["feature_count"],
        "primitive": labels["primitive_profile"],
    }
    for name, items in semantic_labels.items():
        counts = Counter(items)
        eligible = torch.tensor([bool(value) and counts[value] > 1 for value in items])
        match = torch.tensor([bool(items[i]) and items[i] == items[j] for i, j in enumerate(nearest)])
        rows[f"{name}_at_1_all"] = match.float()
        if eligible.any():
            rows[f"{name}_at_1_eligible"] = match[eligible].float()
        if name in {"function", "function_raw"}:
            for k in (5, 10):
                hit = torch.tensor([
                    bool(items[i]) and any(items[i] == items[j] for j in neighbors[i, :k].tolist())
                    for i in range(len(items))
                ])
                rows[f"{name}_recall_at_{k}_eligible"] = hit[eligible].float()
    rows["geometry_structure_at_1"] = (
        (dimension <= 0.2)
        & (rows["history_at_1_all"] > 0)
        & (rows["feature_count_at_1_all"] > 0)
    ).float()
    return {name: float(value.mean()) for name, value in rows.items()}


def latent_diagnostics(values: torch.Tensor, max_rows: int, seed: int) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    if len(values) > max_rows:
        indices = torch.randperm(len(values), generator=generator)[:max_rows]
        sample = values[indices].float()
    else:
        sample = values.float()
    centered = sample - sample.mean(0)
    variance = centered.var(0, unbiased=False)
    nonzero = variance > 1e-12
    # Work in the smaller Gram space for very wide representations.
    singular = torch.linalg.svdvals(centered)
    eigenvalues = singular.square() / max(1, len(sample) - 1)
    probabilities = eigenvalues / eigenvalues.sum().clamp_min(1e-12)
    effective_rank = torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
    participation = eigenvalues.sum().square() / eigenvalues.square().sum().clamp_min(1e-12)
    cumulative = probabilities.cumsum(0)
    pair_count = min(50_000, len(sample) * 4)
    left = torch.randint(len(sample), (pair_count,), generator=generator)
    right = torch.randint(len(sample), (pair_count,), generator=generator)
    similarity = (F.normalize(sample[left], dim=1) * F.normalize(sample[right], dim=1)).sum(1)
    return {
        "rows_used": len(sample),
        "dimensions": sample.shape[1],
        "effective_rank_entropy": float(effective_rank),
        "effective_rank_participation": float(participation),
        "nonzero_variance_dimensions": int(nonzero.sum()),
        "dead_dimension_fraction": float((~nonzero).float().mean()),
        "mean_dimension_variance": float(variance.mean()),
        "median_dimension_variance": float(variance.median()),
        "variance_top_1_fraction": float(probabilities[:1].sum()),
        "variance_top_10_fraction": float(probabilities[:10].sum()),
        "components_for_90_percent_variance": int(torch.searchsorted(cumulative, 0.9)) + 1,
        "random_pair_cosine_mean": float(similarity.mean()),
        "random_pair_cosine_std": float(similarity.std(unbiased=False)),
    }


def class_separation(values: torch.Tensor, labels: list[int]) -> dict[str, float]:
    target = torch.tensor(labels)
    valid = target >= 0
    values = F.normalize(values[valid].float(), dim=1)
    target = target[valid]
    present = torch.unique(target, sorted=True)
    if len(present) < 2:
        raise ValueError("Class separation requires at least two represented classes")
    compact = torch.empty(int(present.max()) + 1, dtype=torch.long)
    compact[present] = torch.arange(len(present))
    target = compact[target]
    classes = len(present)
    centroids = torch.stack([F.normalize(values[target == i].mean(0), dim=0) for i in range(classes)])
    own = (values * centroids[target]).sum(1)
    similarities = values @ centroids.T
    similarities[torch.arange(len(values)), target] = -torch.inf
    margin = own - similarities.max(1).values
    between = centroids @ centroids.T
    off_diagonal = ~torch.eye(classes, dtype=torch.bool)
    return {
        "mean_own_centroid_cosine": float(own.mean()),
        "mean_nearest_other_centroid_cosine": float(similarities.max(1).values.mean()),
        "mean_centroid_margin": float(margin.mean()),
        "positive_margin_fraction": float((margin > 0).float().mean()),
        "mean_interclass_centroid_cosine": float(between[off_diagonal].mean()),
    }
