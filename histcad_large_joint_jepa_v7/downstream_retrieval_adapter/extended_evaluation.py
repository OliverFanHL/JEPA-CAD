"""Interpret direct/adapted retrieval with paired, task-level diagnostics.

This reads frozen representations only. It never trains or changes checkpoints.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
STOPWORDS = {"a", "an", "and", "for", "of", "the", "with", "to", "in", "on", "by", "mm"}


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def words(description: str) -> set[str]:
    text = re.sub(r"\([^)]*\)", " ", description.lower())
    return {token for token in re.findall(r"[a-z]+", text) if token not in STOPWORDS}


def canonical(description: str) -> str:
    text = re.sub(r"\([^)]*\)", "", description.lower())
    text = re.sub(r"\b(?:with|and|for|of|the|a|an)\b", " ", text)
    return " ".join(sorted(set(re.findall(r"[a-z0-9]+", text))))


def nearest(values: torch.Tensor, count: int, chunk: int, device: torch.device) -> torch.Tensor:
    normalized = F.normalize(values.float(), dim=-1)
    bank = normalized.to(device)
    outputs = []
    for start in range(0, len(bank), chunk):
        stop = min(start + chunk, len(bank))
        similarity = bank[start:stop] @ bank.T
        similarity[
            torch.arange(stop - start, device=device), torch.arange(start, stop, device=device)
        ] = -torch.inf
        outputs.append(similarity.topk(count, dim=1).indices.cpu())
    return torch.cat(outputs)


def bootstrap_delta(left: torch.Tensor, right: torch.Tensor, seed: int = 700) -> dict[str, float]:
    delta = (right.float() - left.float()).cpu()
    generator = torch.Generator().manual_seed(seed)
    estimates = []
    for _ in range(1000):
        sample = torch.randint(len(delta), (len(delta),), generator=generator)
        estimates.append(delta[sample].mean())
    interval = torch.quantile(torch.stack(estimates), torch.tensor([0.025, 0.975]))
    return {
        "mean": float(delta.mean()),
        "ci95_low": float(interval[0]),
        "ci95_high": float(interval[1]),
    }


def chance_accuracy(labels: list[str]) -> float:
    counts = Counter(labels)
    count = len(labels)
    return sum(value * (value - 1) for value in counts.values()) / (count * (count - 1))


def evaluate_split(
    source: dict, adapted: dict, split: str, device: torch.device, chunk: int, output_dir: Path
) -> dict:
    indices = [index for index, item in enumerate(source["metadata"]) if item["split"] == split]
    metadata = [source["metadata"][index] for index in indices]
    structural = [source["structural_metadata"][index] for index in indices]
    raw_dims = source["dimension_targets"][indices].float()
    targets = (raw_dims - source["dimension_target_mean"]) / source["dimension_target_std"]
    # Comparing sorted BBox sides avoids penalizing merely different axis orientation.
    bbox = torch.expm1(raw_dims[:, 1:4]).sort(dim=1, descending=True).values
    descriptions = [item.get("functional_type", "") for item in metadata]
    tokens = [words(item) for item in descriptions]
    labels = {
        "function": [canonical(item) for item in descriptions],
        "history": [item["operation_signature"] for item in structural],
        "feature_count": [str(item["num_features"]) for item in structural],
        "primitive": [str(sorted(item["primitive_counts"].items())) for item in structural],
    }
    candidates = {
        name: nearest(adapted["embeddings"][name][indices], 10, chunk, device)
        for name in ("direct_joint", "adapted_joint")
    }
    per_query: dict[str, dict[str, torch.Tensor]] = {}
    for name, neighbors in candidates.items():
        top = neighbors[:, 0].tolist()
        dim_error = (targets[neighbors[:, 0]] - targets).abs().mean(1)
        bbox_ratio = ((bbox[neighbors[:, 0]] + 1.0).log() - (bbox + 1.0).log()).abs().mean(1)
        overlap = torch.tensor(
            [
                len(tokens[index] & tokens[other]) / max(1, len(tokens[index] | tokens[other]))
                for index, other in enumerate(top)
            ]
        )
        rows = {
            "dim_error": dim_error,
            "bbox_log_ratio_error": bbox_ratio,
            "function_match": torch.tensor(
                [
                    bool(labels["function"][i]) and labels["function"][i] == labels["function"][j]
                    for i, j in enumerate(top)
                ]
            ),
            "history_match": torch.tensor(
                [labels["history"][i] == labels["history"][j] for i, j in enumerate(top)]
            ),
            "feature_count_match": torch.tensor(
                [
                    labels["feature_count"][i] == labels["feature_count"][j]
                    for i, j in enumerate(top)
                ]
            ),
            "primitive_match": torch.tensor(
                [labels["primitive"][i] == labels["primitive"][j] for i, j in enumerate(top)]
            ),
            "text_token_jaccard": overlap,
        }
        for k in (5, 10):
            for label_name, metric_name in (
                ("function", "function"),
                ("history", "history"),
                ("feature_count", "feature_count"),
            ):
                rows[f"{metric_name}_recall_at_{k}"] = torch.tensor(
                    [
                        bool(labels[label_name][index])
                        and any(
                            labels[label_name][index] == labels[label_name][other]
                            for other in neighbors[index, :k].tolist()
                        )
                        for index in range(len(neighbors))
                    ]
                )
        rows["dim_error_below_0_2"] = dim_error <= 0.2
        rows["combined_geometry_structure"] = (
            rows["dim_error_below_0_2"] & rows["feature_count_match"] & rows["history_match"]
        )
        rows["combined_function_geometry"] = rows["function_match"] & rows["dim_error_below_0_2"]
        per_query[name] = rows

    direct, task = per_query["direct_joint"], per_query["adapted_joint"]
    metrics = {
        name: {key: float(value.float().mean()) for key, value in rows.items()}
        for name, rows in per_query.items()
    }
    metrics["chance_accuracy"] = {key: chance_accuracy(value) for key, value in labels.items()}
    improvements = {key: bootstrap_delta(direct[key], task[key]) for key in direct}
    dim_change = task["dim_error"] - direct["dim_error"]
    function_counts = Counter(labels["function"])
    same_function_eligible = torch.tensor(
        [function_counts[value] > 1 and bool(value) for value in labels["function"]]
    )
    result = {
        "samples": len(indices),
        "metrics": metrics,
        "paired_adapter_minus_direct": improvements,
        "dimensional_nearest_neighbor_outcomes": {
            "adapter_better": float((dim_change < -1e-6).float().mean()),
            "adapter_worse": float((dim_change > 1e-6).float().mean()),
            "unchanged": float((dim_change.abs() <= 1e-6).float().mean()),
        },
        "exact_function_label_eligible_fraction": float(same_function_eligible.float().mean()),
        "exact_function_label_accuracy_on_eligible": {
            name: float(rows["function_match"][same_function_eligible].float().mean())
            for name, rows in per_query.items()
        },
        "exact_function_recall_on_eligible": {
            name: {
                f"recall_at_{k}": float(
                    rows[f"function_recall_at_{k}"][same_function_eligible].float().mean()
                )
                for k in (5, 10)
            }
            for name, rows in per_query.items()
        },
    }
    if split == "test":
        path = output_dir / "extended_per_query_test.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "uid",
                    *[f"direct_{key}" for key in direct],
                    *[f"adapted_{key}" for key in task],
                ],
            )
            writer.writeheader()
            for index, item in enumerate(metadata):
                writer.writerow(
                    {
                        "uid": item["uid"],
                        **{f"direct_{key}": float(value[index]) for key, value in direct.items()},
                        **{f"adapted_{key}": float(value[index]) for key, value in task.items()},
                    }
                )
    return result


def main() -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    output_dir = resolve(config["paths"]["output_dir"])
    source = torch.load(
        resolve(config["paths"]["representations"]),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    adapted = torch.load(
        output_dir / "adapted_representations.pt", map_location="cpu", weights_only=False, mmap=True
    )
    if source["uids"] != adapted["uids"]:
        raise ValueError("Representations are not UID aligned")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {
        "protocol": "Same-split nearest neighbors; paired direct-versus-adapted; 1000 bootstrap resamples; no training",
        "dim_error_threshold": "0.2 mean absolute standardized 22-D target units (diagnostic threshold, not physical mm)",
        "splits": {},
    }
    for split in ("val", "test"):
        result["splits"][split] = evaluate_split(
            source, adapted, split, device, config["evaluation"]["retrieval_chunk_size"], output_dir
        )
        print(
            json.dumps({"completed_split": split, "samples": result["splits"][split]["samples"]}),
            flush=True,
        )
    (output_dir / "extended_evaluation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
