"""Paired seed replication of the existing frozen-feature classifiers.

This script makes no architecture, feature, or hyperparameter changes. Every
seed trains both arms, with validation-only checkpoint selection. Test metrics
are summarized afterward, never used to select a configuration.
"""

from __future__ import annotations

import argparse
import copy
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from common import ROOT, load_config, save_json
from evaluate import evaluate_models
from train import train_models


METRICS = (
    "val_macro_f1", "val_accuracy", "test_macro_f1",
    "test_accuracy", "test_balanced_accuracy", "test_top3_accuracy",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows to save: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary(values: list[float]) -> tuple[float, float, float, float]:
    data = np.asarray(values, dtype=np.float64)
    return (float(data.mean()), float(data.std(ddof=1)) if len(data) > 1 else 0.0,
            float(data.min()), float(data.max()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/repeated_seeds")
    parser.add_argument("--seeds", nargs="+", type=int, default=[97, 113, 127])
    args = parser.parse_args()
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Provide at least two distinct seeds")
    base = load_config(args.config)
    models = list(base["classifier"]["models"])
    if models != ["pooled_plus_brep_view_geometry",
                  "pooled_plus_brep_view_geometry_topology"]:
        raise ValueError("Replication must compare geometry control against topology arm")
    output = args.output.resolve()
    seed_dirs = {seed: output / f"seed_{seed}" for seed in args.seeds}
    # Do not overwrite any earlier run, including an interrupted one.
    existing = [str(path) for path in seed_dirs.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Seed output already exists; choose a new --output: {existing}")
    output.mkdir(parents=True, exist_ok=True)

    seed_rows: list[dict] = []
    class_rows: list[dict] = []
    for seed in args.seeds:
        config = copy.deepcopy(base)
        config["seed"] = seed
        config["paths"]["output"] = seed_dirs[seed]
        print(f"Paired seed {seed}: training {models[0]} and {models[1]}", flush=True)
        train_models(config)
        evaluate_models(config)
        comparison = _read_csv(seed_dirs[seed] / "model_comparison.csv")
        if [row["model"] for row in comparison] != models:
            raise ValueError(f"Unexpected comparison model order for seed {seed}")
        for row in comparison:
            seed_rows.append({"seed": seed, **row})
            for item in _read_csv(seed_dirs[seed] / row["model"] / "per_class_test.csv"):
                class_rows.append({"seed": seed, "model": row["model"], **item})

    _write_csv(output / "per_seed.csv", seed_rows)
    summary_rows = []
    for model in models:
        subset = [row for row in seed_rows if row["model"] == model]
        for metric in METRICS:
            mean, std, minimum, maximum = _summary([float(row[metric]) for row in subset])
            summary_rows.append({"model": model, "metric": metric, "seeds": len(subset),
                                 "mean": mean, "std": std,
                                 "min": minimum, "max": maximum})
    _write_csv(output / "summary.csv", summary_rows)

    paired_rows = []
    for seed in args.seeds:
        by_model = {row["model"]: row for row in seed_rows if row["seed"] == seed}
        for metric in METRICS:
            paired_rows.append({
                "seed": seed, "metric": metric,
                "topology_minus_geometry": (
                    float(by_model[models[1]][metric]) - float(by_model[models[0]][metric])
                ),
            })
    _write_csv(output / "paired_differences.csv", paired_rows)
    paired_summary = []
    for metric in METRICS:
        values = [float(row["topology_minus_geometry"]) for row in paired_rows
                  if row["metric"] == metric]
        mean, std, minimum, maximum = _summary(values)
        paired_summary.append({"metric": metric, "seeds": len(values),
                               "mean_difference": mean, "std_difference": std,
                               "min_difference": minimum, "max_difference": maximum,
                               "positive_seeds": sum(value > 0 for value in values)})
    _write_csv(output / "paired_summary.csv", paired_summary)

    by_class: dict[tuple[str, str], list[float]] = defaultdict(list)
    support: dict[str, int] = {}
    for row in class_rows:
        key = (row["model"], row["class"])
        by_class[key].append(float(row["f1"]))
        count = int(row["test_n"])
        if row["class"] in support and support[row["class"]] != count:
            raise ValueError("Test support changed between seeds")
        support[row["class"]] = count
    class_summary = []
    for model in models:
        for class_name in sorted(support):
            mean, std, minimum, maximum = _summary(by_class[(model, class_name)])
            class_summary.append({"model": model, "class": class_name,
                                  "test_n": support[class_name],
                                  "f1_mean": mean, "f1_std": std,
                                  "f1_min": minimum, "f1_max": maximum})
    _write_csv(output / "per_class_summary.csv", class_summary)
    save_json(output / "protocol.json", {
        "seeds": args.seeds, "models": models,
        "checkpoint_selection": "Validation macro-F1 independently within every seed",
        "comparison": "Paired differences use the same seed and frozen features in both arms",
        "test_policy": "Test metrics are reported only; never used to choose a model or setting",
        "configuration": str(args.config.resolve()),
    })
    for row in paired_summary:
        if row["metric"] in {"val_macro_f1", "test_macro_f1"}:
            print(f"{row['metric']}: mean topology-minus-geometry="
                  f"{row['mean_difference']:+.4f} across {row['seeds']} seeds", flush=True)
    print(f"Saved paired replication: {output}", flush=True)


if __name__ == "__main__":
    main()
