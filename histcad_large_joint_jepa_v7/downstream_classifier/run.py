"""Reproducible frozen-v7 classification pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from extract import extract_best, load_frozen, resolve_paths
from labels import build_labels, save_labels
from train_classifier import train_all

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify defensible HistCAD labels using frozen JEPA-v7 vectors")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--stage", choices=("all", "labels", "train"), default="all")
    parser.add_argument("--extract", action="store_true", help="Re-extract frozen vectors from best.pt before the selected stage")
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = resolve_paths(settings)
    feature_path = extract_best(settings, paths) if args.extract else paths["representations"]
    source = load_frozen(feature_path, paths["checkpoint"], paths["graph_cache"])
    taxonomy = json.loads(paths["taxonomy"].read_text(encoding="utf-8"))
    targets, report = build_labels(source, taxonomy)
    save_labels(paths["output_dir"], source, targets, report)
    print("Labelled sample coverage:")
    for task, detail in report["tasks"].items():
        counts = detail["per_split"]
        print(f"  {task}: {len(detail['retained_classes'])} classes; train={counts['train']['labelled']:,} val={counts['val']['labelled']:,} test={counts['test']['labelled']:,}")
        if detail["dropped_rare_classes"]:
            print(f"    train-only rare-class exclusions: {detail['dropped_rare_classes']}")
    if args.stage == "labels":
        return
    train_all(settings, paths["output_dir"], source, targets, report)


if __name__ == "__main__":
    main()
