"""Single entry point for the STEP-only frozen JEPA-v7 TMCAD experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import ROOT, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--stage", choices=("prepare", "extract", "train", "evaluate", "all"), default="all")
    parser.add_argument("--skip-render", action="store_true", help="Skip optional OpenCASCADE qualitative sheets")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage in ("prepare", "all"):
        from prepare import prepare_dataset
        prepare_dataset(config)
    if args.stage in ("extract", "all"):
        from extract import extract_features
        extract_features(config)
    if args.stage in ("train", "all"):
        from train import train_classifiers
        train_classifiers(config)
    if args.stage in ("evaluate", "all"):
        from evaluate import evaluate_classifiers
        evaluate_classifiers(config)
        if config["qualitative"]["render_after_evaluation"] and not args.skip_render:
            for name in config["classifier"]["models"]:
                try:
                    from render_cases import render_model
                    render_model(config, name)
                except (ImportError, RuntimeError) as error:
                    print(f"Optional STEP rendering unavailable or incomplete for {name}: {error}. "
                          "Quantitative results and case CSV remain saved.", flush=True)


if __name__ == "__main__":
    main()
