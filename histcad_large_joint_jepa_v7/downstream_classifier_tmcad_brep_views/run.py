"""Entry point for the independent frozen two-view TMCAD experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import ROOT, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config_role_delta.yaml")
    parser.add_argument("--stage", choices=("extract", "train", "evaluate", "all"), default="all")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage in {"extract", "all"}:
        from extract import extract_features
        extract_features(config)
    if args.stage in {"train", "all"}:
        from train import train_models
        train_models(config)
    if args.stage in {"evaluate", "all"}:
        from evaluate import evaluate_models
        evaluate_models(config)


if __name__ == "__main__":
    main()
