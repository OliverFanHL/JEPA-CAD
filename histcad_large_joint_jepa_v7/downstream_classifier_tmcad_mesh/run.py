"""Frozen JEPA-v7 TMCAD classifier with optional actual-surface profile."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import ROOT, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--stage", choices=("mesh", "train", "evaluate", "all"), default="all")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage in {"mesh", "all"}:
        from mesh_features import extract_mesh_profiles
        extract_mesh_profiles(config)
    if args.stage in {"train", "all"}:
        from train import train_models
        train_models(config)
    if args.stage in {"evaluate", "all"}:
        from evaluate import evaluate_models
        evaluate_models(config)


if __name__ == "__main__":
    main()
