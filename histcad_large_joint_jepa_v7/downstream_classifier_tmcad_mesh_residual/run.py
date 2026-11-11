"""Entry point for the TMCAD frozen-control residual experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from data import ROOT, load_config
from experiment import evaluate, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--stage", choices=("train", "evaluate", "all"),
                        default="all")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage in {"train", "all"}:
        train(config)
    if args.stage in {"evaluate", "all"}:
        evaluate(config)


if __name__ == "__main__":
    main()
