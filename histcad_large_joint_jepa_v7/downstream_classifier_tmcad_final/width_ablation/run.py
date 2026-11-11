"""Run the optional 256-D versus 512-D classifier-hidden comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from ablation import evaluate, load_ablation, train
from visualize_width_ablation import visualize


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().parent / "config.yaml")
    parser.add_argument("--stage", choices=("train", "evaluate", "visualize", "all"),
                        default="all")
    args = parser.parse_args()
    base_config, ablation_config = load_ablation(args.config)
    if args.stage in {"train", "all"}:
        train(base_config, ablation_config)
    if args.stage in {"evaluate", "all"}:
        evaluate(base_config, ablation_config)
    if args.stage in {"visualize", "all"}:
        visualize(base_config, ablation_config)


if __name__ == "__main__":
    main()
