from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from src.histcad_large_jepa.cache import build_cache
from src.histcad_large_jepa.config import load_config
from src.histcad_large_jepa.evaluate import evaluate
from src.histcad_large_jepa.train import train


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the self-contained 4096-D HistCAD joint JEPA from scratch"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "full_dataset.yaml")
    parser.add_argument(
        "--cache-only", action="store_true", help="Audit and preprocess without training"
    )
    parser.add_argument("--rebuild-cache", action="store_true", help="Replace an existing v7 cache")
    parser.add_argument("--resume",type=Path, default=Path("outputs/full_dataset_4096/last.pt"), help="Resume a compatible v7 last.pt checkpoint")
    # parser.add_argument("--resume", type=Path, help="Resume a compatible v7 last.pt checkpoint")
    parser.add_argument(
        "--evaluate-only", action="store_true", help="Evaluate existing representations"
    )
    parser.add_argument(
        "--skip-evaluation", action="store_true", help="Stop after training and extraction"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    cache_path = Path(config["paths"]["cache_path"])
    if args.evaluate_only:
        print(json.dumps(evaluate(config), indent=2))
        return
    if args.rebuild_cache or not cache_path.exists():
        preprocessing = config["preprocessing"]
        build_cache(
            dataset_root=config["paths"]["dataset_root"],
            output_path=cache_path,
            max_step_instances=preprocessing["max_step_instances"],
            max_parametric_nodes=preprocessing["max_parametric_nodes"],
            validation_workers=preprocessing["validation_workers"],
            monitor_every_samples=preprocessing["monitor_every_samples"],
        )
    else:
        print(f"Using existing v7 cache: {cache_path.resolve()}")
    if args.cache_only:
        return
    report = train(config, args.resume)
    print(json.dumps(report, indent=2))
    if not args.skip_evaluation:
        print(json.dumps(evaluate(config), indent=2))


if __name__ == "__main__":
    main()
