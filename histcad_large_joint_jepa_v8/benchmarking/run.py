from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.cad_benchmark import (  # noqa: E402
    classification,
    efficiency,
    frozen,
    inventory,
    retrieval,
    semantic,
)
from src.cad_benchmark.config import load_config  # noqa: E402
from src.cad_benchmark.registry import Registry  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-level CAD representation benchmark")
    parser.add_argument("stage", choices=("inventory", "validate", "frozen", "classification",
                                          "retrieval", "semantic", "efficiency", "all"),
                        nargs="?", default="all",
                        help="stage to run; defaults to all available benchmark stages")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--force", action="store_true", help="replace a completed task result")
    args = parser.parse_args()
    config = load_config(args.config)
    registry = Registry(config, apply_shared_coverage=True)
    if args.stage == "inventory":
        inventory.run(config, registry)
        return
    if args.stage == "validate":
        inventory.validate(config, registry)
        return
    stages = ("frozen", "classification", "retrieval", "semantic", "efficiency") if args.stage == "all" else (args.stage,)
    inventory.validate(config, registry)
    available = {stage: registry.names(stage) for stage in stages if stage != "efficiency"}
    missing_optional = [
        name for name, spec in registry.specs.items()
        if not spec.get("required", False) and not registry.available(name)
    ]
    print(json.dumps({
        "benchmark_stages": stages,
        "available_representations": available,
        "skipped_optional_representations": missing_optional,
        "shared_coverage": registry.coverage_summary(),
    }), flush=True)
    runners = {"frozen": frozen.run, "classification": classification.run,
               "retrieval": retrieval.run, "semantic": semantic.run}
    for stage in stages:
        if stage == "efficiency":
            efficiency.run(config, args.force)
        else:
            runners[stage](config, registry, args.force)


if __name__ == "__main__":
    main()
