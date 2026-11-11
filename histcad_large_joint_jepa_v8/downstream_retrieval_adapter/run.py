from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(script: str, *arguments: str) -> None:
    subprocess.run([sys.executable, str(ROOT / script), *arguments], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train controlled retrieval baselines and the optional frozen-v7 adapter"
    )
    parser.add_argument("--skip-rendering", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--skip-adapter", action="store_true")
    args = parser.parse_args()
    run("train_baselines.py")
    if not args.skip_adapter:
        run("train_adapter.py")
    if args.train_only:
        return
    run("evaluate_adapter.py", *(["--exclude-adapter"] if args.skip_adapter else []))
    run("generate_benchmark_report.py")
    if not args.skip_rendering and not args.skip_adapter:
        run("render_adapter_examples.py")


if __name__ == "__main__":
    main()
