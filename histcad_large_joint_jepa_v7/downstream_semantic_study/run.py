"""Run the frozen-JEPA semantic supervision study."""

from __future__ import annotations

import argparse

import ablation
import annotation_transfer
import evaluate
import render
import train
import train_dimensions_only
import train_physical_sizes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        nargs="?",
        choices=(
            "all",
            "train",
            "evaluate",
            "transfer",
            "render",
            "dimensions",
            "physical",
            "ablation",
        ),
        default="all",
    )
    stage = parser.parse_args().stage
    if stage in ("all", "train"):
        train.main()
    if stage in ("all", "evaluate"):
        evaluate.main()
    if stage in ("all", "transfer"):
        annotation_transfer.main()
    if stage in ("all", "render"):
        render.main()
    if stage == "dimensions":
        train_dimensions_only.main()
    if stage == "physical":
        train_physical_sizes.main()
    if stage == "ablation":
        ablation.main()


if __name__ == "__main__":
    main()
