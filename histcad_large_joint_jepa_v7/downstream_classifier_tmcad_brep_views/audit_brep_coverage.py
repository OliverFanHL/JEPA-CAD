"""Read-only, train-split audit of what larger STEP instance caps would retain.

This does not train/evaluate a classifier, change the v7 encoder, or rewrite
the existing TMCAD graph cache. It mirrors the v7 parser's candidate order.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
V7_ROOT = ROOT.parent
sys.path.insert(0, str(V7_ROOT))
from src.histcad_large_jepa.step_graph import (  # noqa: E402
    CURVE_MARKERS, INSTANCE_MARKERS, SURFACE_MARKERS,
)

ENTITY_NAME_RE = re.compile(r"#\d+\s*=\s*([A-Z][A-Z0-9_]*)\s*\(")


def priority(name: str) -> int:
    return next((position for position, marker in enumerate(INSTANCE_MARKERS)
                 if marker in name), len(INSTANCE_MARKERS))


def summarize(samples: list[dict], dataset_root: Path,
              caps: tuple[int, ...]) -> dict:
    values: dict[str, list[dict]] = defaultdict(list)
    for number, sample in enumerate(samples, start=1):
        text = (dataset_root / sample["source_relpath"]).read_text(
            encoding="latin-1", errors="ignore",
        )
        candidates = [name for match in ENTITY_NAME_RE.finditer(text)
                      if any(marker in (name := match.group(1)) for marker in INSTANCE_MARKERS)]
        candidates.sort(key=priority)
        record: dict = {"candidate_count": len(candidates)}
        for cap in caps:
            kept = candidates[:cap]
            record[str(cap)] = {
                "retained": len(kept),
                "surface": sum(any(marker in name for marker in SURFACE_MARKERS) for name in kept),
                "curve": sum(any(marker in name for marker in CURVE_MARKERS) for name in kept),
            }
        values[sample["label"]].append(record)
        if number % 100 == 0 or number == len(samples):
            print(f"Coverage audit {number:,}/{len(samples):,}", flush=True)
    result = {}
    for name, rows in values.items():
        result[name] = {"sampled_train_parts": len(rows),
                        "median_candidates": float(np.median([r["candidate_count"] for r in rows]))}
        for cap in caps:
            items = [r[str(cap)] for r in rows]
            result[name][str(cap)] = {
                "median_retained_fraction": float(np.median([
                    item["retained"] / max(1, row["candidate_count"])
                    for item, row in zip(items, rows, strict=True)])),
                "median_surface_instances": float(np.median([item["surface"] for item in items])),
                "median_curve_instances": float(np.median([item["curve"] for item in items])),
                "fraction_with_no_surface_instance": float(np.mean([item["surface"] == 0 for item in items])),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path,
                        default=V7_ROOT / "downstream_classifier_tmcad/data/processed/manifest.json")
    parser.add_argument("--sample-per-class", type=int, default=30)
    parser.add_argument("--seed", type=int, default=97)
    parser.add_argument("--save", type=Path, default=None)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    root = Path(manifest["dataset_root"])
    rng = random.Random(args.seed)
    selected = []
    for name in manifest["classes"]:
        own = [item for item in manifest["samples"]
               if item["split"] == "train" and item["label"] == name]
        selected.extend(rng.sample(own, min(len(own), args.sample_per_class)))
    result = summarize(selected, root, (256, 512, 1024))
    print(json.dumps(result, indent=2), flush=True)
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
