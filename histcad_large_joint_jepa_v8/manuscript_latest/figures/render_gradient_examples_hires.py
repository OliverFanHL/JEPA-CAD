"""Render the three Figure 6 examples from their original TMCAD STEP files."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from PIL import Image, ImageChops

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "downstream_classifier_tmcad_final"))
from step_render import render_step  # noqa: E402


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    outputs = []
    for relative in ("coupling/337.stp", "pulley/530.stp", "coupling/257.stp"):
        source = ROOT.parent / "TMCAD" / relative
        stem = relative.replace("/", "_").replace(".stp", "")
        raw = HERE / "qualitative_assets" / f"{stem}_hires_raw.png"
        target = HERE / "qualitative_assets" / f"{stem}_hires.png"
        record = render_step(source, raw, image_size=2400,
                             max_triangles=30000, max_edges=350)
        with Image.open(record["views"][0]) as opened:
            part = opened.convert("RGB")
        difference = ImageChops.difference(
            part, Image.new("RGB", part.size, "white"))
        box = difference.getbbox()
        if box is None:
            raise ValueError(f"Blank rendering: {source}")
        margin = 24
        part.crop((max(0, box[0] - margin), max(0, box[1] - margin),
                   min(part.width, box[2] + margin),
                   min(part.height, box[3] + margin))).save(target, optimize=True)
        outputs.append({"step": str(source), "step_sha256": digest(source),
                        "asset": str(target), "asset_sha256": digest(target),
                        "pixel_dimensions": Image.open(target).size,
                        "render": record})
        print(target, flush=True)
    (HERE / "qualitative_assets" / "gradient_hires_provenance.json").write_text(
        json.dumps(outputs, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
