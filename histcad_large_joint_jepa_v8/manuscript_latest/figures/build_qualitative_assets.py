"""Re-render manuscript examples from current v8 result rows and original STEP files."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import torch
from PIL import Image, ImageChops

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS = HERE / "qualitative_assets"
CLASSIFIER = ROOT / "downstream_classifier_tmcad_final"
sys.path.insert(0, str(CLASSIFIER))
from step_render import render_step  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def trim(source: Path, destination: Path) -> None:
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    difference = ImageChops.difference(image, Image.new("RGB", image.size, "white"))
    box = difference.getbbox()
    if box is None:
        raise ValueError(f"Blank STEP rendering: {source}")
    margin = 18
    box = (max(0, box[0] - margin), max(0, box[1] - margin),
           min(image.width, box[2] + margin), min(image.height, box[3] + margin))
    image.crop(box).save(destination, optimize=True)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "downstream_retrieval_adapter/outputs/qualitative_manifest.csv"
    representation_path = ROOT / "downstream_retrieval_adapter/outputs/baseline_representations.pt"
    predictions_path = CLASSIFIER / "outputs/default/test_predictions.csv"
    manifest = list(csv.DictReader(manifest_path.open(newline="", encoding="utf-8")))
    predictions = list(csv.DictReader(predictions_path.open(newline="", encoding="utf-8")))
    representations = torch.load(representation_path, map_location="cpu", weights_only=False,
                                 mmap=True)
    metadata = {row["uid"]: row for row in representations["metadata"]}
    records = []

    def produce(uid: str, path: Path, stem: str, result_source: dict) -> None:
        if not path.is_file():
            raise FileNotFoundError(path)
        output = ASSETS / f"{stem}.png"
        raw = ASSETS / f"{stem}_raw.png"
        audit = render_step(path, raw, image_size=1400,
                            max_triangles=30000, max_edges=350)
        first = Path(audit["views"][0])
        trim(first, output)
        records.append({"uid": uid, "source": str(path), "source_sha256": sha256(path),
                        "result_source": result_source, "asset": str(output),
                        "asset_sha256": sha256(output), "render": audit})
        print(f"Rendered {uid} -> {output.name}", flush=True)

    for case in ("3", "6"):
        rows = {row["representation"]: row for row in manifest
                if row["case"] == case and row["rank"] == "1"}
        direct, adapted = rows["Direct v7 joint"], rows["Adapted v7 joint"]
        for role, uid in (("query", direct["query_uid"]),
                          ("direct", direct["neighbor_uid"]),
                          ("adapted", adapted["neighbor_uid"])):
            produce(uid, Path(metadata[uid]["step_path"]), f"retrieval_{case}_{role}",
                    {"file": str(manifest_path), "case": case, "role": role,
                     "direct": direct, "adapted": adapted})

    for uid in ("coupling/337.stp", "pulley/530.stp", "coupling/257.stp"):
        row = next(item for item in predictions if item["uid"] == uid)
        produce(uid, Path(row["step_path"]), uid.replace("/", "_").replace(".stp", ""),
                {"file": str(predictions_path), "prediction": row})

    audit_path = ASSETS / "asset_provenance.json"
    audit_path.write_text(json.dumps({"selection": {
        "retrieval_cases": [3, 6],
        "classification_uids": ["coupling/337.stp", "pulley/530.stp", "coupling/257.stp"]},
        "records": records}, indent=2), encoding="utf-8")
    print(f"Audit: {audit_path}", flush=True)


if __name__ == "__main__":
    main()
