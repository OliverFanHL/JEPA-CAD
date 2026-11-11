"""Render same-query JCPR, autoencoder, and BRepNet HistCAD neighbors."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from PIL import Image, ImageChops

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BENCH = ROOT / "benchmarking"
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(ROOT / "downstream_classifier_tmcad_final"))
from src.cad_benchmark.config import load_config  # noqa: E402
from src.cad_benchmark.registry import Registry  # noqa: E402
from step_render import render_step  # noqa: E402

QUERIES = ("0040/00402080", "0048/00480829", "0081/00813092",
           "0009/00097910", "0041/00412375", "0046/00462789",
           "0091/00910640", "0007/00079665")
METHODS = {
    "JCPR": "jepa_joint",
    "Graph autoencoder": "graph_autoencoder_joint",
    "BRepNet transfer": "brepnet_official_histcad",
}


def checksum(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    torch.set_num_threads(4)
    registry = Registry(load_config(BENCH / "config.yaml"), apply_shared_coverage=True)
    reps = {key: registry.load(name) for key, name in METHODS.items()}
    reference = reps["JCPR"]
    for rep in reps.values():
        assert rep.uids == reference.uids and rep.splits == reference.splits
    indices = reference.indices("test")
    positions = {uid: i for i, uid in enumerate(reference.uids)}
    source = torch.load(ROOT / "outputs/full_dataset_4096/representations.pt",
                        map_location="cpu", weights_only=False, mmap=True)
    meta = {row["uid"]: row for row in source["metadata"]}
    assets = HERE / "qualitative_assets"
    assets.mkdir(exist_ok=True)
    cases = []
    files = {}
    for case_number, query in enumerate(QUERIES, 1):
        assert query in positions and reference.splits[positions[query]] == "test"
        qi = positions[query]
        case = {"query_uid": query, "candidates": {}}
        for method, rep in reps.items():
            values = F.normalize(rep.values[indices].float(), dim=1)
            query_vector = F.normalize(rep.values[qi:qi+1].float(), dim=1)
            cosine = (query_vector @ values.T).squeeze(0)
            cosine[indices == qi] = -torch.inf
            picked = int(indices[int(cosine.argmax())])
            neighbor = rep.uids[picked]
            target = reference.targets["dimensions"]
            error = float((target[qi] - target[picked]).abs().mean())
            case["candidates"][method] = {
                "neighbor_uid": neighbor, "standardized_dimensional_mae": error,
                "cosine_similarity": float(cosine.max()),
            }
        cases.append(case)
        print("case", case_number, case, flush=True)
    for i, case in enumerate(cases, 1):
        uids = {"query": case["query_uid"]}
        uids.update({method.lower().replace(" ", "_"): value["neighbor_uid"]
                     for method, value in case["candidates"].items()})
        case["assets"] = {}
        for role, uid in uids.items():
            path = Path(meta[uid]["step_path"])
            target = assets / f"matched_retrieval_{i}_{role}.png"
            raw = assets / f"matched_retrieval_{i}_{role}_raw.png"
            for render_size in (1500, 2200, 3000):
                audit = render_step(path, raw, image_size=render_size,
                                    max_triangles=35000, max_edges=400)
                with Image.open(audit["views"][0]) as opened:
                    image = opened.convert("RGB")
                box = ImageChops.difference(image, Image.new("RGB", image.size, "white")).getbbox()
                if box is None:
                    raise RuntimeError(f"Blank rendering: {path}")
                pad = 22
                box = (max(0, box[0]-pad), max(0, box[1]-pad),
                       min(image.width, box[2]+pad), min(image.height, box[3]+pad))
                cropped = image.crop(box)
                if min(cropped.size) >= 200:
                    break
            cropped.save(target, optimize=True)
            case["assets"][role] = {
                "uid": uid, "step_path": str(path), "step_sha256": checksum(path),
                "image": str(target), "image_sha256": checksum(target),
                "render": audit,
            }
            print("rendered", role, uid, flush=True)
    output = HERE / "matched_retrieval_cases.json"
    output.write_text(json.dumps({
        "protocol": "same held-out HistCAD test bank, direct cosine, self excluded",
        "representation_sources": {m: str(registry.specs[n]["artifact"])
                                   for m, n in METHODS.items()},
        "cases": cases,
    }, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
