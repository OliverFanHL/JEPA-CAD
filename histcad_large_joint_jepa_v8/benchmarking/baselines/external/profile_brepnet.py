"""Profile saved BRepNet inference in the official Python 3.7 environment.

Uses processed solids and the pretrained checkpoint. It does not preprocess STEP
files, create embeddings, or train a model. Run this from benchmarking with
`conda run -n brepnet37 python baselines/external/profile_brepnet.py tmcad`,
then run: python run.py efficiency --force
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]


def move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [move(item, device) for item in value]
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("histcad", "tmcad"))
    parser.add_argument("--repo", type=Path, default=ROOT / "external_sources" / "BRepNet")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--profile", type=Path, default=ROOT / "outputs" / "efficiency" / "embedding_profile.json")
    args = parser.parse_args()
    if min(args.samples, args.batch_size, args.repeats) < 1 or args.warmup < 0:
        parser.error("samples, batch-size, repeats must be positive; warmup must be nonnegative")

    repository = args.repo.resolve()
    work_dir = (args.work_dir or ROOT / "external" / "work" / "brepnet" / args.dataset).resolve()
    checkpoint = (args.checkpoint or repository / "example_files" / "pretrained_models"
                  / "pretrained_s2.0.0_extended_step_uv_net_features_0816_183419.ckpt").resolve()
    working = work_dir / "steps" / "temp_working"
    dataset_file = working / "s2.0.0_step_all_features.json"
    manifest_file = work_dir / "staging_manifest.json"
    coverage_file = ROOT / "external" / ("brepnet_" + args.dataset + ".coverage.json")
    for path in (checkpoint, dataset_file, manifest_file, coverage_file):
        if not path.is_file():
            raise FileNotFoundError(str(path))

    # The official package uses paths relative to its repository.
    os.chdir(str(repository))
    sys.path.insert(0, str(repository))
    from eval.evaluate_folder import get_argument_parser
    from dataloaders.brepnet_dataset import BRepNetDataset, brepnet_collate_fn
    from models.brepnet import BRepNet

    feature_list = repository / "feature_lists" / "all.json"
    segment_names = repository / "example_files" / "pretrained_models" / "segment_names.json"
    opts = get_argument_parser().parse_args([
        "--dataset_dir", str(working), "--dataset_file", str(dataset_file),
        "--input_features", str(feature_list), "--segment_names", str(segment_names),
        "--batch_size", str(args.batch_size), "--num_workers", "0",
    ])
    data = BRepNetDataset(opts, "test_set")
    available = set(data.bodies)
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    omitted = set(json.loads(coverage_file.read_text(encoding="utf-8"))["omitted_uids"])
    stems = []
    for row in manifest["samples"]:
        stem = row["stem"]
        if (row["split"] == "test" and row["uid"] not in omitted and stem in available
                and (working / (stem + ".npz")).is_file()
                and (working / "embeddings" / (stem + ".embeddings")).is_file()):
            stems.append(stem)
            if len(stems) == args.samples:
                break
    if not stems:
        raise RuntimeError("No covered, processed test solids were found in " + str(working))
    data.bodies = stems
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=brepnet_collate_fn)
    batches = list(loader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    model = BRepNet.load_from_checkpoint(str(checkpoint), opts=opts)
    model.eval().to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def embed(batch):
        face = model.create_face_embeddings(
            batch["face_features"], batch["face_point_grids"],
            batch["edge_features"], batch["edge_point_grids"],
            batch["coedge_features"], batch["coedge_point_grids"],
            batch["face_kernel_tensor"], batch["edge_kernel_tensor"],
            batch["coedge_kernel_tensor"], batch["coedges_of_edges"],
            batch["coedges_of_small_faces"], batch["coedges_of_big_faces"],
        )
        # Match prepare_brepnet.py: mean+max pooling over each solid's faces.
        return [torch.cat((face[part["face_indices"]].mean(0),
                           face[part["face_indices"]].max(0)[0]))
                for part in batch["split_batch"]]

    with torch.no_grad():
        for index in range(args.warmup):
            embed(move(batches[index % len(batches)], device))
        synchronize()
        elapsed = 0.0
        measured_solids = 0
        for _ in range(args.repeats):
            for host in batches:
                batch = move(host, device)
                synchronize()
                started = time.perf_counter()
                embed(batch)
                synchronize()
                elapsed += time.perf_counter() - started
                measured_solids += len(host["file_stems"])

    profile_file = args.profile.resolve()
    profile = json.loads(profile_file.read_text(encoding="utf-8")) if profile_file.is_file() else {"methods": {}}
    method_key = "brepnet_official" if args.dataset == "histcad" else "brepnet_official_tmcad"
    profile.setdefault("methods", {})[method_key] = {
        "embedding_seconds_per_sample": elapsed / measured_solids,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
        "parameters": parameters,
        "device": str(device),
        "samples": len(stems),
        "batch_size": args.batch_size,
        "warmup_batches": args.warmup,
        "repeats": args.repeats,
        "dataset": args.dataset,
        "protocol": "official_encoder_plus_mean_max_face_pooling_v1",
        "excludes": ["STEP preprocessing", "checkpoint loading", "data collation", "host-to-device transfer"],
    }
    profile_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = profile_file.with_suffix(".json.partial")
    temporary.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    temporary.replace(profile_file)
    print(json.dumps({"profile": str(profile_file), "method": method_key,
                      "dataset": args.dataset, "samples": len(stems),
                      "embedding_seconds_per_sample": elapsed / measured_solids,
                      "parameters": parameters}), flush=True)


if __name__ == "__main__":
    main()
