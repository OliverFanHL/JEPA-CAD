"""Materialize official BRepNet model-level embeddings for HistCAD or TMCAD.

The official repository performs raw-STEP preprocessing and exports one face-embedding matrix per
solid. This wrapper preserves the benchmark UID order, runs that official pipeline in its own Python
environment, and converts each face matrix to a fixed model vector using deterministic mean+max
pooling. It never trains BRepNet.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

from benchmarking.src.cad_benchmark.config import load_config  # noqa: E402
from benchmarking.src.cad_benchmark.registry import Registry  # noqa: E402


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_revision(repository: Path) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={repository}", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def checkpoint_tensor_count(path: Path) -> int:
    load_options: dict[str, Any] = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_options["weights_only"] = False
    payload = torch.load(path, **load_options)
    state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise ValueError(f"Cannot identify a state dictionary in {path}")
    return sum(value.numel() for value in state.values() if isinstance(value, torch.Tensor))


def source_rows(dataset: str, config: dict[str, Any]) -> tuple[list[dict[str, Any]], Any]:
    registry = Registry(config)
    if dataset == "histcad":
        reference = registry.load("jepa_joint")
        payload = torch.load(config["paths"]["jepa_representations"], map_location="cpu",
                             mmap=True, weights_only=False)
        rows = [
            {"uid": item["uid"], "split": item["split"], "step_path": item["step_path"]}
            for item in payload["metadata"]
        ]
        return rows, reference
    reference = registry.load("jepa_step_tmcad")
    manifest_path = Path(config["tmcad_supervised_from_scratch"]["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_root = Path(manifest["dataset_root"])
    rows = [
        {
            "uid": item["uid"],
            "split": item["split"],
            "step_path": str(dataset_root / item["source_relpath"]),
        }
        for item in manifest["samples"]
    ]
    return rows, reference


def stage_steps(rows: list[dict[str, Any]], step_dir: Path, link_mode: str) -> Path:
    step_dir.mkdir(parents=True, exist_ok=True)
    staged = []
    for index, row in enumerate(rows):
        source = Path(row["step_path"])
        if not source.is_file():
            raise FileNotFoundError(f"Missing STEP input for {row['uid']}: {source}")
        stem = f"{index:08d}"
        destination = step_dir / f"{stem}.step"
        if not destination.exists():
            if link_mode == "copy":
                shutil.copy2(source, destination)
            else:
                try:
                    os.link(source, destination)
                except OSError as error:
                    raise OSError(
                        f"Cannot hard-link {source} to {destination}; rerun with --link-mode copy"
                    ) from error
        staged.append({"stem": stem, "uid": row["uid"], "split": row["split"], "source": str(source)})
        if (index + 1) % 5000 == 0:
            print(json.dumps({"stage": "brepnet_inputs", "complete": index + 1, "total": len(rows)}), flush=True)
    manifest_path = step_dir.parent / "staging_manifest.json"
    manifest_path.write_text(json.dumps({"samples": staged}, indent=2), encoding="utf-8")
    return manifest_path


def run_official(
    external_python: Path,
    repository: Path,
    dataset: str,
    step_dir: Path,
    standardization: Path,
    checkpoint: Path,
    feature_list: Path,
    official_args: list[str],
    preprocess_workers: int,
    retry_failed: bool,
) -> None:
    command = [
        str(external_python),
        str(Path(__file__).resolve()),
        dataset,
        "--official-worker",
        "--repo", str(repository),
        "--step-dir", str(step_dir),
        "--standardization", str(standardization),
        "--checkpoint", str(checkpoint),
        "--feature-list", str(feature_list),
        "--preprocess-workers", str(preprocess_workers),
    ]
    for value in official_args:
        command.extend(("--official-arg", value))
    if retry_failed:
        command.append("--retry-failed")
    subprocess.run(command, check=True, cwd=repository)


def official_worker(args: argparse.Namespace) -> None:
    repository = args.repo.resolve()
    os.chdir(repository)
    sys.path.insert(0, str(repository))
    from eval.evaluate_folder import evaluate_folder  # noqa: PLC0415

    if args.preprocess_workers > 1 or args.retry_failed:
        from pipeline.extract_brepnet_data_from_step import extract_brepnet_data_from_step

        working_dir = args.step_dir.resolve() / "temp_working"
        working_dir.mkdir(parents=True, exist_ok=True)
        extract_brepnet_data_from_step(
            args.step_dir.resolve(), working_dir,
            feature_list_path=args.feature_list.resolve(),
            force_regeneration=False,
            num_workers=args.preprocess_workers,
            retry_failed=args.retry_failed,
        )

    evaluate_folder(
        args.step_dir.resolve(),
        args.standardization.resolve(),
        model=args.checkpoint.resolve(),
        input_features=args.feature_list.resolve(),
        extra_args=args.official_arg,
    )


def pool_face_embeddings(path: Path) -> torch.Tensor:
    values = np.loadtxt(path, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError(f"Invalid official BRepNet face embeddings: {path}")
    tensor = torch.from_numpy(values)
    return torch.cat((tensor.mean(0), tensor.amax(0)))


def collect(
    dataset: str,
    rows: list[dict[str, Any]],
    reference: Any,
    embeddings_dir: Path,
    output: Path,
    repository: Path,
    checkpoint: Path,
    allow_partial: bool = False,
) -> None:
    if reference.uids != [row["uid"] for row in rows]:
        raise ValueError("BRepNet source order is not aligned with the canonical reference")
    vectors, retained_indices, missing = [], [], []
    for index, row in enumerate(rows):
        path = embeddings_dir / f"{index:08d}.embeddings"
        if not path.is_file():
            item = {"uid": row["uid"], "expected": str(path)}
            failure_path = embeddings_dir.parent / "preprocessing_failures" / f"{index:08d}.json"
            if failure_path.is_file():
                failure = json.loads(failure_path.read_text(encoding="utf-8"))
                item["preprocessing_error"] = failure.get("error")
                item["failure_record"] = str(failure_path)
            missing.append(item)
            continue
        try:
            vector = pool_face_embeddings(path)
        except (OSError, ValueError) as error:
            missing.append({"uid": row["uid"], "expected": str(path),
                            "embedding_error": f"{type(error).__name__}: {error}"})
            continue
        vectors.append(vector)
        retained_indices.append(index)
    audit = output.with_suffix(".missing.json")
    if missing:
        audit.parent.mkdir(parents=True, exist_ok=True)
        audit.write_text(json.dumps({"missing_count": len(missing), "samples": missing}, indent=2), encoding="utf-8")
    elif audit.is_file():
        audit.unlink()
    if missing and not allow_partial:
        raise RuntimeError(
            f"Official BRepNet preprocessing/inference omitted {len(missing)} rows; "
            f"rerun with --allow-partial to use the shared subset. See {audit}"
        )
    if not vectors:
        raise RuntimeError("No BRepNet embeddings were produced")
    values = torch.stack(vectors).half()
    indices = torch.tensor(retained_indices, dtype=torch.long)
    coverage = {
        "dataset": dataset,
        "source_rows": len(rows),
        "included_rows": len(retained_indices),
        "omitted_rows": len(missing),
        "omitted_uids": [item["uid"] for item in missing],
        "artifact": str(output),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    torch.save({
        "format_version": "cad_benchmark_embeddings_v1",
        "uids": [reference.uids[index] for index in retained_indices],
        "splits": [reference.splits[index] for index in retained_indices],
        "labels": {key: [items[index] for index in retained_indices]
                   for key, items in reference.labels.items()},
        "targets": {key: tensor[indices] for key, tensor in reference.targets.items()},
        "embeddings": {"brepnet_official": values},
        "metadata": {
            "method": "BRepNet",
            "dataset": dataset,
            "source_repository": "https://github.com/AutodeskAILab/BRepNet",
            "source_revision": repository_revision(repository),
            "local_extractor_sha256": file_sha256(
                repository / "pipeline" / "extract_brepnet_data_from_step.py"
            ),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "checkpoint_tensor_parameter_count": checkpoint_tensor_count(checkpoint),
            "preprocessing": "official OpenCascade BRepNet evaluate_folder pipeline with local STEP location normalization",
            "pooling": "concatenated unweighted mean and elementwise max over official face embeddings",
            "training_data": "Fusion 360 Gallery segmentation s2.0.0",
            "pretraining_objective": "supervised face segmentation",
            "test_labels_used": False,
            "coverage": {key: value for key, value in coverage.items() if key != "omitted_uids"},
        },
    }, temporary)
    temporary.replace(output)
    coverage_path = output.with_suffix(".coverage.json")
    coverage_temporary = coverage_path.with_suffix(".json.partial")
    coverage_temporary.write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    coverage_temporary.replace(coverage_path)
    print(json.dumps({"saved": str(output), "rows": len(values),
                      "omitted": len(missing), "coverage": str(coverage_path),
                      "dimensions": values.shape[1]}), flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("dataset", choices=("histcad", "tmcad"))
    result.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    result.add_argument("--repo", type=Path, default=ROOT / "external_sources" / "BRepNet")
    result.add_argument("--external-python", type=Path, default=Path(sys.executable))
    result.add_argument("--work-dir", type=Path)
    result.add_argument("--output", type=Path)
    result.add_argument("--checkpoint", type=Path)
    result.add_argument("--standardization", type=Path)
    result.add_argument("--feature-list", type=Path)
    result.add_argument("--step-dir", type=Path, help=argparse.SUPPRESS)
    result.add_argument("--link-mode", choices=("hardlink", "copy"), default="hardlink")
    result.add_argument("--official-arg", action="append", default=[])
    result.add_argument("--preprocess-workers", type=int, default=1,
                        help="parallel STEP feature extraction processes (try 4)")
    result.add_argument("--retry-failed", action="store_true",
                        help="retry STEP files recorded as preprocessing failures")
    result.add_argument("--stage-only", action="store_true")
    result.add_argument("--collect-only", action="store_true")
    result.add_argument("--allow-partial", action="store_true",
                        help="save successful rows and evaluate all methods on their shared subset")
    result.add_argument("--official-worker", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--force", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.preprocess_workers < 1:
        raise ValueError("--preprocess-workers must be at least 1")
    if args.official_worker:
        official_worker(args)
        return
    config = load_config(args.config)
    repository = args.repo.resolve()
    if not repository.is_dir():
        raise FileNotFoundError(
            f"Official BRepNet checkout not found: {repository}. See external/README.md."
        )
    checkpoint = (args.checkpoint or repository / "example_files" / "pretrained_models"
                  / "pretrained_s2.0.0_extended_step_uv_net_features_0816_183419.ckpt").resolve()
    standardization = (args.standardization or repository / "example_files" / "feature_standardization"
                       / "s2.0.0_step_all_features.json").resolve()
    feature_list = (args.feature_list or repository / "feature_lists" / "all.json").resolve()
    for path in (checkpoint, standardization, feature_list):
        if not path.is_file():
            raise FileNotFoundError(path)
    work_dir = (args.work_dir or ROOT / "external" / "work" / "brepnet" / args.dataset).resolve()
    output = (args.output or ROOT / "external" / f"brepnet_{args.dataset}.pt").resolve()
    if output.is_file() and not args.force:
        print(f"Reusing existing BRepNet artifact: {output}")
        return
    rows, reference = source_rows(args.dataset, config)
    step_dir = work_dir / "steps"
    if not args.collect_only:
        stage_steps(rows, step_dir, args.link_mode)
        if args.stage_only:
            print(json.dumps({"staged": len(rows), "step_dir": str(step_dir)}), flush=True)
            return
        run_official(
            args.external_python.resolve(), repository, args.dataset, step_dir,
            standardization, checkpoint, feature_list, args.official_arg,
            args.preprocess_workers, args.retry_failed,
        )
    collect(
        args.dataset, rows, reference, step_dir / "temp_working" / "embeddings",
        output, repository, checkpoint, args.allow_partial,
    )


if __name__ == "__main__":
    main()
