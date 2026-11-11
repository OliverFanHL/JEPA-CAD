from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from .io import atomic_json, write_csv


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _epoch_metrics(path: Path) -> tuple[float | None, int | None, int | None]:
    if not path.is_file():
        return None, None, None
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows = [row for row in rows if row.get("event") == "epoch" and row.get("seconds")]
    if not rows:
        return None, None, None
    last_step = rows[-1].get("global_step")
    return sum(float(row["seconds"]) for row in rows), len(rows), int(last_step) if last_step else None


def run(config: dict[str, Any], force: bool = False) -> dict[str, Any]:
    output = Path(config["paths"]["output_dir"]) / "efficiency" / "results.json"
    if output.is_file() and not force:
        return _json(output)
    profile_path = output.with_name("embedding_profile.json")
    profiles = _json(profile_path).get("methods", {}) if profile_path.is_file() else {}
    result: dict[str, Any] = {"methods": {}, "measurement_policy": {
        "measured": "read directly from an existing checkpoint, log, or summary",
        "unavailable": "left null; no retraining or timing run was performed",
        "derived_optimizer_steps": "computed from completed full epochs, train rows, and DataLoader batch size",
        "embedding_cost": "model forward time on existing checkpoints; excludes loading, graph collation, and transfers",
        "estimated_times": "never substituted for a missing measurement",
    }}
    rows = []
    for name, spec in config["efficiency_sources"].items():
        summary_path = Path(spec["training_summary"]) if spec.get("training_summary") else None
        summary = _json(summary_path) if summary_path and summary_path.is_file() else {}
        seconds, logged_epochs, logged_steps = (
            _epoch_metrics(Path(spec["metrics_csv"])) if spec.get("metrics_csv")
            else (None, None, None)
        )
        if seconds is None:
            measured = summary.get("measured_training_seconds")
            history = summary.get("history", [])
            if measured is None and history and all(row.get("seconds") is not None for row in history):
                measured = sum(row["seconds"] for row in history)
            if isinstance(measured, (int, float)) and measured >= 0:
                seconds = float(measured)
        checkpoint = Path(spec["checkpoint"]) if spec.get("checkpoint") else None
        checkpoint_bytes = checkpoint.stat().st_size if checkpoint and checkpoint.is_file() else None
        parameters = None
        for key in spec.get("parameter_keys", []):
            if key in summary:
                parameters = summary[key] if parameters is None else parameters + summary[key]
        optimizer_steps = summary.get("optimizer_steps", logged_steps)
        steps_status = "logged" if optimizer_steps is not None else "unavailable"
        if optimizer_steps is None and spec.get("training_batch_size") and summary.get("train_samples"):
            epochs = summary.get("epochs_completed", logged_epochs)
            if epochs is not None:
                optimizer_steps = math.ceil(summary["train_samples"] / spec["training_batch_size"]) * epochs
                steps_status = "derived_from_full_epochs_and_batch_size"
        training_status = spec.get("training_status")
        if training_status == "no_training":
            seconds = 0.0
            optimizer_steps = 0
            steps_status = "not_applicable_no_training"
        profile = profiles.get(name, {})
        profile_seconds = profile.get("embedding_seconds_per_sample")
        profile_checkpoint = profile.get("checkpoint_bytes")
        profile_valid = (
            isinstance(profile_seconds, (int, float)) and profile_seconds >= 0
            and (checkpoint is None or (checkpoint.is_file() and (
                profile_checkpoint == checkpoint_bytes
                and profile.get("checkpoint_mtime_ns") == checkpoint.stat().st_mtime_ns
            )))
        )
        embedding_seconds = profile_seconds if profile_valid else summary.get("embedding_seconds_per_sample")
        embedding_status = (
            "measured_existing_checkpoint" if profile_valid and checkpoint else
            "measured_forward" if profile_valid else
            "measured" if embedding_seconds is not None else
            "stale_profile" if profile and checkpoint is not None else
            "requires_official_extractor_timing" if name == "brepnet_official" else "unavailable"
        )
        if parameters is None and profile_valid:
            parameters = profile.get("parameters")
        tmcad_profile = profiles.get("brepnet_official_tmcad", {}) if name == "brepnet_official" else {}
        tmcad_valid = (
            isinstance(tmcad_profile.get("embedding_seconds_per_sample"), (int, float))
            and tmcad_profile["embedding_seconds_per_sample"] >= 0
            and checkpoint is not None and checkpoint.is_file()
            and tmcad_profile.get("checkpoint_bytes") == checkpoint_bytes
            and tmcad_profile.get("checkpoint_mtime_ns") == checkpoint.stat().st_mtime_ns
        )
        if parameters is None and tmcad_valid:
            parameters = tmcad_profile.get("parameters")
        row = {
            "method": name,
            "representations": ",".join(spec.get("representations", [])),
            "training_seconds": seconds,
            "training_seconds_status": (
                "not_applicable_no_training" if training_status == "no_training" else
                "measured" if seconds is not None else training_status or "unavailable"
            ),
            "epochs": summary.get("epochs_completed", logged_epochs),
            "optimizer_steps": optimizer_steps,
            "optimizer_steps_status": steps_status,
            "parameters": parameters,
            "checkpoint_bytes": checkpoint_bytes,
            "embedding_seconds_per_sample": embedding_seconds,
            "embedding_cost_status": embedding_status,
            "embedding_device": profile.get("device") if profile_valid else None,
            "embedding_profile_rows": profile.get("samples") if profile_valid else None,
            "embedding_profile_batch_size": profile.get("batch_size") if profile_valid else None,
            "tmcad_embedding_seconds_per_sample": (
                tmcad_profile["embedding_seconds_per_sample"] if tmcad_valid else None
            ),
            "tmcad_embedding_cost_status": (
                "measured_existing_checkpoint" if tmcad_valid else
                "requires_official_extractor_timing" if name == "brepnet_official" else "not_profiled"
            ),
            "tmcad_embedding_profile_rows": tmcad_profile.get("samples") if tmcad_valid else None,
            "tmcad_embedding_profile_batch_size": tmcad_profile.get("batch_size") if tmcad_valid else None,
            "tmcad_embedding_device": tmcad_profile.get("device") if tmcad_valid else None,
            "source": str(summary_path) if summary_path else str(checkpoint) if checkpoint else "",
            "embedding_profile": str(profile_path) if profile_valid or tmcad_valid else "",
        }
        result["methods"][name] = row
        rows.append(row)
    atomic_json(output, result)
    write_csv(output.with_name("summary.csv"), rows)
    return result
