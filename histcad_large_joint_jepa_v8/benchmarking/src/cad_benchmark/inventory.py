from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .io import atomic_json, write_csv
from .registry import Registry


def run(config: dict[str, Any], registry: Registry) -> dict[str, Any]:
    rows = registry.inventory()
    artifacts = [
        {
            **spec,
            "artifact": str(spec["artifact"]),
            "available": Path(spec["artifact"]).is_file(),
        }
        for spec in config.get("required_artifacts", [])
    ]
    result = {
        "config": str(config["_config_path"]),
        "ready": sum(row["available"] for row in rows),
        "active_required": len(rows),
        "missing_required": [row["name"] for row in rows if row["required"] and not row["available"]],
        "required_artifacts": artifacts,
        "missing_required_artifacts": [row["name"] for row in artifacts if not row["available"]],
        "representations": rows,
        "shared_coverage": registry.coverage_summary(),
        "rule": "Existing compatible artifacts are reused. Every active representation is required for the complete benchmark; preparation commands are reported for missing artifacts.",
    }
    output = Path(config["paths"]["output_dir"]) / "inventory.json"
    atomic_json(output, result)
    write_csv(output.with_suffix(".csv"), rows)
    print(json.dumps({"inventory": str(output), "ready": result["ready"],
                      "active_required": result["active_required"]}), flush=True)
    return result


def validate(config: dict[str, Any], registry: Registry) -> dict[str, Any]:
    result = run(config, registry)
    if result["missing_required"] or result["missing_required_artifacts"]:
        representation_commands = {
            row["name"]: row["prepare"]
            for row in result["representations"]
            if row["name"] in result["missing_required"] and row.get("prepare")
        }
        artifact_commands = {
            row["name"]: row.get("prepare")
            for row in result["required_artifacts"]
            if not row["available"]
        }
        raise FileNotFoundError(
            f"Missing required representations: {result['missing_required']}; "
            f"missing required controls: {result['missing_required_artifacts']}. "
            f"Preparation commands: {representation_commands | artifact_commands}"
        )
    for task in ("frozen", "classification", "retrieval", "semantic"):
        if registry.names(task):
            registry.validate_alignment(task)
    return result
