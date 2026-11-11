from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML and resolve every configured path relative to the project root."""
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    project_root = config_path.parent.parent
    config["config_path"] = config_path
    config["project_root"] = project_root
    for name, value in config["paths"].items():
        candidate = Path(value)
        config["paths"][name] = candidate if candidate.is_absolute() else project_root / candidate
    model = config["model"]
    expected_parametric = (model["parametric_pooling_slots"] + 1) * model["parametric_node_dim"]
    expected_step = (model["step_pooling_slots"] + 1) * model["step_node_dim"]
    if model["parametric_output_dim"] != expected_parametric:
        raise ValueError(
            f"parametric_output_dim must preserve pooled source features: expected {expected_parametric}"
        )
    if model["step_output_dim"] != expected_step:
        raise ValueError(
            f"step_output_dim must preserve pooled source features: expected {expected_step}"
        )
    training = config["training"]
    positive_training_keys = (
        "monitor_every_steps",
        "batch_size",
        "gradient_accumulation_steps",
        "epochs",
    )
    if any(training[name] < 1 for name in positive_training_keys):
        raise ValueError(f"{', '.join(positive_training_keys)} must all be positive")
    if not 0 < training["minimum_learning_rate"] <= training["learning_rate"]:
        raise ValueError("Learning rates must satisfy 0 < minimum_learning_rate <= learning_rate")
    if config["preprocessing"]["monitor_every_samples"] < 1:
        raise ValueError("preprocessing.monitor_every_samples must be positive")
    if model["embedding_dim"] != 4096:
        raise ValueError("This experiment is defined around a 4096-D final embedding")
    if model["parametric_output_dim"] + model["step_output_dim"] <= model["embedding_dim"]:
        raise ValueError("Source branches must jointly exceed the final 4096-D capacity")
    baselines = config["evaluation"].get("optional_baselines", {})
    config["evaluation"]["optional_baselines"] = {
        name: candidate if (candidate := Path(value)).is_absolute() else project_root / candidate
        for name, value in baselines.items()
    }
    return config


def require_keys(config: dict[str, Any], section: str, names: tuple[str, ...]) -> None:
    missing = [name for name in names if name not in config[section]]
    if missing:
        raise KeyError(f"Missing {section} configuration keys: {', '.join(missing)}")
