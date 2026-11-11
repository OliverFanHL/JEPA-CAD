from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .io import atomic_json, write_csv
from .metrics import dimension_reference_neighbors, semantic_study_metrics
from .projection import FixedProjector
from .registry import Registry
from .training import device_for, seed_everything


class SemanticAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden: int, output_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(hidden, output_dim))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(values), dim=1)


@torch.inference_mode()
def predict(model: nn.Module, values: torch.Tensor, batch: int, device: torch.device) -> torch.Tensor:
    model.eval()
    return torch.cat([model(values[i:i + batch].to(device)).cpu() for i in range(0, len(values), batch)])


def train_one(values: torch.Tensor, teacher: torch.Tensor, split: dict[str, torch.Tensor],
              settings: dict[str, Any], seed: int) -> tuple[torch.Tensor, dict[str, Any]]:
    seed_everything(seed)
    device = device_for(settings["device"])
    model = SemanticAdapter(values.shape[1], settings["hidden_dim"], teacher.shape[1], settings["dropout"]).to(device)
    train_index = split["train"][teacher[split["train"]].norm(dim=1) > 0]
    val_index = split["val"][teacher[split["val"]].norm(dim=1) > 0]
    loader = DataLoader(TensorDataset(train_index), batch_size=settings["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    best, best_epoch, best_state, stale, history = float("inf"), 0, None, 0, []
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        total = 0.0
        batches = 0
        for (index,) in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(values[index].to(device))
            target = F.normalize(teacher[index].float().to(device), dim=1)
            cosine = (1 - (prediction * target).sum(1)).mean()
            relational = F.smooth_l1_loss(prediction @ prediction.T, target @ target.T)
            loss = cosine + settings["relational_weight"] * relational
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"])
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        val_prediction = predict(model, values[val_index], settings["batch_size"], device)
        val_target = F.normalize(teacher[val_index].float(), dim=1)
        val = float((1 - (val_prediction * val_target).sum(1)).mean())
        history.append({"epoch": epoch, "train_loss": total / batches, "validation_cosine_loss": val})
        if val < best - settings["minimum_improvement"]:
            best, best_epoch, best_state, stale = val, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= settings["patience"]:
            break
    if best_state is None:
        raise RuntimeError("No semantic adapter checkpoint selected")
    model.load_state_dict(best_state)
    all_values = predict(model, values, settings["batch_size"], device)
    return all_values, {"best_epoch": best_epoch, "best_validation_cosine_loss": best,
                        "trainable_parameters": sum(p.numel() for p in model.parameters()), "history": history}


def run(config: dict[str, Any], registry: Registry, force: bool = False) -> dict[str, Any]:
    settings = config["semantic_retrieval"]
    output = Path(config["paths"]["output_dir"]) / "semantic_retrieval" / "results.json"
    if output.is_file() and not force:
        return json.loads(output.read_text(encoding="utf-8"))
    registry.validate_alignment("semantic")
    names = registry.names("semantic")
    teacher_payload = torch.load(settings["teacher_targets"], map_location="cpu", mmap=True, weights_only=False)
    reference = registry.load(names[0])
    teacher_lookup = {uid: index for index, uid in enumerate(teacher_payload["uids"])}
    if len(teacher_lookup) != len(teacher_payload["uids"]) or any(
            uid not in teacher_lookup for uid in reference.uids):
        raise ValueError("Frozen semantic teacher targets do not cover the shared UID subset")
    teacher_indices = torch.tensor([teacher_lookup[uid] for uid in reference.uids], dtype=torch.long)
    teacher = teacher_payload["targets"][teacher_indices]
    split = {name: reference.indices(name) for name in ("train", "val", "test")}
    geometry_targets = reference.targets[settings["geometry_target"]][split["test"]].float()
    dimension_references = dimension_reference_neighbors(geometry_targets, settings["chunk_size"])
    missing_labels = set(settings["retrieval_labels"]) - reference.labels.keys()
    if missing_labels:
        raise ValueError(f"Canonical semantic reference lacks labels: {sorted(missing_labels)}")
    test_labels = {
        key: [values[int(index)] for index in split["test"]]
        for key, values in reference.labels.items()
        if key in settings["retrieval_labels"]
    }
    result: dict[str, Any] = {"protocol": {
        "teacher": "existing frozen all-MiniLM-L6-v2 targets; no teacher retraining",
        "supervision": "identical train-only text targets and optimizer budget",
        "evaluation": "same held-out queries/gallery and semantic labels for all representations",
        "selection": "validation cosine loss only",
    }, "representations": {}, "shared_coverage": registry.coverage_summary()}
    rows = []
    for name in names:
        representation = registry.load(name)
        projected = FixedProjector(representation.dim, settings["common_width"], config["seed"])(representation.values)
        semantic_values, training = train_one(projected, teacher, split, settings["training"], config["seed"])
        test = split["test"]
        direct = semantic_study_metrics(
            representation.values[test], geometry_targets, test_labels,
            dimension_references, settings["chunk_size"],
        )
        adapted = semantic_study_metrics(
            semantic_values[test], geometry_targets, test_labels,
            dimension_references, settings["chunk_size"],
        )
        result["representations"][name] = {"family": representation.family, "training": training,
                                            "direct": direct, "semantic_adapter": adapted}
        rows.append({"representation": name, "family": representation.family,
                     "direct_semantic_hit_at_1": direct[settings["primary_metric"]],
                     "adapted_semantic_hit_at_1": adapted[settings["primary_metric"]],
                     "semantic_gain_at_1": adapted[settings["primary_metric"]] - direct[settings["primary_metric"]],
                     "direct_dimension_recall_at_10": direct["dimension_recall_at_10"],
                     "adapted_dimension_recall_at_10": adapted["dimension_recall_at_10"],
                     "adapter_parameters": training["trainable_parameters"]})
        print(json.dumps({"stage": "semantic", "complete": name}), flush=True)
    atomic_json(output, result)
    write_csv(output.with_name("summary.csv"), rows)
    return result
