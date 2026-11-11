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
from .metrics import controlled_retrieval_metrics, dimension_reference_neighbors
from .projection import FixedProjector
from .registry import Registry
from .training import device_for, seed_everything


class RetrievalAdapter(nn.Module):
    def __init__(self, width: int, hidden: int, target_dim: int, dropout: float):
        super().__init__()
        self.body = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, width),
        )
        self.gate = nn.Parameter(torch.tensor(-2.0))
        self.target_head = nn.Linear(width, target_dim)

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(values + self.gate.sigmoid() * self.body(values), dim=1)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode(values)
        return encoded, self.target_head(encoded)


def approximate_positives(targets: torch.Tensor, projections: int, window: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    vectors = [torch.eye(targets.shape[1])[i] for i in range(min(3, targets.shape[1]))]
    vectors += [torch.randn(targets.shape[1], generator=generator) for _ in range(projections)]
    best_score = torch.full((len(targets),), torch.inf)
    best_index = torch.arange(len(targets))
    for vector in vectors:
        order = torch.argsort(targets @ vector)
        for offset in range(1, window + 1):
            left, right = order[:-offset], order[offset:]
            score = (targets[left] - targets[right]).abs().mean(1)
            improve = score < best_score[left]
            best_score[left[improve]] = score[improve]
            best_index[left[improve]] = right[improve]
            improve = score < best_score[right]
            best_score[right[improve]] = score[improve]
            best_index[right[improve]] = left[improve]
    return best_index


def negatives(count: int, positives: torch.Tensor, number: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    result = torch.randint(count, (count, number), generator=generator)
    rows = torch.arange(count)[:, None]
    invalid = (result == rows) | (result == positives[:, None])
    while invalid.any():
        result[invalid] = torch.randint(count, (int(invalid.sum()),), generator=generator)
        invalid = (result == rows) | (result == positives[:, None])
    return result


def loss_batch(model: RetrievalAdapter, values: torch.Tensor, targets: torch.Tensor,
               positive: torch.Tensor, negative: torch.Tensor, index: torch.Tensor,
               settings: dict[str, Any], device: torch.device) -> torch.Tensor:
    p, n = positive[index], negative[index]
    query = values[index].to(device)
    combined = torch.cat((query, values[p].to(device), values[n.flatten()].to(device)))
    encoded = model.encode(combined)
    count, negative_count = len(index), n.shape[1]
    anchor, pos = encoded[:count], encoded[count:2 * count]
    neg = encoded[2 * count:].reshape(count, negative_count, -1)
    logits = torch.cat(((anchor * pos).sum(1, keepdim=True), (anchor[:, None] * neg).sum(2)), 1)
    contrastive = F.cross_entropy(logits / settings["temperature"], torch.zeros(count, dtype=torch.long, device=device))
    prediction = model.target_head(anchor)
    regression = F.smooth_l1_loss(prediction, targets[index].to(device))
    anchoring = (1 - (anchor * query).sum(1)).mean()
    return (settings["contrastive_weight"] * contrastive + settings["target_weight"] * regression
            + settings["anchor_weight"] * anchoring)


@torch.inference_mode()
def validation_loss(model: RetrievalAdapter, values: torch.Tensor, targets: torch.Tensor,
                    positive: torch.Tensor, negative: torch.Tensor,
                    settings: dict[str, Any], device: torch.device) -> float:
    model.eval()
    total = 0.0
    batches = 0
    for (index,) in DataLoader(TensorDataset(torch.arange(len(values))), batch_size=settings["batch_size"]):
        total += float(loss_batch(model, values, targets, positive, negative, index, settings, device))
        batches += 1
    return total / batches


@torch.inference_mode()
def encode(model: RetrievalAdapter, values: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    model.eval()
    return torch.cat([model.encode(values[i:i + batch_size].to(device)).cpu() for i in range(0, len(values), batch_size)])


def train_one(values: torch.Tensor, targets: torch.Tensor, split: dict[str, torch.Tensor],
              settings: dict[str, Any], seed: int) -> tuple[RetrievalAdapter, dict[str, Any]]:
    seed_everything(seed)
    device = device_for(settings["device"])
    model = RetrievalAdapter(values.shape[1], settings["hidden_dim"], targets.shape[1], settings["dropout"]).to(device)
    train_values, val_values = values[split["train"]], values[split["val"]]
    train_targets, val_targets = targets[split["train"]], targets[split["val"]]
    train_positive = approximate_positives(train_targets, settings["positive_projections"], settings["positive_window"], seed)
    val_positive = approximate_positives(val_targets, settings["positive_projections"], settings["positive_window"], seed + 1)
    train_negative = negatives(len(train_values), train_positive, settings["negatives_per_query"], seed + 2)
    val_negative = negatives(len(val_values), val_positive, settings["negatives_per_query"], seed + 3)
    loader = DataLoader(TensorDataset(torch.arange(len(train_values))), batch_size=settings["batch_size"],
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    best, best_epoch, best_state, stale, history = float("inf"), 0, None, 0, []
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        total = 0.0
        batches = 0
        for (index,) in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_batch(model, train_values, train_targets, train_positive, train_negative, index, settings, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"])
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        val = validation_loss(model, val_values, val_targets, val_positive, val_negative, settings, device)
        history.append({"epoch": epoch, "train_loss": total / batches, "validation_loss": val})
        if val < best - settings["minimum_improvement"]:
            best, best_epoch, best_state, stale = val, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= settings["patience"]:
            break
    if best_state is None:
        raise RuntimeError("No retrieval adapter checkpoint selected")
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_validation_loss": best,
                   "trainable_parameters": sum(p.numel() for p in model.parameters()), "history": history}


def run(config: dict[str, Any], registry: Registry, force: bool = False) -> dict[str, Any]:
    settings = config["retrieval_adaptation"]
    output = Path(config["paths"]["output_dir"]) / "retrieval_adaptation" / "results.json"
    if output.is_file() and not force:
        return json.loads(output.read_text(encoding="utf-8"))
    registry.validate_alignment("retrieval")
    names = registry.names("retrieval")
    reference = registry.load(names[0])
    targets = reference.targets[settings["target"]].float()
    split = {name: reference.indices(name) for name in ("train", "val", "test")}
    test_targets = targets[split["test"]]
    dimension_references = dimension_reference_neighbors(test_targets, settings["chunk_size"])
    missing_labels = set(settings["retrieval_labels"]) - reference.labels.keys()
    if missing_labels:
        raise ValueError(f"Canonical retrieval reference lacks labels: {sorted(missing_labels)}")
    test_labels = {
        key: [values[int(index)] for index in split["test"]]
        for key, values in reference.labels.items()
        if key in settings["retrieval_labels"]
    }
    result: dict[str, Any] = {"protocol": {
        "supervision": "same standardized geometry targets and approximate positive-pair construction",
        "capacity": "deterministic frozen projection followed by an identical residual adapter",
        "selection": "minimum validation objective; test untouched until selection",
    }, "representations": {}, "shared_coverage": registry.coverage_summary()}
    rows = []
    for name in names:
        representation = registry.load(name)
        projected = FixedProjector(representation.dim, settings["common_width"], config["seed"])(representation.values)
        model, training = train_one(projected, targets, split, settings["training"], config["seed"])
        adapted_test = encode(model, projected[split["test"]], settings["training"]["batch_size"], device_for(settings["training"]["device"]))
        direct = controlled_retrieval_metrics(
            representation.values[split["test"]], test_targets, test_labels,
            dimension_references, settings["chunk_size"],
        )
        adapted = controlled_retrieval_metrics(
            adapted_test, test_targets, test_labels, dimension_references, settings["chunk_size"]
        )
        row = {"family": representation.family, "training": training, "direct": direct, "adapted": adapted,
               "adaptation_gain": {
                   "dimension_mae_at_1": (
                       direct["dimension_retrieval"]["mae_at_1"]
                       - adapted["dimension_retrieval"]["mae_at_1"]
                   ),
                   "dimension_recall_at_10": (
                       adapted["dimension_neighbor_retrieval"]["recall_at_10"]
                       - direct["dimension_neighbor_retrieval"]["recall_at_10"]
                   ),
                   "canonical_function_hit_at_1": (
                       adapted["canonical_functional_type_evaluation_only"]["hit_at_1"]
                       - direct["canonical_functional_type_evaluation_only"]["hit_at_1"]
                   ),
               }}
        result["representations"][name] = row
        rows.append({"representation": name, "family": representation.family,
                     "direct_dimension_mae_at_1": direct["dimension_retrieval"]["mae_at_1"],
                     "adapted_dimension_mae_at_1": adapted["dimension_retrieval"]["mae_at_1"],
                     "dimension_mae_improvement": row["adaptation_gain"]["dimension_mae_at_1"],
                     "direct_dimension_recall_at_10": direct["dimension_neighbor_retrieval"]["recall_at_10"],
                     "adapted_dimension_recall_at_10": adapted["dimension_neighbor_retrieval"]["recall_at_10"],
                     "adapter_parameters": training["trainable_parameters"]})
        print(json.dumps({"stage": "retrieval", "complete": name}), flush=True)
    atomic_json(output, result)
    write_csv(output.with_name("summary.csv"), rows)
    return result
