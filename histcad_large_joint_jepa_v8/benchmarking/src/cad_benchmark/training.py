from __future__ import annotations

import copy
import random
from collections import Counter
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .metrics import classification_metrics


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for(value: str) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if value == "auto" else torch.device(value)


class Probe(nn.Module):
    def __init__(self, width: int, classes: int, kind: str, hidden: int, dropout: float):
        super().__init__()
        if kind == "linear":
            self.network = nn.Linear(width, classes)
        elif kind == "adapter":
            self.network = nn.Sequential(
                nn.LayerNorm(width), nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden, classes),
            )
        else:
            raise ValueError(kind)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


@torch.inference_mode()
def infer(model: nn.Module, values: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    model.eval()
    output = []
    for start in range(0, len(values), batch_size):
        output.append(model(values[start:start + batch_size].to(device)).cpu())
    return torch.cat(output)


def train_probe(
    values: torch.Tensor,
    target: torch.Tensor,
    split_indices: dict[str, torch.Tensor],
    classes: int,
    kind: str,
    settings: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    seed_everything(seed)
    device = device_for(settings["device"])
    model = Probe(values.shape[1], classes, kind, settings["hidden_dim"], settings["dropout"]).to(device)
    train_index = split_indices["train"]
    counts = torch.bincount(target[train_index], minlength=classes).float().clamp_min(1)
    weights = counts.sum() / (classes * counts)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    loader = DataLoader(
        TensorDataset(values[train_index], target[train_index]), batch_size=settings["batch_size"],
        shuffle=True, generator=torch.Generator().manual_seed(seed),
    )
    best_state, best_score, best_epoch, stale, history = None, -float("inf"), 0, 0, []
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        total, count = 0.0, 0
        for batch_values, batch_target in loader:
            batch_values, batch_target = batch_values.to(device), batch_target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_values)
            loss = F.cross_entropy(logits, batch_target, weight=weights.to(device), label_smoothing=settings["label_smoothing"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"])
            optimizer.step()
            total += float(loss.detach()) * len(batch_target)
            count += len(batch_target)
        val_logits = infer(model, values[split_indices["val"]], settings["batch_size"], device)
        val_metrics = classification_metrics(val_logits, target[split_indices["val"]], classes)
        history.append({"epoch": epoch, "train_loss": total / count, "validation_macro_f1": val_metrics["macro_f1"]})
        if val_metrics["macro_f1"] > best_score + settings["minimum_improvement"]:
            best_score, best_epoch, stale = val_metrics["macro_f1"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= settings["patience"]:
            break
    if best_state is None:
        raise RuntimeError("Probe training did not produce a checkpoint")
    model.load_state_dict(best_state)
    logits = {split: infer(model, values[index], settings["batch_size"], device) for split, index in split_indices.items()}
    result = {
        "kind": kind,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_score,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
        "validation": classification_metrics(logits["val"], target[split_indices["val"]], classes),
        "test": classification_metrics(logits["test"], target[split_indices["test"]], classes),
    }
    return result, logits


def knn_classification(
    values: torch.Tensor, target: torch.Tensor, split_indices: dict[str, torch.Tensor], k: int, chunk_size: int
) -> dict[str, Any]:
    train = F.normalize(values[split_indices["train"]].float(), dim=1)
    train_target = target[split_indices["train"]]
    classes = int(target.max()) + 1
    output: dict[str, Any] = {}
    for split in ("val", "test"):
        query = F.normalize(values[split_indices[split]].float(), dim=1)
        predictions = []
        for start in range(0, len(query), chunk_size):
            neighbors = (query[start:start + chunk_size] @ train.T).topk(k, dim=1).indices
            votes = F.one_hot(train_target[neighbors], classes).float().sum(1)
            predictions.append(votes)
        output[split] = classification_metrics(torch.cat(predictions), target[split_indices[split]], classes)
    return output


def encode_labels(labels: list[Any], splits: list[str], minimum_train_count: int = 2) -> tuple[torch.Tensor, list[str]]:
    counts = Counter(value for value, split in zip(labels, splits, strict=True) if split == "train" and value not in (None, ""))
    classes = [str(value) for value, count in counts.most_common() if count >= minimum_train_count]
    mapping = {value: index for index, value in enumerate(classes)}
    target = torch.tensor([mapping.get(str(value), -1) for value in labels])
    return target, classes
