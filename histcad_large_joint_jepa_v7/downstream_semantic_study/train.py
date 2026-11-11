"""Controlled semantic-only and semantic+geometry adapter training."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from model import SemanticAdapter
from torch.nn import functional as F

from data import config, geometry_targets, payload, prepare_text_targets, resolve, split_indices

PHYSICAL_SIZE_INDICES = (0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16)


def losses(
    output: dict[str, torch.Tensor],
    text: torch.Tensor,
    dimensions: torch.Tensor,
    feature: torch.Tensor,
    operation: torch.Tensor,
    settings: dict,
    mode: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    student = output["embedding"].float()
    target = text.float()
    target = F.normalize(target, dim=-1)
    semantic = (1.0 - (student * target).sum(-1)).mean()
    # The pairwise term transfers the teacher's neighborhood structure without
    # treating duplicate functional descriptions as false negatives.
    relational = F.mse_loss(student @ student.T, target @ target.T)
    objective = semantic + settings["semantic_pair_weight"] * relational
    terms = {"semantic_cosine": semantic, "semantic_relational": relational}
    if mode in (
        "semantic_plus_dimensions",
        "semantic_plus_geometry",
        "semantic_plus_physical_sizes",
    ):
        if mode == "semantic_plus_physical_sizes":
            dimension = F.smooth_l1_loss(
                output["dimension"][:, PHYSICAL_SIZE_INDICES].float(),
                dimensions[:, PHYSICAL_SIZE_INDICES].float(),
            )
        else:
            dimension = F.smooth_l1_loss(output["dimension"].float(), dimensions.float())
        objective = objective + settings["dimension_weight"] * dimension
        terms["dimension"] = dimension
    if mode == "semantic_plus_geometry":
        count = F.cross_entropy(output["feature_count"].float(), feature)
        boolean = F.cross_entropy(output["operation"].float(), operation)
        objective = (
            objective
            + settings["feature_count_weight"] * count
            + settings["operation_weight"] * boolean
        )
        terms.update(feature_count=count, operation=boolean)
    terms["loss"] = objective
    return objective, {name: float(value.detach()) for name, value in terms.items()}


@torch.inference_mode()
def validate(
    model: SemanticAdapter,
    source: torch.Tensor,
    text: torch.Tensor,
    dimensions: torch.Tensor,
    feature: torch.Tensor,
    operation: torch.Tensor,
    indices: torch.Tensor,
    settings: dict,
    mode: str,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    seen = 0
    for batch in indices.split(settings["batch_size"]):
        output = model(source[batch].to(device, non_blocking=True))
        _, row = losses(
            output,
            text[batch].to(device),
            dimensions[batch].to(device),
            feature[batch].to(device),
            operation[batch].to(device),
            settings,
            mode,
        )
        seen += len(batch)
        for name, value in row.items():
            totals[name] = totals.get(name, 0.0) + value * len(batch)
    return {name: value / seen for name, value in totals.items()}


@torch.inference_mode()
def extract(
    model: SemanticAdapter, source: torch.Tensor, indices: torch.Tensor, device: torch.device
) -> torch.Tensor:
    model.eval()
    outputs = []
    for batch in indices.split(1024):
        outputs.append(model(source[batch].to(device))["embedding"].float().cpu().half())
    return torch.cat(outputs)


def train_one(
    name: str,
    source: torch.Tensor,
    text: torch.Tensor,
    dimensions: torch.Tensor,
    feature: torch.Tensor,
    operation: torch.Tensor,
    splits: dict[str, torch.Tensor],
    settings: dict,
    output_dir: Path,
    device: torch.device,
) -> dict:
    if name not in (
        "semantic_only",
        "semantic_plus_dimensions",
        "semantic_plus_physical_sizes",
        "semantic_plus_geometry",
    ):
        raise ValueError(f"Unsupported training condition: {name}")
    torch.manual_seed(settings["seed"])
    model = SemanticAdapter(settings["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["training"]["learning_rate"],
        weight_decay=settings["training"]["weight_decay"],
    )
    train_indices = splits["train"][(text[splits["train"]].float().norm(dim=1) > 0)]
    val_indices = splits["val"][(text[splits["val"]].float().norm(dim=1) > 0)]
    generator = torch.Generator().manual_seed(settings["seed"])
    best_loss = float("inf")
    best_epoch = 0
    history = []
    checkpoint = output_dir / f"{name}.pt"
    for epoch in range(1, settings["training"]["epochs"] + 1):
        model.train()
        order = train_indices[torch.randperm(len(train_indices), generator=generator)]
        totals: dict[str, float] = {}
        for step, batch in enumerate(order.split(settings["training"]["batch_size"]), start=1):
            optimizer.zero_grad(set_to_none=True)
            output = model(source[batch].to(device, non_blocking=True))
            loss, row = losses(
                output,
                text[batch].to(device),
                dimensions[batch].to(device),
                feature[batch].to(device),
                operation[batch].to(device),
                settings["training"],
                name,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for key, value in row.items():
                totals[key] = totals.get(key, 0.0) + value * len(batch)
            if step % 100 == 0:
                print(
                    json.dumps(
                        {
                            "mode": name,
                            "epoch": epoch,
                            "step": step,
                            "steps": (len(order) + settings["training"]["batch_size"] - 1)
                            // settings["training"]["batch_size"],
                            "train_loss_so_far": round(
                                totals["loss"] / (step * settings["training"]["batch_size"]), 5
                            ),
                        }
                    ),
                    flush=True,
                )
        train_row = {key: value / len(order) for key, value in totals.items()}
        val_row = validate(
            model,
            source,
            text,
            dimensions,
            feature,
            operation,
            val_indices,
            settings["training"],
            name,
            device,
        )
        history.append({"epoch": epoch, "train": train_row, "val": val_row})
        print(json.dumps({"mode": name, **history[-1]}), flush=True)
        if val_row["loss"] < best_loss:
            best_loss, best_epoch = val_row["loss"], epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_config": settings["model"],
                    "mode": name,
                    "best_epoch": epoch,
                    "best_validation_loss": best_loss,
                    "jepa_encoder_frozen": True,
                    "text_teacher_frozen": True,
                },
                checkpoint,
            )
        if epoch - best_epoch >= settings["training"]["patience"]:
            break
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=False)["state_dict"]
    )
    return {
        "name": name,
        "checkpoint": str(checkpoint),
        "train_samples": len(train_indices),
        "validation_samples": len(val_indices),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "history": history,
        "embeddings": {
            split: extract(model, source, splits[split], device) for split in ("val", "test")
        },
    }


def main() -> None:
    settings = config()
    random.seed(settings["seed"])
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    torch.set_num_threads(8)
    output_dir = resolve(settings["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    source = payload(settings)
    splits = split_indices(source)
    teacher_path = prepare_text_targets(settings, source, splits)
    teacher = torch.load(teacher_path, map_location="cpu", mmap=True, weights_only=False)
    if teacher["uids"] != source["uids"]:
        raise ValueError("Text target cache UID mismatch")
    embedding = source["embeddings"]["joint"]
    dimensions, feature, operation = geometry_targets(source)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for name in ("semantic_only", "semantic_plus_geometry"):
        results[name] = train_one(
            name,
            embedding,
            teacher["targets"],
            dimensions,
            feature,
            operation,
            splits,
            settings,
            output_dir,
            device,
        )
    torch.save(
        {
            "uids": {
                split: [source["uids"][i] for i in splits[split].tolist()]
                for split in ("val", "test")
            },
            "embeddings": {name: row.pop("embeddings") for name, row in results.items()},
            "jepa_encoder_frozen": True,
            "text_teacher_frozen": True,
        },
        output_dir / "study_embeddings.pt",
    )
    (output_dir / "training_summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
