from __future__ import annotations

import csv
import json
import math
import random
import shutil
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import PairedGraphDataset, collate_paired, make_node_mask, to_device
from .losses import training_loss
from .model import WideJointJEPA, parameter_report


class RunLogger:
    def __init__(self, path: Path, append: bool):
        self.path = path
        if not append:
            path.write_text("", encoding="utf-8")

    def write(self, message: str) -> None:
        timestamped = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {message}"
        print(timestamped, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(timestamped + "\n")


def _loader(
    dataset: PairedGraphDataset, settings: dict[str, Any], shuffle: bool, seed: int
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=settings["batch_size"],
        shuffle=shuffle,
        num_workers=settings["num_workers"],
        collate_fn=collate_paired,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=settings["device"] in {"auto", "cuda"} and torch.cuda.is_available(),
        persistent_workers=settings["num_workers"] > 0,
    )


def _device(name: str) -> torch.device:
    return (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if name == "auto"
        else torch.device(name)
    )


def _autocast_context(device: torch.device, settings: dict[str, Any]):
    if not settings.get("mixed_precision", False) or device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _momentum(step: int, total_steps: int, start: float, end: float) -> float:
    progress = step / max(1, total_steps - 1)
    return end - (end - start) * (math.cos(math.pi * progress) + 1.0) / 2.0


def _lr_multiplier(step: int, total_steps: int, warmup_steps: int, minimum_ratio: float) -> float:
    if warmup_steps and step < warmup_steps:
        return max(1e-6, (step + 1) / warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return minimum_ratio + (1.0 - minimum_ratio) * (math.cos(math.pi * progress) + 1.0) / 2.0


def _masked_forward(
    model: WideJointJEPA,
    host_batch: dict[str, Any],
    device: torch.device,
    mask_ratio: float,
    generator: torch.Generator,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    parametric_mask = make_node_mask(host_batch["parametric"], mask_ratio, generator)
    step_mask = make_node_mask(host_batch["step"], mask_ratio, generator)
    batch = to_device(host_batch, device)
    outputs = model(
        batch["parametric"], batch["step"], parametric_mask.to(device), step_mask.to(device)
    )
    return outputs, batch


@torch.inference_mode()
def validate(
    model: WideJointJEPA,
    loader: DataLoader,
    device: torch.device,
    settings: dict[str, Any],
    seed: int,
) -> dict[str, float]:
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    totals: dict[str, float] = {}
    predicted, target, samples = [], [], 0
    for host_batch in loader:
        with _autocast_context(device, settings):
            outputs, batch = _masked_forward(
                model, host_batch, device, settings["mask_ratio"], generator
            )
            _, metrics = training_loss(
                outputs,
                batch["dimension_target"],
                model.correlation_permutations(0),
                settings["temperature"],
                settings["correlation_block_size"],
                settings["loss_weights"],
            )
        count = len(batch["indices"])
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.0) + value * count
        samples += count
        predicted.append(outputs["dimension_prediction"].cpu())
        target.append(batch["dimension_target"].cpu())
    predicted_tensor, target_tensor = torch.cat(predicted), torch.cat(target)
    residual = (predicted_tensor - target_tensor).square().sum(0)
    total = (target_tensor - target_tensor.mean(0)).square().sum(0).clamp_min(1e-8)
    result = {name: value / samples for name, value in totals.items()}
    result["dimension_macro_r2"] = float((1.0 - residual / total).mean())
    result["dimension_standardized_mae"] = float((predicted_tensor - target_tensor).abs().mean())
    return result


@torch.inference_mode()
def extract_representations(
    model: WideJointJEPA,
    loaders: dict[str, DataLoader],
    payload: dict[str, Any],
    device: torch.device,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    output_path: Path,
    logger: RunLogger,
    storage_dtype: str,
    source_checkpoint: Path,
    source_checkpoint_epoch: int,
    model_config: dict[str, Any],
) -> None:
    model.eval()
    count = len(payload["uids"])
    dimensions = {
        "joint": model.dimension_head.in_features,
        "parametric": model.parametric_encoder.output_dim,
        "step": model.step_encoder.output_dim,
    }
    dtype = {"float16": torch.float16, "float32": torch.float32}.get(storage_dtype)
    if dtype is None:
        raise ValueError("representation_storage_dtype must be float16 or float32")
    embeddings = {
        name: torch.empty((count, width), dtype=dtype) for name, width in dimensions.items()
    }
    predictions = torch.empty((count, len(payload["dimension_target_names"])))
    for split, loader in loaders.items():
        completed = 0
        for host_batch in loader:
            batch = to_device(host_batch, device)
            outputs = model.embed(batch["parametric"], batch["step"])
            indices = batch["indices"].cpu()
            for name in embeddings:
                embeddings[name][indices] = outputs[name].cpu().to(dtype)
            predictions[indices] = outputs["dimension_prediction"].cpu()
            completed += len(indices)
        logger.write(f"Representation extraction: split={split} samples={completed:,}")
    torch.save(
        {
            "format_version": 7,
            "uids": payload["uids"],
            "metadata": payload["metadata"],
            "structural_metadata": payload["structural_metadata"],
            "dimension_targets": payload["dimension_targets"],
            "dimension_target_names": payload["dimension_target_names"],
            "dimension_target_mean": target_mean,
            "dimension_target_std": target_std,
            "dimension_predictions_standardized": predictions,
            "embedding_storage_dtype": storage_dtype,
            "source_checkpoint": str(source_checkpoint.resolve()),
            "source_checkpoint_name": source_checkpoint.name,
            "source_checkpoint_epoch": source_checkpoint_epoch,
            "model_config": model_config,
            "embeddings": embeddings,
        },
        output_path,
    )


def _checkpoint(
    model: WideJointJEPA,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_loss: float,
    stale: int,
    history: list[dict[str, float]],
    model_config: dict[str, Any],
    payload: dict[str, Any],
    train_dataset: PairedGraphDataset,
    mask_generator: torch.Generator,
    train_loader: DataLoader,
) -> dict[str, Any]:
    return {
        "format_version": 7,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "gradient_scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_validation_loss": best_loss,
        "stale_epochs": stale,
        "history": history,
        "model_config": model_config,
        "step_entity_types": payload["step_entity_types"],
        "dimension_target_mean": train_dataset.target_mean,
        "dimension_target_std": train_dataset.target_std,
        "mask_generator_state": mask_generator.get_state(),
        "train_loader_generator_state": train_loader.generator.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def train(config: dict[str, Any], resume_path: str | Path | None = None) -> dict[str, Any]:
    seed, settings = config["seed"], config["training"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _device(settings["device"])
    payload = torch.load(
        config["paths"]["cache_path"], map_location="cpu", weights_only=False, mmap=True
    )
    train_dataset = PairedGraphDataset(payload, "train")
    val_dataset = PairedGraphDataset(
        payload, "val", train_dataset.target_mean, train_dataset.target_std
    )
    test_dataset = PairedGraphDataset(
        payload, "test", train_dataset.target_mean, train_dataset.target_std
    )
    loaders = {
        "train": _loader(train_dataset, settings, True, seed),
        "val": _loader(val_dataset, settings, False, seed),
        "test": _loader(test_dataset, settings, False, seed),
    }
    model_config = dict(config["model"])
    model_config["correlation_permutations"] = settings["correlation_permutations"]
    model = WideJointJEPA(
        len(payload["step_entity_types"]), model_config, len(payload["dimension_target_names"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    accumulation_steps = settings["gradient_accumulation_steps"]
    updates_per_epoch = math.ceil(len(loaders["train"]) / accumulation_steps)
    total_steps = settings["epochs"] * updates_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _lr_multiplier(
            step,
            total_steps,
            settings["warmup_steps"],
            settings["minimum_learning_rate"] / settings["learning_rate"],
        ),
    )
    float16_scaling = (
        settings.get("mixed_precision", False)
        and device.type == "cuda"
        and not torch.cuda.is_bf16_supported()
    )
    scaler = torch.amp.GradScaler("cuda", enabled=float16_scaling)
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config["config_path"], output_dir / "config_used.yaml")
    resuming = resume_path is not None
    logger = RunLogger(output_dir / "training.log", append=resuming)
    progress_path = output_dir / "training_progress.jsonl"
    if not resuming:
        progress_path.write_text("", encoding="utf-8")
    mask_generator = torch.Generator().manual_seed(seed + 10)
    start_epoch, global_step, best_loss, stale, history = 1, 0, float("inf"), 0, []
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        if checkpoint.get("format_version") != 7 or checkpoint["model_config"] != model_config:
            raise ValueError("Resume checkpoint is not compatible with this v7 model configuration")
        if tuple(checkpoint["step_entity_types"]) != tuple(payload["step_entity_types"]):
            raise ValueError("Resume checkpoint STEP vocabulary does not match the current cache")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("gradient_scaler"):
            scaler.load_state_dict(checkpoint["gradient_scaler"])
        start_epoch, global_step = checkpoint["epoch"] + 1, checkpoint["global_step"]
        best_loss, stale, history = (
            checkpoint["best_validation_loss"],
            checkpoint["stale_epochs"],
            checkpoint["history"],
        )
        mask_generator.set_state(checkpoint["mask_generator_state"])
        loaders["train"].generator.set_state(checkpoint["train_loader_generator_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        random.setstate(checkpoint["python_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        if torch.cuda.is_available() and checkpoint["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        logger.write(f"Resumed checkpoint {Path(resume_path).resolve()} at epoch {start_epoch}")

    audit = payload["dataset_audit"]
    report = parameter_report(model)
    logger.write(
        f"Device={device} train={len(train_dataset):,} val={len(val_dataset):,} test={len(test_dataset):,} "
        f"excluded={audit['excluded_samples']:,} trainable_parameters={report['online_trainable_parameters']:,}"
    )
    logger.write(
        f"Optimization: batch={settings['batch_size']} accumulation={accumulation_steps} "
        f"effective_batch={settings['batch_size'] * accumulation_steps} "
        f"mixed_precision={settings.get('mixed_precision', False) and device.type == 'cuda'}"
    )
    logger.write(
        f"Dimensions: parametric={report['parametric_output_dim']} STEP={report['step_output_dim']} "
        f"concatenated={report['joint_input_dim']} hidden={model_config['joint_hidden_dim']} "
        f"final={report['embedding_dim']}"
    )
    steps_per_epoch, monitor_every = len(loaders["train"]), settings["monitor_every_steps"]
    for epoch in range(start_epoch, settings["epochs"] + 1):
        model.train()
        totals: dict[str, float] = {}
        rolling: dict[str, float] = {}
        samples = rolling_steps = 0
        started = interval_started = time.time()
        optimizer.zero_grad(set_to_none=True)
        for epoch_step, host_batch in enumerate(loaders["train"], start=1):
            with _autocast_context(device, settings):
                outputs, batch = _masked_forward(
                    model, host_batch, device, settings["mask_ratio"], mask_generator
                )
                loss, metrics = training_loss(
                    outputs,
                    batch["dimension_target"],
                    model.correlation_permutations(global_step),
                    settings["temperature"],
                    settings["correlation_block_size"],
                    settings["loss_weights"],
                )
                remainder = steps_per_epoch % accumulation_steps
                final_group_size = remainder or accumulation_steps
                in_final_group = epoch_step > steps_per_epoch - final_group_size
                divisor = final_group_size if in_final_group else accumulation_steps
                backward_loss = loss / divisor
            scaler.scale(backward_loss).backward()
            should_update = epoch_step % accumulation_steps == 0 or epoch_step == steps_per_epoch
            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                model.update_targets(
                    _momentum(global_step, total_steps, settings["ema_start"], settings["ema_end"])
                )
                global_step += 1
            count = len(batch["indices"])
            samples += count
            rolling_steps += 1
            for name, value in metrics.items():
                totals[name] = totals.get(name, 0.0) + value * count
                rolling[name] = rolling.get(name, 0.0) + value
            if epoch_step % monitor_every == 0 or epoch_step == steps_per_epoch:
                average = {name: value / rolling_steps for name, value in rolling.items()}
                lr = optimizer.param_groups[0]["lr"]
                logger.write(
                    f"epoch={epoch:03d}/{settings['epochs']:03d} step={epoch_step:05d}/{steps_per_epoch:05d} "
                    f"loss={average['loss']:.4f} joint={average['joint_jepa']:.4f} "
                    f"param={average['parametric_jepa']:.4f} step_jepa={average['step_jepa']:.4f} "
                    f"dim={average['dimension']:.4f} slots={average['slot_diversity']:.4f} "
                    f"lr={lr:.3e} interval={time.time() - interval_started:.1f}s"
                )
                step_row = {
                    "event": "step",
                    "epoch": epoch,
                    "epoch_step": epoch_step,
                    "global_step": global_step,
                    "learning_rate": lr,
                    **average,
                }
                with progress_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(step_row) + "\n")
                rolling, rolling_steps, interval_started = {}, 0, time.time()
        train_metrics = {name: value / samples for name, value in totals.items()}
        val_metrics = validate(model, loaders["val"], device, settings, seed + 1000 + epoch)
        row = {
            "event": "epoch",
            "epoch": epoch,
            "global_step": global_step,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **{f"train_{name}": value for name, value in train_metrics.items()},
            **{f"val_{name}": value for name, value in val_metrics.items()},
            "seconds": time.time() - started,
        }
        history.append(row)
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        improved = val_metrics["loss"] < best_loss - settings["minimum_validation_improvement"]
        if improved:
            best_loss, stale = val_metrics["loss"], 0
        else:
            stale += 1
        state = _checkpoint(
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            global_step,
            best_loss,
            stale,
            history,
            model_config,
            payload,
            train_dataset,
            mask_generator,
            loaders["train"],
        )
        torch.save(state, output_dir / "last.pt")
        if improved:
            torch.save(state, output_dir / "best.pt")
        logger.write(
            f"epoch={epoch:03d} complete train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_joint={val_metrics['joint_jepa']:.4f} "
            f"val_param={val_metrics['parametric_jepa']:.4f} val_step={val_metrics['step_jepa']:.4f} "
            f"val_dim={val_metrics['dimension']:.4f} val_R2={val_metrics['dimension_macro_r2']:.4f} "
            f"best={best_loss:.4f} checkpoint={'best+last' if improved else 'last'}"
        )
        if stale >= settings["patience"]:
            logger.write(
                f"Early stopping after {stale} validation epochs without sufficient improvement"
            )
            break

    best_checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    test_metrics = validate(model, loaders["test"], device, settings, seed + 2000)
    report = parameter_report(model)
    report.update(
        {
            "train_samples": len(train_dataset),
            "validation_samples": len(val_dataset),
            "test_samples": len(test_dataset),
            "excluded_samples": audit["excluded_samples"],
            "epochs_completed": len(history),
            "best_epoch": best_checkpoint["epoch"],
            "best_validation_loss": best_loss,
            "test_metrics": test_metrics,
            "model_selection": "minimum validation objective; test evaluated only after selection",
            "effective_batch_size": settings["batch_size"]
            * settings["gradient_accumulation_steps"],
            "representation_audit": payload.get("representation_audit", {}),
            "model_config": model_config,
            "training_config": settings,
        }
    )
    (output_dir / "training_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    with (output_dir / "training_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    extraction_loaders = {
        "train": _loader(train_dataset, settings, False, seed),
        "val": loaders["val"],
        "test": loaders["test"],
    }
    extract_representations(
        model,
        extraction_loaders,
        payload,
        device,
        train_dataset.target_mean,
        train_dataset.target_std,
        output_dir / "representations.pt",
        logger,
        settings["representation_storage_dtype"],
        output_dir / "best.pt",
        best_checkpoint["epoch"],
        model_config,
    )
    logger.write(
        f"Training complete: best_epoch={best_checkpoint['epoch']} test_loss={test_metrics['loss']:.4f}"
    )
    return report
