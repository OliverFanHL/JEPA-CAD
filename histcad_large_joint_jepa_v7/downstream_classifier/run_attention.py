"""Train and evaluate a role-token attention probe on frozen JEPA-v7 outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from attention_classifier import ParametricResidualAttentionClassifier, RoleTokenAttentionClassifier, TOKEN_NAMES
from extract import load_frozen, resolve_paths
from labels import TASKS, build_labels, load_taxonomy, save_labels
from train_classifier import _class_weights, _device, _metrics, _split_indices

ROOT = Path(__file__).resolve().parent
NAMES = ("parametric", "step", "joint")


def _batch(source: dict, indices: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(source["embeddings"][name][indices].float().to(device, non_blocking=True) for name in NAMES)


def _autocast(device: torch.device, enabled: bool):
    if device.type != "cuda" or not enabled:
        return nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _weighted_loss(logits: dict[str, torch.Tensor], labels: torch.Tensor,
                   weights: dict[str, torch.Tensor], functional_label_smoothing: float = 0.0,
                   parametric_auxiliary_weight: float = 0.0) -> torch.Tensor:
    losses = []
    for column, task in enumerate(TASKS):
        valid = labels[:, column] >= 0
        if valid.any():
            smoothing = functional_label_smoothing if task == "functional_form" else 0.0
            losses.append(F.cross_entropy(logits[task][valid].float(), labels[valid, column],
                                          weight=weights[task], label_smoothing=smoothing))
    objective = torch.stack(losses).mean()
    valid_form = labels[:, 0] >= 0
    if parametric_auxiliary_weight and valid_form.any():
        objective = objective + parametric_auxiliary_weight * F.cross_entropy(
            logits["functional_form_parametric"][valid_form].float(), labels[valid_form, 0],
            weight=weights["functional_form"], label_smoothing=functional_label_smoothing)
    return objective


@torch.inference_mode()
def evaluate(model: nn.Module, source: dict, targets: torch.Tensor, indices: torch.Tensor,
             report: dict, device: torch.device, batch_size: int, weights: dict[str, torch.Tensor],
             mixed_precision: bool, details: bool = False,
             functional_label_smoothing: float = 0.0,
             parametric_auxiliary_weight: float = 0.0) -> tuple[dict, dict | None]:
    model.eval()
    predictions: dict[str, list[np.ndarray]] = {task: [] for task in TASKS}
    probabilities: list[np.ndarray] = []
    attention: list[np.ndarray] = []
    loss_total, examples = 0.0, 0
    for start in range(0, len(indices), batch_size):
        rows = indices[start:start + batch_size]
        batch = _batch(source, rows, device)
        labels = targets[rows].to(device)
        with _autocast(device, mixed_precision):
            logits, weights_by_token = model(*batch, return_attention=details)
            loss = _weighted_loss(logits, labels, weights, functional_label_smoothing,
                                  parametric_auxiliary_weight)
        loss_total += float(loss.item()) * len(rows)
        examples += len(rows)
        for task in TASKS:
            predictions[task].append(logits[task].argmax(-1).cpu().numpy())
        if details:
            probabilities.append(logits["functional_form"].float().softmax(-1).max(-1).values.cpu().numpy())
            attention.append(weights_by_token.float().cpu().numpy())
    predicted = {task: np.concatenate(chunks) for task, chunks in predictions.items()}
    metrics = _metrics(predicted, targets, indices, report)
    metrics["weighted_loss"] = loss_total / examples
    if not details:
        return metrics, None
    return metrics, {"predictions": predicted, "form_confidence": np.concatenate(probabilities),
                     "query_attention": np.concatenate(attention)}


def _qualitative(output: Path, source: dict, targets: torch.Tensor, indices: torch.Tensor,
                 detail: dict, report: dict) -> None:
    """Deterministic representative correct and mistaken part-form examples."""
    names = report["tasks"]["functional_form"]["retained_classes"]
    parents = report["tasks"]["functional_form"]["parent_by_class"]
    truth = targets[indices, 0].numpy()
    pred = detail["predictions"]["functional_form"]
    confidence = detail["form_confidence"]
    candidates = {"correct": [], "within_parent_error": [], "cross_parent_error": []}
    for local, (actual, forecast) in enumerate(zip(truth, pred, strict=True)):
        if actual < 0:
            continue
        bucket = ("correct" if actual == forecast else "within_parent_error"
                  if parents[names[actual]] == parents[names[forecast]] else "cross_parent_error")
        candidates[bucket].append(local)
    chosen: list[tuple[str, int]] = []
    # Rotate classes so qualitative cases are not dominated by plates/blocks.
    for bucket in candidates:
        ordered = sorted(candidates[bucket], key=lambda i: (-confidence[i], source["uids"][int(indices[i])]))
        seen: set[int] = set()
        for local in ordered:
            actual = int(truth[local])
            if actual not in seen:
                chosen.append((bucket, local))
                seen.add(actual)
            if len(seen) >= 6:
                break
    with (output / "qualitative_cases.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("case_type", "uid", "functional_description", "true_form", "predicted_form",
                         "true_parent", "predicted_parent", "prediction_confidence",
                         "top_non_query_attention_role", "step_path"))
        for bucket, local in chosen:
            index = int(indices[local])
            actual, forecast = names[int(truth[local])], names[int(pred[local])]
            weights = detail["query_attention"][local].copy()
            weights[0] = 0.0
            writer.writerow((bucket, source["uids"][index], source["metadata"][index].get("functional_type", ""),
                             actual, forecast, parents[actual], parents[forecast], float(confidence[local]),
                             TOKEN_NAMES[int(weights.argmax())], source["metadata"][index]["step_path"]))
    attention_mean = detail["query_attention"].mean(0)
    (output / "attention_diagnostic.json").write_text(json.dumps({
        "warning": "Final-layer mean query attention is a diagnostic distribution, not a causal feature attribution.",
        "roles": list(TOKEN_NAMES),
        "mean_query_attention": {name: float(value) for name, value in zip(TOKEN_NAMES, attention_mean)},
        "qualitative_cases": len(chosen),
    }, indent=2), encoding="utf-8")


def _comparison(output: Path, attention_result: dict, baseline_path: Path) -> None:
    if not baseline_path.is_file():
        return
    baseline_mapping_path = baseline_path.parent / "label_mappings.json"
    if not baseline_mapping_path.is_file():
        return
    baseline_taxonomy = json.loads(baseline_mapping_path.read_text(encoding="utf-8"))["taxonomy_version"]
    if baseline_taxonomy != attention_result["protocol"]["taxonomy_version"]:
        return  # Never label cross-taxonomy scores as a direct comparison.
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["representations"]
    rows = []
    for name, run in [("role_token_attention", attention_result), *baseline.items()]:
        for task in TASKS:
            test = run["test"][task]
            rows.append({"model": name, "task": task, "test_accuracy": test["accuracy"],
                         "test_balanced_accuracy": test["balanced_accuracy"], "test_macro_f1": test["macro_f1"],
                         "parent_macro_f1": test.get("parent_macro_f1", ""),
                         "validation_macro_f1": run["validation"][task]["macro_f1"]})
    with (output / "comparison_to_existing_probes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="32-epoch frozen JEPA-v7 role-token attention classification")
    parser.add_argument("--config", type=Path, default=ROOT / "config_attention.yaml")
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths = resolve_paths(settings)
    output = paths["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    source = load_frozen(paths["representations"], paths["checkpoint"], paths["graph_cache"])
    taxonomy = load_taxonomy(paths["taxonomy"])
    targets, report = build_labels(source, taxonomy)
    baseline_mapping_path = ROOT / "outputs" / "label_mappings.json"
    if baseline_mapping_path.is_file():
        baseline_mapping = json.loads(baseline_mapping_path.read_text(encoding="utf-8"))
        for task in TASKS:
            if baseline_mapping["tasks"][task]["retained_classes"] != report["tasks"][task]["retained_classes"]:
                raise ValueError("Attention taxonomy differs from the original classification baseline")
    save_labels(output, source, targets, report)
    seed = settings["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _device(settings["training"]["device"])
    model_settings = settings["model"]
    model_class = (ParametricResidualAttentionClassifier if model_settings.get("parametric_form_residual", False)
                   else RoleTokenAttentionClassifier)
    model = model_class(
        report, width=model_settings["token_width"], heads=model_settings["attention_heads"],
        depth=model_settings["attention_layers"], feedforward=model_settings["feedforward_width"],
        dropout=model_settings["dropout"],
        **({"parametric_hidden": model_settings.get("parametric_hidden", 384),
            "parametric_dropout": model_settings.get("parametric_dropout", 0.3)}
           if model_settings.get("parametric_form_residual", False) else {}),
    ).to(device)
    train_settings = settings["training"]
    indices = _split_indices(source)
    weights = _class_weights(targets, indices["train"], report, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_settings["learning_rate"], weight_decay=train_settings["weight_decay"])
    float16 = device.type == "cuda" and train_settings["mixed_precision"] and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=float16)
    generator = torch.Generator().manual_seed(seed)
    max_epochs, batch_size = train_settings["epochs"], train_settings["batch_size"]
    if max_epochs != 32:
        print(f"Configured epoch budget: {max_epochs} (default experiment uses 32)")
    history, best_score, best_epoch, stale = [], -math.inf, 0, 0
    checkpoint_path = output / "best_attention.pt"
    print(f"Device={device} frozen_v7_epoch={source['source_checkpoint_epoch']} "
          f"train={len(indices['train']):,} val={len(indices['val']):,} test={len(indices['test']):,} "
          f"trainable_probe_parameters={sum(p.numel() for p in model.parameters()):,}", flush=True)
    for epoch in range(1, max_epochs + 1):
        progress = (epoch - 1) / max(1, max_epochs - 1)
        minimum, peak = train_settings["minimum_learning_rate"], train_settings["learning_rate"]
        lr = peak if epoch <= train_settings["warmup_epochs"] else minimum + (peak - minimum) * (1 + math.cos(math.pi * progress)) / 2
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        shuffled = indices["train"][torch.randperm(len(indices["train"]), generator=generator)]
        loss_sum, count = 0.0, 0
        for start in range(0, len(shuffled), batch_size):
            rows = shuffled[start:start + batch_size]
            batch = _batch(source, rows, device)
            labels = targets[rows].to(device)
            with _autocast(device, train_settings["mixed_precision"]):
                logits, _ = model(*batch)
                loss = _weighted_loss(logits, labels, weights, train_settings.get("functional_label_smoothing", 0.0),
                                      train_settings.get("parametric_auxiliary_weight", 0.0))
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), train_settings["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.item()) * len(rows)
            count += len(rows)
        val, _ = evaluate(model, source, targets, indices["val"], report, device, batch_size,
                          weights, train_settings["mixed_precision"],
                          functional_label_smoothing=train_settings.get("functional_label_smoothing", 0.0),
                          parametric_auxiliary_weight=train_settings.get("parametric_auxiliary_weight", 0.0))
        train_loss = loss_sum / count
        score = val["mean_macro_f1"]
        row = {"epoch": epoch, "learning_rate": lr, "train_weighted_loss": train_loss,
               "validation_weighted_loss": val["weighted_loss"], "validation_mean_macro_f1": score,
               **{f"validation_{task}_accuracy": val[task]["accuracy"] for task in TASKS},
               **{f"validation_{task}_macro_f1": val[task]["macro_f1"] for task in TASKS}}
        history.append(row)
        print(f"epoch={epoch:02d}/{max_epochs} train_loss={train_loss:.4f} val_loss={val['weighted_loss']:.4f} "
              f"val_macro_F1={score:.4f} form_F1={val['functional_form']['macro_f1']:.4f} "
              f"history_F1={val['modeling_sequence']['macro_f1']:.4f} lr={lr:.2e}", flush=True)
        if score > best_score + train_settings.get("early_stopping_min_delta", 1e-5):
            best_score, best_epoch, stale = score, epoch, 0
            torch.save({"model": model.state_dict(), "model_config": model_settings,
                        "epoch": epoch, "validation_mean_macro_f1": score,
                        "source_checkpoint": source["source_checkpoint"],
                        "taxonomy_version": report["taxonomy_version"],
                        "class_names": {task: report["tasks"][task]["retained_classes"] for task in TASKS}}, checkpoint_path)
        else:
            stale += 1
        patience = train_settings.get("early_stopping_patience")
        if patience is not None and epoch >= train_settings.get("early_stopping_min_epochs", 1) and stale >= patience:
            print(f"Early stopping at epoch {epoch}; selected epoch {best_epoch} "
                  f"(no validation macro-F1 gain >= {train_settings.get('early_stopping_min_delta', 1e-5):.4f} "
                  f"for {patience} epochs)", flush=True)
            break
    with (output / "training_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    best = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["model"])
    model.to(device)
    train_metrics, _ = evaluate(model, source, targets, indices["train"], report, device, batch_size, weights,
                                train_settings["mixed_precision"], functional_label_smoothing=train_settings.get("functional_label_smoothing", 0.0),
                                parametric_auxiliary_weight=train_settings.get("parametric_auxiliary_weight", 0.0))
    val_metrics, _ = evaluate(model, source, targets, indices["val"], report, device, batch_size, weights,
                              train_settings["mixed_precision"], functional_label_smoothing=train_settings.get("functional_label_smoothing", 0.0),
                              parametric_auxiliary_weight=train_settings.get("parametric_auxiliary_weight", 0.0))
    # The held-out test set is evaluated once, after validation-based selection.
    test_metrics, detail = evaluate(model, source, targets, indices["test"], report, device, batch_size, weights,
                                    train_settings["mixed_precision"], details=True,
                                    functional_label_smoothing=train_settings.get("functional_label_smoothing", 0.0),
                                    parametric_auxiliary_weight=train_settings.get("parametric_auxiliary_weight", 0.0))
    np.savez_compressed(output / "test_predictions.npz", uids=np.asarray([source["uids"][i] for i in indices["test"].tolist()]),
                        true=targets[indices["test"]].numpy(),
                        **{f"pred_{task}": detail["predictions"][task] for task in TASKS})
    _qualitative(output, source, targets, indices["test"], detail, report)
    result = {"protocol": {"frozen_checkpoint": source["source_checkpoint"],
                           "taxonomy_version": report["taxonomy_version"],
                           "split": "same v7 UID train/val/test and same taxonomy as linear/MLP probes",
                           "selection": "validation mean macro-F1 with configured minimum improvement and optional patience",
                           "test_use": "evaluated once after checkpoint selection",
                           "model": "10 semantically named frozen v7 summary tokens plus learned query token"
                                    + (" and direct parametric form residual" if model_settings.get("parametric_form_residual") else "")},
              "best_epoch": best_epoch, "best_validation_mean_macro_f1": best_score,
              "train": train_metrics, "validation": val_metrics, "test": test_metrics,
              "history": history, "model_config": model_settings, "training_config": train_settings}
    (output / "attention_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _comparison(output, result, ROOT / "outputs" / "classification_results.json")
    print(f"Selected epoch {best_epoch}; completed {len(history)} epochs; test mean macro-F1={test_metrics['mean_macro_f1']:.4f} "
          f"form accuracy={test_metrics['functional_form']['accuracy']:.4f} "
          f"form macro-F1={test_metrics['functional_form']['macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
