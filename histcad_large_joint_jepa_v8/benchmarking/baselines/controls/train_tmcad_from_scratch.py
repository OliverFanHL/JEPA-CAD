"""Fully supervised TMCAD control: the JEPA STEP architecture with random initialization."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

from benchmarking.src.cad_benchmark.metrics import classification_metrics  # noqa: E402
from benchmarking.src.cad_benchmark.training import device_for, seed_everything  # noqa: E402
from src.histcad_large_jepa.model import StepEncoder  # noqa: E402
from src.histcad_large_jepa.step_graph import collate_step_graphs  # noqa: E402


class TMCADGraphs(Dataset):
    def __init__(self, manifest: dict, prepared: Path, split: str):
        self.samples = [(i, item) for i, item in enumerate(manifest["samples"]) if item["split"] == split]
        self.prepared = prepared
        self.classes = manifest["classes"]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, position: int):
        index, sample = self.samples[position]
        payload = torch.load(self.prepared / sample["graph_relpath"], map_location="cpu", weights_only=True)
        return payload["graph"], self.classes.index(sample["label"]), index


def collate(rows):
    graphs, labels, indices = zip(*rows, strict=True)
    return collate_step_graphs(list(graphs)), torch.tensor(labels), torch.tensor(indices)


class Classifier(nn.Module):
    def __init__(self, encoder: StepEncoder, width: int, classes: int, dropout: float):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.LayerNorm(encoder.output_dim), nn.Linear(encoder.output_dim, width),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(width, classes))

    def forward(self, graph):
        return self.head(self.encoder(graph))


def move(graph, device):
    return {key: value.to(device) for key, value in graph.items()}


@torch.inference_mode()
def evaluate(model, loader, device, classes):
    model.eval()
    logits, labels = [], []
    for graph, target, _ in loader:
        logits.append(model(move(graph, device)).cpu())
        labels.append(target)
    return classification_metrics(torch.cat(logits), torch.cat(labels), classes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base = args.config.resolve().parent
    settings = config["tmcad_supervised_from_scratch"]

    def resolve(value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (base / path).resolve()

    output = resolve(settings["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "evaluation.json").is_file():
        print(f"Reusing existing supervised control: {output / 'evaluation.json'}")
        return
    manifest = json.loads(resolve(settings["manifest"]).read_text(encoding="utf-8"))
    cache = torch.load(resolve(config["paths"]["histcad_cache"]), map_location="cpu", mmap=True, weights_only=False)
    jepa = torch.load(resolve(config["paths"]["jepa_representations"]), map_location="cpu", mmap=True, weights_only=False)
    model_config = jepa["model_config"]
    classes = len(manifest["classes"])
    seed_everything(config["seed"])
    device = device_for(settings["device"])
    encoder = StepEncoder(len(cache["step_entity_types"]), model_config["step_node_dim"],
                          model_config["step_output_dim"], model_config["step_depth"],
                          model_config["attention_heads"], model_config["step_pooling_slots"],
                          model_config["dropout"])
    model = Classifier(encoder, settings["hidden_dim"], classes, settings["dropout"]).to(device)
    datasets = {split: TMCADGraphs(manifest, resolve(settings["prepared_graphs"]), split) for split in ("train", "val", "test")}
    loaders = {split: DataLoader(data, batch_size=settings["batch_size"], shuffle=split == "train",
                                 num_workers=settings["num_workers"], collate_fn=collate,
                                 generator=torch.Generator().manual_seed(config["seed"])) for split, data in datasets.items()}
    train_labels = torch.tensor([
        manifest["classes"].index(sample["label"])
        for _, sample in datasets["train"].samples
    ])
    counts = torch.bincount(train_labels, minlength=classes).float().clamp_min(1)
    weights = counts.sum() / (classes * counts)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    best, best_epoch, best_state, stale, history = -1.0, 0, None, 0, []
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        total = 0.0
        samples = 0
        for graph, target, _ in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            target = target.to(device)
            loss = F.cross_entropy(model(move(graph, device)), target, weight=weights.to(device),
                                   label_smoothing=settings["label_smoothing"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"])
            optimizer.step()
            total += float(loss.detach()) * len(target)
            samples += len(target)
        val = evaluate(model, loaders["val"], device, classes)
        history.append({"epoch": epoch, "train_loss": total / samples, "validation_macro_f1": val["macro_f1"]})
        if val["macro_f1"] > best + settings["minimum_improvement"]:
            best, best_epoch, best_state, stale = val["macro_f1"], epoch, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        print(json.dumps(history[-1]), flush=True)
        if stale >= settings["patience"]:
            break
    if best_state is None:
        raise RuntimeError("No supervised checkpoint selected")
    model.load_state_dict(best_state)
    result = {"method": "same_step_encoder_supervised_from_random_initialization",
              "best_epoch": best_epoch, "validation": evaluate(model, loaders["val"], device, classes),
              "test": evaluate(model, loaders["test"], device, classes),
              "trainable_parameters": sum(p.numel() for p in model.parameters()), "history": history,
              "test_used_for_selection": False}
    torch.save({"state_dict": best_state, "model_config": model_config, "classes": manifest["classes"],
                "best_epoch": best_epoch}, output / "best.pt")
    (output / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
