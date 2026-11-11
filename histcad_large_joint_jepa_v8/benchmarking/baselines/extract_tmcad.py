"""Extract TMCAD STEP embeddings from AE, contrastive, or random HistCAD encoders."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

from src.histcad_large_jepa.model import StepEncoder  # noqa: E402
from src.histcad_large_jepa.step_graph import collate_step_graphs  # noqa: E402


class Graphs(Dataset):
    def __init__(self, samples: list[dict], prepared: Path):
        self.samples, self.prepared = samples, prepared

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return torch.load(self.prepared / self.samples[index]["graph_relpath"],
                          map_location="cpu", weights_only=True)["graph"]


def resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=("autoencoder", "contrastive", "random"))
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base = args.config.resolve().parent
    settings = config["tmcad_embedding_extraction"]
    output = resolve(base, settings["outputs"][args.method])
    if output.is_file():
        print(f"Reusing TMCAD embeddings: {output}")
        return
    manifest = json.loads(resolve(base, settings["manifest"]).read_text(encoding="utf-8"))
    source = torch.load(resolve(base, config["paths"]["jepa_representations"]), map_location="cpu", mmap=True, weights_only=False)
    cache = torch.load(resolve(base, config["paths"]["histcad_cache"]), map_location="cpu", mmap=True, weights_only=False)
    model_config = source["model_config"]
    torch.manual_seed(config["seed"])
    encoder = StepEncoder(len(cache["step_entity_types"]), model_config["step_node_dim"],
                          model_config["step_output_dim"], model_config["step_depth"],
                          model_config["attention_heads"], model_config["step_pooling_slots"],
                          model_config["dropout"])
    checkpoint_path = None
    if args.method != "random":
        checkpoint_path = resolve(base, settings["checkpoints"][args.method])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint["model"] if args.method == "autoencoder" else checkpoint["state_dict"]
        weights = {key.removeprefix("step_encoder."): value for key, value in state.items()
                   if key.startswith("step_encoder.")}
        encoder.load_state_dict(weights, strict=True)
    device = torch.device("cuda" if settings["device"] == "auto" and torch.cuda.is_available() else settings["device"] if settings["device"] != "auto" else "cpu")
    encoder.to(device).eval().requires_grad_(False)
    data = DataLoader(Graphs(manifest["samples"], resolve(base, settings["prepared_graphs"])),
                      batch_size=settings["batch_size"], shuffle=False,
                      num_workers=settings["num_workers"], collate_fn=collate_step_graphs)
    values = torch.empty((len(manifest["samples"]), model_config["step_output_dim"]), dtype=torch.float16)
    cursor = 0
    with torch.inference_mode():
        for graph in data:
            graph = {key: value.to(device) for key, value in graph.items()}
            embedding = F.normalize(encoder(graph).float(), dim=1).cpu().half()
            values[cursor:cursor + len(embedding)] = embedding
            cursor += len(embedding)
            if cursor % settings["progress_every"] < len(embedding) or cursor == len(values):
                print(json.dumps({"method": args.method, "extracted": cursor, "total": len(values)}), flush=True)
    name = f"{args.method}_step" if args.method != "random" else "random_step"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": "cad_benchmark_embeddings_v1",
        "uids": [item["uid"] for item in manifest["samples"]],
        "splits": [item["split"] for item in manifest["samples"]],
        "labels": {"class": [manifest["classes"].index(item["label"]) for item in manifest["samples"]]},
        "targets": {}, "embeddings": {name: values},
        "metadata": {"method": args.method, "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
                     "architecture": "HistCAD v7 STEP encoder", "preprocessing": "exact saved TMCAD graph cache"},
    }, output)


if __name__ == "__main__":
    main()
