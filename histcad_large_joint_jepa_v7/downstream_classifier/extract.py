"""Load the already extracted frozen best.pt vectors, or regenerate them."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
V7_ROOT = ROOT.parent
sys.path.insert(0, str(V7_ROOT))

from src.histcad_large_jepa.data import PairedGraphDataset, collate_paired  # noqa: E402
from src.histcad_large_jepa.model import WideJointJEPA  # noqa: E402
from src.histcad_large_jepa.train import RunLogger, extract_representations  # noqa: E402


def resolve_paths(settings: dict) -> dict[str, Path]:
    return {
        key: (Path(value) if Path(value).is_absolute() else (ROOT / value).resolve())
        for key, value in settings["paths"].items()
    }


def _device(name: str) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)


def extract_best(settings: dict, paths: dict[str, Path]) -> Path:
    """This is inference only: model.eval(), no gradient, no optimizer or adaptation."""
    checkpoint_path = paths["checkpoint"]
    if checkpoint_path.name != "best.pt":
        raise ValueError("Frozen extraction must use the JEPA-v7 best.pt checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    cache = torch.load(paths["graph_cache"], map_location="cpu", weights_only=False, mmap=True)
    if checkpoint.get("format_version") != 7 or cache.get("format_version") != 7:
        raise ValueError("Expected matching JEPA-v7 checkpoint and graph cache")
    model = WideJointJEPA(
        len(cache["step_entity_types"]),
        checkpoint["model_config"],
        len(cache["dimension_target_names"]),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = _device(settings["classification"]["device"])
    model.to(device)
    batch_size = settings["classification"]["extraction_batch_size"]
    workers = settings["classification"]["extraction_num_workers"]
    mean, std = checkpoint["dimension_target_mean"], checkpoint["dimension_target_std"]
    loaders = {}
    for split in ("train", "val", "test"):
        dataset = PairedGraphDataset(cache, split, mean, std)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            collate_fn=collate_paired,
            pin_memory=device.type == "cuda",
        )
    target = paths["extracted_representations"]
    target.parent.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(target.parent / "extraction.log", append=False)
    with torch.inference_mode():
        extract_representations(
            model, loaders, cache, device, mean, std, target, logger,
            "float16", checkpoint_path, int(checkpoint["epoch"]), checkpoint["model_config"],
        )
    return target


def load_frozen(path: Path, best_checkpoint: Path, graph_cache: Path) -> dict:
    source = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if source.get("format_version") != 7:
        raise ValueError("Representation cache is not JEPA-v7")
    if source.get("source_checkpoint_name") != "best.pt":
        raise ValueError("Representations were not extracted from best.pt")
    if Path(source["source_checkpoint"]).resolve() != best_checkpoint.resolve():
        raise ValueError("Representations refer to a different best.pt path")
    if not graph_cache.is_file() or not best_checkpoint.is_file():
        raise FileNotFoundError("The original v7 graph cache and best.pt must both exist")
    if len(source["uids"]) != len(source["metadata"]) or len(source["uids"]) != len(source["structural_metadata"]):
        raise ValueError("Representation metadata are not aligned")
    if source["embeddings"]["joint"].shape != (len(source["uids"]), 4096):
        raise ValueError("Expected 4096-D frozen joint embeddings for every retained item")
    if len(set(source["uids"])) != len(source["uids"]):
        raise ValueError("Duplicate UIDs in the representation cache")
    return source
