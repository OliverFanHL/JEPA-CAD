"""Train a small functional-label readout on frozen adapted v7 embeddings.

The original JEPA and current adapter remain unchanged. Only train labels fit the
readout; validation selects checkpoint and fusion weight; test is evaluated once.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch
import yaml
from extended_evaluation import canonical, nearest, resolve, words
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent


class LabelTokenProbe(nn.Module):
    def __init__(self, input_dim: int, labels: int):
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, labels))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values.float())


def selected(payload: dict, split: str) -> torch.Tensor:
    return torch.tensor(
        [index for index, item in enumerate(payload["metadata"]) if item["split"] == split]
    )


def target_matrix(
    metadata: list[dict], indices: torch.Tensor, vocabulary: dict[str, int]
) -> torch.Tensor:
    output = torch.zeros((len(indices), len(vocabulary)))
    for row, index in enumerate(indices.tolist()):
        for token in words(metadata[index].get("functional_type", "")):
            if token in vocabulary:
                output[row, vocabulary[token]] = 1.0
    return output


@torch.inference_mode()
def predict(
    model: nn.Module, values: torch.Tensor, batch_size: int, device: torch.device
) -> torch.Tensor:
    model.eval()
    outputs = []
    for start in range(0, len(values), batch_size):
        outputs.append(model(values[start : start + batch_size].to(device)).sigmoid().cpu())
    return torch.cat(outputs)


def retrieval_quality(
    vector: torch.Tensor,
    metadata: list[dict],
    structural: list[dict],
    targets: torch.Tensor,
    device: torch.device,
    chunk: int,
) -> dict[str, float]:
    top = nearest(vector, 1, chunk, device)[:, 0].tolist()
    tokens = [words(item.get("functional_type", "")) for item in metadata]
    labels = [canonical(item.get("functional_type", "")) for item in metadata]
    dimension = (targets[torch.tensor(top)] - targets).abs().mean(1)
    history = torch.tensor(
        [
            structural[index]["operation_signature"] == structural[neighbor]["operation_signature"]
            for index, neighbor in enumerate(top)
        ]
    )
    feature = torch.tensor(
        [
            structural[index]["num_features"] == structural[neighbor]["num_features"]
            for index, neighbor in enumerate(top)
        ]
    )
    return {
        "text_token_jaccard_at_1": sum(
            len(tokens[index] & tokens[neighbor]) / max(1, len(tokens[index] | tokens[neighbor]))
            for index, neighbor in enumerate(top)
        )
        / len(top),
        "exact_function_at_1_all_queries": sum(
            bool(labels[index]) and labels[index] == labels[neighbor]
            for index, neighbor in enumerate(top)
        )
        / len(top),
        "dimension_mae_at_1": float(dimension.mean()),
        "history_accuracy_at_1": float(history.float().mean()),
        "feature_count_accuracy_at_1": float(feature.float().mean()),
        "geometry_structure_at_1": float(((dimension <= 0.2) & history & feature).float().mean()),
    }


def main() -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    settings = config["semantic_probe"]
    output_dir = resolve(config["paths"]["output_dir"])
    source = torch.load(
        resolve(config["paths"]["representations"]),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    adapted = torch.load(
        output_dir / "adapted_representations.pt", map_location="cpu", weights_only=False, mmap=True
    )
    if source["uids"] != adapted["uids"]:
        raise ValueError("Source and adapted embeddings are not UID aligned")
    embeddings = adapted["embeddings"]["adapted_joint"]
    splits = {name: selected(source, name) for name in ("train", "val", "test")}
    frequency = Counter()
    for index in splits["train"].tolist():
        frequency.update(words(source["metadata"][index].get("functional_type", "")))
    vocabulary = {
        token: index
        for index, token in enumerate(
            sorted(
                token
                for token, count in frequency.items()
                if count >= settings["min_train_token_frequency"]
            )
        )
    }
    if not vocabulary:
        raise ValueError("No functional-label tokens met the configured frequency threshold")
    train_targets = target_matrix(source["metadata"], splits["train"], vocabulary)
    val_targets = target_matrix(source["metadata"], splits["val"], vocabulary)
    train_features = embeddings[splits["train"]]
    val_features = embeddings[splits["val"]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LabelTokenProbe(embeddings.shape[1], len(vocabulary)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=1e-4
    )
    positive = train_targets.sum(0)
    negative = len(train_targets) - positive
    pos_weight = (
        (negative / positive.clamp_min(1)).clamp(max=settings["max_positive_weight"]).to(device)
    )
    train_loader = DataLoader(
        TensorDataset(train_features, train_targets),
        batch_size=settings["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(config["seed"] + 100),
    )
    history = []
    best_loss = float("inf")
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        training_loss = 0.0
        for features, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(
                model(features.to(device)), labels.to(device), pos_weight=pos_weight
            )
            loss.backward()
            optimizer.step()
            training_loss += float(loss.detach()) * len(features)
        model.eval()
        with torch.inference_mode():
            validation_loss = 0.0
            for start in range(0, len(val_features), settings["batch_size"]):
                stop = start + settings["batch_size"]
                loss = F.binary_cross_entropy_with_logits(
                    model(val_features[start:stop].to(device)),
                    val_targets[start:stop].to(device),
                    pos_weight=pos_weight,
                )
                validation_loss += float(loss) * len(val_features[start:stop])
            validation_loss /= len(val_features)
        history.append(
            {
                "epoch": epoch,
                "train_bce": training_loss / len(train_features),
                "val_bce": validation_loss,
            }
        )
        print(json.dumps(history[-1]), flush=True)
        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "vocabulary": vocabulary,
                    "epoch": epoch,
                    "val_bce": best_loss,
                },
                output_dir / "semantic_probe.pt",
            )
    best = torch.load(output_dir / "semantic_probe.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    # IDF is learned from training labels only. It prevents common words such as
    # "rectangular" from dominating every semantic-neighborhood comparison.
    inverse_frequency = torch.tensor(
        [((len(train_features) + 1) / (frequency[token] + 1)) ** 0.5 for token in vocabulary]
    )
    result = {
        "protocol": "Frozen v7 JEPA and frozen existing adapter; train-only functional descriptions fit a linear multi-label readout; validation-only fusion selection",
        "train_samples": len(splits["train"]),
        "validation_samples": len(splits["val"]),
        "test_samples": len(splits["test"]),
        "token_vocabulary_size": len(vocabulary),
        "best_epoch": best["epoch"],
        "history": history,
        "validation_grid": {},
    }
    selected_weight = None
    for split in ("val", "test"):
        indices = splits[split]
        metadata = [source["metadata"][index] for index in indices.tolist()]
        structural = [source["structural_metadata"][index] for index in indices.tolist()]
        targets = (source["dimension_targets"][indices] - source["dimension_target_mean"]) / source[
            "dimension_target_std"
        ]
        direct = F.normalize(adapted["embeddings"]["direct_joint"][indices].float(), dim=-1)
        base = F.normalize(embeddings[indices].float(), dim=-1)
        probabilities = predict(model, embeddings[indices], settings["batch_size"], device)
        semantic = F.normalize(probabilities * inverse_frequency, dim=-1)
        chunk = config["evaluation"]["retrieval_chunk_size"]
        if split == "val":
            result["validation_grid"]["direct"] = retrieval_quality(
                direct, metadata, structural, targets, device, chunk
            )
            result["validation_grid"]["adapted"] = retrieval_quality(
                base, metadata, structural, targets, device, chunk
            )
            result["validation_grid"]["semantic_only"] = retrieval_quality(
                semantic, metadata, structural, targets, device, chunk
            )
            baseline = result["validation_grid"]["adapted"]
            eligible = []
            for weight in settings["fusion_weights"]:
                vector = F.normalize(torch.cat((base, weight * semantic), dim=-1), dim=-1)
                quality = retrieval_quality(vector, metadata, structural, targets, device, chunk)
                result["validation_grid"][str(weight)] = quality
                if (
                    quality["dimension_mae_at_1"]
                    <= baseline["dimension_mae_at_1"]
                    * (1 + settings["max_dimension_mae_relative_increase"])
                    and quality["history_accuracy_at_1"]
                    >= baseline["history_accuracy_at_1"] - settings["max_history_accuracy_drop"]
                ):
                    eligible.append((quality["text_token_jaccard_at_1"], weight))
            selected_weight = max(eligible)[1] if eligible else 0.0
            result["selected_fusion_weight"] = selected_weight
            print(
                json.dumps(
                    {
                        "selected_fusion_weight": selected_weight,
                        "validation": result["validation_grid"][str(selected_weight)],
                    }
                ),
                flush=True,
            )
        else:
            assert selected_weight is not None
            fused = F.normalize(torch.cat((base, selected_weight * semantic), dim=-1), dim=-1)
            result["test"] = {
                "direct": retrieval_quality(direct, metadata, structural, targets, device, chunk),
                "adapted": retrieval_quality(base, metadata, structural, targets, device, chunk),
                "semantic_only": retrieval_quality(
                    semantic, metadata, structural, targets, device, chunk
                ),
                "semantic_fused": retrieval_quality(
                    fused, metadata, structural, targets, device, chunk
                ),
            }
            print(json.dumps({"test": result["test"]}), flush=True)
    (output_dir / "semantic_probe_evaluation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
