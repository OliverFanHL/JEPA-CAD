"""Train the isolated semantic+dimension condition without count/operation losses."""

from __future__ import annotations

import json

import torch
from train import train_one

from data import config, geometry_targets, payload, resolve, split_indices


def main() -> None:
    settings = config()
    torch.set_num_threads(8)
    output_dir = resolve(settings["paths"]["output_dir"])
    source = payload(settings)
    splits = split_indices(source)
    teacher = torch.load(
        output_dir / "functional_text_teacher.pt",
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    if (
        teacher["uids"] != source["uids"]
        or teacher.get("preprocess", "raw") != settings["text_teacher"]["preprocess"]
    ):
        raise ValueError("Text teacher cache does not match this study configuration")
    dimensions, feature, operation = geometry_targets(source)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    row = train_one(
        "semantic_plus_dimensions",
        source["embeddings"]["joint"],
        teacher["targets"],
        dimensions,
        feature,
        operation,
        splits,
        settings,
        output_dir,
        device,
    )
    embeddings = row.pop("embeddings")
    torch.save(
        {
            "uids": {
                name: [source["uids"][i] for i in splits[name].tolist()] for name in ("val", "test")
            },
            "embeddings": embeddings,
            "mode": "semantic_plus_dimensions",
            "jepa_encoder_frozen": True,
            "text_teacher_frozen": True,
        },
        output_dir / "dimension_only_embeddings.pt",
    )
    (output_dir / "dimension_only_training_summary.json").write_text(
        json.dumps(row, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
