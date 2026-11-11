"""Independent annotation-field retrieval diagnostics (evaluation only)."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
from evaluate import paired_interval
from sentence_transformers import SentenceTransformer
from torch.nn import functional as F

from data import config, payload, resolve, split_indices

STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "of",
    "the",
    "with",
    "to",
    "in",
    "on",
    "by",
    "mm",
    "part",
    "model",
}


def tokens(text: str) -> set[str]:
    return {value for value in re.findall(r"[a-z]+", text.lower()) if value not in STOPWORDS}


def annotations(path: Path, uids: list[str]) -> dict[str, dict[str, str]]:
    wanted = set(uids)
    found = {}
    csv.field_size_limit(2_147_483_647)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["uid"] in wanted:
                found[row["uid"]] = row
    return found


def main() -> None:
    settings = config()
    output_dir = resolve(settings["paths"]["output_dir"])
    source = payload(settings)
    indices = split_indices(source)["test"].tolist()
    uids = [source["uids"][index] for index in indices]
    rows = annotations(resolve(settings["paths"]["annotations"]), uids)
    neighbor_payload = torch.load(
        output_dir / "test_neighbors.pt", map_location="cpu", weights_only=False
    )
    if neighbor_payload["uids"] != uids:
        raise ValueError("Test-neighbor and annotation UID mismatch")
    fields = ("Geometric Feature", "Modeling Process", "NLT")
    arrays = {field: [rows.get(uid, {}).get(field, "") for uid in uids] for field in fields}
    token_sets = {field: [tokens(text) for text in arrays[field]] for field in fields}
    # The teacher is used here only as an evaluation yardstick on Geometric
    # Feature descriptions, not as training supervision or retrieval input.
    teacher = SentenceTransformer(
        settings["text_teacher"]["model_id"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        local_files_only=True,
    )
    teacher.max_seq_length = settings["text_teacher"]["max_seq_length"]
    unique = list(dict.fromkeys(arrays["Geometric Feature"]))
    encoded = teacher.encode(
        unique,
        batch_size=settings["text_teacher"]["batch_size"],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    lookup = {item: index for index, item in enumerate(unique)}
    shape_text = F.normalize(
        torch.from_numpy(np.asarray(encoded))[
            [lookup[text] for text in arrays["Geometric Feature"]]
        ],
        dim=-1,
    )
    per_query = {}
    summary = {
        "evaluation_only_fields": list(fields),
        "nonempty": {field: sum(bool(item) for item in arrays[field]) for field in fields},
        "metrics": {},
        "paired_vs_direct": {},
        "caveat": "Geometric Feature and Modeling Process are held out of training, but are annotations of the same CAD parts, not independent human relevance judgments; NLT contains extensive repeated and dimensional text.",
    }
    for name, neighbor_row in neighbor_payload["neighbors"].items():
        top = neighbor_row["indices"][:, 0].tolist()
        metrics = {}
        for field in fields:
            values = token_sets[field]
            metrics[field + "_token_jaccard_at_1"] = torch.tensor(
                [
                    len(values[i] & values[j]) / max(1, len(values[i] | values[j]))
                    for i, j in enumerate(top)
                ]
            )
        metrics["Geometric Feature_text_embedding_cosine_at_1"] = (
            shape_text * shape_text[top]
        ).sum(1)
        per_query[name] = metrics
        summary["metrics"][name] = {key: float(value.mean()) for key, value in metrics.items()}
        if name != "direct_jepa":
            summary["paired_vs_direct"][name] = {
                key: paired_interval(per_query["direct_jepa"][key], value)
                for key, value in metrics.items()
            }
    (output_dir / "annotation_transfer.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["metrics"]), flush=True)


if __name__ == "__main__":
    main()
