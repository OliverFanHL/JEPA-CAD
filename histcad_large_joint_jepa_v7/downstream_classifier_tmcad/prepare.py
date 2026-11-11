"""Validate TMCAD STEP files and make an exact-file-duplicate-safe split."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from common import read_pretrained_metadata, save_json, sha256_file, v7_modules
from step_statements import PARSER_VERSION, STEP_MATCHER


class FrozenVocabulary(dict):
    """Map out-of-vocabulary STEP types to embedding row zero, without retraining it."""

    def __init__(self, names: tuple[str, ...]):
        super().__init__({name: index for index, name in enumerate(names)})
        self.unknown: set[str] = set()

    def __missing__(self, key: str) -> int:
        self.unknown.add(key)
        return 0


def _source_files(root: Path, classes: list[str], extensions: list[str]) -> list[tuple[str, Path]]:
    if not root.is_dir():
        raise FileNotFoundError(f"TMCAD folder not found: {root}")
    extensions = {suffix.lower() for suffix in extensions}
    entries: list[tuple[str, Path]] = []
    for name in classes:
        folder = root / name
        if not folder.is_dir():
            raise FileNotFoundError(f"Expected TMCAD class folder is missing: {folder}")
        files = sorted(path for path in folder.rglob("*")
                       if path.is_file() and path.suffix.lower() in extensions)
        if not files:
            raise ValueError(f"No STEP files in configured class {name}: {folder}")
        entries.extend((name, path) for path in files)
    return entries


def _split_groups(records: list[dict], classes: list[str], fractions: dict, seed: int) -> None:
    if set(fractions) != {"train", "val", "test"} or abs(sum(fractions.values()) - 1.0) > 1e-6:
        raise ValueError("Split fractions must define train, val, test and sum to 1")
    rng = np.random.default_rng(seed)
    for name in classes:
        groups: dict[str, list[dict]] = defaultdict(list)
        for item in records:
            if item["label"] == name:
                groups[item["sha256"]].append(item)
        keys = sorted(groups)
        if len(keys) < 3:
            raise ValueError(f"Not enough unique valid {name} files for a three-way split")
        rng.shuffle(keys)
        count = len(keys)
        n_train = max(1, round(fractions["train"] * count))
        n_val = max(1, round(fractions["val"] * count))
        n_train = min(n_train, count - 2)
        n_val = min(n_val, count - n_train - 1)
        for position, key in enumerate(keys):
            split = "train" if position < n_train else "val" if position < n_train + n_val else "test"
            for item in groups[key]:
                item["split"] = split


def prepare_dataset(config: dict) -> dict:
    dataset = config["paths"]["dataset_root"]
    processed = config["paths"]["processed_dir"]
    processed.mkdir(parents=True, exist_ok=True)
    classes = list(config["dataset"]["classes"])
    found = _source_files(dataset, classes, config["dataset"]["extensions"])
    summary, vocabulary = read_pretrained_metadata(config)
    _, _, parse_step_graph = v7_modules(config)
    vocab_sha = hashlib.sha256("\n".join(vocabulary).encode()).hexdigest()
    max_instances = int(config["dataset"]["max_step_instances"])
    max_bytes = int(config["dataset"]["max_file_mb"] * 1024 * 1024)
    # Keep v1 caches untouched; the old regex could save incomplete graphs.
    graph_root = processed / "graphs_v2"
    records: list[dict] = []
    exclusions: list[dict] = []
    previously_checked: dict[str, dict] = {}
    for ordinal, (label, path) in enumerate(found, start=1):
        relative = path.relative_to(dataset).as_posix()
        reason = None
        try:
            if path.stat().st_size == 0:
                reason = "empty_file"
            elif path.stat().st_size > max_bytes:
                reason = "oversize_file"
            else:
                with path.open("rb") as handle:
                    if b"ISO-10303-21" not in handle.read(512):
                        reason = "not_step_header"
            if reason:
                raise ValueError(reason)
            digest = sha256_file(path)
            if digest in previously_checked:
                payload = previously_checked[digest]
            else:
                graph_path = graph_root / digest[:2] / f"{digest}.pt"
                if graph_path.is_file():
                    try:
                        cached = torch.load(graph_path, map_location="cpu", weights_only=True)
                        payload = (cached if cached.get("vocabulary_sha256") == vocab_sha
                                   and cached.get("max_step_instances") == max_instances
                                   and cached.get("parser_version") == PARSER_VERSION else None)
                    except (OSError, RuntimeError, ValueError, EOFError):
                        # An interrupted prior write should be rebuilt, not
                        # silently treated as an invalid source STEP file.
                        payload = None
                else:
                    payload = None
                if payload is None:
                    mapping = FrozenVocabulary(vocabulary)
                    graph = parse_step_graph(path, mapping, max_instances)
                    assignments = STEP_MATCHER.last_assignment_count
                    parsed_fraction = (STEP_MATCHER.last_parsed_count / assignments) if assignments else 0.0
                    roles = graph["node_role"]
                    unknown = graph["node_type"] == 0
                    type_nodes = roles == 1
                    instance_nodes = roles == 2
                    stats = {
                        "num_step_entities": int(graph["num_step_entities"]),
                        "num_step_types": int(graph["num_step_types"]),
                        "num_brep_candidates": int(graph["num_brep_candidates"]),
                        "num_brep_instances": int(graph["num_brep_instances"]),
                        "brep_truncated": bool(graph["brep_truncated"]),
                        "unknown_type_fraction": float(unknown[type_nodes].float().mean()) if type_nodes.any() else 1.0,
                        "unknown_instance_fraction": float(unknown[instance_nodes].float().mean()) if instance_nodes.any() else 0.0,
                        "unknown_types": sorted(mapping.unknown),
                        "entity_assignment_count": assignments,
                        "entity_parse_fraction": parsed_fraction,
                    }
                    needed = ("node_type", "node_role", "numeric", "edge_index", "edge_type", "edge_weight")
                    payload = {"graph": {key: graph[key] for key in needed}, "stats": stats,
                               "vocabulary_sha256": vocab_sha, "max_step_instances": max_instances,
                               "parser_version": PARSER_VERSION}
                    if stats["num_step_entities"] < config["dataset"]["min_step_entities"]:
                        raise ValueError("too_few_step_entities")
                    if stats["num_brep_candidates"] < config["dataset"]["min_brep_candidates"]:
                        raise ValueError("no_brep_candidates")
                    if stats["entity_parse_fraction"] < config["dataset"]["min_entity_parse_fraction"]:
                        raise ValueError("low_entity_parse_coverage")
                    if not torch.isfinite(graph["numeric"]).all() or not torch.isfinite(graph["edge_weight"]).all():
                        raise ValueError("nonfinite_graph_features")
                    graph_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = graph_path.with_suffix(".partial.pt")
                    torch.save(payload, temporary)
                    temporary.replace(graph_path)
                previously_checked[digest] = payload
            stats = payload["stats"]
            if stats["num_step_entities"] < config["dataset"]["min_step_entities"]:
                raise ValueError("too_few_step_entities")
            if stats["num_brep_candidates"] < config["dataset"]["min_brep_candidates"]:
                raise ValueError("no_brep_candidates")
            if stats["entity_parse_fraction"] < config["dataset"]["min_entity_parse_fraction"]:
                raise ValueError("low_entity_parse_coverage")
            records.append({"uid": relative, "source_relpath": relative, "label": label,
                            "sha256": digest, "graph_relpath": f"graphs_v2/{digest[:2]}/{digest}.pt",
                            **stats})
        except (OSError, ValueError, RuntimeError, KeyError, IndexError, OverflowError, UnicodeError) as error:
            exclusions.append({"source_relpath": relative,
                               "reason": reason or type(error).__name__, "detail": str(error)[:300]})
        if ordinal % 250 == 0 or ordinal == len(found):
            print(f"STEP audit {ordinal:,}/{len(found):,} retained={len(records):,} excluded={len(exclusions):,}", flush=True)

    # Byte-identical geometry carrying different class labels has no defensible single target.
    by_hash: dict[str, set[str]] = defaultdict(set)
    for item in records:
        by_hash[item["sha256"]].add(item["label"])
    ambiguous = {digest for digest, labels in by_hash.items() if len(labels) > 1}
    if ambiguous:
        kept = []
        for item in records:
            if item["sha256"] in ambiguous:
                exclusions.append({"source_relpath": item["source_relpath"],
                                   "reason": "cross_class_exact_duplicate", "detail": item["sha256"]})
            else:
                kept.append(item)
        records = kept
    _split_groups(records, classes, config["dataset"]["split_fractions"], int(config["seed"]))
    records.sort(key=lambda item: (classes.index(item["label"]), item["source_relpath"]))
    counts = {name: dict(Counter(item["split"] for item in records if item["label"] == name)) for name in classes}
    for name in classes:
        if any(counts[name].get(split, 0) == 0 for split in ("train", "val", "test")):
            raise ValueError(f"No usable {name} sample in at least one split; inspect audit")
    manifest = {"format": "tmcad_v7_step_only_1", "dataset_root": str(dataset),
                "source_checkpoint": summary["checkpoint"], "source_epoch": summary["epoch"],
                "vocabulary_sha256": vocab_sha, "parser_version": PARSER_VERSION,
                "classes": classes, "samples": records}
    save_json(processed / "manifest.json", manifest)
    reasons = Counter(item["reason"] for item in exclusions)
    oov_by_class = {}
    for name in classes:
        subset = [item for item in records if item["label"] == name]
        oov_by_class[name] = {"mean_unknown_type_fraction": float(np.mean([x["unknown_type_fraction"] for x in subset])),
                              "mean_entity_parse_fraction": float(np.mean([x["entity_parse_fraction"] for x in subset])),
                              "fraction_above_warning_threshold": float(np.mean([
                                  x["unknown_type_fraction"] > config["dataset"]["warn_unknown_type_fraction"]
                                  for x in subset])),
                              "brep_truncated_fraction": float(np.mean([x["brep_truncated"] for x in subset]))}
    audit = {"discovered": len(found), "retained": len(records), "excluded": len(exclusions),
             "parser_version": PARSER_VERSION,
             "exclusion_reasons": dict(reasons), "exclusions": exclusions,
             "class_split_counts": counts, "oov_by_class": oov_by_class,
             "warning": "Unknown STEP type IDs map to the frozen embedding's zero row; numeric, role and connectivity data remain. High or class-dependent OOV rates limit transfer validity."}
    save_json(processed / "audit.json", audit)
    print(f"Prepared {len(records):,}/{len(found):,} valid models; exclusions={dict(reasons)}", flush=True)
    return manifest
