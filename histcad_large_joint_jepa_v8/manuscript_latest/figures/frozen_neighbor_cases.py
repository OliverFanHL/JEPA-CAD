"""Exact same-part 5-NN decisions from the saved matched TMCAD embeddings."""
from pathlib import Path
import json
import sys

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
BENCH = HERE.parents[1] / "benchmarking"
sys.path.insert(0, str(BENCH / "src"))
from cad_benchmark.config import load_config  # noqa: E402
from cad_benchmark.registry import Registry  # noqa: E402
from cad_benchmark.training import knn_classification  # noqa: E402

METHODS = ("jepa_step_tmcad", "autoencoder_step_tmcad", "brepnet_official_tmcad")
CASES = ("bearing/55.stp", "pulley/423.stp", "coupling/257.stp")


def main() -> None:
    config = load_config(BENCH / "config.yaml")
    registry = Registry(config, apply_shared_coverage=True)
    registry.validate_alignment("classification")
    saved = json.loads((BENCH / "outputs/tmcad_classification/results.json").read_text(
        encoding="utf-8"))
    classes = registry.load(METHODS[0]).metadata["classes"]
    results = {}
    for name in METHODS:
        rep = registry.load(name)
        labels = torch.tensor(rep.labels["class"], dtype=torch.long)
        splits = {split: rep.indices(split) for split in ("train", "val", "test")}
        measured = knn_classification(rep.values, labels, splits, 5, 256)["test"]["macro_f1"]
        expected = saved["representations"][name]["knn"]["test"]["macro_f1"]
        assert abs(measured - expected) < 1e-10, (name, measured, expected)
        train = F.normalize(rep.values[splits["train"]].float(), dim=1)
        entries = []
        for uid in CASES:
            index = rep.uids.index(uid)
            assert rep.splits[index] == "test"
            query = F.normalize(rep.values[index:index+1].float(), dim=1)
            nearest = (query @ train.T).topk(5).indices[0]
            votes = torch.bincount(labels[splits["train"]][nearest], minlength=len(classes))
            predicted = int(votes.argmax())
            entries.append({"uid": uid, "true_class": classes[int(labels[index])],
                            "prediction": classes[predicted],
                            "votes_for_prediction": int(votes[predicted]),
                            "votes_for_truth": int(votes[int(labels[index])]),
                            "neighbor_uids": [rep.uids[int(splits["train"][j])]
                                              for j in nearest],
                            "neighbor_classes": [classes[int(labels[int(splits["train"][j])])]
                                                 for j in nearest]})
        results[name] = {"test_macro_f1": measured, "cases": entries}
        print(name, [(x["uid"], x["prediction"], x["votes_for_prediction"])
                     for x in entries], flush=True)
    (HERE / "frozen_neighbor_cases.json").write_text(json.dumps({
        "protocol": "shared-coverage TMCAD test subset, frozen 5-NN",
        "classes": classes, "methods": results}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
