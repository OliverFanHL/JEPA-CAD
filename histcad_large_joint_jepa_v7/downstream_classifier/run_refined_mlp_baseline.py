"""Re-fit the previous joint MLP on refined labels for a same-taxonomy baseline."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from extract import load_frozen, resolve_paths
from labels import build_labels, load_taxonomy, save_labels
from train_classifier import _train_one

ROOT = Path(__file__).resolve().parent


def main() -> None:
    import yaml
    settings = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    paths = resolve_paths(settings)
    source = load_frozen(paths["representations"], paths["checkpoint"], paths["graph_cache"])
    taxonomy = load_taxonomy(ROOT / "taxonomy_refinement_v2.json")
    targets, report = build_labels(source, taxonomy)
    output = ROOT / "outputs" / "mlp_refined_v2"
    save_labels(output, source, targets, report)
    seed = settings["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    options = {**settings["classification"], "seed": seed}
    result = _train_one("joint", "mlp", source["embeddings"]["joint"], source,
                        targets, report, options, output)
    (output / "baseline_results.json").write_text(json.dumps({
        "protocol": "Same refined v2 labels, UID split, frozen best.pt joint vector, and 8-epoch MLP settings as original baseline",
        "taxonomy_version": report["taxonomy_version"],
        "run": result,
    }, indent=2), encoding="utf-8")
    print(f"Refined-label joint MLP: best_epoch={result['best_epoch']} "
          f"test_form_accuracy={result['test']['functional_form']['accuracy']:.4f} "
          f"test_form_macro_F1={result['test']['functional_form']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
