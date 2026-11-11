from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


DISPLAY = {
    "pretrained_jepa_joint": "Pretrained JEPA (joint)",
    "graph_autoencoder_joint": "Graph autoencoder (joint)",
    "input_graph_statistics": "Input graph statistics",
    "pca_graph_statistics": "PCA graph statistics",
    "adapted_jepa_joint": "JEPA + downstream adapter",
}


def value(number: float, lower_is_better: bool = False) -> str:
    suffix = " ↓" if lower_is_better else " ↑"
    return f"{number:.4f}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the representation benchmark report")
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    configured = Path(config["paths"]["output_dir"])
    output = configured if configured.is_absolute() else (ROOT / configured).resolve()
    result = json.loads((output / "benchmark_evaluation.json").read_text(encoding="utf-8"))
    rows = result["splits"]["test"]["summary"]
    lines = [
        "# HistCAD representation benchmark",
        "",
        "The primary question is whether the pretrained JEPA representation improves CAD retrieval over alternative representation-learning methods. All methods use the same deterministic split, the same held-out test queries/candidates, cosine retrieval, and the same metric implementation. The adapter comparison is retained only as a secondary downstream ablation.",
        "",
        "## Primary representation comparison",
        "",
        "| Representation | Dim. neighbor R@10 | Dim. MAE@1 | Operation NN | Feature-sequence NN | Canonical function NN | Canonical function success@5 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    primary = result["protocol"]["primary_comparison"]
    for name in primary:
        row = rows[name]
        lines.append(
            "| "
            + " | ".join(
                (
                    DISPLAY.get(name, name),
                    value(row["dimension_recall_at_10"]),
                    value(row["dimension_mae_at_1"], True),
                    value(row["operation_history_nn_accuracy"]),
                    value(row["feature_sequence_nn_accuracy"]),
                    value(row["canonical_function_nn_accuracy"]),
                    value(row["canonical_function_success_at_5"]),
                )
            )
            + " |"
        )
    if result["protocol"]["secondary_comparison"]:
        lines.extend(
            [
                "",
                "## Secondary downstream adaptation ablation",
                "",
                "This comparison asks whether a task-specific adapter can reorganize an already pretrained JEPA representation; it is not the main evidence for the pretraining objective.",
                "",
                "| Representation | Dim. neighbor R@10 | Dim. MAE@1 | Operation NN | Feature-sequence NN | Canonical function NN |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for name in result["protocol"]["secondary_comparison"]:
            row = rows[name]
            lines.append(
                "| "
                + " | ".join(
                    (
                        DISPLAY.get(name, name),
                        value(row["dimension_recall_at_10"]),
                        value(row["dimension_mae_at_1"], True),
                        value(row["operation_history_nn_accuracy"]),
                        value(row["feature_sequence_nn_accuracy"]),
                        value(row["canonical_function_nn_accuracy"]),
                    )
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "## Protocol and interpretation",
            "",
            "- The graph autoencoder is trained from scratch on the same train split and uses the same parametric/STEP graph encoders and 4096-D joint width as JEPA. Its checkpoint is selected using validation reconstruction loss.",
            "- Input graph statistics are a deterministic, non-neural anchor standardized with train-split statistics only.",
            "- PCA is fit only on standardized train-split graph statistics and supplies a classical unsupervised representation baseline.",
            "- Retrieval labels (operation history, feature sequence, functional type, and related labels) are evaluation-only for every primary method.",
            "- Validation and test retrieval are computed independently within each split with self-matches excluded.",
            "- The test split is not used for optimization or model selection.",
            "- Published numbers should only be added as reference rows when their dataset split, candidate pool, relevance definition, and metrics are genuinely comparable; otherwise they should be discussed separately rather than mixed into this table.",
            "",
            "## Reproducibility artifacts",
            "",
            "- `baseline_training_summary.json`: baseline optimization and selection history",
            "- `benchmark_evaluation.json`: full validation/test metrics and protocol metadata",
            "- `benchmark_comparison.csv`: compact method-by-method table",
            "- `best_graph_autoencoder.pt`: validation-selected autoencoder checkpoint",
            "- `baseline_representations.pt`: UID-aligned baseline embeddings",
        ]
    )
    (output / "REPRESENTATION_BENCHMARK_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
