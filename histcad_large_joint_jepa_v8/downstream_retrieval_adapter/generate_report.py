from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402, I001


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (ROOT / candidate).resolve()


def comparison_chart(direct: dict, adapted: dict) -> None:
    metrics = [
        ("Dimension\nRecall@10", "dimension_recall_at_10"),
        ("Dimension\nsuccess@5", "dimension_success_at_5"),
        ("History\nNN accuracy", "operation_history_nn_accuracy"),
        ("History\nsuccess@5", "operation_history_success_at_5"),
        ("Feature sequence\nNN accuracy", "feature_sequence_nn_accuracy"),
        ("Final operation\nNN accuracy", "final_operation_nn_accuracy"),
        ("Feature count\nNN accuracy", "feature_count_nn_accuracy"),
        ("Function\nsuccess@5", "canonical_function_success_at_5"),
    ]
    x = np.arange(len(metrics))
    width = 0.36
    figure, axis = plt.subplots(figsize=(13, 5.8), dpi=150)
    direct_values = [direct[key] for _, key in metrics]
    adapted_values = [adapted[key] for _, key in metrics]
    direct_bars = axis.bar(
        x - width / 2, direct_values, width, label="Direct v7 joint", color="#94a3b8"
    )
    adapted_bars = axis.bar(
        x + width / 2, adapted_values, width, label="Adapted v7 joint", color="#2563eb"
    )
    axis.bar_label(
        direct_bars,
        labels=[f"{100 * value:.1f}%" for value in direct_values],
        padding=3,
        fontsize=8,
    )
    axis.bar_label(
        adapted_bars,
        labels=[f"{100 * value:.1f}%" for value in adapted_values],
        padding=3,
        fontsize=8,
    )
    axis.set_xticks(x, [label for label, _ in metrics], fontsize=9)
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Accuracy / recall")
    axis.set_title(
        "Frozen v7 embedding: direct retrieval versus lightweight adapted retrieval", fontsize=14
    )
    axis.legend(loc="upper left")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(OUTPUT / "adapter_comparison.png", bbox_inches="tight")
    figure.savefig(OUTPUT / "adapter_comparison.svg", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    global OUTPUT
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    OUTPUT = resolve(config["paths"]["output_dir"])
    evaluation = json.loads((OUTPUT / "evaluation.json").read_text(encoding="utf-8"))
    training = json.loads((OUTPUT / "training_summary.json").read_text(encoding="utf-8"))
    audit = json.loads((OUTPUT / "render_audit.json").read_text(encoding="utf-8"))
    summary = evaluation["splits"]["test"]["summary"]
    direct, adapted, delta = (
        summary["direct_joint"],
        summary["adapted_joint"],
        summary["adapted_minus_direct"],
    )
    comparison_chart(direct, adapted)

    rows = [
        ("Dimensional Recall@10", "dimension_recall_at_10", True),
        ("Dimensional success@5", "dimension_success_at_5", True),
        ("Dimensional MAE@1", "dimension_mae_at_1", False),
        ("Operation-history NN accuracy", "operation_history_nn_accuracy", True),
        ("Operation-history success@5", "operation_history_success_at_5", True),
        ("Feature-sequence NN accuracy", "feature_sequence_nn_accuracy", True),
        ("Final-operation NN accuracy", "final_operation_nn_accuracy", True),
        ("Feature-count NN accuracy", "feature_count_nn_accuracy", True),
        ("Primitive-profile NN accuracy", "primitive_profile_nn_accuracy", True),
        ("Dominant-constraint NN accuracy", "dominant_constraint_nn_accuracy", True),
        ("Canonical-function NN accuracy", "canonical_function_nn_accuracy", True),
        ("Canonical-function success@5", "canonical_function_success_at_5", True),
    ]
    lines = [
        "# Lightweight downstream retrieval adapter for HistCAD v7",
        "",
        "## Result",
        "",
        "The experiment tests whether the frozen v7 joint representation contains information that can "
        "be organized more effectively by a small task-specific retrieval model. Positive table deltas "
        "mean that adaptation helped; negative deltas expose specialization tradeoffs.",
        "",
        "![Direct versus adapted retrieval](adapter_comparison.png)",
        "",
        "## Direct versus adapted test retrieval",
        "",
        "| Metric | Direct pretrained embedding | With downstream adapter | Change |",
        "|---|---:|---:|---:|",
    ]
    for label, key, percentage in rows:
        if percentage:
            lines.append(
                f"| {label} | {100 * direct[key]:.1f}% | {100 * adapted[key]:.1f}% | "
                f"{100 * delta[key]:+.1f} pp |"
            )
        else:
            relative = 100 * (adapted[key] / direct[key] - 1)
            lines.append(f"| {label} | {direct[key]:.3f} | {adapted[key]:.3f} | {relative:+.1f}% |")
    lines.extend(
        [
            "",
            f"Near-tie queries changed from {direct['near_tie_queries']:,} to "
            f"{adapted['near_tie_queries']:,}; the median top-1/top-2 margin changed from "
            f"{direct['median_top1_top2_margin']:.4f} to "
            f"{adapted['median_top1_top2_margin']:.4f}.",
            "",
            "## What was trained",
            "",
            "The original v7 encoder and `best.pt` weights were frozen. The adapter has "
            f"{training['adapter_trainable_parameters']:,} trainable parameters and keeps the output at "
            f"{training['embedding_dim']} dimensions. It contains one chunk-level multi-head attention "
            "block, attention pooling, "
            "a small residual MLP, and lightweight auxiliary heads.",
            "",
            "Adapter supervision used:",
            "",
            "- approximate dimensional/structural positive pairs from the 22-D target vector;",
            "- contrastive retrieval against fixed negative samples;",
            "- dimensional-summary regression;",
            "- feature-count and final-Boolean-operation classification; and",
            "- a small residual-anchor loss to retain the original representation.",
            "",
            "Functional labels and text were not used for adapter training. Exact operation history, "
            "feature sequence, primitive profile, constraints, and functional labels were evaluation "
            "targets only. Changes in exact history and feature-sequence retrieval therefore measure "
            "transfer from the supervised dimensional/coarse-structural adaptation.",
            "",
            "## Protocol",
            "",
            f"Training used {training['train_samples']:,} training parts. Checkpoint selection used "
            f"{training['validation_samples']:,} validation parts and selected epoch "
            f"{training['best_epoch']} at validation loss {training['best_validation_loss']:.6f}. The "
            "adapter trainer accessed zero test samples. The final direct and adapted embeddings were "
            f"then evaluated once on the unchanged {evaluation['splits']['test']['samples']:,}-part test "
            "split using the same self-match removal, "
            "cosine similarity, dimensional reference neighbors, labels, and metrics as the previous "
            "benchmark.",
            "",
            "Because feature count, final operation, and dimensional targets are used to train the adapter, "
            "their gains are task-adapted results—not zero-shot representation metrics. The encoder remains "
            "frozen, so the experiment specifically measures how much task-relevant information can be "
            "recovered from the pretrained latent vector.",
            "",
            "## Qualitative retrieval",
            "",
            f"Six comparison sheets contain {audit['valid']} unique, successfully rendered STEP parts; "
            f"{audit['invalid']} displayed parts failed validation. Each sheet uses the same query and "
            "shows direct and adapted top-3 results. Captions report dimensional distance, feature-count "
            "agreement, and exact operation-history agreement.",
            "",
        ]
    )
    for case in range(1, 7):
        lines.extend(
            [
                f"### Retrieval case {case}",
                "",
                f"![Adapter retrieval case {case}](retrieval_sheets/case_{case:02d}.png)",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "Direct retrieval measures general-purpose latent organization, while the adapted results "
            "measure task-specific extractability. No single adapted metric should replace the direct "
            "benchmark: direct retrieval "
            "measures general-purpose latent organization, while adapted retrieval measures task-specific "
            "extractability. Both should be reported.",
            "",
            "## Artifacts",
            "",
            "- `best_adapter.pt`: validation-selected adapter checkpoint",
            "- `adapted_representations.pt`: UID-aligned direct and adapted embeddings",
            "- `training_summary.json`: complete training and validation history",
            "- `evaluation.json`: complete validation/test metrics and deltas",
            "- `comparison.csv`: compact direct-versus-adapted table",
            "- `qualitative_manifest.csv`: identities and correctness for all displayed retrievals",
            "- `render_audit.json`: CAD import and rendering checks",
            "- `retrieval_sheets/INDEX.md`: qualitative gallery",
        ]
    )
    (OUTPUT / "ADAPTER_EVALUATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps({"status": "complete", "report": str(OUTPUT / "ADAPTER_EVALUATION_REPORT.md")})
    )


if __name__ == "__main__":
    main()
