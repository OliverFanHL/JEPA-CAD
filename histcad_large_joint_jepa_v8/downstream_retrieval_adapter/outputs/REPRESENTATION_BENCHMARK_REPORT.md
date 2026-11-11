# HistCAD representation benchmark

The primary question is whether the pretrained JEPA representation improves CAD retrieval over alternative representation-learning methods. All methods use the same deterministic split, the same held-out test queries/candidates, cosine retrieval, and the same metric implementation. The adapter comparison is retained only as a secondary downstream ablation.

## Primary representation comparison

| Representation | Dim. neighbor R@10 | Dim. MAE@1 | Operation NN | Feature-sequence NN | Canonical function NN | Canonical function success@5 |
|---|---:|---:|---:|---:|---:|---:|
| Pretrained JEPA (joint) | 0.2673 ↑ | 0.2649 ↓ | 0.8522 ↑ | 0.8803 ↑ | 0.3218 ↑ | 0.5812 ↑ |
| Graph autoencoder (joint) | 0.2064 ↑ | 0.2813 ↓ | 0.8561 ↑ | 0.8832 ↑ | 0.3120 ↑ | 0.5913 ↑ |
| PCA graph statistics | 0.2041 ↑ | 0.2893 ↓ | 0.8349 ↑ | 0.8728 ↑ | 0.3260 ↑ | 0.5910 ↑ |
| Input graph statistics | 0.2041 ↑ | 0.2893 ↓ | 0.8349 ↑ | 0.8728 ↑ | 0.3255 ↑ | 0.5910 ↑ |

## Secondary downstream adaptation ablation

This comparison asks whether a task-specific adapter can reorganize an already pretrained JEPA representation; it is not the main evidence for the pretraining objective.

| Representation | Dim. neighbor R@10 | Dim. MAE@1 | Operation NN | Feature-sequence NN | Canonical function NN |
|---|---:|---:|---:|---:|---:|
| Pretrained JEPA (joint) | 0.2673 ↑ | 0.2649 ↓ | 0.8522 ↑ | 0.8803 ↑ | 0.3218 ↑ |
| JEPA + downstream adapter | 0.5248 ↑ | 0.1696 ↓ | 0.9193 ↑ | 0.9681 ↑ | 0.3135 ↑ |

## Protocol and interpretation

- The graph autoencoder is trained from scratch on the same train split and uses the same parametric/STEP graph encoders and 4096-D joint width as JEPA. Its checkpoint is selected using validation reconstruction loss.
- Input graph statistics are a deterministic, non-neural anchor standardized with train-split statistics only.
- PCA is fit only on standardized train-split graph statistics and supplies a classical unsupervised representation baseline.
- Retrieval labels (operation history, feature sequence, functional type, and related labels) are evaluation-only for every primary method.
- Validation and test retrieval are computed independently within each split with self-matches excluded.
- The test split is not used for optimization or model selection.
- Published numbers should only be added as reference rows when their dataset split, candidate pool, relevance definition, and metrics are genuinely comparable; otherwise they should be discussed separately rather than mixed into this table.

## Reproducibility artifacts

- `baseline_training_summary.json`: baseline optimization and selection history
- `benchmark_evaluation.json`: full validation/test metrics and protocol metadata
- `benchmark_comparison.csv`: compact method-by-method table
- `best_graph_autoencoder.pt`: validation-selected autoencoder checkpoint
- `baseline_representations.pt`: UID-aligned baseline embeddings
