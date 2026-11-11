# Final results and manuscript evidence

All paths below are under `histcad_large_joint_jepa_v8/` unless stated otherwise.

| Evidence | Saved outputs |
| --- | --- |
| Pretraining (20 executed epochs, validation-selected epoch 20) | `outputs/full_dataset_4096/training_summary.json`, `training_metrics.csv`, `config_used.yaml`, retrieval JSON/CSV |
| Frozen representations: 12 active HistCAD/TMCAD representations | `benchmarking/outputs/frozen_representation/results.json`, `summary.csv` |
| Shared TMCAD readouts: five representations | `benchmarking/outputs/tmcad_classification/results.json`, `summary.csv` |
| Matched numeric retrieval adapters: seven representations | `benchmarking/outputs/retrieval_adaptation/results.json`, `summary.csv` |
| Matched functional-text adapters: seven representations | `benchmarking/outputs/semantic_retrieval/results.json`, `summary.csv` |
| Efficiency: seven method families, measured/unavailable status preserved | `benchmarking/outputs/efficiency/` |
| Shared coverage and exact representation inventory | `benchmarking/outputs/inventory.json`, `benchmarking/external/*.json` |
| Full-cohort TMCAD classifier | `downstream_classifier_tmcad_final/outputs/default/` (test predictions, confusion matrices, changed predictions, validation/test comparisons, per-class results, final checkpoint/history) |
| Original full-cohort retrieval adapter | `downstream_retrieval_adapter/outputs/` (evaluation, comparisons, per-query results, qualitative renderings, selected adapter checkpoint) |
| Additional semantic/factorial analyses | `downstream_semantic_study/outputs/`, `outputs_clean/` |
| Manuscript figures and their numerical evidence | `manuscript_latest/figures/`, including residual sensitivity, integrated gradients, spatial response maps, frozen-neighbor comparisons, retrieval-case/asset provenance |

The final full-cohort classifier increases accuracy from 0.8860 to 0.9202 and macro
F1 from 0.8657 to 0.9106 on 1,404 held-out parts. It corrects 55 control errors and
introduces seven. Its residual branch has 33,418 trained parameters and 128/128
hidden widths; older source comments mentioning 96/96 are historical descriptions.

On the common 8,082-part HistCAD bank, direct joint-JEPA dimensional recall@10 is
0.2673; its matched numeric adapter reaches 0.5796. The functional-text adapter
has a different objective: canonical-function match@1 rises from 0.3212 to 0.3989
while dimensional recall decreases. These are separate relation-specific results.

Published BRT/UV-Net/AAGNet values in the manuscript use their original protocols;
they are not additional matched local experimental runs. BRepNet is an official
segmentation-pretrained external reference with its own preprocessing and partial
coverage. Its official source and checkpoint locations are recorded in
`benchmarking/baselines/sources.yaml`; its source tree and data are not vendored.

The final 12-representation benchmark and manuscript-supporting studies are
included. Exploratory runs, width-sweep outputs, redundant checkpoints, full
embedding matrices, and rendered QA previews are omitted. Original JSON/CSV
records are preserved byte for byte. One stale smoke-test assertion was updated
from a 10-epoch expectation to the saved 20-epoch configuration; this does not
change training or results.
