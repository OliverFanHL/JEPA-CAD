# Frozen JEPA-v7 TMCAD classification: complementary B-Rep graph views

This is a separate downstream experiment. It imports no code or checkpoints
from either earlier TMCAD classifier folder. It reuses the **same validated
TMCAD manifest and split** and the same frozen JEPA-v7 STEP checkpoint.
The existing implementations, caches, checkpoints, and results are unchanged.

## Why this experiment

The latest local-token run did **not** support a larger classifier: pooled-only
test macro-F1 was 0.818; attending to all available pre-pooling nodes fell to
0.810, with validation macro-F1 peaking at epoch 8 while training loss kept
decreasing. The earlier 52-descriptor attention control achieved 0.826 test
macro-F1, suggesting some useful B-Rep structure was not accessible in the
original pooled features, but simply expanding descriptors is not the goal.

The original v7 parser orders B-Rep instances by topology-first marker
priority and retains the first 256. In a **train-only, 20-per-class** coverage
audit, median curve-instance counts at the 256 cap were 0.5 for coupling,
9.5 for pulley, and 0 for gear. At 512 they were 1, 56, and 31; at 1024
they were 57.5, 88, and 217.5. A larger cap therefore can add curve
information, but 512 remains particularly incomplete for coupling, while
1024 quadruples instance count relative to JEPA pretraining. Since the
local-token readout already overfit at 256, feeding it four times as many
similar-priority nodes is not the first supported change.

This version instead holds **256 instances**—the cap used to pretrain v7—and
builds one complementary graph with a deterministic 128 topology / 64 surface
/ 64 curve allocation (unused slots are filled by original priority). It
prefers topological references connected to selected geometry, while keeping
all type-level aggregates and typed-reference edges. The original graph is
also retained. The frozen STEP encoder is run on both; nothing is fine-tuned.
The supervised comparison changes only attention keys/values:

- `pooled_fullwidth`: four 512-D roles from the original graph attend to each
  other; the original 2048-D pooled vector reaches the head unchanged.
- `pooled_plus_brep_view`: the same four original queries attend to the four
  512-D pooled roles from the complementary graph; the original vector and
  classifier parameterization are otherwise unchanged.

This controlled design tests **which geometry is selected before JEPA
pooling**, without increasing node count, embedding width, handcrafted
descriptor count, or classifier capacity. The alternative view is nonetheless
a changed input distribution for the frozen encoder, so any benefit must be
established on validation and then confirmed once on test. It is not a
guaranteed improvement.

## Run

Inspect `config.yaml` paths, then from this folder:

```powershell
python -m pip install -r requirements.txt
python audit_brep_coverage.py --sample-per-class 20 --save coverage_train_sample.json
python run.py --stage extract
python run.py --stage train
python run.py --stage evaluate
```

The audit is optional; its supplied result was computed only on sampled
training files. `python run.py --stage all` runs extraction, training, and
evaluation sequentially. The extraction caches complementary graphs by
file hash under this folder, then saves both frozen 2048-D vectors in this
folder's `data/processed`. Parsing all STEP sources can be slow, but it is
performed only once per unique file. The source dataset's split, original
graph caches, and original JEPA checkpoint remain read-only.

Training uses the same seed, inverse-square-root class weights, smoothing,
dropout, weight decay, and validation-macro-F1 early stopping as the recent
full-width classifier. Check `outputs/default/model_comparison.csv` and both
models' per-class files; do not select the view by the test set. Qualitative
case CSVs list actual STEP paths and, for the dual-view model, attention over
the four complementary summary roles. Attention is diagnostic, not causal
attribution. No training or evaluation was run while preparing this code.

## Minimal follow-up: direct complementary readout

The saved two-view run reached validation macro-F1 0.822 versus 0.809 for
the pooled-only control. Its test macro-F1 was 0.827 versus 0.822. Benefits
varied by class, so these results do not justify increasing the graph cap,
adding another local-token model, or a wider attention stack. Attention
diagnostics also often assign most weight to the global role; they cannot
show whether the complementary roles survive the readout.

`config_direct_readout.yaml` therefore compares the existing two-view model
with `pooled_plus_brep_view_direct`. The latter retains the same frozen
original and complementary 2048-D vectors, the same cross-attention pathway,
and the same classifier head, but adds a zero-initialized linear logit readout
of all four complementary roles (about 20,000 additional weights for ten
classes). This small residual path can expose view information that attention
may suppress without supplying handcrafted geometry or changing JEPA. It is
an experiment, not an established improvement. Select by validation macro-F1
and inspect class-wise results; the test split remains final assessment only.

The existing frozen feature cache is reused. From this folder, run:

```powershell
python run.py --config config_direct_readout.yaml --stage train
python run.py --config config_direct_readout.yaml --stage evaluate
```

Results go to `outputs/direct_readout`, leaving `outputs/default` intact.
The follow-up code was prepared without running training or evaluation.

## Role-aligned complementary difference

The completed direct-readout comparison did not improve aggregate validation
macro-F1: 0.819 versus 0.822 for the existing two-view model. Test macro-F1
was 0.825 versus 0.827. Some classes improved while others worsened. Thus,
more generic readout capacity is not supported by these results. The saved
attention examples often emphasize the global STEP role, which can obscure
the change in selected local B-Rep instances.

`config_role_delta.yaml` adds one controlled alternative:
`pooled_plus_brep_view_role_delta`. It keeps the existing cross-attention and
original 2048-D feature path, but adds a zero-initialized linear classifier
on the signed difference between the original and complementary **first
three corresponding roles** (typed entities, B-Rep instances, connectivity).
The global role is excluded from this extra path. This adds only 1,536 ×
number-of-classes weights. It reuses the same frozen cached views and
validation-selected training protocol; it does not add descriptors, increase
the 256-instance cap, or alter JEPA. It specifically tests whether the
complementary graph's changed information is being masked by attention and
the largely unchanged global summary.

```powershell
python run.py --config config_role_delta.yaml --stage train
python run.py --config config_role_delta.yaml --stage evaluate
```

The new run writes to `outputs/role_delta`; previous checkpoints/results
remain intact. No training or evaluation was performed while making this
revision. If validation macro-F1 does not improve, prefer the existing
two-view model rather than adding further fusion complexity. The parser's
12-number per-node summary and the v7 pooling can still lose exact local
geometry; a small downstream head cannot recover information absent from
both frozen views.
