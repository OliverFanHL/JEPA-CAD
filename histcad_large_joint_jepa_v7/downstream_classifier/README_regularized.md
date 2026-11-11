# Regularized, validation-audited attention classifier

This is the targeted follow-up to [the 32-epoch attention run](README_attention.md).
It uses the same frozen JEPA-v7 `best.pt` vectors, original UID
train/validation/test split, role-token attention architecture (2,707,492
trainable downstream parameters), four tasks, and inverse-square-root class
weights. JEPA is never fine-tuned. The maximum budget remains 32 epochs, but
training can stop early using **validation mean macro-F1 only**.

## Diagnosis and targeted changes

The previous attention model was not under-capacity: at its selected epoch,
part-form macro-F1 was 0.780 on training versus 0.557 on validation, and
weighted validation loss rose as training loss continued to fall. Increasing
hidden size or layer count was therefore not justified.

Validation failure inspection also found an annotation-rule error. The old
last-noun rule mapped phrases such as *square block extrusion* to `profile`
because it considered *extrusion* the last noun. In the refined taxonomy,
explicit specific form nouns (e.g. *block*, *pin*, *beam*) take precedence over
generic fallback words (*profile/extrusion/section*, *cylinder*, *base*).
This changed 2,517 **training** labels. The rule was fixed from training and
validation analysis, not test examples. The minimum class support is 500
training samples, preserving the same 23 leaf class names as the previous
taxonomy. There is no new `other` or `unassigned` prediction class. Part-form
coverage is 140,701/145,915 train (96.4%), 7,837/8,130 validation (96.4%),
and 7,778/8,086 test (96.2%). Exact rules are in
[`taxonomy_refinement_v2.json`](taxonomy_refinement_v2.json) and
[`labels.py`](labels.py).

The revised attention training uses dropout 0.20 (previously 0.10) to reduce
co-adaptation of the ten CAD summary tokens; 0.05 label smoothing **only for
the noisy part-form task** to temper high-confidence errors; and patience 7
with a minimum 0.001 validation macro-F1 improvement for early stopping.
The existing 1e-4 weight decay, class weights, optimizer, width, and two-layer
attention architecture were left unchanged. No capacity increase or broad
hyperparameter search was performed.

## Same-taxonomy results

A new joint MLP baseline was trained on the **identical refined labels**;
its saved `labels.csv` has the same hash as the attention run's label file.
All test metrics below are on the original held-out 8,086 UIDs, with per-task
coverage as recorded in the label mapping.

| Refined-label frozen-v7 readout | Form accuracy | Form balanced accuracy | Form macro-F1 | Sketch macro-F1 | Constraint macro-F1 | History macro-F1 | Mean macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Joint MLP, 8-epoch budget | 71.2% | 58.1% | 0.518 | 0.996 | 0.973 | 0.922 | 0.852 |
| Regularized role-token attention | **74.7%** | **61.9%** | **0.561** | **0.999** | **0.994** | **0.997** | **0.888** |

Attention selected epoch 13 and stopped at epoch 20. Its selected-checkpoint
train/validation/test form macro-F1 values were 0.640/0.573/0.561. Validation
loss was roughly flat near the selected epoch; the remaining form gap is 0.067,
not zero. This is descriptive evidence that overfitting was reduced, **not** a
controlled attribution to any one change, since the taxonomy also changed.
The old-taxonomy attention result (form F1 0.548) is context only and must
not be presented as a same-label numerical baseline: 140 held-out form labels
also change under the deterministic v2 rule. Validation examples helped
identify the label-rule error, so validation is not an independent label-design
holdout; this remains an exploratory benchmark rather than a blind external
test of the taxonomy.

Full metrics and the 20-epoch trajectory are saved in
[`attention_results.json`](outputs/attention_regularized_v2/attention_results.json),
[`training_history.csv`](outputs/attention_regularized_v2/training_history.csv),
and [`training_curves.png`](outputs/attention_regularized_v2/training_curves.png).
The same-taxonomy MLP record is
[`baseline_results.json`](outputs/mlp_refined_v2/baseline_results.json).

## Class support and weak-class analysis

The complete 23-class train/validation/test distribution, precision, recall,
F1, top confusion, STEP graph truncation rate, and same-taxonomy MLP recall
are in
[`class_distribution_and_performance.csv`](outputs/attention_regularized_v2/class_distribution_and_performance.csv)
and the
[`classwise_validation_recall.png`](outputs/attention_regularized_v2/classwise_validation_recall.png)
chart. Lowest validation recalls are:

| Leaf | Train / val / test | Val recall | Main issue |
|---|---:|---:|---|
| cylinder | 1,457 / 71 / 70 | 0.169 | Generic shape label overlaps tube, pin, rod, spacer; no STEP truncation in these validation parts. |
| beam | 779 / 43 / 56 | 0.372 | Often visually equivalent to a long bar/profile. |
| sleeve | 916 / 50 / 57 | 0.380 | Hollow sleeves, bushings, and shafts overlap. |
| plug | 988 / 64 / 60 | 0.422 | A short cylindrical plug can look like a disc/spacer without usage context. |
| profile | 1,725 / 89 / 93 | 0.506 | Very broad descriptor; 42.7% of validation cases exceed the v7 256 B-Rep-instance cap. |
| shim | 830 / 54 / 43 | 0.556 | Thin plates and shims can have identical visible form. |

The rare-class weighting already used in both models did not solve these
confusions; several weak classes have around 1,000-1,700 training examples,
so sample count is not the only cause. All eight selected validation STEP
examples in the [failure sheet](outputs/attention_regularized_v2/validation_failure_sheet.png)
were successfully imported and rendered with OpenCASCADE; there is no evidence
from those cases of file corruption. See
[`validation_failure_examples.csv`](outputs/attention_regularized_v2/validation_failure_examples.csv)
and [`failure_review.md`](failure_review.md) for a manual label check. The
full rendered held-out qualitative sheet is
[`qualitative_sheet.png`](outputs/attention_regularized_v2/qualitative_sheet.png).

Improvement is not uniform: on validation, attention recall is higher than
the refined-label MLP for 12 of 23 leaves, tied for one, and lower for ten;
some small classes have noisy recall estimates. Increasing class weights or
model size further would not reliably fix ambiguous noun definitions.

## Reproduce

From this directory:

```powershell
python run_attention.py --config config_regularized.yaml
python run_refined_mlp_baseline.py
python analyze_regularized.py
python plot_attention_history.py --output outputs/attention_regularized_v2
python render_attention_cases.py --output outputs/attention_regularized_v2
```

The attention training entry point and configuration are
[`run_attention.py`](run_attention.py) and
[`config_regularized.yaml`](config_regularized.yaml). Rerunning writes to the
dedicated `outputs/attention_regularized_v2` folder; the original 32-epoch
experiment and earlier classifier checkpoints remain unchanged.
