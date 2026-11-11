# Frozen JEPA-v7 classification probe

For the stronger 32-epoch role-token attention follow-up, see
[`README_attention.md`](README_attention.md).
For the validation-audited regularized revision, see
[`README_regularized.md`](README_regularized.md).

This experiment measures which HistCAD attributes can be decoded from the
**frozen** `best.pt` JEPA-v7 representation. It does not update JEPA weights.
The default path reuses `../outputs/full_dataset_4096/representations.pt`,
which the v7 training run extracted from `best.pt` at epoch 20. `--extract`
regenerates the same vectors from the v7 graph cache and checkpoint without
training the encoder; that takes considerably longer and creates another large
representation file.

## Taxonomy and reliability

The exact machine-readable taxonomy is in [`taxonomy.json`](taxonomy.json).
[`labels.py`](labels.py) applies it to the aligned v7 metadata and records the
final mapping and per-split counts in `outputs/label_mappings.json`. The
classes are selected using **training counts only** (minimum 300); rare classes
are dropped rather than placed in a vague `other` class. A missing/ambiguous
label is represented by `-1` and masked in that task's loss and metrics. The
same part can still be used in the other tasks. No predicted class is called
`unassigned`.

The hierarchy has three major categories:

| Major category | Task / subclasses | Source and interpretation |
|---|---|---|
| Part-form noun | 23 supported leaves under `planar_and_support`, `prismatic_stock`, `axial_and_annular` | The head phrase of the `Functional Type` annotation. In compounds such as *shim plate*, the last explicit noun (*plate*) is the primary form. Phrases offering alternatives with *or* abstain. These are **form nouns**, not validated functional-use labels. |
| Geometric structure | Four sketch-composition classes and four constraint-regime classes | Counts and names of sketch primitives and constraints in the JSON history, as preserved in v7 structural metadata. |
| CAD modeling history | Five common operation-sequence classes | Recorded `NewBody`, `Join`, and `Cut` patterns in the JSON history. The rare `Intersect` class is excluded. These are **not** real machining, molding, or turning process labels. |

`Geometric Feature`, `Modeling Process`, and `NLT` are descriptive free text,
often paraphrasing the CAD history. They are not used as closed-world negative
labels here. Manufacturing process labels, materials, and actual downstream
part-use labels are not available in this aligned HistCAD split, so the study
does not claim to classify them. Functional-form labels are automatically
generated annotations, not independent human-verified ground truth.

The train-only support check retains 23 part-form leaves (140,981/145,915
training rows, 96.6% coverage). Seven rare noun classes are excluded. All
sketch and constraint rows are labelled. Modeling-history coverage is 99.9%
after dropping 111 training intersections. The saved mapping contains exact
train/validation/test counts for every retained leaf.

## Protocol

The original v7 deterministic split is retained: 145,915 training, 8,130
validation, and 8,086 test parts. Taxonomy rules and the 300-example threshold
were fixed from training annotations and training structural metadata. A
linear multi-task probe is fit separately to the frozen `joint`, `parametric`,
and `step` vectors. A 512-wide one-hidden-layer MLP probes the frozen joint
vector for nonlinear decodability. Each has separate heads for the four tasks.
Validation mean macro-F1 selects the probe epoch; test predictions and metrics
are computed only afterward. A training-majority baseline is reported for
every task. No text, label, or test metric enters JEPA pretraining in this
experiment.

Metrics in `outputs/classification_results.json` and the compact
`outputs/classification_summary.csv` include accuracy, balanced
accuracy, macro/weighted F1, per-class recall, confusion matrices, labelled
coverage, and majority-class baselines. For part-form nouns, parent-category
accuracy and macro-F1 are also reported. `outputs/*_test_predictions.npz`
contains aligned UIDs, targets, and predictions; checkpoints contain only
probe weights, not JEPA weights.

## Run

From this directory, in the existing PyTorch environment:

```powershell
python run.py --stage labels
python run.py --stage train
```

`python run.py` performs both stages. Add `--extract` only if the v7
representation cache is absent or must be regenerated from `best.pt`.
All paths and hyperparameters are in [`config.yaml`](config.yaml), resolved
relative to this directory. The code imports only the v7 source modules for
optional frozen extraction; it does not import earlier downstream experiments.

## Current interpretation

Held-out test results (best epoch selected on validation only):

| Frozen vector + probe | Form accuracy / macro-F1 | Sketch macro-F1 | Constraint macro-F1 | Modeling macro-F1 |
|---|---:|---:|---:|---:|
| Joint 4096-D + linear | 65.8% / 0.455 | 0.994 | 0.968 | 0.874 |
| Parametric 2560-D + linear | 70.0% / 0.500 | 0.999 | 0.993 | 0.989 |
| STEP 2048-D + linear | 57.2% / 0.396 | 0.965 | 0.609 | 0.583 |
| Joint 4096-D + small MLP | 69.1% / 0.500 | 0.997 | 0.977 | 0.921 |

The test majority-class accuracy for the form task is 34.4%.
The joint linear probe reaches 0.839 macro-F1 for the three parent form
categories; the joint MLP reaches 0.856. These numbers apply only to samples
with a supported form label (7,789 of 8,086 test parts).

On this split, the parametric-only linear probe outperforms the joint linear
probe, especially on history-derived labels. That result is plausible because
these labels are direct functions of the parametric inputs. The joint MLP
improves over the joint linear probe but does not make the joint representation
best on every task. Very high sketch/constraint/operation scores mean these
input-derived attributes are recoverable; they are **not** evidence of
independent semantic understanding or generalization to new CAD families.
The part-form task is harder and more relevant to semantic transfer, but its
automatically generated annotation and UID-level split can still overstate
generalization to truly novel design families.
