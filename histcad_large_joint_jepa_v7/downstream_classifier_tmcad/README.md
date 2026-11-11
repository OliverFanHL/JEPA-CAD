# TMCAD classification with the frozen JEPA-v7 STEP encoder

This folder is a separate downstream experiment. It does **not** modify or
fine-tune the HistCAD JEPA-v7 checkpoint. It uses only the pretrained
**2048-dimensional STEP/B-Rep branch**, not the HistCAD JSON-history branch or
the 4096-dimensional joint vector.

## Local dataset and compatibility inspection

The local `../../TMCAD` folder contains 9,799 `.stp` files in the ten classes
listed in `config.yaml`: bearing (857), bolt_screw (2,070), bracket (1,102),
coupling (474), flange (979), gear (962), nut (899), pulley (544), shaft (893),
and spring (1,019). `bolt_screw` contains nested `bolt/` and `screw/` folders;
both are assigned the *same* combined class. No official split files are
present in this local folder. The code creates a deterministic, per-class
70/15/15 train/validation/test split and keeps byte-identical files in the
same split. Exact file duplicates with conflicting folder labels are excluded.
Near-duplicate solids with different STEP headers are **not** guaranteed to be
grouped; the protocol is exploratory rather than directly comparable with a
published split. The code also does not hash all 162,131 HistCAD pretraining
solids against TMCAD, so independence from pretraining-source duplicates is
not certified.

The STEP syntax is compatible with v7's parser, but the domain is not an
exact match. The frozen HistCAD encoder knows 53 STEP entity IDs; sample TMCAD
files contain unseen IDs such as `CONICAL_SURFACE` and `TOROIDAL_SURFACE`.
An unseen type is mapped to frozen ID 0, while node role, numeric fields,
typed-reference graph and connectivity remain available. The preparation
audit reports unknown-type rates by class, and the final evaluation reports
accuracy separately for lower/higher unknown-type cases. This limitation must
be considered when interpreting transfer performance.

At least one local file, `gear/582.stp`, is an Apple metadata blob rather than
an ISO 10303 STEP file. Preparation checks headers, parsed entity/B-Rep counts,
finite graph values, entity parse coverage, and records exact exclusions in
`data/processed/audit.json`. The local statement scanner handles both `);`
and `) ;` entity endings. Corrected graphs are saved under `graphs_v2/`;
graphs from the older regex-based preparation under `graphs/` are ignored and
left untouched.
It does not claim that every retained solid has passed an OpenCASCADE shape
repair; selected qualitative cases are independently imported and rendered by
OpenCASCADE, with any failures listed in a render audit.

## Pipeline

1. `prepare.py` discovers files, validates their STEP graphs using the **same
   v7 parser and frozen entity vocabulary**, saves one graph cache per unique
   file hash, and writes `manifest.json` and `audit.json`.
2. `extract.py` loads only `step_encoder.*` weights from the v7 `best.pt`,
   freezes every parameter, and saves L2-normalized 2048-D embeddings. The
   source checkpoint and manifest hashes are checked downstream.
3. `train.py` fits a linear probe and a compact attention classifier on the
   identical frozen vectors and split. The attention tokens are the v7 STEP
   encoder's actual contiguous 512-D roles: typed entities, B-Rep instances,
   connectivity, and global STEP context. These are *pooled role summaries*,
   not one token per raw STEP entity. Training uses class-balanced weights,
   dropout/weight decay, and validation macro-F1 early stopping. It does not
   inspect test labels.
4. `evaluate.py` evaluates the selected heads on test, compares against direct
   cosine 1-NN on the frozen representation, and saves accuracy, balanced
   accuracy, macro/weighted F1, top-3 accuracy, per-class precision/recall/F1,
   confusion matrices, training-class centroid cosine heatmap,
   transfer-quality breakdowns, and representative
   correct/error/nearest-neighbor cases.
5. `render_cases.py` optionally produces per-class sheets showing the actual
   STEP solids for selected test parts and their nearest training neighbors.
   Failed imports are audited and never displayed as if they were valid CAD.

## Run locally

From this folder, use the Python environment containing the v7 dependencies:

```powershell
python -m pip install -r requirements.txt
python run.py --stage prepare
```

Inspect `data/processed/audit.json`, particularly exclusions and per-class
unknown STEP entity rates. Then run:

```powershell
python run.py --stage extract
python run.py --stage train
python run.py --stage evaluate
```

OpenCASCADE rendering needs `OCP` (for example, the `cadquery-ocp` package).
If available, the evaluate stage renders the qualitative sheets automatically.
Use `--skip-render` to retain all quantitative and case-CSV results without
rendering, or rerun `python render_cases.py --model attention` later. A single
`python run.py --stage all` runs every stage in order. Change paths, batch
sizes, or the output directory in `config.yaml` as needed; paths are resolved
relative to that configuration file, not the current terminal directory.

Key outputs are under `outputs/default/`: `model_comparison.csv`,
`frozen_step_1nn.json`, `frozen_step_class_centroids.png`, and each model's `evaluation.json`,
`per_class_test.csv`, `confusion_matrix.png`, `qualitative_cases.csv`,
optional `qualitative_<class>.png` sheets, and `render_audit.json`.

## Structural-information control

`config_structural.yaml` runs two additional downstream heads on the **same
saved split and same frozen 2048-D JEPA vectors**. No STEP or JEPA model is
fine-tuned. `structural_features.py` reads the prepared graph cache and adds
52 train-only-scaled descriptors: counts of known B-Rep entity types, presence
of previously unseen geometric B-Rep types, graph-reference counts, and
graph-size/retention statistics. Presentation/style/annotation entity types
are excluded. The simple linear head is an information-content control; the
attention head keeps the previous four-token architecture and adds a small
residual descriptor head. This is a test of whether information was lost in
the pooled STEP embedding, **not** evidence that the standalone frozen branch
fully represents surface geometry or shape semantics.

With `prepare` and `extract` already complete, run only:

```powershell
python run.py --config config_structural.yaml --stage train
python run.py --config config_structural.yaml --stage evaluate
```

The local controlled run selected both heads by validation macro-F1 and then
evaluated the held-out test once:

| Input/head | Validation macro-F1 | Test accuracy | Test macro-F1 | Coupling F1 | Pulley F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen STEP / linear | 0.725 | 0.759 | 0.727 | 0.551 | 0.599 |
| Frozen STEP / attention | 0.808 | 0.839 | 0.812 | 0.641 | 0.676 |
| Frozen STEP + geometric graph descriptors / linear | 0.765 | 0.794 | 0.757 | 0.552 | 0.593 |
| Frozen STEP + geometric graph descriptors / attention | 0.823 | 0.850 | 0.826 | 0.675 | 0.705 |

These results support a limited information-bottleneck diagnosis: extra B-Rep
type/connectivity summaries help both head capacities without changing the
encoder. Coupling and pulley nevertheless remain difficult. The training
counts are 317 and 359 respectively versus 1,405 for the largest class, but
inverse-square-root class weights were already used, errors cross multiple
classes, and coupling's high-OOV examples are not uniformly worse. For most
samples only 256 of thousands of B-Rep candidates become instance nodes.
Rendered remaining errors include coupling/flange and pulley/flange shapes;
some distinctions rely on local surface/feature geometry that these summaries
do not capture. Arbitrary augmentation of frozen vectors would not restore
missing geometric detail, and naive oversampling could memorize ambiguous
minority examples. Neither was added.

The labels are folder-level part categories, not HistCAD's annotation-derived
functional nouns. This experiment measures **STEP-branch transfer**, not full
joint CAD representation transfer. Near-duplicate STEP solids may still occur
across the exploratory split.
