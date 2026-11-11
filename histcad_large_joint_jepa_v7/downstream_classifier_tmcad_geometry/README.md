# Frozen JEPA-v7 TMCAD classifier: full-file geometric control

The two-view and role-delta experiments are unchanged. This independent
downstream run reuses their existing frozen original/complementary STEP pools,
the same TMCAD manifest and split, and the same JEPA-v7 checkpoint. It never
updates JEPA. The new information is an uncapped, small set of physical
geometry descriptors read from each STEP file; it is not another learned
readout of the same frozen vector.

## Why this comparison

The latest role-delta arm reached validation macro-F1 0.821 versus 0.822 for
the existing two-view control; the earlier direct readout also failed to
improve validation macro-F1. Increasing downstream fusion capacity has no
support from these runs. The frozen STEP branch, however, sees at most 256
instance nodes and encodes each node's numbers into twelve generic summary
statistics. The train-only cap audit shows many coupling and gear parts have
well over 2,000 candidate B-Rep instances. Even the balanced 256-instance
view cannot represent all local radii or point extents. Increasing the cap
would change the frozen encoder's pretraining input distribution and make
extraction more expensive; it is not the first controlled test.

`geometry_features.py` therefore reads *all* parseable `CARTESIAN_POINT`,
`CIRCLE`, `CYLINDRICAL_SURFACE`, `TOROIDAL_SURFACE`, and `SPHERICAL_SURFACE`
entities. It computes 37 label-free descriptors: 3D point count, sorted
bbox aspect/relative-span summaries, and radius quantiles/count/diversity
for five primitive families. Radii are normalized by full-file bbox diagonal
(with a radius-derived fallback) for unit and scale robustness. STEP
references and quoted strings are stripped before parsing numbers, so entity
IDs and text cannot masquerade as dimensions. No CAD filename, label, or
functional text enters the descriptors. Standardization uses training rows
only, with a five-standard-deviation clip; validation/test statistics do not
affect preprocessing.

The classifier comparison changes only a zero-initialized linear residual
head on these standardized descriptors (37 × 10 weights). The original
2048-D JEPA pool, complementary 2048-D pool, attention, hidden width,
regularization, split, and validation-based model selection are unchanged.
This tests whether physically meaningful local geometry absent from the
frozen view helps classification, without an increased node cap or more
classifier depth. It is exploratory, not a claimed performance gain.

## Run

The prepared TMCAD manifest and frozen two-view cache must exist at the
configurable paths in `config.yaml`. From this folder:

```powershell
python -m pip install -r requirements.txt
python run.py --stage geometry
python run.py --stage train
python run.py --stage evaluate
```

`python run.py --stage all` runs these stages in order. Geometry preprocessing
parses source STEP files once per unique file hash and reports progress every
250 manifest items; subsequent runs reuse `data/processed/geometry.npy`.
The two classifier arms and all results are written under `outputs/default`.
Select using validation macro-F1; inspect per-class precision/recall/F1 and
the confusion matrix, then use the held-out test result once. If the geometry
arm does not improve validation performance, retain the two-view baseline.

No training or evaluation was run while preparing this code.

## Follow-up: low-rank geometry × JEPA interaction

The completed 37-descriptor run improved validation macro-F1 from 0.822 to
0.827 and held-out test macro-F1 from 0.827 to 0.840. Improvement across
several classes supports the presence of complementary physical information.
However, training loss continued to fall after validation performance
leveled off, so a larger classifier, more attention layers, or a larger
instance cap is not warranted by these results.

`config_interaction.yaml` compares the existing additive geometry model with
one small interaction arm using exactly the same frozen features and labels.
The added pathway multiplies a rank-16 projection of the 37 descriptors by a
rank-16 projection of the existing 256-D supervised JEPA readout, then maps
that product to class logits. Its output layer starts at zero, and the
original linear geometry head remains. This adds 4,848 parameters for ten classes,
testing whether the *meaning* of a radius/extent descriptor depends on the
JEPA structural context without expanding the frozen representation or
extracting new features. All regularization and stopping settings remain
matched to the control.

The existing geometry cache is reused. From this folder:

```powershell
python run.py --config config_interaction.yaml --stage train
python run.py --config config_interaction.yaml --stage evaluate
```

Results go to `outputs/interaction`; `outputs/default` is unchanged. Select
by validation macro-F1 and retain the additive model if the interaction
does not help validation. No training or evaluation was run for this update.
