# Frozen JEPA-v7 TMCAD classification: full-file B-Rep topology control

This independent downstream experiment preserves the prior frozen JEPA-v7
checkpoint, original and complementary 256-instance STEP embeddings, the
37 full-file radius/bbox descriptors, TMCAD split, and classifier architecture.
It does not train or alter JEPA. Existing experiment code, caches, checkpoints,
and results remain unchanged.

## Tested question

The low-rank JEPA × geometry interaction fell below the additive geometry
control: validation macro-F1 0.824 versus 0.827 and test macro-F1 0.818
versus 0.840. Increasing fusion capacity is therefore not the supported next
step. The geometric descriptors helped, but they contain only marginal radius
and bbox statistics. The frozen STEP encoder uses a 256-instance cap, so its
instance graph cannot retain every local face/edge relationship on complex
parts. The remaining comparison targets this missing *input information*.

`topology_features.py` reads the full STEP DATA section, follows
`ADVANCED_FACE → FACE_BOUND → EDGE_LOOP → ORIENTED_EDGE → EDGE_CURVE`
references, identifies the underlying surface and edge-curve types, and
computes 23 compact, label-free descriptors: face/edge coverage, boundary and
shared-edge fractions, face-component and degree summaries, surface-pair
adjacency fractions, and curve types on shared edges. It does not enlarge the
frozen graph or invent parametric history. Parts whose STEP export lacks these
relations receive zero raw topology features; cache metadata reports the
coverage and number of nonconstant training features. Standardization uses
training rows only.

The control is the existing geometry-augmented classifier. The new arm adds
only a zero-initialized linear residual readout of the 23 topology values,
leaving JEPA, attention, geometry readout, hidden width, optimizer, and early
stopping unchanged. Select by validation macro-F1 and inspect class-wise
results; no improvement should be inferred until the user runs the test.

## Run

The paths in `config.yaml` point to the existing TMCAD manifest, frozen
two-view feature cache, and geometry cache. They can be changed if those
artifacts were moved. From this folder:

```powershell
python -m pip install -r requirements.txt
python run.py --stage topology
python run.py --stage train
python run.py --stage evaluate
```

`python run.py --stage all` also verifies/reuses the existing geometry cache.
Full-file topology parsing reports progress every 250 parts and is cached
locally as `data/processed/topology.npy`. Results go to `outputs/default`.
No training or evaluation was run while preparing this implementation.

## Repeat-seed check before another model change

The completed topology run reached validation macro-F1 0.833 versus 0.827
for the geometry control, but held-out test macro-F1 was 0.841 versus 0.840.
The approximately 0.001 test difference, mixed class-level changes, and
continued train–validation gap do not justify another classifier, larger
cap, or descriptor family on the basis of one seed. The next useful step is
to measure run-to-run variability of the *unchanged* two arms.

`run_repeats.py` trains both arms from the same frozen cached inputs with
paired seeds, independently selects each checkpoint by validation macro-F1,
and reports per-seed metrics, means/standard deviations, paired differences,
and per-class F1 spread. It does not select an architecture by test results
or modify JEPA, descriptors, the split, or training hyperparameters.

```powershell
python run_repeats.py --seeds 97 113 127
```

Results are written to `outputs/repeated_seeds`; the script refuses to
overwrite an existing seed directory. Use `--output` to choose another
destination if needed. No repeat training or evaluation was run while
preparing this script.

## Fixed seed ensemble, without another model change

The completed three-seed replication did not support the topology arm:
mean topology-minus-geometry validation macro-F1 was -0.0008, while the
test difference was +0.0020 with a 0.0044 standard deviation and changed
sign across seeds. The geometry-only heads themselves varied more than this
difference. Thus, more topology features or deeper fusion are not justified
by the current evidence.

`evaluate_seed_ensemble.py` uses only the three already-trained
`pooled_plus_brep_view_geometry` checkpoints from `outputs/repeated_seeds`.
It takes their **equal-weight probability average**, fixed in advance, and
reports validation/test metrics, per-class F1, confusion matrix, predictions,
and representative cases. There is no new training, no learned weighting,
and no test-based choice of seeds or model. This tests whether averaging the
observed seed variability improves robustness without touching JEPA or the
downstream architecture.

```powershell
python evaluate_seed_ensemble.py --seeds 97 113 127
```

Results go to `outputs/geometry_seed_ensemble`. The script refuses to
overwrite existing ensemble results; use `--output` for a new destination.
No ensemble evaluation was run while preparing this code.
