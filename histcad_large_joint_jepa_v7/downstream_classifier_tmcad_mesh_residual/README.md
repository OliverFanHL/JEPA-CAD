# Frozen TMCAD mesh classifier: controlled nonlinear readout

This folder is a separate experiment. It **does not modify** JEPA-v7, the
existing TMCAD mesh-classifier checkpoint, cached input features, or previous
results. The old model is reconstructed exactly from its saved state dict and
remains frozen throughout. This folder contains its own data validation,
model definitions, training, evaluation, configuration, and outputs; it does
not import Python modules from earlier experiment folders. The pre-existing
manifest and NumPy feature caches are read-only inputs.

## Why these two small variants

The validation audit found correct-class surface-profile nearest neighbors
for 10/16 coupling and 14/30 pulley mistakes. Equal-count reference pools
changed nearest-neighbor accuracy only modestly. The present mesh head is a
linear 84-D-to-class logit addition, with no nonlinear interaction with the
37-D geometry block or frozen JEPA context. These results motivate a narrow
readout test, not a larger JEPA branch, more oversampling, or another global
logit bias.

- `physical_residual`: 37-D geometry + 84-D surface profile through a small
  nonlinear MLP, added to the **fixed baseline logits**.
- `conditional_residual`: the same physical information, with a compact
  additional projection of the fixed 256-D JEPA/two-view classifier context.
  This tests whether cross-source conditioning adds value. That context is
  upstream of the original geometry and mesh logit additions; it is not the
  whole original classifier embedding.

Both residual outputs start at exactly zero. Only the residual parameters
are trained. Dropout, weight decay, a residual-magnitude penalty, and early
stopping limit adaptation to what validation supports. Both variants use the
identical frozen inputs, TMCAD split, sqrt-inverse class weights, label
smoothing, and metrics. There is no extra label supervision or augmentation.

## Run

Check the relative input paths in `config.yaml`. From this folder:

```powershell
python -m pip install -r requirements.txt
python run.py --stage all
```

Or use `--stage train` followed by `--stage evaluate`. Training materializes
only train/validation frozen features under `data/processed`; it never
extracts test rows or uses test labels for adaptation/selection. The baseline
validation macro-F1 must reproduce the existing
0.857506 within the configured tolerance, otherwise execution stops. Both
adapters train for at most 32 epochs and save their validation-best weights
inside this folder. Running `train` again against an existing adapter
checkpoint refuses to overwrite it; choose another output path in the config
if repeating the experiment.

Evaluation selects a single variant **using validation macro-F1 only**. A
candidate must gain at least 0.003 over the frozen baseline; the conditional
variant must additionally exceed the physical-only variant by 0.002 to
justify its extra complexity. Otherwise the simpler model or unchanged
baseline is selected. Only after that choice does evaluation open the test
split. Test results report the original baseline and the selected model, not
test scores for unselected variants.

Inspect `outputs/default/validation_comparison.csv`,
`validation_per_class.csv`, `test_comparison.csv`, `test_per_class.csv`,
`changed_predictions.csv`, and `qualitative_review.csv`. The latter lists
original STEP paths for corrected, worsened, and still-wrong coupling/pulley
parts so they can be inspected in a CAD viewer. Existing audit contact sheets
may contain holes due to sparse display-triangle sampling; do not relabel
parts from those pictures alone. The TMCAD split prevents exact-file
duplicates crossing splits, but near-identical parameter variants may still
exist, so nearest-neighbor audit numbers are diagnostic rather than a
guarantee of generalization.

No training or evaluation was run when this code was prepared.
