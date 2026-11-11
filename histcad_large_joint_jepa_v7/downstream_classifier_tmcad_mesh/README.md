# Frozen JEPA-v7 TMCAD classification with actual STEP surface profiles

This is a separate downstream experiment. The frozen JEPA-v7 checkpoint,
two-view 2048-D STEP features, 37 full-file radius/bbox descriptors, TMCAD
manifest and split, attention readout, class weighting, and early-stopping
rule remain unchanged. Existing code, processed data, and results are untouched.

## Why this input is different

The equal-weight three-seed geometry ensemble reached test macro-F1 0.850,
but pulley (0.736), coupling (0.776), gear (0.810), and bearing (0.828)
remain weak. Errors cluster among rotational parts: pulley is often called
gear or bearing; coupling is called pulley or bearing. Saved STEP renders show
grooves, teeth, and stepped coaxial profiles that are visually distinctive.
The frozen JEPA graph is capped at 256 local instances. Its per-node numeric
input compresses numbers into generic summary statistics. The 37 added
descriptors improved classification, but their radius quantiles and bbox
ratios do not encode *where* material lies along a part or the local shape of
grooves and teeth. The 23 full-file adjacency counts gave no stable gain
across seeds, and a larger learned interaction worsened performance. More
attention, global counts, or more seeds are therefore not the priority.

The new `mesh_features.py` uses OpenCASCADE to tessellate the actual STEP
solid once and calculate an **area-weighted, orientation-invariant surface
profile**: an 8×8 axial/radial area map, 8-bin surface-normal alignment,
8 axial radial-spread values, and four global shape/complexity values. This
is a compact 84-D descriptor, not a voxelized substitute for JEPA. A fixed
coarse relative deflection and deterministic triangle cap control runtime.
The classifier receives the unchanged frozen JEPA view and the existing
radius/bbox descriptors; a zero-initialized linear head on the new surface
profile is the *only* model change. This directly tests whether local
three-dimensional shape missing from the earlier inputs explains the hard
classes, without increasing downstream network capacity materially.

Geometry extraction uses no labels. Scaling uses valid training rows only.
If OpenCASCADE cannot tessellate a file, that sample's surface vector is set
to zero **after** standardization, so its new head contributes zero and both
comparison arms keep the original split and sample count. The cache metadata
records failures by class and refuses to proceed if failures exceed 10%; a
large failure rate must be investigated rather than silently treated as an
improvement. The saved checkpoint is never updated.

## Run

Check `config.yaml` paths. The existing frozen two-view and 37-D descriptor
caches must be present. From this folder:

```powershell
python -m pip install -r requirements.txt
python run.py --stage mesh
python run.py --stage train
python run.py --stage evaluate
```

`python run.py --stage all` runs the same stages. Meshing all STEP files may
take considerable time; it is cached under this folder's `data/processed`,
reports progress every 100 parts, and reuses descriptors for repeated file
hashes. Outputs go to `outputs/default`. Compare validation macro-F1 first,
then held-out class-wise test results; pay special attention to pulley,
coupling, gear, and bearing. If the mesh view does not help validation, keep
the simpler geometry-only classifier.

No preprocessing, training, or evaluation was run while preparing this code.

## Targeted pulley decision-boundary check

The latest mesh run improved test macro-F1 from 0.840 to 0.866. Its 9,312
surface profiles were valid, so missing mesh rows do not explain the remaining
errors. Pulley has relatively high precision but lower recall on validation
(0.827/0.589) and test (0.862/0.718): it is often underpredicted. Coupling's
validation precision is lower than its recall, and its errors are spread across
several other classes. The training already uses square-root inverse-frequency
weights. These observations do not justify more oversampling, a larger head,
or a change to the frozen encoder at this point.

`calibrate_pulley.py` is a small, controlled check of a *decision-boundary*
limitation. It reuses the saved mesh model and frozen features, selects one
nonnegative pulley logit bias using validation macro-F1 only, then reports
unchanged versus adjusted predictions on the untouched test split. If the
validation improvement is below `minimum_val_macro_f1_gain`, it selects zero
bias. It does not retrain or change any saved model. The sweep and all
class-wise effects are retained, so a pulley gain that damages other classes
is visible. It also writes a confusion plot and representative predictions.

```powershell
python calibrate_pulley.py
```

Results are written to `outputs/pulley_calibration` without overwriting the
existing experiment. If validation does not support a bias, the remaining
errors are more likely due to class overlap or missing discriminative local
information; this script makes no unsupported architectural change.

## Hard-class geometry and label audit

The pulley-bias sweep selected zero: a decision-threshold shift did not
produce a worthwhile validation improvement. Before modifying the model,
run a read-only audit of the difficult validation cases:

```powershell
python audit_hard_classes.py
```

This reuses the saved mesh classifier and all existing feature caches. It
compares coupling and pulley validation queries only with *training-set*
neighbors in five existing spaces: the frozen 2048-D STEP vector, 37-D
geometry descriptors, 84-D surface profile, balanced concatenation of the
two physical descriptor blocks, and the classifier's 256-D penultimate
readout. A deterministic equal-count training-neighbor pool is also reported
as a class-density control. It removes majority-class examples; it does not
augment minority classes or change any model. There is no fitting,
retraining, relabeling, or test-set selection.

`outputs/hard_class_audit/summary.json` reports class counts, confusions,
and nearest-training-label accuracy by feature space. `all_focus_cases.csv`
contains every focus-class validation query, its model prediction and
nearest training examples, including distances to its nearest true-class
and competing-class examples. `manual_review.csv` selects both confident
and borderline mistakes plus correct controls; its blank review columns are
for human judgments, not automatic label changes. Each contact sheet shows
the query, nearest true-class training part, and nearest rival-class training
part in three orthographic views of the actual STEP solid, using uniform
per-part scaling. The three views use the saved STEP coordinate frame, which
may differ across files; consult the linked original STEP files if the
orientation makes a comparison ambiguous. Any rendering failures are
explicitly listed in `render_failures.csv`.

Interpret the neighbor results together with the contact sheets: a correct
physical neighbor but incorrect supervised-readout neighbor suggests the
classifier may not exploit available geometry; wrong neighbors in every
space with visibly distinct shapes suggest missing local features; visually
indistinguishable parts with different class labels suggest overlap or a
label issue. None of these signals alone proves a label is wrong. The audit
does not select a new architecture or alter the frozen JEPA checkpoint.
