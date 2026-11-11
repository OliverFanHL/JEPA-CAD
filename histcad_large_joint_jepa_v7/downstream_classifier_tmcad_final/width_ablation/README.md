# Controlled 256-D versus 512-D classifier-hidden comparison

This is an optional experiment **inside the finalized version**. It does not
change the frozen JEPA-v7 STEP encoder or the default final model/checkpoint.
It compares the 256-D and 512-D *supervised hidden layers*, not JEPA output
dimensions. Both arms receive the same original 2048-D frozen STEP vector,
the same four complementary 512-D STEP summaries, and the same 37+84 physical
descriptors. The concatenated supervised head input stays 4096-D in both.

The saved 256-D classifier cannot be resized to 512-D without inventing new
weights. Consequently, **both classifier heads are freshly trained** on the
same fixed TMCAD train/validation split under the same optimizer, epochs,
regularization, class weights, seed, and checkpoint rule. The *only* Stage-A
architectural difference is `Linear(4096, 256)` versus `Linear(4096, 512)`.
Then each validation-best classifier is frozen, and an identical
`121 -> 96 -> 96 -> 10` physical residual is trained for that arm. The 96-D
physical width stays fixed, isolating the supervised head-width question.

From this folder:

```powershell
python -m pip install -r requirements.txt
python run.py --stage all
```

`--stage all` trains, evaluates, and prepares the visual comparison. Or run
`--stage train`, `--stage evaluate`, and `--stage visualize` separately.
The visualization source is in the parent finalized folder:
`../visualize_width_ablation.py` and `../step_render.py`. After evaluation,
you can also run `python ../visualize_width_ablation.py` directly.
The rendering stage additionally requires OpenCASCADE's `OCP` module
(`cadquery-ocp`) in the active CAD environment. Missing OCP does not
invalidate completed training or numeric evaluation; after installing it,
rerun only `python run.py --stage visualize`. This code imports only
modules from its parent finalized folder; it does not import any earlier
experiment. Existing source checkpoints, split manifests, and feature arrays
are read-only. New checkpoints/logs/results go under `outputs/default` here.

Training never extracts test rows. Evaluation first selects 512 only if its
validation macro-F1 beats 256 by at least 0.003; otherwise it retains 256.
It then reports both predeclared widths on the held-out test for comparison,
with the validation-selected width marked explicitly. Do not revise the
width choice after seeing those test results. The experiment also writes
class-wise metrics, confusion plots, prediction changes, and STEP paths for
corrected/worsened coupling and pulley examples.

The visualization stage writes `outputs/default/visualization/index.html`,
`metric_comparison.png`, `class_change_summary.csv`, class-specific sheets,
individual two-view case cards, and `render_audit.json`. The index shows
per-class recall changes and corrected/worsened counts. It draws **actual
STEP solids**, not approximations
made from frozen descriptors. Images use bounded tessellation for display only;
model inputs and metrics remain unchanged. Each class contributes up to one
test example from each available outcome: 512 corrected, 512 worsened, both
wrong, and both correct. The cards print the ground-truth class, both
predictions, and both confidences. All selected cases, including failed STEP
imports, are listed in `rendered_cases.csv` and `render_audit.json`; missing
renders are never shown as if they were successful.

If an earlier visualization attempt reported `ValueError: setting an array
element with a sequence`, rerun only the visualization command. The top-level
renderer now draws variable-length STEP edge curves individually.

This is a fresh two-stage width ablation, not a direct continuation of the
already trained 256-D final classifier. Differences from the previously
reported 0.9088 test macro-F1 may arise from retraining the supervised base;
the relevant comparison is **256 versus 512 within this run**. The current
test split has already been observed in prior work, so it is not a new
independent model-development test. Use validation for the width decision.

No training or evaluation was run while preparing this code.
