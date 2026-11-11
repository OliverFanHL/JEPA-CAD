# TMCAD downstream classifier: fresh 512-D head and 196-D physical residual

This is a standalone implementation. It imports no Python module from the
previous TMCAD experiment folders, changes none of their checkpoints or
results, and has no ablation subfolder. It reads the existing TMCAD split and
frozen JEPA-v7 STEP feature/physical-descriptor caches through configurable
relative paths. Check `config.yaml` before running on another machine.

The original HistCAD JEPA joint embedding is 4096-D, but TMCAD does not
provide the paired parametric JSON needed to compute it. This downstream
experiment uses the frozen **2048-D STEP branch** plus four complementary
frozen 512-D STEP role summaries. Attention produces four 512-D contexts;
concatenating these with the original 2048-D vector gives the classifier's
**4096-D STEP-derived input**. This is not the HistCAD 4096-D joint embedding.
The fresh supervised classifier uses `Linear(4096, 512)` and ten class logits.
It also receives the existing 37 geometric and 84 surface-profile descriptors
through separate direct-logit readouts. The JEPA encoder remains frozen, and
no old 256-D classifier checkpoint is loaded.

After selecting the best 512-D classifier epoch on validation, that
classifier is frozen. A separate residual uses the **121 real physical fields**
through `121 -> 196 -> 196 -> 10` learned layers, adding a small class-score
correction. Increasing these learned widths does not increase the raw physical
input width. The same TMCAD train/validation/test split and class-weight rule
are retained. Both stages use validation macro-F1 for checkpoint selection;
test rows are not extracted in either training stage. The residual is selected
for deployment only if its validation macro-F1 exceeds the base by 0.003.

From this folder, in the CAD-enabled Python environment:

```powershell
python -m pip install -r requirements.txt
python run.py --stage all
```

Or run `--stage train-base`, then `--stage train-adapter`, then
`--stage evaluate`. Training refuses to overwrite existing checkpoints.
If the first stage is already complete, run only the remaining stage.

The evaluator reports both base and physical-residual test scores **after**
fixing the validation-only selection. Outputs under `outputs/default` include
accuracy, balanced accuracy, macro-F1, weighted F1, top-3 accuracy,
per-class precision/recall/F1, confusion plots, every test prediction,
checkpoint provenance, and training histories. `evaluate()` itself also
renders class/outcome-stratified held-out STEP parts from two opposite views.
`outputs/default/visualization/index.html` contains the visual comparison,
per-class recall changes, corrected/worsened counts, and individual part
cards. `render_audit.json` records every import failure rather than silently
omitting it. Rendering uses OpenCASCADE (`cadquery-ocp`) and affects display
only—not the inputs, predictions, or metrics.

The original 256-D and this 512-D supervised head differ in training history
and capacity. Treat any comparison to older published numbers as exploratory;
the existing test set has been seen in earlier design iterations.

No training or evaluation was run when this implementation was prepared.
