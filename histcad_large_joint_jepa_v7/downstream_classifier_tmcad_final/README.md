# Final frozen-JEPA STEP classifier for TMCAD

This folder is a self-contained **downstream implementation**. It imports no
Python code from previous experiment folders and writes all new caches,
checkpoints, metrics, and case lists here. It reads the existing TMCAD split,
frozen feature arrays, and trained baseline checkpoint without modifying
them. No JEPA parameter or baseline-classifier parameter is updated.

## Evidence for the selected design

The controlled nonlinear-readout experiment selected `physical_residual` on
validation. Compared with the saved mesh classifier, validation macro-F1
rose from **0.8575 to 0.8855** and test macro-F1 from **0.8657 to 0.9088**.
Test coupling recall rose from **0.743 to 0.843** and pulley recall from
**0.718 to 0.872**. The alternative that also conditioned on the classifier's
intermediate context scored **0.8822** validation macro-F1, below the
physical-only model. Its extra branch is therefore omitted. The physical
model's training and validation losses continued to improve through the
32-epoch run; the validation-best checkpoint was epoch 30. These results
support finalizing this architecture **for the current TMCAD STEP-only
classification protocol**, not claiming that it is a universal CAD classifier.
The previous experiment selected the physical-only variant by validation,
but its test results are now known. A fresh run of this finalized code on the
same test split is a reproducibility check, not a new independent test of
model selection. A truly independent generalization estimate needs another
held-out dataset or a newly established family-disjoint split.

## Exact dimensions and data flow

The original HistCAD JEPA-v7 model has a **4096-D joint output**, formed
from a 2560-D parametric-history branch and a 2048-D STEP/B-Rep branch.
TMCAD supplies STEP files but **no matching HistCAD parametric JSON history**.
The 4096-D joint embedding therefore cannot be computed honestly for TMCAD;
this classifier uses the pretrained **2048-D STEP branch**. It does not pad
the missing parametric branch or call a 2048-to-4096 expansion a joint
embedding.

The original frozen STEP vector is 2048-D. A complementary graph selection
from the same frozen STEP encoder yields four 512-D role summaries (another
2048-D, not a new modality). Four 512-D attention readouts are concatenated
with the uncompressed original 2048-D vector, giving **4096-D at the
supervised classifier head**. This 4096-D classifier input is *not* the
4096-D HistCAD JEPA joint embedding. The saved baseline maps it through a
256-D task-specific hidden layer to ten class logits. The 256-D layer is
retained solely to reproduce the validated baseline, not treated as the JEPA
output. It is not passed to the final residual branch because the conditional
experiment did not improve validation performance.

Separately, the full-file physical cache contains **37** train-standardized
STEP dimension/radius/bbox descriptors and **84** train-standardized surface
profile descriptors. Their true input width is **121**, with no padding or
invented expansion. The selected small nonlinear residual is
`121 -> 96 -> 96 -> 10` (GELU/dropout between layers). Its zero-initialized
last layer adds class-score corrections to the frozen baseline logits.
The 96-D hidden widths are justified as a modest, successful supervised
readout of 121 actual fields; they do not compress the 2048-D STEP vector.
Both hidden widths may be increased to 128, 192, or 256 through `config.yaml`
for a controlled comparison, but this retrains a new residual and should use
a separate output directory. The true 121-D input must not be increased by
padding: it can change only if new, independently computed STEP descriptors
are added to the preprocessing/cache and validated for nonredundant content.

## Train and evaluate

Check the relative paths in `config.yaml`. From this folder:

```powershell
python -m pip install -r requirements.txt
python run.py --stage all
```

Alternatively run `--stage train` and then `--stage evaluate`. Training
extracts only train/validation frozen scores and physical fields into this
folder's `data/processed`. It verifies that the frozen control reproduces
the prior validation macro-F1 before fitting the residual. Training uses the
same split, seed, class weights, 32-epoch limit, and validation macro-F1
checkpoint rule as the successful controlled experiment. If a checkpoint
already exists here, training refuses to overwrite it; choose another output
path for a new run.

Evaluation compares the saved baseline with the final residual. The final
residual must exceed baseline validation macro-F1 by at least 0.003 to be
selected; only then is the held-out test split materialized. Outputs include
overall accuracy, balanced accuracy, macro-F1, top-3 accuracy, per-class
precision/recall/F1, confusion plots, training history, and all test
predictions. `evaluate()` also writes an intuitive visual review under
`outputs/default/visualization`: an HTML index, class-wise recall comparison,
corrected/worsened counts, and two-view renders of actual STEP test parts.
The case sheets show ground truth, frozen-control prediction, final prediction,
and both confidences. `render_audit.json` records every selected sample and
any failed STEP import, so missing renders cannot be mistaken for valid
examples. This stage requires OpenCASCADE's `OCP` module (`cadquery-ocp`).
`qualitative_review.csv` additionally lists corrected, worsened, and
still-wrong coupling/pulley STEP paths. The rendering changes neither inputs
nor labels. The manifest
keeps exact duplicate files in one split, but near-identical variants may
still cross splits; this limitation should accompany reported scores.

No training or evaluation was run when this final code was prepared.

## Optional supervised hidden-width comparison

The [width ablation](width_ablation/README.md) independently trains the same
downstream classifier with a 256-D or 512-D task-specific hidden layer, then
fits the same 121→96→96→10 physical residual to each frozen classifier.
It keeps the 2048-D JEPA STEP input and all other inputs unchanged. Its
checkpoints and results are separate from the default finalized model, and
the width decision uses validation macro-F1 only. Run it from
`width_ablation` with `python run.py --stage all` if you want to test whether
the wider supervised layer is useful. That command also produces a local HTML
visual review with class-wise changes and two-view renders of actual test STEP
parts. The visualization code lives directly in this folder in
`visualize_width_ablation.py` and `step_render.py`; only its width-comparison
results are stored under `width_ablation/outputs`. The rendering stage requires
OpenCASCADE's `OCP` module.
