# JEPA-CAD

Joint predictive representation learning for CAD using parametric histories and
typed STEP/B-Rep graphs. This repository contains the implementation, final
experimental matrix, and saved results used in the manuscript
*Joint Predictive Representation Learning for CAD*.

## Start here

- [Final experiment matrix](FINAL_EXPERIMENT_MATRIX.csv): 38 representation/task
  and efficiency entries across the completed comparison.
- [Results guide](RESULTS.md): final benchmark and manuscript evidence locations.
- [Dataset links](DATASETS.md): download HistCAD and TMCAD separately.
- [Reproduction guide](REPRODUCTION.md): installation, verification, and rerunning.
- [Integrity manifest](FILE_MANIFEST.json): checksums and origin of the archived files.

## Implementation

`histcad_large_joint_jepa_v8/` contains the final joint encoder, pretraining,
12-representation benchmark, retrieval/semantic adapters, TMCAD classifier,
tests, and manuscript figure-generation code. The joint encoder combines a
2,560-dimensional parametric branch and 2,048-dimensional STEP branch through a
learned 4,096-dimensional projection.

`histcad_large_joint_jepa_v7/` preserves preprocessing and downstream code needed
by the final pipeline's relative paths. These folder names preserve the original
working layout; they do not denote independent pretraining replicates. The v8
study reused the v7 pretrained checkpoint. No JEV-CAD code or results are included.

## Data and artifacts

Source CAD files, original parametric JSON histories, annotation datasets, and
full processed graph caches are excluded. Small derived frozen logits and
physical descriptors for reproducing the final classifier are included.
Selected downstream checkpoints are included. Multi-gigabyte encoder checkpoints
and full-corpus embedding matrices are omitted; their producing code, saved
configuration, training history, and evaluation results are retained.

The final reports preserve distinct cohorts: HistCAD's original 8,086 test parts,
the shared-coverage benchmark's 8,082, TMCAD's full 1,404, and its shared-coverage
benchmark's 1,090. Compare methods within the corresponding protocol.

Run `python verify_release.py` for portable file-integrity and saved-result checks.
Run `python verify_classifier.py` with NumPy and PyTorch to reproduce all 1,404
final TMCAD predictions from the released checkpoint and frozen inputs.

## Existing pretrained comparison arm

On the Windows working copy, the original graph cache and pretrained `best.pt`
are copied into the v8 project's matching paths. `jepa_joint` explicitly uses
that checkpoint as a frozen comparison arm. Run `python use_pretrained.py` to
check artifact compatibility. It fails if either existing file is missing;
it never starts preprocessing or training. See `LOCAL_PRETRAINED_ARTIFACTS.json`
for the verified copy sizes and hashes. These large local files are excluded
from GitHub.

If representations need to be regenerated, `python use_pretrained.py --extract`
runs inference on the copied cache with the frozen checkpoint. This is optional;
the artifact copy and compatibility check do not run extraction.
