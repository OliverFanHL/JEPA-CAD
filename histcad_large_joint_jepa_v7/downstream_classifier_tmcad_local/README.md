# TMCAD frozen JEPA-v7 classification with local STEP tokens

This is an independent downstream experiment. It reads the existing validated
TMCAD manifest and graph cache from `../downstream_classifier_tmcad/data/processed`
but imports **no source code or classifier checkpoints** from that folder.
It uses the existing JEPA-v7 `best.pt` checkpoint only as a **frozen** STEP
encoder. It does not fine-tune JEPA, alter the old TMCAD split, or modify the
older downstream implementation and results.

## Exact representation boundary

In v7, `StepEncoder` gives every STEP graph node a 512-D state after typed
embedding, numeric embedding, relational message passing, and node normalization.
The encoder's role-aware pool compresses all these states into three 512-D
summaries (typed entities, B-Rep instances, connectivity) plus one 512-D
global-node state: **4 x 512 = 2048-D**. The previous downstream attention
classifier received only that L2-normalized 2048-D vector, reshaped it to
four 512-D slots, and then **linearly projected each slot from 512 to 256-D**
before attention. Thus both the JEPA role pool and the downstream 512->256
projection could discard local information. The 52 added B-Rep descriptors
were fed to a separate residual head, not recovered from the pooled vector.

This experiment saves the original normalized 2048-D vector **unchanged** and,
through a read-only forward hook immediately before the v7 pool, saves **every
512-D post-GNN node state** in each prepared TMCAD graph. Nodes are not
subsampled or width-projected by this downstream code. Both compared heads use
512-D attention. `pooled_fullwidth` attends the four pooled roles to one
another; `pooled_plus_local` uses those same role queries to attend to all
pre-pooling node states. In both cases the entire original 2048-D vector
also reaches the final classifier head. Only the source of the attention
keys/values changes. Frozen features are stored as configurable float16 or
float32; float16 changes storage precision, not node count or feature width.

The prepared graph cache itself still contains at most 256 B-Rep instance
nodes per part because of the original v7 parser. This experiment cannot
recover instances absent from that graph. The local-token comparison tests
whether the *existing* unpooled JEPA features help. If it does not, a future
experiment should improve STEP parsing and surface/face coverage rather than
add another classifier layer or interpolate frozen embeddings. No geometric
augmentation or additional handcrafted descriptors are introduced here.

## Inputs and evaluation

- Source STEP: TMCAD's ten folder-level classes, exactly the same prepared
  manifest and train/validation/test split as the previous experiment.
- Frozen source: v7 STEP branch only. TMCAD has no HistCAD parametric history,
  so neither the HistCAD JSON branch nor the joint 4096-D embedding is used.
- Training: two compact attention heads are fitted from scratch to frozen
  features, with the same inverse-square-root class weighting, label
  smoothing, dropout, weight decay, and validation-macro-F1 early stopping.
- Test: evaluated only after checkpoint selection; accuracy, balanced
  accuracy, macro-F1, top-3 accuracy, per-class precision/recall/F1, confusion
  matrix, predictions, and representative STEP paths are saved. The local
  head's example CSV also records its most-attended frozen graph nodes.
  Attention weights are diagnostic, not causal explanations.

This is an exploratory split: byte-identical files are grouped/excluded by
the older preparation, but geometrically near-identical files with different
STEP bytes may cross splits. No official TMCAD split was supplied locally.

## Run

From this folder, after confirming the paths in `config.yaml`:

```powershell
python -m pip install -r requirements.txt
python run.py --stage extract
python run.py --stage train
python run.py --stage evaluate
```

`python run.py --stage all` runs the same sequence. Extraction can require a
few gigabytes of disk for all node features; set `extraction.storage_dtype`
to `float32` if the extra precision is worth the larger cache. The entry point
checks the prepared manifest, v7 checkpoint/vocabulary, array widths, sample
order, and checkpoint/feature compatibility. Outputs stay under this folder's
`data/processed` and `outputs/default` by default.

No training or evaluation was run while preparing this revision. Compare the
two new heads using **validation macro-F1 first**; report the held-out test
result once, alongside the earlier pooled-attention and structural-control
results. Do not use the test split to select a new parser or architecture.
