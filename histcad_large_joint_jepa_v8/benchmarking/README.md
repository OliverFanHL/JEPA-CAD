# Comprehensive JEPA CAD benchmark

This folder adds an artifact-first, three-level comparison without modifying the completed v8
pretraining or downstream experiments. It does not run anything automatically. The main entry point
is `run.py`, and `config.yaml` is the single experiment registry.

## What is compared

1. **Frozen representation quality**: direct cosine retrieval, train-bank kNN, a capacity-matched
   linear probe, an equally sized nonlinear adapter probe, effective rank, variance concentration,
   duplicate/tie counts, similarity margins, and class-centroid separation.
2. **Controlled transfer**: the exact TMCAD split and a common fixed-width adapter; and the exact
   HistCAD retrieval/semantic splits with identical supervision, adapter capacity, optimization,
   validation selection, and the existing downstream studies' metric definitions for every
   representation.
3. **Task-specific supervised controls**: the completed TMCAD supervised result is imported, and a
   script is provided for the same STEP encoder trained end-to-end from random initialization.

The three questions answered by the final tables are therefore separated: frozen quality, benefit as
a controlled transferable initialization, and comparison with supervised task learning.

## Reused artifacts

The default registry reuses these completed results:

- JEPA `best.pt`, training log/summary, and all 162,131 saved joint/branch representations;
- the completed same-encoder graph autoencoder checkpoint and embeddings;
- the completed standardized graph-statistics and train-only PCA embeddings;
- the existing TMCAD JEPA STEP embeddings and exact split;
- the existing TMCAD final supervised evaluation;
- the frozen semantic text-teacher targets; and
- existing retrieval-adapter and semantic-study documentation as provenance (new controlled adapters
  are trained per representation because the old adapters were JEPA-only).

No pretraining stage is rerun when its compatible artifact exists. Each baseline preparation script
refuses to replace a completed artifact.

## Source audit comes first

Read [`baselines/BASELINE_SOURCE_AUDIT.md`](baselines/BASELINE_SOURCE_AUDIT.md) before adding an
external method. The current decisions are:

- BRepNet: official checkpoint plus a complete raw-STEP preprocessing/embedding path; active external
  supervised-transfer baseline for HistCAD and TMCAD.
- Brep2Shape: promising official self-supervised checkpoint, but inactive because its release expects
  precomputed Bezier/DGL inputs and provides no raw-STEP generator.
- UVStyle-Net: inactive for new datasets because its official own-data B-Rep-to-DGL feature pipeline
  is still marked “Coming soon.”
- DeepCAD: official code plus pretrained weights; optional only on a compatible command-sequence
  shared subset.
- UV-Net: official code, but no released checkpoint; architecture-only supervised control unless it
  is newly trained under a declared objective.
- GC-CAD: no author implementation/checkpoint located; never substitute another method under its
  name.
- The provided graph-contrastive baseline uses the exact HistCAD encoders/cache and is explicitly
  called a controlled contrastive objective baseline, not GC-CAD.

## Safe first commands

These commands do not train or evaluate models:

```powershell
cd histcad_large_joint_jepa_v8/benchmarking
python run.py inventory
python run.py validate
```

`inventory` records present/missing artifacts and dimensions. `validate` additionally enforces exact
UID and split alignment among available representations after applying any shared coverage subset.

Running `python run.py` without a stage runs `all`, not `inventory`. Use the explicit commands above
when you only want to inspect readiness.

## Prepare missing controlled baselines

Run only the missing methods you intend to include:

```powershell
# Same architecture/input, graph contrastive pretraining
python baselines/contrastive/train.py

# Seeded random-encoder frozen control (feature extraction only)
python baselines/contrastive/train.py --random-only

# Extract existing AE/contrastive/random STEP encoders on the saved TMCAD graph cache
python baselines/extract_tmcad.py autoencoder
python baselines/extract_tmcad.py contrastive
python baselines/extract_tmcad.py random

# Official external BRepNet, after creating its official environment/checking out its repository
python baselines/external/prepare_brepnet.py histcad --external-python C:\path\to\brepnet\python.exe
python baselines/external/prepare_brepnet.py tmcad --external-python C:\path\to\brepnet\python.exe
```

If BRepNet already produced face embeddings but rejected some STEP files, collect the successful
rows without repeating inference:

```powershell
python baselines/external/prepare_brepnet.py histcad --collect-only --allow-partial
python baselines/external/prepare_brepnet.py tmcad --collect-only --allow-partial
python run.py validate
python run.py frozen --force
```

The collectors write `external/brepnet_<dataset>.pt` for successful rows and
`external/brepnet_<dataset>.coverage.json` with the omitted UIDs. All benchmark representations
for that dataset are filtered to the same UIDs before evaluation. The original `.missing.json`
files retain the individual failure details. The reduced cohort and row counts appear in
`outputs/inventory.json` and each stage result. Use `--force` when replacing results from a
previous cohort.

Contrastive training writes concise JSON-line heartbeats to the console and to
`outputs/baselines/contrastive/progress.jsonl`: one after the first step, every
`baseline_training.contrastive.monitor_every_steps` steps, at validation start, and at epoch end.
Each training heartbeat includes the recent loss, throughput, ETA, and peak CUDA memory (when
available). Increase `monitor_every_steps` for less frequent output.

The contrastive implementation avoids per-batch device synchronization, transfers only graph inputs,
uses non-blocking pinned-memory copies on CUDA, and keeps the selected checkpoint on CPU. These are
execution-only changes: the graph views, batch size, NT-Xent objective, optimizer, gradient clipping,
validation batches, early stopping, and checkpoint-selection rule are unchanged. The larger
`extraction_batch_size` applies only after model selection. `num_workers` remains zero by default for
reliable Windows execution; if local profiling shows input starvation, it can be increased and
`prefetch_factor` will take effect without changing sample order or supervision.

The graph AE, PCA, and graph-statistics artifacts already exist and need no preparation. BRepNet has
the dedicated official-pipeline wrapper above. Brep2Shape and UVStyle-Net are not active because
their releases do not provide raw-STEP preprocessing for new datasets.

If another method later releases an applicable official pipeline or author-produced embeddings,
preserve the exact benchmark order and validate its output with the generic converter:

```powershell
python baselines/external/convert_embeddings.py `
  --values external/raw/new_method_histcad.npy `
  --manifest external/manifests/histcad.json `
  --metadata external/provenance/new_method.json `
  --name new_method `
  --output external/new_method_histcad.pt
```

The converter requires repository revision, checkpoint hash, training data, preprocessing, objective,
and parameter count. Main-table imports must contain every UID in exactly the saved split order.

## Run the benchmark manually

```powershell
python run.py frozen
python run.py classification
python run.py retrieval
python run.py semantic
python run.py efficiency
```

Or, after reviewing the inventory, use `python run.py all`. Existing task result JSON files are reused;
pass `--force` only when intentionally replacing a result. Test rows are never used to choose epochs
or hyperparameters.

`python run.py` is equivalent to `python run.py all`. All 12 active representations are required;
partial BRepNet artifacts count when collected with `--allow-partial` and shared coverage is applied.
Validation stops and reports preparation commands if any artifact is missing. Published methods
without an official raw-STEP materialization path are kept in `baselines/sources.yaml` and do not
inflate the active count.

## Output structure

```text
outputs/
  inventory.{json,csv}
  baselines/
  frozen_representation/{results.json,summary.csv}
  tmcad_classification/{results.json,summary.csv}
  retrieval_adaptation/{results.json,summary.csv}
  semantic_retrieval/{results.json,summary.csv}
  efficiency/{results.json,summary.csv}
```

The efficiency table has one row for each active method family. Refreshing an existing
`results.json` requires `--force`; a plain run reuses the saved file. To measure HistCAD
embedding forward time from the existing checkpoints, then rebuild only the efficiency table:

```powershell
python profile_efficiency.py --device auto
python run.py efficiency --force
```

The profile uses covered HistCAD test rows and times model forward calls only. Checkpoint loading,
graph collation, and host-to-device transfer are excluded. It profiles JEPA, graph autoencoder,
contrastive, the seeded random encoder, input graph statistics, and PCA graph statistics.

BRepNet inference can be profiled from the already prepared solids and official checkpoint using
its Python 3.7 environment. From `benchmarking` in PowerShell:

```powershell
conda run -n brepnet37 python baselines/external/profile_brepnet.py histcad
conda run -n brepnet37 python baselines/external/profile_brepnet.py tmcad
python run.py efficiency --force
```

Run through `conda run` (or activate `brepnet37` first). On Windows, calling that environment's
`python.exe` directly while another environment is active may leave its DLL directories off PATH
and fail while importing NumPy or PyTorch.

The two BRepNet profiles record checkpoint parameter count and seconds per covered test solid;
`summary.csv` places HistCAD timing in `embedding_seconds_per_sample` and TMCAD timing in
`tmcad_embedding_seconds_per_sample`. The timing includes the official BRepNet encoder and the
benchmark's mean+max face pooling. It excludes STEP preprocessing, checkpoint loading, data
collation, and transfers. Neither profiler trains anything. The original BRepNet pretraining
duration and graph-autoencoder training duration were not saved, so those historical times remain
unavailable. Autoencoder optimizer steps are derived from its saved train-row, batch-size, and
completed-epoch counts. Training time is zero for untrained controls and tagged accordingly.

## Fairness rules

- Dataset rows, splits, labels, query/gallery construction, and validation rules must match exactly.
- Original embeddings are used for direct cosine retrieval and kNN.
- A deterministic, non-trainable projection maps each embedding to 256 dimensions before learned
  probes/adapters, giving every method the same trainable downstream parameter count.
- External architectures keep their native input/preprocessing and are documented rather than
  falsely made identical.
- A restricted shared-subset result (for example DeepCAD) is a separate table and cannot be mixed
  with the full-cohort main table.
- `functional_type` is evaluation-only in frozen retrieval. Semantic-adapter training uses only the
  saved train-split text-teacher targets.
