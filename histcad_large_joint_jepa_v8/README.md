# HistCAD large joint JEPA (v7, 4096-D)

> A comprehensive, artifact-reusing three-level comparison framework is available in
> [`benchmarking/`](benchmarking/README.md). It covers frozen representations, controlled TMCAD and
> retrieval/semantic transfer, supervised/no-pretraining controls, and measured-only efficiency
> reporting. The framework does not run experiments automatically.

This is a self-contained, from-scratch successor to v6. It preserves HistCAD's parametric history
and typed STEP/B-Rep graph rather than converting parts to voxels or point-only geometry.

## Capacity chosen from the full-data audit

The v6 branch vectors were learned summaries, not zero-padded raw vectors. The new widths therefore
increase learned relational channel capacity while keeping the number of summary roles tied to CAD
semantics:

```text
Parametric graph encoder (512-D nodes)
  feature-operation summary       512
  sketch-geometry summary         512
  constraint/dimension summary    512
  whole-relationship summary      512
  explicit global/history token   512
  branch output                  2560

STEP/B-Rep encoder (512-D nodes)
  typed-entity summary             512
  B-Rep-instance summary           512
  whole-connectivity summary       512
  explicit global/type token       512
  branch output                   2048

L2-normalized concatenation       4608
learned nonlinear joint projection 4608 -> 4096 -> 4096
final joint embedding             4096
```

There is no branch-vector padding or duplicated token. A learned absent-category token represents a
genuinely missing constraint or other role. A slot-diversity penalty discourages different summaries
from collapsing to the same information.

The detailed measurements and design rationale are in `DATA_AND_CAPACITY_AUDIT.md`.

## Run from scratch

From this directory:

```powershell
python -m pip install -e ".[dev]"
python run.py
```

`run.py` will:

1. audit every JSON/STEP pair;
2. build `data/processed/full_paired_graphs_v7.pt` if it does not exist;
3. train for 10 epochs on every valid training-split part;
4. select `best.pt` using validation loss only;
5. evaluate the selected model once on the test split;
6. extract 4096-D joint, 2560-D parametric, and 2048-D STEP representations; and
7. run the direct retrieval evaluator unless `--skip-evaluation` is supplied.

No v6 checkpoint is loaded. Resume accepts only a v7 checkpoint with the identical architecture:

```powershell
python run.py --resume outputs/full_dataset_4096/last.pt
```

Useful modes:

```powershell
python run.py --cache-only
python run.py --rebuild-cache --cache-only
python run.py --evaluate-only
python run.py --skip-evaluation
```

All paths are relative to this project and can be changed in `configs/full_dataset.yaml`. The default
micro-batch is 4 with four-way gradient accumulation (effective batch 16) and automatic CUDA mixed
precision. Reduce `batch_size` if necessary without changing the effective batch by increasing
`gradient_accumulation_steps`.

## Outputs

Preprocessing writes the cache plus `dataset_audit.json`, `representation_audit.json`, retained and
excluded manifests. Training writes logs, progress JSONL, CSV metrics, `last.pt`, `best.pt`, a summary,
and `representations.pt` under `outputs/full_dataset_4096`.

The representation file stores embeddings in float16 to keep the three full-corpus matrices practical;
training and evaluation convert them to float32 when needed.

## Downstream retrieval adapter

After v7 training finishes:

```powershell
cd downstream_retrieval_adapter
python run.py
```

This freezes the v7 representation and trains a lightweight attention/MLP retrieval adapter. See the
adapter README for individual stages and protocol details.
