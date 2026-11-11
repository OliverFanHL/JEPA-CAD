# CAD retrieval representation benchmark

The primary benchmark asks whether the pretrained v7 JEPA representation is better for CAD retrieval
than alternative representations under a controlled protocol. It compares:

1. **Pretrained JEPA joint (4096-D)**: the proposed frozen representation.
2. **Graph autoencoder joint (4096-D)**: trained from scratch with the same parametric and STEP graph
   inputs, the same graph encoder architecture, the same joint width, and the same train/validation split.
   It reconstructs deterministic input-graph descriptors and is selected by validation reconstruction
   loss. It does not initialize from JEPA.
3. **PCA graph statistics (256-D)**: a classical unsupervised baseline fit on the train split only.
4. **Input graph statistics**: a deterministic, non-neural retrieval anchor standardized on train only.

The previous Direct Joint versus Adapted Joint comparison remains available as a **secondary downstream
ablation**. It measures task-specific extractability from JEPA, not whether JEPA pretraining is superior
to alternative representation learning.

All methods use the exact v7 UID ordering and deterministic split. Validation/test retrieval uses the same
within-split candidate pool, cosine similarity, leave-self-out neighbors, dimensional reference neighbors,
and metric implementation. Operation history, feature sequence, functional type, and other retrieval labels
are evaluation-only for the primary methods. Test data is not used for optimization or model selection.

## Run locally

Do this after the main v7 experiment has produced `best.pt`, `representations.pt`, and the v7 graph cache:

```powershell
python run.py
```

The command trains the baselines, trains the optional adapter, evaluates every representation, and writes
`outputs/REPRESENTATION_BENCHMARK_REPORT.md`. To omit the secondary adapter experiment:

```powershell
python run.py --skip-adapter --skip-rendering
```

Or run the primary benchmark stages separately:

```powershell
python train_baselines.py
python evaluate_adapter.py
python generate_benchmark_report.py
```

No experiment is run automatically by this repository revision. Paths, training budget, descriptor width,
and evaluation settings are in `config.yaml`.

The graph autoencoder checkpoint is selected solely by validation reconstruction loss. The evaluator writes:

- `baseline_training_summary.json`: training history and selection provenance;
- `baseline_representations.pt`: UID-aligned autoencoder, PCA, and graph-statistic representations;
- `benchmark_evaluation.json`: full metrics for validation and test;
- `benchmark_comparison.csv`: compact shared-protocol comparison;
- `REPRESENTATION_BENCHMARK_REPORT.md`: a publication-oriented table and protocol summary.

## Optional adapter ablation and qualitative rendering

The adapter keeps the JEPA encoder frozen, reshapes its 4096-D embedding into 32 chunks, applies attention
and a residual MLP, then performs cosine retrieval. Functional labels and test data are not used to train or
select it. Training uses dimensional targets, feature-count buckets, final Boolean operation, approximate
positive pairs, and fixed negatives.

Run only its legacy stages with:

```powershell
python train_adapter.py
python render_adapter_examples.py
python generate_report.py
```

Rendering requires optional visualization dependencies:

```powershell
python -m pip install -e "..[visualization]"
```

## Extended diagnostics and optional semantic supervision

The original adapter checkpoint and original JEPA encoder are unchanged. The
following scripts analyze the existing direct and adapted embeddings and train
only a small, separate readout on frozen adapted embeddings:

```powershell
python extended_evaluation.py
python semantic_probe.py
```

`extended_evaluation.py` writes paired nearest-neighbor accuracy, Recall@5/10,
dimensional error, bootstrap confidence intervals, and per-query diagnostics to
`outputs/extended_evaluation.json` and `outputs/extended_per_query_test.csv`.
The thresholded dimensional metrics use standardized dimension targets, not mm.

`semantic_probe.py` learns a linear multilabel readout of frequent tokens in
the `Functional Type` annotation using train labels only. Validation selects
the readout checkpoint and semantic-fusion weight; test is evaluated only after
selection. It writes `outputs/semantic_probe.pt` and
`outputs/semantic_probe_evaluation.json`. This is an exploratory downstream
supervision test, **not** a change to JEPA pretraining or the original adapter.

Interpretation, protocol caveats, and results are in
`outputs/ADAPTER_DIAGNOSTIC_REPORT.md`.

## Published reference results

Do not insert external numbers into the main comparison unless the dataset split, candidate gallery,
query set, relevance definition, and metric computation are sufficiently equivalent. Results from a
different CAD corpus or a different retrieval protocol should be discussed as context, not presented as a
direct baseline. The local baselines above are provided because they can be evaluated without those
comparability ambiguities.
