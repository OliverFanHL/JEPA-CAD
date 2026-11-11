# Frozen-JEPA semantic downstream study

This is a controlled downstream experiment using the existing v7 `best.pt`
4096-D representation cache. The v7 encoder is **never updated**. The generic
English sentence teacher `sentence-transformers/all-MiniLM-L6-v2` is also
frozen. Only a 2.31M-parameter CAD-to-semantic adapter is trained.

The default training teacher uses the HistCAD `Functional Type` annotation
with numbers, measurement units, and parenthetical phrases stripped before
text encoding. Train text fits the adapter; validation text selects the
checkpoint; test text is never encoded for training or model selection. A
second adapter has the same model and seed, with the same semantic target plus dimensions, feature count, and
final operation. Their best checkpoints are chosen by validation loss.

At retrieval time, every vector is computed from the CAD representation alone;
**no text query or part annotation is needed**. The existing geometry adapter
and direct JEPA retrieval are read-only comparison conditions. Two score-space
fusions are selected using validation only, within predeclared dimensional
degradation limits. Test is never used for checkpoint or fusion selection;
reporting scripts may be rerun on the fixed test embeddings. Candidate banks
use the same split, with self-match excluded.

## Run

From this folder, with the same environment used for v7 and the locally cached
sentence-transformer:

```powershell
python run.py all
```

Stages may be run independently after their inputs are available:

```powershell
python run.py train
python run.py evaluate
python run.py transfer
python run.py render
python run.py dimensions
python run.py physical
python run.py ablation
```

Paths, training hyperparameters, and fusion constraints are in `config.yaml`.
Its clean semantic run writes `outputs_clean/`. The preliminary raw-text run,
which is confounded by dimensions embedded in functional descriptions, remains
in `outputs/`; its configuration is preserved as `config_raw.yaml`. To
reproduce that secondary run, set `HISTCAD_STUDY_CONFIG=config_raw.yaml` before
invoking `run.py`. After both runs exist, use `python run.py ablation` with the
default clean configuration. The `all` stage does not implicitly rerun the
raw-text or dimension-only experiments. For the expanded dimension-only
comparison, run `python run.py dimensions` and `python run.py physical` once with each configuration before
`python run.py ablation`.
The text teacher is read from the local Hugging Face cache (`local_files_only`
is enabled), so the study does not download model weights implicitly.

## Outputs

- `outputs_clean/functional_text_teacher.pt`: train/validation-only frozen text targets
- `outputs_clean/semantic_only.pt`, `outputs_clean/semantic_plus_geometry.pt`: downstream checkpoints
- `outputs_clean/semantic_plus_dimensions.pt`, `outputs_clean/dimension_only_embeddings.pt`: isolated dimension-supervision checkpoint and vectors
- `outputs_clean/semantic_plus_physical_sizes.pt`, `outputs_clean/physical_size_embeddings.pt`: size/ratio-only supervision without count fields
- `outputs_clean/training_summary.json`: epoch-level train and validation losses
- `outputs_clean/study_embeddings.pt`: validation/test CAD-derived downstream vectors
- `outputs_clean/evaluation.json`: direct, existing adapter, controlled conditions, paired intervals
- `outputs_clean/annotation_transfer.json`: evaluation-only `Geometric Feature`, `Modeling Process`, and `NLT` diagnostics
- `outputs_clean/test_neighbors.pt`, `outputs_clean/qualitative_manifest.csv`: selected retrievals
- `outputs_clean/retrieval_sheets/`: four STEP-rendered cases, including wins and trade-offs
- `outputs_clean/RESULTS.md`: interpretation and limitations
- `outputs_clean/factorial_ablation.json` and `outputs_clean/FACTORIAL_ABLATION.md`: matched full/stripped-text × numeric-target experiment

The `Functional Type` labels contain free-form dimensions and wording; canonical
functional matching removes parenthetical detail, while a separate raw-label
metric keeps it. Neither measure alone proves engineering interchangeability.
The rendered shapes are normalized in size for visibility, so dimensional
distance must be read from the per-neighbor numeric annotation.
