# External baseline integration

External implementations remain outside this repository so their authors' environments,
preprocessing, and licenses stay intact. Only row-aligned embeddings and provenance enter the
benchmark.

## Accepted sources

- **BRepNet:** active. The official repository provides raw-STEP preprocessing, a pretrained
  segmentation checkpoint, and face-embedding export. The wrapper below produces aligned HistCAD and
  TMCAD artifacts. Record it as supervised segmentation transfer, not self-supervised learning. Its
  CC BY-NC-SA license may restrict use.
- **Brep2Shape:** inactive. The checkpoint is public, but the release expects precomputed Bezier
  triangles, topology tensors, DGL graphs, and line graphs. No official raw-STEP generator for those
  inputs was located.
- **UVStyle-Net:** inactive. Its README explicitly marks the own-data B-Rep-to-DGL feature pipeline
  “Coming soon,” so the released pretrained model cannot be applied faithfully to new HistCAD/TMCAD
  STEP files.
- **DeepCAD:** use the official author repository and checkpoint only on a separately reported
  command-sequence-compatible subset.
- **UV-Net:** there is official source but no official pretrained checkpoint currently located. A
  newly trained model is a supervised architecture control or a newly declared pretraining baseline,
  not an off-the-shelf frozen UV-Net baseline.
- **GC-CAD/SS-CAD:** keep these absent until author code/checkpoints or author-produced embeddings are
  available. Do not rename the controlled HistCAD contrastive experiment as either method.
- **DualBrep:** optional autoencoder-family comparison with official VAE weights. Keep its
  implicit-field input family explicit.
- **AutoBrep:** excluded from the main table because its decoder-only checkpoint does not define a
  canonical model-level encoder embedding.

The authoritative decisions and URLs are in
[`../BASELINE_SOURCE_AUDIT.md`](../BASELINE_SOURCE_AUDIT.md) and [`../sources.yaml`](../sources.yaml).

## Materialize BRepNet

Keep the authors' environment separate from the JEPA environment:

```powershell
cd ../../
git clone https://github.com/AutodeskAILab/BRepNet.git external_sources/BRepNet
cd external_sources/BRepNet
conda env create -f environment.yml
```

Then return to `benchmarking/` and point the wrapper at that environment's Python executable:

```powershell
python baselines/external/prepare_brepnet.py histcad `
  --external-python C:\path\to\envs\brepnet\python.exe

python baselines/external/prepare_brepnet.py tmcad `
  --external-python C:\path\to\envs\brepnet\python.exe
```

The wrapper:

1. stages the canonical STEP rows using hard links (use `--link-mode copy` if necessary);
2. invokes the official `eval.evaluate_folder` preprocessing/inference path. The local
   `extract_brepnet_data_from_step.py` has a compatibility fix that bakes STEP compound
   locations into geometry before feature extraction, since `occwl` requires edges without
   attached locations;
3. reads the official per-face `.embeddings` files;
4. concatenates deterministic mean and max face pooling into one model vector;
5. requires every benchmark UID to succeed by default, or records a shared subset with `--allow-partial`; and
6. writes `external/brepnet_histcad.pt` or `external/brepnet_tmcad.pt` with the original labels,
   targets, splits, repository revision, and checkpoint digest.

For debugging or resuming a long preprocessing run, use `--stage-only` and `--collect-only`. Official
trainer options can be forwarded repeatedly, for example
`--official-arg=--gpus --official-arg=1`. If the official OpenCascade extractor rejects any parts,
the wrapper writes a `.missing.json` audit. `--collect-only --allow-partial` saves the successful
rows and a `.coverage.json` file; the benchmark applies those omissions to every representation
of the same dataset.

STEP feature extraction defaults to one process. To resume it with four processes, stop the current
run with Ctrl+C and run:

```powershell
python baselines/external/prepare_brepnet.py tmcad --preprocess-workers 4
```

Completed `.npz` features are reused, and interrupted or invalid `.npz` files are regenerated.
`--preprocess-workers` affects STEP feature extraction; it does not set BRepNet's inference
`--num_workers` option. Reduce the process count if memory becomes tight.

Unsupported or failing STEP files are recorded individually under
`external/work/brepnet/<dataset>/steps/temp_working/preprocessing_failures/`. A rerun skips those
files and keeps processing the remaining parts. After fixing an input or extractor issue, use
`--retry-failed` to try the recorded files again. The final `.missing.json` report includes each
failed UID and its preprocessing error. The full-cohort benchmark artifact still requires every
UID unless `--allow-partial` is used. The shared subset is explicitly reported in benchmark outputs.

## Export contract

Run the official preprocessing and inference in the external repository, then export one model-level
embedding per canonical benchmark UID. The JSON manifest passed to `convert_embeddings.py` must use
the canonical row order and contain:

```json
{
  "uids": ["uid-1", "uid-2"],
  "splits": ["train", "test"],
  "labels": {"class": [0, 1]},
  "targets": {}
}
```

For other future external methods, copy HistCAD labels and standardized geometry targets from the JEPA reference artifact rather
than recomputing them. For TMCAD, copy UIDs, splits, and labels from the saved metadata. Do not infer a
new split. The provenance JSON must contain non-empty values for:

```json
{
  "source_repository": "https://github.com/owner/repository",
  "source_revision": "pinned commit SHA",
  "checkpoint": "checkpoint filename or stable source URL",
  "checkpoint_sha256": "64-character hexadecimal digest",
  "preprocessing": "exact command/config and pooling rule",
  "training_data": "data used by the checkpoint authors",
  "pretraining_objective": "published training objective",
  "parameter_count": 123456
}
```

Convert the exported `.npy` array with:

```powershell
python convert_embeddings.py `
  --values raw/brepnet_histcad.npy `
  --manifest manifests/histcad.json `
  --metadata provenance/brepnet.json `
  --name brepnet_official `
  --output ../../external/brepnet_histcad.pt
```

The main registry then enforces exact UID and split equality against the local representation. A
partial DeepCAD cohort belongs in a separate configuration and output directory; it must never be
added to the full-cohort main table.
