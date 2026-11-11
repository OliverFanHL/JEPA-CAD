# Download and use the existing pretrained arm on Linux

The graph cache and pretrained model are hosted at
[IELTS8/JEPA-CAD on Hugging Face](https://huggingface.co/IELTS8/JEPA-CAD).
Download size: 10,014,680,021 bytes (about 10.0 GB). Allow additional free disk
space for the download cache and any later extracted representations.

```bash
git clone https://github.com/OliverFanHL/JEPA-CAD.git
cd JEPA-CAD
python -m pip install --upgrade huggingface_hub
python download_pretrained.py
```

The downloader uses the exact revision and SHA-256 checksums in
`REMOTE_PRETRAINED_ARTIFACTS.json`. If interrupted, rerun the same command; the
Hugging Face client reuses downloaded data. Only the two artifact files are
downloaded, preserving the code checkout's README and other files.

Files are placed at the paths already configured for `jepa_joint`:

```text
histcad_large_joint_jepa_v8/data/processed/full_paired_graphs_v7.pt
histcad_large_joint_jepa_v8/outputs/full_dataset_4096/best.pt
```

Install the code dependencies in your preferred Linux Python/PyTorch environment,
then check the existing arm:

```bash
python -m pip install -e "./histcad_large_joint_jepa_v8[dev]"
python use_pretrained.py
```

This check loads artifact metadata, verifies schema compatibility, and reports
the existing epoch-20 checkpoint and original 162,131-part graph cache. It does
not preprocess, train, or extract representations.

To recheck downloaded file integrity without any network access:

```bash
python download_pretrained.py --verify-only
```

If the benchmark representation matrix is needed, run frozen inference directly
on the downloaded cache:

```bash
python use_pretrained.py --extract --device cuda
```

Extraction is optional and potentially time-consuming. It preserves the frozen
checkpoint, never creates an optimizer, and does not read or preprocess original
CAD files. It writes `outputs/full_dataset_4096/representations.pt` under the v8
project. Other benchmark methods require their separately prepared artifacts;
these two downloads establish the existing pretrained JEPA arm.
