# Reproduce JEPA-CAD

## Install

Use Python 3.11 or later. From the repository root:

```bash
python -m pip install -e "./histcad_large_joint_jepa_v8[dev,visualization]"
python -m pip install scikit-learn scipy matplotlib pillow scienceplots pypdf
python verify_release.py
python verify_classifier.py
```

Functional-text teacher regeneration additionally needs `sentence-transformers`.
The external BRepNet baseline uses the official repository/environment described
in `histcad_large_joint_jepa_v8/benchmarking/baselines/BASELINE_SOURCE_AUDIT.md`.
Install external code separately; the wrappers default to
`benchmarking/external_sources/BRepNet`.

## Check code and saved outputs

From `histcad_large_joint_jepa_v8/`:

```bash
python -m pytest tests benchmarking/tests -q
```

`verify_release.py` requires only the Python standard library. It verifies the
archive hashes, complete final matrix, JSON/CSV summary alignment, saved TMCAD
metrics and paired corrections, and manuscript figure files. It does not rerun
pretraining or infer missing measurements.

`verify_classifier.py` loads the selected 128/128 physical residual and released
derived test inputs to reproduce the baseline and final decisions on all 1,404
test parts. This verification does not need the source datasets or large encoder.

## Regenerate the full pipeline

Download the [source datasets](DATASETS.md), retain the original split identities,
and update dataset paths in the YAML files and historical manifests. The preserved
two-project layout lets the final configuration reference v7 preprocessing and
frozen-control paths. Regenerate the excluded graph/feature caches before running
steps that require them. The recorded run configuration is
`histcad_large_joint_jepa_v8/outputs/full_dataset_4096/config_used.yaml`.

From `histcad_large_joint_jepa_v8/`, `python run.py --cache-only` builds paired
graphs, and `python run.py` trains/evaluates/extracts the joint representation.
After those artifacts exist, prepare the selected controlled baselines using the
commands in `benchmarking/README.md`, then run from `benchmarking/`:

```bash
python run.py inventory
python run.py validate
python run.py all
```

`all` can train downstream models. Preserve the released result directories and
use a separate checkout or output directory for reruns. Shared coverage is applied
to every representation before comparison. The standalone retrieval and semantic
study folders provide their own `run.py`, configuration, and README files.

The large pretrained encoder and full-corpus representations are not bundled.
The recorded metrics can be inspected directly, but full retrieval recomputation
requires regenerating those artifacts. Historical rendering/audit scripts may
also require local source CAD files and path remapping. The included final figure
PDFs/images and provenance records can be inspected without those inputs.
