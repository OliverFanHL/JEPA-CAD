# Baseline source and compatibility audit

This audit is the gate for adding a named external baseline. A method is not called an
implementation of a published baseline unless its authors released code or a compatible checkpoint.
All external outputs must be converted to `cad_benchmark_embeddings_v1` and pass exact UID/split
validation before any metric is computed.

Audit date: **2026-09-22**. Sources were checked at the authors' repositories, repository release
pages, paper project links, and paper/supplement pages. A future release can change a decision, but it
must be recorded in `sources.yaml` with a pinned revision and checkpoint digest before use.

## Decision table

| Baseline | Located source | Checkpoint | Decision | Benchmark scope |
|---|---|---:|---|---|
| Graph autoencoder | Local `downstream_retrieval_adapter` implementation and completed checkpoint | Yes | **Primary** | All HistCAD frozen/retrieval/semantic tasks; extract its STEP encoder on TMCAD before classification |
| Graph statistics + PCA | Local completed artifact | Yes | **Primary** | Frozen HistCAD controls; no learned pretraining claim |
| Same-encoder graph contrastive control | Implemented in this benchmark from the existing HistCAD encoders and cache | Not yet | **Primary after manual training** | HistCAD and TMCAD. It is a controlled contrastive objective comparison, **not** labeled GC-CAD |
| Random same-architecture encoder | Existing HistCAD encoder initialized with the benchmark seed | Not applicable | **Primary control** | Frozen and downstream controls after one feature-extraction pass; no pretraining |
| Brep2Shape (ICML 2026) | Official repository: <https://github.com/thuml/Brep2Shape> | Yes, author-linked pretrained checkpoint | **Candidate, not currently applicable to raw STEP** | The release consumes precomputed Bezier triangles, topology tensors, DGL graphs, and line graphs. No official raw-STEP generator for those inputs was located, so it is not an active benchmark slot |
| UVStyle-Net | Official Autodesk repository: <https://github.com/AutodeskAILab/UVStyle-Net> | Yes, official download script provides pretrained models and precomputed Gram representations | **Candidate, not applicable to new raw STEP data** | The official README still marks the own-data B-Rep-to-DGL feature pipeline “Coming soon.” Precomputed ABC/SolidLETTERS Gram matrices cannot be aligned to HistCAD/TMCAD |
| BRepNet | Official Autodesk repository: <https://github.com/AutodeskAILab/BRepNet> | Yes, repository includes a Fusion 360 segmentation checkpoint | **Active external supervised-transfer baseline** | Complete official raw-STEP preprocessing and face-embedding export are available. `prepare_brepnet.py` materializes exact-coverage HistCAD/TMCAD model vectors using benchmark-defined, documented mean+max pooling |
| DeepCAD | Official repository: <https://github.com/rundiwu/DeepCAD> | Yes, authors provide a pretrained autoencoder download | **Optional external representation baseline** | HistCAD samples with compatible command sequences only. Never compare on TMCAD or silently drop incompatible rows; use an explicitly reported shared subset |
| UV-Net | Official Autodesk repository: <https://github.com/AutodeskAILab/UV-Net>; empty releases page: <https://github.com/AutodeskAILab/UV-Net/releases> | No released checkpoint or GitHub release located | **Architecture control only** | Train from scratch on the benchmark split for a task-specific supervised comparison, or pretrain under an explicitly declared new objective. Do not report it as a frozen pretrained baseline |
| GC-CAD | Paper: <https://arxiv.org/abs/2406.08863> | No author code or checkpoint located | **Not directly reproducible** | Keep a generic import slot if authors release embeddings later. Use the controlled contrastive baseline meanwhile and label it accurately |
| SS-CAD (Jones et al., CVPR 2023) | Paper and supplement located; no author release located | No | **Not directly reproducible** | Do not use the small third-party feasibility repository as a paper reproduction |
| Third-party SS-CAD feasibility code | <https://github.com/zhenshihaowanlee/Self-supervised-BRep-learning-for-CAD> | No large-scale checkpoint | **Excluded** | Repository explicitly describes itself as a feasibility test on simple generated shapes and lists large-scale training as future work |
| DualBrep | Official Autodesk repository: <https://github.com/AutodeskAILab/DualBrep> | Yes, author-hosted VAE checkpoints | **Secondary external autoencoder** | Potential frozen autoencoding comparator, but its sampled implicit dual-field input and vecset latent differ substantially from graph encoders. Report separately by input family |
| AutoBrep | Official Autodesk repository: <https://github.com/AutodeskAILab/AutoBrep> | Yes | **Excluded from the main representation table** | Decoder-only autoregressive generation has no documented canonical solid encoder embedding. Include only if an author-supported extraction protocol is established |

## Applicability decision

1. **BRepNet** is the only located external release currently active in the main registry because it
   exposes the complete raw-STEP path and official embedding export.
2. **Brep2Shape** would be a stronger self-supervised comparator, but a checkpoint alone is
   insufficient: its required precomputed Bezier/DGL inputs cannot be faithfully recreated from the
   released repository.
3. **UVStyle-Net** is retrieval-relevant, but its own-data feature pipeline is explicitly unreleased.
4. **DeepCAD** remains a separately labeled command-sequence-compatible subset analysis.
5. **DualBrep** is technically runnable from STEP but requires a C++ Voronoi pipeline and changes the
   input family to sampled implicit fields; it is an exploratory autoencoder comparison, not a core
   graph-representation baseline.

## Why BRepNet remains preferable to an unsourced UV-Net checkpoint

The official UV-Net repository contains the model, STEP preprocessing, classification, and
segmentation code, but no released checkpoint was found and its GitHub releases page is empty.
Therefore UV-Net cannot supply a no-retraining frozen baseline. BRepNet is from the same Autodesk
B-Rep ecosystem, its official repository includes OpenCascade preprocessing and a pretrained
checkpoint, and its documentation explicitly supports exporting embeddings. It is a mature
immediately sourceable **supervised-transfer** reference, with the caveat that its checkpoint was
trained for face segmentation; Brep2Shape is the preferred external self-supervised comparator.

## Input compatibility

- JEPA, local AE, local contrastive, graph statistics, and random controls can use the existing
  HistCAD graph cache and exact deterministic split.
- Brep2Shape requires Bezier primitive, face/edge topology, graph, and line-graph inputs. Until the
  authors release their raw-data generator, its paper-reported TMCAD score is related work rather
  than an executable baseline on this repository's split.
- BRepNet uses its official OpenCascade/UV-grid/coedge preprocessing from STEP. Its inputs are not
  fabricated from the compact v7 cache.
- UVStyle-Net cannot process these new parts through an official route while its own-data feature
  pipeline remains unreleased.
- DualBrep requires its official STEP-to-sampled-dual-field preprocessing and must be labeled as an
  implicit-field autoencoder rather than a graph encoder.
- DeepCAD consumes sketch/extrude command sequences, so it is applicable only where the source JSON
  can be converted by the official DeepCAD vectorizer.
- GC-CAD uses face/curve attributed B-Rep graphs, feature masking, structural augmentation, and
  NT-Xent. A same-encoder contrastive control can test the objective family fairly, but it is not a
  faithful GC-CAD reproduction.

## External artifact acceptance contract

An imported artifact must contain:

```text
format_version: cad_benchmark_embeddings_v1
uids: unique source identifiers in benchmark order
splits: train/val/test for every row
embeddings: {method_name: rank-2 finite tensor}
labels: optional row-aligned label arrays
targets: optional row-aligned continuous targets
metadata:
  source_repository
  source_revision
  checkpoint
  checkpoint_sha256
  preprocessing
  training_data
  pretraining_objective
  parameter_count
```

The benchmark refuses partial UID overlap for the main table. A deliberately restricted shared-subset
analysis must use a separate configuration and must report its retained sample count.
