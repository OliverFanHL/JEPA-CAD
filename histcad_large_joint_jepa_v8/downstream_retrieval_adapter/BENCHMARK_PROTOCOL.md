# Retrieval benchmark protocol

## Research question

Does the pretrained multimodal JEPA representation provide better CAD retrieval behavior than controlled
alternative representations?

The main comparison is `pretrained_jepa_joint` against `graph_autoencoder_joint`,
`pca_graph_statistics`, and `input_graph_statistics`. `adapted_jepa_joint` is a secondary ablation because
it starts from the JEPA representation and adds downstream supervision.

## Controlled factors

| Factor | Rule |
|---|---|
| Samples and split | Use the v7 cache's deterministic train/validation/test assignment for every method. |
| Inputs | JEPA and the graph autoencoder receive the same cached parametric and STEP graphs. |
| Neural architecture | The autoencoder reuses the v7 parametric encoder, STEP encoder, joint projection architecture, and 4096-D joint width, initialized from scratch. |
| Model selection | JEPA uses its validation objective; the autoencoder uses validation reconstruction loss. Test is never used for selection. |
| Retrieval | L2-normalize each representation, use cosine similarity, exclude the query itself, and retrieve only within the evaluated split. |
| Metrics | Use one shared call to `histcad_large_jepa.benchmark.retrieval_metrics` for every method. |
| Labels | Retrieval labels are never used to train a primary representation. They are read only by the evaluator. |
| Queries/candidates | Every method is evaluated on the same UID-aligned validation and test tensors. |

## Autoencoder target

The reconstruction target is a deterministic summary of the input graphs: categorical node/edge
distributions, masked numeric means and standard deviations, and log node/edge counts. STEP entity types
use a fixed-size hashed histogram. Targets are standardized with train-split mean and variance only.
This makes the baseline a graph-input reconstruction model rather than an autoencoder over pretrained JEPA
vectors.

## Reporting

Report validation and test metrics for all methods, but use test only once the configuration is fixed.
The primary table should contain all retrieval metrics of interest rather than selecting whichever metric
favors a method. Clearly mark dimensional MAE as lower-is-better and recall/accuracy/success metrics as
higher-is-better.

If multiple random seeds are later run, keep the split fixed, vary only initialization/data-loader seeds,
and report mean, standard deviation, and per-seed results. Do not select the best test seed.

External published results belong in the primary table only if they use a sufficiently equivalent dataset,
split, query/candidate pool, relevance definition, and metric implementation. Otherwise cite them in a
separate related-work comparison without a numeric claim of superiority.
