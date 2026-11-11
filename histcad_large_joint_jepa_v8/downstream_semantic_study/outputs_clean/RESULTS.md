# Controlled semantic supervision on frozen v7 HistCAD features

**Follow-up:** A matched dimension-only condition was subsequently trained to
separate measurement supervision from feature-count/operation supervision.
See [FACTORIAL_ABLATION.md](FACTORIAL_ABLATION.md) for the six-condition result;
the tables below describe the original four-condition retrieval comparison.

## Setup and geometric-information question

The v7 JEPA encoder is frozen. It originally ingested parametric-history and
STEP/B-Rep graphs, including sketches, constraints, dimensions, feature
operations, typed entities, and connectivity. Its pretraining objective also
predicted dimensional targets; the recorded held-out dimensional macro R² was
0.951. Thus geometric/parametric information was **explicitly available to
JEPA**, but this does not prove that the final cosine neighborhood preserves
every detail. The direct v7 Dimensional Recall@10 of 28.0% shows that a
retrieval-specific organization still matters.

This study trains a 2.31M-parameter CAD-to-semantic adapter on frozen v7
embeddings. A locally cached, frozen
[`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
teacher supplies 384-D targets from the dataset's short `Functional Type`
descriptions. To isolate **semantic** supervision, all parenthetical phrases,
numbers, and explicit measurement units are removed before the teacher sees
text: for example, `Rectangular block (124 x 64 x 124 mm)` becomes
`Rectangular block`. This yields 62,129 distinct nonempty train/validation
teacher texts. The adapter uses 145,899 nonempty train examples (16 train
descriptions are empty); 8,130 validation parts select checkpoints and fusion
weights. The 8,086 test parts are held out. Neither JEPA nor the text teacher
is updated, and no part text is supplied at retrieval time.

Two matched new adapters use the same architecture, initialization seed, and
six-epoch budget. The first minimizes cosine error to the frozen text teacher
plus pairwise text-neighborhood distillation. The second adds numeric
dimension, feature-count, and final-operation heads. Both select the lowest
validation-loss checkpoint. Fixed fusion weights are selected on validation
under at most +10% dimensional NN error and -5 points Dimensional Recall@10;
the selected weights were 1.0 for direct JEPA + semantic and 0.5 for the
existing adapter + semantic. These are score-space fusions, not retraining of
the existing adapter.

## Held-out test results

| CAD-derived retrieval vector | Dimensional Recall@10 ↑ | Dimensional NN error ↓ | Canonical function NN ↑ | Exact raw function NN ↑ | Canonical function Recall@5 ↑ | Unused `Geometric Feature` text cosine ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Direct frozen v7 JEPA | 28.0% | 0.254 | 31.2% | 20.4% | 56.8% | 0.753 |
| Existing geometry adapter | **53.0%** | **0.169** | 31.5% | 19.5% | 56.4% | 0.747 |
| Clean semantic-only adapter | 16.4% | 0.348 | **37.0%** | **22.1%** | **65.2%** | 0.771 |
| Clean semantic + numeric-geometry adapter | 29.8% | 0.220 | 36.4% | 21.8% | 64.4% | **0.771** |
| Direct JEPA + clean semantic score | 26.9% | 0.258 | 34.3% | 21.1% | 61.4% | 0.765 |
| Existing adapter + clean semantic score | 51.2% | 0.171 | 33.1% | 20.1% | 59.7% | 0.756 |

Function metrics condition on a matching label being present in the test
candidate bank. Only about 46.7% of queries have an available canonical
function match; exact raw descriptions are even sparser. Canonical labels
drop parenthetical text and dimensions, so a broad `Rectangular block` match
is not evidence of dimensional interchangeability. Dimensional error is mean
absolute difference over 22 standardized targets, **not millimeters**.
Dimensional Recall@10 is the fraction of the ten true dimensional neighbors
recovered among the retrieved ten.

The clean semantic-only adapter improves `Functional Type` token Jaccard@1
from 0.410 to 0.456 versus direct JEPA: +0.0465 (paired 95% bootstrap interval
+0.0411 to +0.0525). It also improves the held-out, **unused-for-training**
`Geometric Feature` annotation's token Jaccard by +0.0260 (+0.0220 to
+0.0297) and its frozen text-embedding cosine from 0.753 to 0.771. This is
evidence of transfer to descriptions beyond the exact supervision field,
though those annotations were authored for the same CAD items and are not
independent expert relevance ratings.

There is a substantial geometry trade-off. Clean semantic-only Dimensional
Recall@10 falls from 28.0% to 16.4%; the geometry-supervised companion raises
it to 29.8% with little loss of semantic accuracy, but remains far below the
existing geometry adapter's 53.0%. This means numeric geometry objectives can
reorganize a downstream vector, even with geometry already in JEPA, but they
are **not required for the semantic gain**.

The most practical balanced condition is the existing geometry adapter plus
a small clean-semantic score. Relative to the old adapter, canonical-function
Recall@5 improves by **3.29 percentage points** (95% interval +2.68 to
+3.90), token Jaccard@1 rises by +0.0199 (+0.0173 to +0.0226), and canonical
function NN improves by 1.59 points (+1.09 to +2.15). Dimensional Recall@10
falls by 1.86 points (-2.06 to -1.67), and standardized nearest dimensional
error rises by 0.0027. Exact raw-function NN rises only 0.56 points, with a
95% interval crossing zero; avoid claiming a robust exact-description gain.
Combined dimension/history/count success drops slightly from 64.5% to 63.8%.
No condition improves *all* semantic and geometric metrics simultaneously.

## Qualitative checks

Four deliberately mixed STEP-rendered sheets show the same query and top
three results under direct JEPA, old geometry adapter, clean semantic-only,
and old+semantic score. All four sheets rendered successfully after skipping
three STEP files that this OpenCASCADE visualization path could not triangulate;
those files were not removed from quantitative evaluation.

- [Semantic win](retrieval_sheets/case_01_semantic_win.png): semantic training moves a rectangular block above a generically labeled cube, but dimensional distances remain essential to interpreting quality.
- [Balanced win](retrieval_sheets/case_02_balanced_win.png): the score fusion promotes a matching rectangular plate while keeping a low dimensional distance.
- [Geometry trade-off](retrieval_sheets/case_03_geometry_tradeoff.png): semantic-only neighbors share the broad `Rectangular plate` label but may be far apart in physical dimensions; the old adapter and score fusion preserve closer dimensional neighbors.
- [Seeded control query](retrieval_sheets/case_04_random_eligible.png): not selected for an improvement claim.

Green borders indicate **canonical annotation matches**, not verified
engineering equivalence. Rendered shapes are normalized in display size and
orientation, so visually similar thumbnails can hide large real-scale
differences. The numeric dimensional distance shown under every retrieved part
must be read alongside the shape.

## Conclusions and limits

Removing sizes from text **did not eliminate** the semantic improvement.
This supports using short functional descriptions as additional downstream
supervision while leaving JEPA frozen. A pretrained generic sentence model is
a useful low-cost teacher, but the test is CAD-to-CAD retrieval; it does not
show text-query-to-CAD search or that text semantics were part of JEPA itself.

`Geometric Feature` was evaluation-only in this controlled run. Introducing it
as a training target would mix detailed shape supervision with function and
make the semantic-only question harder to answer. `NLT` heavily restates
geometry and dimensions; its token overlap is around 0.9 even for the direct
baseline, so it is a poor first semantic benchmark. The existing geometry
adapter differs in architecture and training protocol (14 epochs versus six
for the new adapters), and both new validation losses were still decreasing at
epoch six. These are exploratory trade-offs, not a converged architecture
ranking. Stronger future evidence would require expert functional-family or
relevance judgments and dimension-tolerance-aware retrieval metrics.

Full metrics, paired intervals, training curves, text-target audit, rendered
case manifest, and render failures are retained beside this report.
