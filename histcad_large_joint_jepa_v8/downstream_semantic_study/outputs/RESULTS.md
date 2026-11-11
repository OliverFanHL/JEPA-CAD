# Semantic supervision of frozen v7 HistCAD embeddings

**Preliminary, confounded text-teacher run.** This run encoded raw `Functional
Type` descriptions, many of which contain exact dimensions. It is preserved
for comparison, but the stricter semantic-only conclusions should use
`../outputs_clean/RESULTS.md`, where numbers and measurement parentheticals
are removed from teacher text before training.

## Question and method

The v7 JEPA encoder directly ingests parametric history graphs (including
sketches, constraints, dimensions, and feature operations) and STEP/B-Rep
graphs. Its pretraining objective also has a dimensional-prediction term; the
recorded test dimensional macro R² is 0.951. Geometry is therefore **explicit
in the input and objective**, although this does not guarantee every shape or
dimension is easily recovered by cosine nearest-neighbor search. Annotation
text is not part of JEPA pretraining.

The controlled downstream teacher is the frozen
[`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
sentence encoder. It maps HistCAD `Functional Type` descriptions to 384-D
semantic targets. A 2.31M-parameter CAD-to-text adapter learns from **145,899
nonempty train labels** (16 train descriptions are empty). There are 100,988
distinct nonempty train/validation descriptions; validation has 8,130 parts and
test has 8,086. The text teacher, v7 JEPA, and old adapter stay frozen. Six
epochs were run, and the best validation epoch was six for both new conditions.
All retrieval-time vectors are predicted from CAD alone—no test text is input
to the model. Test is not used for checkpoint or fusion selection.

The semantic-only adapter minimizes cosine error to its text teacher and
transfers teacher pairwise similarities. The matched semantic+geometry adapter
adds numeric dimension, feature-count, and operation heads to that same model.
For fusion, validation chooses a fixed semantic score weight subject to at
most 10% worse dimensional NN error and at most 5 percentage points lower
Dimensional Recall@10 than its base. It chose weights 1.0 for direct JEPA +
semantic and 0.5 for existing adapter + semantic.

## Held-out test retrieval

| CAD-derived representation | Dimensional Recall@10 ↑ | Dimensional NN error ↓ | Canonical function NN ↑ | Raw function NN ↑ | Canonical function Recall@5 ↑ | Unused `Geometric Feature` text cosine ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Direct frozen v7 JEPA | 28.0% | 0.254 | 31.2% | 20.4% | 56.8% | 0.753 |
| Existing geometry adapter | **53.0%** | **0.169** | 31.5% | 19.5% | 56.4% | 0.747 |
| New semantic-only adapter | 18.5% | 0.326 | 36.8% | **23.2%** | **65.4%** | **0.773** |
| New semantic + numeric-geometry adapter | 31.4% | 0.215 | **37.1%** | 22.6% | 63.8% | 0.772 |
| Direct JEPA + semantic score | 27.2% | 0.256 | 34.2% | 21.3% | 61.0% | 0.765 |
| Existing adapter + semantic score | 51.3% | 0.171 | 32.6% | 20.1% | 59.6% | 0.756 |

Nearest-neighbor function percentages condition on a matching label being
available in the **test candidate bank**. Only 46.7% of test queries have an
available canonical-function match. Dimensional error is mean absolute error
over 22 **standardized** targets, not millimeters. Dimensional Recall@10 is
the fraction of the ten true nearest dimensional neighbors recovered in the
retrieved top ten. Raw functional matching keeps parenthetical/dimensional
wording; canonical matching removes it. Both are imperfect proxies for function.

Against the existing geometry adapter, validation-selected semantic fusion
improves canonical-function Recall@5 by **3.23 points** (paired 95% bootstrap
interval +2.62 to +3.82) and `Functional Type` token Jaccard@1 by **0.0166**
(+0.0142 to +0.0190). It costs **1.75 points** of Dimensional Recall@10
(-1.94 to -1.57) and +0.0024 standardized dimensional NN error. Exact raw
function NN rises only from 19.5% to 20.1%: useful but modest. The combined
dimension/history/count success drops from 64.5% to 63.8%.

Semantic-only supervision has a clearer semantic gain over direct JEPA:
`Functional Type` token Jaccard@1 +0.0459 (95% interval +0.0405 to +0.0514),
canonical function NN +5.6 points, and held-out `Geometric Feature` token
Jaccard +0.0286 (+0.0249 to +0.0328). The `Geometric Feature` sentence cosine
also rises from 0.753 to 0.773, despite that field **not being used for
training**. That is evidence of some transfer beyond exact text-label
memorization, though both fields were authored about the same CAD parts.

The semantic+numeric-geometry condition preserves almost the same semantic
score as semantic-only while improving dimensional Recall@10 from 18.5% to
31.4%. It still trails the existing geometry-focused adapter at 53.0%. This
shows that geometry targets can change how the *downstream* vector is organized,
even though geometric data were already available to JEPA; they are not needed
to obtain the semantic gain itself. A fixed score fusion is the simpler option
when the old adapter's dimensional behavior should mostly be retained.

## Qualitative STEP-rendered checks

All 29 required distinct STEP parts in the four sheets rendered successfully.
Each sheet shows direct JEPA, existing adapter, semantic-only, and
existing+semantic results for the same query; green borders mark canonical
functional-label matches, **not** verified engineering equivalence.

- [Case 01: semantic win](retrieval_sheets/case_01_semantic_win.png) — the semantic branch moves a rectangular block ahead of a square extruded plate, but its second and third neighbors show that generic function wording can overrule dimensions.
- [Case 02: balanced win](retrieval_sheets/case_02_balanced_win.png) — a rectangular-block query gains a same-function first neighbor in the fusion; the rendered shapes are similar, while the numeric dimensional distance is needed to assess scale.
- [Case 03: geometry trade-off](retrieval_sheets/case_03_geometry_tradeoff.png) — semantic-only retrieval chooses other thin plates but can worsen dimensional distance from roughly 0.11 to 0.48; fusion restores the original dimension-oriented neighbors.
- [Case 04: random eligible query](retrieval_sheets/case_04_random_eligible.png) — a seeded, non-win-selected control example.

The renderer normalizes display scale and viewpoint, so drawings alone can
hide large physical dimension differences. The sheet's standardized
dimensional distance and labels must be inspected alongside the shape.

## Interpretation and next step

Semantic information **does** improve retrieval from frozen JEPA features,
particularly functional wording and held-out geometric-description relevance.
It does **not** improve both semantic and dimensional metrics simultaneously
without a trade-off in this study. The best low-cost balance is the old adapter
plus a validation-selected semantic score (or direct JEPA plus semantic score
if operation/dimensional specialization is not wanted). A user-facing system
could expose the semantic weight as a retrieval preference rather than claim a
single universal ranking.

`NLT` is not a good first semantic target: it extensively restates geometric
and numeric construction details, and its token-set overlap was around 0.9
even for baseline neighbors, offering little discrimination. `Modeling Process`
overlap changed only modestly. `Geometric Feature` was retained **evaluation
only** here; it may be a later source of shape-oriented supervision if the
desired task is shape/feature retrieval, but using it now would confound the
isolated semantic-supervision test. A stronger next controlled experiment
would use expert-labeled functional families or relevance judgments, especially
because the current free-form labels contain many unique dimensional variants.

This is exploratory: the old adapter was trained with a different architecture
and 14 epochs; the new adapters used six epochs and their validation losses
were still decreasing. The study establishes directional trade-offs, not a
converged architecture ranking. All per-condition metrics and paired intervals
are in `evaluation.json`; unused-field tests are in `annotation_transfer.json`.
