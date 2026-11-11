# v7 downstream adapter: retrieval diagnostic

## Protocol and data

The main v7 JEPA encoder and the existing retrieval adapter were already trained.
This diagnostic does not update either checkpoint. It compares cosine nearest
neighbors from the direct 4096-D v7 joint embedding and from the adapter's
4096-D embedding. Queries and candidate banks use the same split (self-match
excluded): 8,130 validation parts and 8,086 test parts. The original adapter
trained on **all 145,915 training parts**, not the earlier 50k development
subset. Its early-stopped checkpoint was selected by validation objective; test
was not used for selection. The best epoch was 11 of 14 run: train objective
0.0389 versus validation 0.0547. Validation did not improve further despite
continued training-loss reduction, so more epochs alone are not the first fix.

The adapter objective uses approximate positive pairs derived from 22
standardized dimensional targets, feature-count bucket, and final Boolean
operation. Its auxiliary heads predict dimensions, feature-count bucket, and
final operation; an anchor term discourages excessive drift from v7. Functional
labels and other annotation text were not used by the original adapter.
The adapter reshapes the final embedding into 32 coordinate chunks and applies
attention; these chunks are **not** CAD graph nodes or semantic feature tokens.

## Test retrieval: intuitive task metrics

| Test metric | Direct v7 | Adapted v7 | Interpretation |
|---|---:|---:|---|
| Dimensional Recall@10 | 27.9% | 53.0% | Fraction of the 10 dimension-reference neighbors recovered in the top 10; original benchmark definition. |
| Nearest dimensional error | 0.254 | 0.169 | Mean absolute distance across 22 standardized dimensional targets; lower is better, not mm. |
| Nearest dimensions within 0.2 | 47.8% | 66.4% | Mean 22-target error at most 0.2 standardized units. |
| Nearest sorted bounding-box log-ratio error | 0.642 | 0.372 | Lower means closer overall part scale/side lengths. |
| Exact operation-history NN accuracy | 84.5% | 91.1% | Same full operation signature. |
| Exact feature-count NN accuracy | 87.4% | 96.6% | Same integer feature count. |
| Dimension + history + feature-count NN success | 46.6% | 64.5% | All three conditions satisfied for the first neighbor. |
| Exact primitive-profile NN accuracy | 64.8% | 58.0% | Same sketch primitive-count profile; **declines**. |
| Functional-description token Jaccard@1 | 0.410 | 0.401 | Mean token overlap of `Functional Type` labels; **declines**. |
| Exact functional-label NN accuracy, eligible queries | 31.2% | 31.6% | Only queries with another same-label item in the test bank. |
| Exact functional-label Recall@5, eligible queries | 56.8% | 56.4% | No functional improvement. |
| Exact functional-label Recall@10, eligible queries | 65.6% | 64.8% | No functional improvement. |

The first line is from the original evaluator; all other rows come from
`extended_evaluation.json`. The extended history/count/primitive rows include
every test query, whereas some original evaluator categorical metrics condition
on label eligibility. Do not mix the numerical values of those two definitions.
Only **46.7%** of test queries even have a duplicate canonical functional label
in the test bank, so unconditional exact-label accuracy is not a suitable sole
measure. Across *all* queries it was 14.56% direct versus 14.73% adapted, a
0.17-point change. The paired 95% bootstrap interval for that change is
[-0.37, +0.72] points, consistent with no effect.

For the main improvements, the paired 95% bootstrap intervals (adapter minus
direct) are: dimensional NN error -0.0853 [-0.0883, -0.0821], dimensional
threshold success +18.6 [+17.7, +19.5] points, and joint dimension/history/
count success +17.8 [+17.0, +18.7] points. Primitive-profile accuracy changes
-6.84 [-7.57, -6.16] points. For individual queries, the adapter improves
dimensional NN error for 48.4%, worsens it for 5.9%, and leaves it effectively
unchanged for 45.7%. This is substantial but targeted improvement, not an
all-purpose gain.

## Why some rendered neighbors look poor

The six existing image sheets are valid STEP-based renders, but they were
selected to illustrate improvement in the adapter's geometry/operation target,
not sampled representatively for function. For example, `case_01.png` retrieves
parts with closer dimensional signatures and matching operation histories after
adaptation, but they are visually different from the L-bracket query. By
contrast, `case_05.png` shows a U-bracket where the adapted first neighbor is
visually and functionally plausible. Both outcomes are consistent with the
aggregate metrics. A future gallery should stratify randomly over wins,
unchanged cases, regressions, and functional-label errors.

The major limitation is **objective mismatch**: positive pairs and auxiliary
heads reward dimensional closeness, count, and final operation, not sketch
primitive profile, detailed shape, or function. Counting/history metrics are
also easy on this dataset (random-pair exact history and count baselines are
about 52% and 54%), so high percentages alone should not be mistaken for full
shape understanding. The main v7 joint embedding has an estimated test-set
effective participation rank of about 16 of 4096 ambient dimensions; the
adapter increases this to about 24, still highly correlated. This suggests
limited independent variation but does not prove specific source information
is absent. The paired test improvements and similar validation/test results do
not support *too little adapter-training data* as the primary cause. More
adapter data is unavailable within the present split; merely training longer
against the same target would not address semantic shape/function mismatch.

## Downstream functional-label supervision test

The annotation CSV contains `NLT`, `Modeling Process`, `Geometric Feature`,
and `Functional Type`. The existing v7 cache retains only `Functional Type` for
evaluation. To test a modest downstream semantic strategy without changing
JEPA, a linear 136-token multilabel readout was trained on frozen adapted
embeddings using only frequent tokens from **train** `Functional Type` labels.
Validation selected the checkpoint and a small fusion weight (0.5); test was
queried only after selection. This probe is exploratory and is not the original
adapter.

| Test metric | Direct | Adapted | Semantic readout only | Adapted + readout |
|---|---:|---:|---:|---:|
| Functional token Jaccard@1 | 0.410 | 0.401 | **0.434** | 0.413 |
| Exact function@1, all queries | 14.6% | 14.7% | **16.0%** | 15.1% |
| Dimensional NN error | 0.254 | **0.169** | 0.264 | 0.174 |
| Dimension + history + count success | 46.6% | **64.5%** | 44.7% | 62.5% |

Semantic supervision can extract some functional information from the frozen
representation, but semantic-only retrieval sacrifices the geometric gains.
The validation-selected fusion yields only a small semantic benefit at a small
cost to geometry/structure. The many unique free-form functional labels (61,700
distinct in training, with 53,620 singleton examples) also make exact-label
matching a difficult and sometimes brittle target. Text-token overlap is a
useful supplementary metric, but wording overlap is not identical to CAD
functional equivalence.

## Recommendation

For dimensional/operation-oriented retrieval, retain the current adapter; it
has clear, stable task-level gains. Do not add more unlabeled epochs as the
first remedy for qualitative function/shape failures. First define the target
retrieval notion and construct positives that enforce it: for example a
multi-positive, same-structure-and-dimension objective with explicit sketch
primitive/feature-sequence terms, and a separate supervised functional branch
or score fusion when function is desired. Treat `NLT` and `Geometric Feature`
as potential downstream supervision, not as evidence that the present JEPA
embedding explicitly contains semantics; these texts partly restate geometry
and dimensions and must be kept out of test-pair construction. Evaluate such a
change on held-out labels with both geometry/structure and functional metrics,
and with representative retrieval galleries.

## Reproduction artifacts

- `extended_evaluation.py` -> `outputs/extended_evaluation.json`, `outputs/extended_per_query_test.csv`
- `semantic_probe.py` -> `outputs/semantic_probe.pt`, `outputs/semantic_probe_evaluation.json`
- Existing adapter -> `outputs/best_adapter.pt`, `outputs/evaluation.json`
- Existing STEP-based gallery -> `outputs/retrieval_sheets/case_01.png` through `case_06.png`

The exact `dimension <= 0.2` threshold is a transparent diagnostic choice on
standardized targets, not a physical tolerance or a paper-protocol metric.
