# Where to retain dimensions: downstream adapter ablation

**Answer:** The recommendation refers to **downstream adapter training**.
The v7 JEPA pretraining stage already used parametric/B-Rep geometry and a
dimensional-prediction loss; it was not changed. If dimensional similarity is
an intended retrieval property, the downstream adapter should be given the
CAD's explicit numeric dimensional targets rather than relying on natural
language to carry them. The full text can still be retained as a separate
semantic target.

We compared eight downstream models in a matched 2×4
ablation. Each used the same frozen 4096-D JEPA vectors, frozen 384-D MiniLM
text teacher, 2.31M-parameter adapter architecture, 145,899 labeled training
parts, seed, validation/test split, and six-epoch training budget. Only two
factors changed: whether `Functional Type` text kept its original measurements
or was stripped of numerical measurements, and whether the adapter received
no numeric CAD loss, a **physical-size-only** loss (15 values), the full
22-value CAD measurement descriptor, or the 22-value descriptor plus separate
feature-count and operation losses. No test labels
selected a checkpoint or fusion weight. The same test candidate bank and
evaluation code were applied to all eight saved checkpoints.

| Downstream supervision | Dimensional Recall@10 ↑ | Dimensional NN error ↓ | Canonical function NN ↑ | Canonical function Recall@5 ↑ | Functional token Jaccard@1 ↑ |
|---|---:|---:|---:|---:|---:|
| Size-stripped functional text only | 16.4% | 0.348 | **37.0%** | 65.2% | **0.456** |
| Full functional text only | 18.5% | 0.326 | 36.8% | **65.4%** | 0.456 |
| Size-stripped text + physical sizes only | 30.5% | 0.221 | 36.7% | 63.9% | 0.451 |
| Full text + physical sizes only | **31.9%** | 0.218 | 36.8% | 64.2% | 0.451 |
| Size-stripped text + dimensions only | 30.1% | 0.217 | 36.6% | 64.3% | 0.452 |
| Full text + dimensions only | **31.6%** | **0.212** | 36.9% | 64.2% | 0.452 |
| Size-stripped text + numeric CAD targets | 29.8% | 0.220 | 36.4% | 64.4% | 0.453 |
| Full text + numeric CAD targets | **31.4%** | **0.215** | **37.1%** | 63.8% | 0.454 |

The physical-size-only rows supervise bounding-box scale/sides/ratios and
aggregate constraint/feature sizes—**no count or operation targets**. They
still do not supply every individual hole diameter; those details remain in
the frozen CAD input, not necessarily in this 15-value downstream target. The
22-value descriptor adds several counts, so its rows are not a pure
physical-measurement control. The last two rows further add separate
feature-count and final-operation heads.

Holding the numeric-target factor off, retaining measurements in the text
teacher increased Dimensional Recall@10 by **2.09 points** (paired 95%
bootstrap interval +1.89 to +2.28) and reduced standardized dimensional NN
error by 0.0222. With numeric targets already present, the extra full-text
measurements added **1.60 points** (+1.42 to +1.77) and reduced error by
0.0053. Neither text change produced a reliable canonical-function NN or
token-Jaccard gain: the corresponding paired intervals include zero.
For the cleaner physical-size-only condition, retaining measurements in text
added 1.40 points (+1.22 to +1.59) beyond the explicit numeric targets.

Explicit **physical-size-only** CAD supervision had a much larger dimensional effect:

- With stripped text, it added **14.15 points** of Dimensional Recall@10
  (paired 95% interval +13.81 to +14.49) and reduced dimensional NN error by
  0.1270.
- With full text, it added **13.46 points** (+13.14 to +13.78) and reduced
  dimensional NN error by 0.1081.

The full 22-value descriptor also improved dimensional retrieval:

- With stripped text, it added **13.77 points** of Dimensional Recall@10
  (paired 95% interval +13.44 to +14.10) and reduced dimensional NN error by
  0.1312.
- With full text, it added **13.10 points** (+12.79 to +13.41) and reduced
  dimensional NN error by 0.1140.

Adding feature-count and operation targets on top of the 22-value descriptor did **not**
further improve dimensional retrieval here: full-text Dimensional Recall@10
changed from 31.6% to 31.4%, while history NN accuracy rose from 86.8% to
93.4% and exact feature-count NN accuracy rose from 90.3% to 97.5%.
The earlier bundled comparison, which used all three targets, gave:

- With stripped text, it added **13.46 points** of Dimensional Recall@10
  (+13.10 to +13.82) and reduced dimensional NN error by 0.1275.
- With full text, it added **12.97 points** (+12.66 to +13.30) and reduced
  dimensional NN error by 0.1107.

Thus, dimensions in natural language were useful but only a partial substitute
for explicit numeric CAD targets. The extra benefit of including sizes in text
became smaller once numeric supervision was available. This is consistent
with some information overlap, but the ablation does not prove exactly which
measurements the generic sentence model encodes. Units and numerical
equivalence should be handled using CAD quantities, not inferred solely from
text similarity.

The numerical heads did not simply reproduce the existing geometry adapter:
its Dimensional Recall@10 remains 53.0%, above the best new physical-size
condition's 31.9%. The full-text physical-size condition improves semantic
retrieval, however (canonical function NN 36.8% versus 31.5% for the existing
adapter). The earlier
validation-selected **existing adapter + semantic score** is still the best
low-cost balance when dimensional recall must stay near the existing model:
with the clean semantic teacher, it yields 51.2% dimensional recall and 33.1%
canonical function NN; with full-text semantic teacher, 51.3% and 32.6%.

All dimensional errors are over 22 standardized CAD targets, not millimeters.
Canonical function matching removes parenthetical size phrases, and only
queries with an available matching label are scored. Raw exact-label matching
is reported separately in `factorial_ablation.json`. All new adapters were
still improving on validation at epoch six, so these numbers identify a
directional effect rather than a fully converged optimum. The full paired
contrasts, confidence intervals, interaction terms, and evaluation-only
annotation-field results are in `factorial_ablation.json`.
