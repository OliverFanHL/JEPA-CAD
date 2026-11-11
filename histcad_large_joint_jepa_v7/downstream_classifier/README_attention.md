# Role-token attention classification (32 epochs)

This follow-up is directly comparable to the earlier frozen-v7 linear and
MLP probes in [README.md](README.md). The same `best.pt`-derived embeddings,
145,915 / 8,130 / 8,086 UID split, taxonomy, train-only rare-class rule,
class weighting, validation model-selection metric, and test metrics are used.
Only the downstream readout changes. JEPA remains frozen; no raw annotation
text or class label is passed to the classifier at inference.

## Architecture

The previous retrieval adapter attended over arbitrary coordinate chunks of
the 4096-D vector. Here the tokens have an actual CAD role, using the v7
encoder's contiguous 512-D role-aware summaries:

- Five parametric tokens: feature operations, sketch geometry, constraints,
  all relational nodes, and the global/history node.
- Four STEP tokens: typed entities, B-Rep instances, connectivity, and the
  global/type node.
- One additional global token projected from the 4096-D joint embedding.
- One learned classification-query token.

Separate source projections map each frozen summary to 256 dimensions. Two
8-head residual self-attention blocks with 768-wide feed-forward networks
allow cross-role and cross-source interaction. The query output feeds four
task heads. The probe has 2,707,492 trainable parameters; the JEPA encoder has
zero trainable parameters in this run. [`attention_classifier.py`](attention_classifier.py)
contains the exact model.

Training used 32 epochs, batch size 1024, AdamW, cosine learning-rate decay
from 3e-4 to 3e-5, class-weighted cross-entropy, and mixed precision. The
checkpoint with the highest **validation mean macro-F1** is retained. The
test split is evaluated once after all 32 epochs. Run with:

```powershell
python run_attention.py
python plot_attention_history.py
python render_attention_cases.py
```

Configuration is in [`config_attention.yaml`](config_attention.yaml).

## Held-out test comparison

| Frozen v7 readout | Form accuracy | Form macro-F1 | Sketch macro-F1 | Constraint macro-F1 | History macro-F1 | Mean macro-F1 |
|---|---:|---:|---:|---:|---:|---:|
| Joint linear | 65.8% | 0.455 | 0.994 | 0.968 | 0.874 | 0.823 |
| Joint MLP | 69.1% | 0.500 | 0.997 | 0.977 | 0.921 | 0.849 |
| Parametric linear | 70.0% | 0.500 | 0.999 | 0.993 | 0.989 | 0.870 |
| Role-token attention | **73.5%** | **0.548** | **0.999** | **0.994** | **0.997** | **0.885** |

The attention probe improves form accuracy by 4.4 percentage points over the
joint MLP and 3.5 points over the parametric linear probe. Its form parent
macro-F1 is 0.875, versus 0.856 for the joint MLP. Exact values, balanced
accuracy, per-class recall, confusion matrices, baselines, and counts are in
[`attention_results.json`](outputs/attention_role_tokens_32ep/attention_results.json)
and [`comparison_to_existing_probes.csv`](outputs/attention_role_tokens_32ep/comparison_to_existing_probes.csv).

## Training behavior and limitations

The best validation mean macro-F1, 0.8867, occurs at epoch 32. Held-out test
mean macro-F1 is 0.8847. However, validation weighted loss bottoms near
epoch 9 and rises to 0.2846 by epoch 32 while training loss continues down to
0.1317. At the selected checkpoint, form macro-F1 is 0.780 on training,
0.557 on validation, and 0.548 on test. Thus the 32-epoch run improved the
selected macro-F1 but **did not eliminate overfitting**. The trajectory is in
[`training_history.csv`](outputs/attention_role_tokens_32ep/training_history.csv)
and [`training_curves.png`](outputs/attention_role_tokens_32ep/training_curves.png).

The sketch, constraint, and modeling-history labels are deterministic
functions of the parametric input and should not be interpreted as independent
semantic ground truth. The 23 form labels come from generated free-text
`Functional Type` descriptions (96.3% test coverage), and compound descriptions
can be ambiguous. High-confidence errors such as *bracket* versus *plate* and
*washer* versus *ring* expose both model confusions and taxonomy limitations.
An attention weight is only a diagnostic; it does not prove a token caused a
classification. STEP files in the qualitative sheet were independently
imported by OpenCASCADE, triangulated, and checked. No unrendered sample was
substituted; the [`render_audit.json`](outputs/attention_role_tokens_32ep/render_audit.json)
documents 12 valid rendered cases.

The [qualitative sheet](outputs/attention_role_tokens_32ep/qualitative_sheet.png)
shows four correct cases, four same-parent mistakes, and four cross-parent
mistakes. The full labels, descriptions, confidences, STEP paths, and a
final-block attention diagnostic are saved in
[`qualitative_cases.csv`](outputs/attention_role_tokens_32ep/qualitative_cases.csv).
