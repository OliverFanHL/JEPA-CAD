# Manuscript figure and evidence audit

- All LaTeX figure paths exist; figure, table, and equation environments balance.
- All 26 cross-references resolve to unique labels; all 22 cited keys exist in `ref.bib`.
- Published TMCAD values (BRT 0.8345, UV-Net 0.8131, AAGNet 0.7472) appear in Table 3 beside the separately labeled present test protocol.
- The [BRT source paper](https://arxiv.org/pdf/2504.07134) reports those values in Table 1 and uses separate bolt/screw classes; the prepared study cohort's distinct labels are stated in the manuscript.
- The framework uses `jcpr_framework.png`; deprecated transfer and qualitative composites are not referenced.
- All 32 matched retrieval cells are separate original-STEP renderings (at least 490 pixels on the long side and 200 on the short side). Source and asset SHA-256 hashes match the current retrieval manifest.
- The TMCAD held-out prediction file has 1,404 rows, including 55 corrections and 7 regressions; residual sensitivity was recomputed from the saved checkpoint and frozen inputs.
- Three control/residual integrated-gradient comparisons reproduce saved predictions; each attribution sum matches the mean-input-to-part margin change within 0.01 logit.
- The three spatial-attribution parts have independently recomputed 5-NN decisions for JCPR, the graph autoencoder, and BRepNet; each method's full-test 5-NN macro-F1 exactly matches the saved benchmark.
- JCPR face-response maps reproduce cached frozen embeddings within 0.002 and project only uniquely matched STEP faces; at least 70% of each displayed part's faces are verified.
- All plotted manuscript figures are single-page PDFs with embedded Arial and compact page boxes; the CAD renderings are embedded in PDF figure artwork:
  - histcad_numeric_benchmark.pdf: 6.46 × 4.76 in, Arial embedded
  - semantic_tradeoff.pdf: 6.32 × 4.88 in, Arial embedded
  - tmcad_confusion_change.pdf: 6.75 × 3.81 in, Arial embedded
  - tmcad_residual_interpretability.pdf: 6.69 × 2.80 in, Arial embedded
  - tmcad_gradient_attribution.pdf: 6.83 × 3.65 in, Arial embedded; CAD surface images embedded
  - jcpr_spatial_gradcam.pdf: 6.91 × 6.31 in, Arial embedded; CAD surface images embedded
- The revised 19-page manuscript PDF was compiled and the figure/table pages were visually reviewed.
