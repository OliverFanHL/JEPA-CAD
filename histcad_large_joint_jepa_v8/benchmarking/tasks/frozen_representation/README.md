# Frozen representation task

Runs direct held-out cosine retrieval, train-bank kNN, fixed-capacity linear/nonlinear probes, latent
rank and variance diagnostics, duplicate/tie analysis, similarity margins, and label separation.
Original dimensions are preserved for retrieval/kNN; only learned probes receive the deterministic
common-width projection.
