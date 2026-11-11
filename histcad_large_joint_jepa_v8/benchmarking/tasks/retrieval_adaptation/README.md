# Retrieval adaptation

Reports original direct retrieval, the same fixed-capacity residual adapter for every representation,
and the per-metric gain after adaptation. Approximate positive neighbors are constructed once from the
same standardized CAD geometry targets. Evaluation reuses the existing retrieval study's within-split,
leave-self-out protocol and metric family: dimensional neighbor recall/error, operation history,
feature sequence/count, primitive/constraint profile, canonical function, duplicates, and tie margins.
Checkpoint selection uses validation loss only.
