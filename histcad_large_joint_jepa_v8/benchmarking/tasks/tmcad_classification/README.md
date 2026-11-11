# TMCAD classification transfer

All methods use the saved TMCAD train/validation/test split and labels. Frozen linear probes and
controlled adapters have exactly the same trainable capacity and training budget. The existing
task-specific supervised result is imported separately. `baselines/controls/train_tmcad_from_scratch.py`
provides the end-to-end random-initialization STEP-encoder control.
