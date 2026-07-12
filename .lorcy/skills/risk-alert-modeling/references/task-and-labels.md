# Task And Labels

The task predicts whether a breeding farm will experience abortion abnormal risk in three future windows.

Labels:

- `abort_1_7`: risk in the next 1-7 days.
- `abort_8_14`: risk in the next 8-14 days.
- `abort_15_21`: risk in the next 15-21 days.

This is a multi-label binary prediction task. A sample may have multiple positive labels across windows.

Evaluation uses per-window Precision, Recall, F1, and AUC. Later windows are harder; avoid optimizing only `abort_1_7` if it harms 8-14 or 15-21 day performance.

The model output must be a logit tensor with shape `(batch_size, 3)` in the order:

1. `abort_1_7`
2. `abort_8_14`
3. `abort_15_21`
