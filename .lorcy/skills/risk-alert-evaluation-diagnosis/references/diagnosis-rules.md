# Diagnosis Rules

## Precision And Recall

- High precision, low recall: model is conservative; consider threshold adjustment, positive weighting, or recall-oriented loss changes.
- Low precision, high recall: model over-alerts; consider threshold increase, regularization, or better feature fusion.
- Both low: representation is weak or training is unstable.

## AUC And F1

- High AUC, low F1: ranking is useful but threshold is poor.
- Low AUC: model representation or features are not separating classes well.
- F1 improves while AUC drops: thresholding may improve apparent binary decisions but probability ranking worsened.

## Horizon Differences

- 1-7 days strong, 15-21 days weak: temporal or long-range signals are insufficient.
- All horizons weak: check feature parsing, masks, label order, and training loop correctness first.
- One horizon collapses: consider task-specific head or horizon-specific threshold.

## Training Logs

- Train loss decreases but validation metrics degrade: overfitting.
- Loss stagnates: underfitting, learning rate, architecture capacity, or feature handling issue.
- Metrics fluctuate heavily: training instability, small validation positives, or threshold sensitivity.

## Recommended Next Actions

Prefer changes that are small enough to implement reliably in one iteration. Mention exact model/training edits, not vague advice.
