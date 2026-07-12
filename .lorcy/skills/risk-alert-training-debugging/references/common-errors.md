# Common Errors

## Initialization Errors

- Missing `super().__init__()` before assigning modules.
- Parameter key mismatch such as `reserve_sow_sqty` vs `reverse_sow_sqty` in comments.
- Embedding input dtype not converted to `long`.

## Shape Errors

- Concatenating tensors with mismatched dimensions.
- Forgetting to unsqueeze masks to `(batch_size, 1)`.
- Returning probabilities or a tuple instead of a single `(batch_size, 3)` logit tensor.

## Mask Errors

- Treating missing values as normal zeros without mask use.
- Applying a feature mask with the wrong feature index.
- Multiplying discrete IDs by masks before embedding; embed first, then mask embeddings.

## Training Errors

- Applying sigmoid before `BCEWithLogitsLoss`.
- Saving whole model object instead of `state_dict`.
- Validation AUC crashing when a label has only one class; guard with try/except or class checks.

## Prediction Errors

- Shuffling prediction DataLoader.
- Merging prediction rows by transformed `org_inv_dk`.
- Saving probability/decision columns in an unexpected order or with wrong names.

## Artifact Errors

If execution succeeds but artifacts are missing, check that the generated code saved:

- model weights,
- prediction result CSV,
- training log through the execution wrapper,
- model parameter JSON.
