# Coding Contract

## CLI and configuration

Generated model code must parse `--mode`, `--data-dir`, `--transform-path`, `--output-dir`, `--checkpoint`, `--epochs`, `--batch-size`, `--learning-rate`, `--num-workers`, `--dropout`, `--embedding-size`, `--decision-threshold`, and `--seed`.

Use executor-provided values directly. Create no `model_v_*` directory inside model code. Write only inside `--output-dir`.

## Model and training

- Define `RiskAlertMultiLabelModel(nn.Module)`, call `super().__init__()`, accept `params`, and return three logits.
- Load `RiskAlertTransformPipeline.from_json(Path(transform_path).read_text(...))` at runtime.
- Derive categorical embedding cardinalities from `transform.trans.features.features[name].category_encode.size + 1` at runtime.
- Derive value/mask column indices from the loaded DataFrame column names. Do not encode numeric positions such as `(0, 1)` in a global `FEATURES` dictionary.
- Use `BCEWithLogitsLoss`; apply sigmoid only for metrics and prediction.
- Save the best validation-loss state dict as `model_best.pth`.
- Write `train_history.csv` and `train_history.json` with epoch, train loss, validation loss, Precision, Recall, F1, AUC, and learning rate.
- Keep prediction row order; never merge on transformed `org_inv_dk`.

## Inputs and outputs

Read standard filenames below `--data-dir`; path values are parameterized, but filenames and schemas are protected. Load transform metadata from `--transform-path`. In predict mode load `--checkpoint`; after training predict from `model_best.pth`.

Forbidden generation patterns:

- embedding sizes copied from one transform file, such as `org_inv_dk: 304`, `city: 75`, or `l3_org_inv_dk: 24`;
- a global `FEATURES = {...}` containing fixed numeric value/mask indices;
- parsing transform JSON as a plain dictionary and manually copying its values into source code;
- falling back to guessed category counts when transform loading fails.

Fail clearly when the transform file, required feature metadata, category encoder, feature column, or mask column is missing. Do not silently substitute constants.

Write `predict_result.csv` with original index/label columns plus all three `_decision` and `_probability` pairs. Do not write weights or predictions to global `path_config` destinations.
