# Project Template Contract

## Standard input files

Resolve these files below `--data-dir`:

- `risk_alert_transformed_train_X_mask_null.csv`
- `risk_alert_train_y.csv`
- `risk_alert_transformed_test_X_mask_null.csv`
- `risk_alert_test_y.csv`
- `risk_alert_predict_feature_transformed_masknull.csv`
- `risk_alert_predict_index_sample_data.csv`

Fill missing feature values with zero. Load `RiskAlertTransformPipeline` from `--transform-path` and use its category cardinalities for `org_inv_dk`, `city`, and `l3_org_inv_dk`. `season` and `month` are unchanged domain features, so use their stable domain cardinalities 5 and 13. This is the only allowed fixed cardinality exception.

Build input layout from the actual CSV headers. For every feature name, resolve `columns.get_loc(name)` and `columns.get_loc(f"{name}_mask")`; validate that every pair exists. The architecture may group feature names, but must not hardcode their numeric positions.

## Modes

- `train`: load train/test data, train, and write checkpoint/history.
- `predict`: load prediction data and the explicit checkpoint, then write predictions.
- `train-predict`: train first, then predict with the best checkpoint.

Always select `cuda:0` when CUDA is available, otherwise CPU. Set Python, NumPy, and PyTorch seeds; seed all CUDA devices when available.

## Stable model behavior

Use shuffled training batches and non-shuffled validation/prediction batches. Concatenate prediction outputs to `predict_index_label_data.reset_index(drop=True)` by row order. Output labels map in order to 1-7, 8-14, and 15-21 days.

The executor, not model code, owns run creation, provenance, transform snapshot, stdout, and stderr. Model code owns `model_best.pth`, `train_history.csv/json`, and `predict_result.csv`.
