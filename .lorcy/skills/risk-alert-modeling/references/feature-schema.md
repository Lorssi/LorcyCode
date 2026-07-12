# Feature Schema

Base feature order before masks:

1. `org_inv_dk`
2. `city`
3. `season`
4. `l3_org_inv_dk`
5. `month`
6. `check_out_ratio_7d`
7. `reserve_sow_sqty`
8. `abortion_rate_ma_diff`
9. `abortion_rate_past_1d`
10. `abortion_rate_past_2d`
11. `abortion_rate_past_3d`
12. `abortion_rate_past_4d`
13. `abortion_rate_past_5d`
14. `abortion_rate_past_6d`
15. `abortion_rate_past_7d`

Actual tensor layout interleaves each feature with a mask:

```text
feature_1, feature_1_mask, feature_2, feature_2_mask, ...
```

Mask convention:

- `1`: valid observed value.
- `0`: missing value.

Discrete features:

- `org_inv_dk`
- `city`
- `season`
- `l3_org_inv_dk`
- `month`

Continuous features:

- `check_out_ratio_7d`
- `reserve_sow_sqty`
- `abortion_rate_ma_diff`
- `abortion_rate_past_1d` through `abortion_rate_past_7d`

Use the seven `abortion_rate_past_*` features as a short temporal signal when helpful. If vectorizing them, prefer a shared projection or a small sequence module rather than separate unrelated projections for each day.

Prediction data:

- `predict_transformed_Mask_Null` has the same transformed feature order as training data.
- `predict_index_label_data` contains index columns and labels.
- Do not merge by transformed `org_inv_dk`; align prediction features and index labels by row order.
