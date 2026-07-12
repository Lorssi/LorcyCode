# Model Constraints

These constraints are mandatory for generated model code.

- Keep the model class name exactly `RiskAlertMultiLabelModel`.
- The forward method accepts a flat 2D tensor `x` with shape `(batch_size, features_with_masks)`.
- The forward method returns only one tensor with shape `(batch_size, 3)`.
- The output is logits. Do not apply sigmoid inside `forward` when training uses `BCEWithLogitsLoss`.
- During prediction, apply sigmoid outside the model to obtain probabilities.
- Preserve label order: `abort_1_7`, `abort_8_14`, `abort_15_21`.
- Preserve prediction result column names expected by evaluation:
  - `abort_1_7_decision`
  - `abort_1_7_probability`
  - `abort_8_14_decision`
  - `abort_8_14_probability`
  - `abort_15_21_decision`
  - `abort_15_21_probability`
- Do not alter protected data loading, transform loading, model parameter saving, or result saving blocks supplied by the project template.
- Save model parameters with `torch.save(self.model.state_dict(), path)`, not a whole model object.
