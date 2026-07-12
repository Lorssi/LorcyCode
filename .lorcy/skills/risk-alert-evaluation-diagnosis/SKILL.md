---
name: risk-alert-evaluation-diagnosis
description: Evaluate and diagnose archived PRRS abortion abnormal risk-alert model runs. Use when reading a model_v_* run directory, computing per-horizon Precision, Recall, F1 and AUC, interpreting training history/config/design, and writing actionable next-iteration advice into the same run.
---

# Risk Alert Evaluation Diagnosis

## Procedure

1. Run `scripts/evaluate_model.py --run-dir <model_v_*>`.
2. Read `evaluate_result.csv`, `train_history.csv`, `resolved_config.json`, and `design.md` when present.
3. Read `references/diagnosis-rules.md` and compare all three horizons.
4. Identify threshold, imbalance, overfitting, underfitting, missing-value, or temporal-modeling issues.
5. Return a compact metric summary, diagnosis, 2-4 implementation-ready actions, and one risk note.

The preferred interface reads `<run-dir>/predict_result.csv` and writes `evaluate_result.txt` and `evaluate_result.csv` back to that run. `--prediction-csv` and `--output-dir` remain available for compatibility.

Required prediction columns are the three `abort_*` labels and their matching `_decision` and `_probability` columns.
