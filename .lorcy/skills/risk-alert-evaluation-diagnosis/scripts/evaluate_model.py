#!/usr/bin/env python
"""Evaluate PRRS abortion abnormal alert prediction results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

TASKS = [
    ("abort_1_7", "abort_1_7_decision", "abort_1_7_probability", "1-7"),
    ("abort_8_14", "abort_8_14_decision", "abort_8_14_probability", "8-14"),
    ("abort_15_21", "abort_15_21_decision", "abort_15_21_probability", "15-21"),
]


def evaluate(df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    lines: list[str] = []
    row: dict[str, float] = {}

    for label_col, decision_col, prob_col, period in TASKS:
        if not all(col in df.columns for col in [label_col, decision_col, prob_col]):
            lines.append(f"warning: missing columns for {label_col}; skipped")
            continue

        y_true = df[label_col]
        y_pred = df[decision_col]
        y_prob = df[prob_col]

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = -1.0

        row[f"{period}_Precision"] = precision
        row[f"{period}_Recall"] = recall
        row[f"{period}_F1"] = f1
        row[f"{period}_AUC"] = auc

        auc_text = "Undefined" if auc < 0 else f"{auc:.4f}"
        lines.extend([
            f"=== {label_col} ===",
            f"Precision: {precision:.4f}",
            f"Recall: {recall:.4f}",
            f"F1: {f1:.4f}",
            f"AUC: {auc_text}",
            "",
        ])

    return "\n".join(lines), pd.DataFrame([row]).round(4)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", help="Preferred: model_v_* directory containing predict_result.csv.")
    parser.add_argument("--prediction-csv", help="Compatibility input; defaults to <run-dir>/predict_result.csv.")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if not args.run_dir and not args.prediction_csv:
        parser.error("one of --run-dir or --prediction-csv is required")
    run_dir = Path(args.run_dir).resolve() if args.run_dir else None
    prediction_csv = Path(args.prediction_csv).resolve() if args.prediction_csv else run_dir / "predict_result.csv"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (run_dir or prediction_csv.parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(prediction_csv)
    text, metrics = evaluate(df)
    (output_dir / "evaluate_result.txt").write_text(text, encoding="utf-8")
    metrics.to_csv(output_dir / "evaluate_result.csv", index=False)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
