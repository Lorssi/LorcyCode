# Parameterized Model Template

Generate the model from this interface. Keep architecture-specific sections flexible, but keep the parser, paths, modes, and artifact names stable.

```python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lorcy_code.dependencies.feature.risk_alert_transformer import RiskAlertTransformPipeline

DATA_FILES = {
    "train_x": "risk_alert_transformed_train_X_mask_null.csv",
    "train_y": "risk_alert_train_y.csv",
    "test_x": "risk_alert_transformed_test_X_mask_null.csv",
    "test_y": "risk_alert_test_y.csv",
    "predict_x": "risk_alert_predict_feature_transformed_masknull.csv",
    "predict_index": "risk_alert_predict_index_sample_data.csv",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "predict", "train-predict"), default="train-predict")
    parser.add_argument("--data-dir", default="lorcy_code/data/interim/PRRS_Abortion_Abnormal_Alert/risk_alert")
    parser.add_argument("--transform-path", default="lorcy_code/data/model/PRRS_Abortion_Abnormal_Alert/risk_alert/risk_alert_nfm_transform.json")
    parser.add_argument("--output-dir", default="lorcy_code/data/result")
    parser.add_argument("--checkpoint")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--embedding-size", type=int, default=128)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else output_dir / "model_best.pth"
    if args.mode == "predict" and not args.checkpoint:
        raise ValueError("--checkpoint is required in predict mode")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.mode in ("train", "train-predict"):
        train_x = pd.read_csv(data_dir / DATA_FILES["train_x"]).fillna(0)
        train_y = pd.read_csv(data_dir / DATA_FILES["train_y"])
        test_x = pd.read_csv(data_dir / DATA_FILES["test_x"]).fillna(0)
        test_y = pd.read_csv(data_dir / DATA_FILES["test_y"])
    if args.mode in ("predict", "train-predict"):
        predict_x = pd.read_csv(data_dir / DATA_FILES["predict_x"]).fillna(0)
        predict_index = pd.read_csv(data_dir / DATA_FILES["predict_index"])

    feature_columns = train_x.columns if args.mode in ("train", "train-predict") else predict_x.columns
    params = vars(args).copy()
    params.update(output_size=3)
    trainer = RiskAlertMultiLabelTrainer(params=params, transform_path=Path(args.transform_path),
                                         checkpoint_path=checkpoint, output_dir=output_dir,
                                         feature_columns=feature_columns)
    trainer.build_model()
    if args.mode in ("train", "train-predict"):
        trainer.train(args.epochs, train_x, train_y, test_x, test_y)
    if args.mode in ("predict", "train-predict"):
        trainer.predict(predict_x, predict_index)


if __name__ == "__main__":
    main()
```

Implement `RiskAlertMultiLabelModel`, datasets, collate functions, and `RiskAlertMultiLabelTrainer` around this entry point. The trainer constructor must accept `transform_path`, `checkpoint_path`, `output_dir`, and `feature_columns`. Its `build_model()` must use the exact runtime-loading approach in `assets/model_reference.md`; never emit category counts or numeric feature positions obtained from the current JSON/CSV. Save only these stable artifacts:

- `output_dir/model_best.pth`
- `output_dir/train_history.csv`
- `output_dir/train_history.json`
- `output_dir/predict_result.csv`

Record history as one row per epoch. Include aggregate Precision, Recall, F1, and AUC; per-horizon metrics may be added. Preserve all column and row-order requirements from `coding-contract.md`.
