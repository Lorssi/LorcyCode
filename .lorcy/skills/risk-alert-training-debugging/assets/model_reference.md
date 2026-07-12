# Runtime Transform Model Reference

This is the required reference for generated risk-alert model code. Adapt the architecture, but preserve the runtime transform and column-layout logic. Do not paste values from a specific transform JSON into the generated source.

## Current imports and paths

```python
from pathlib import Path
from collections.abc import Sequence

import pandas as pd
import torch

from lorcy_code.dependencies.feature.risk_alert_transformer import (
    RiskAlertTransformPipeline,
)
```

The transform path must come from CLI `--transform-path`. Data comes from CLI `--data-dir`; outputs go to CLI `--output-dir`. Do not import old `PRRS_Abortion_Abnormal_Alert.config`, `agent_paht_config`, or `util` modules, and do not save through global config paths.

## Required runtime metadata loading

```python
DISCRETE_FEATURES = ("org_inv_dk", "city", "season", "l3_org_inv_dk", "month")
CONTINUOUS_FEATURES = (
    "check_out_ratio_7d",
    "reserve_sow_sqty",
    "abortion_rate_ma_diff",
)
TEMPORAL_FEATURES = tuple(f"abortion_rate_past_{day}d" for day in range(1, 8))
MODEL_FEATURES = DISCRETE_FEATURES + CONTINUOUS_FEATURES + TEMPORAL_FEATURES


def load_transform(transform_path: Path) -> RiskAlertTransformPipeline:
    if not transform_path.is_file():
        raise FileNotFoundError(f"transform file not found: {transform_path}")
    try:
        return RiskAlertTransformPipeline.from_json(
            transform_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise ValueError(f"invalid transform file {transform_path}: {exc}") from exc


def category_cardinality(transform: RiskAlertTransformPipeline, name: str) -> int:
    feature_dict = transform.trans.features.features
    if name not in feature_dict:
        raise KeyError(f"transform is missing categorical feature: {name}")
    encoder = feature_dict[name].category_encode
    if encoder is None:
        raise ValueError(f"transform feature has no category encoder: {name}")
    return encoder.size + 1  # id 0 is reserved for unknown/missing values


def build_feature_layout(columns: Sequence[str]) -> dict[str, dict[str, int]]:
    columns = pd.Index(columns)
    layout = {}
    for name in MODEL_FEATURES:
        mask_name = f"{name}_mask"
        if name not in columns:
            raise ValueError(f"feature CSV is missing column: {name}")
        if mask_name not in columns:
            raise ValueError(f"feature CSV is missing mask column: {mask_name}")
        layout[name] = {
            "value_idx": int(columns.get_loc(name)),
            "mask_idx": int(columns.get_loc(mask_name)),
        }
    return layout
```

`season` and `month` are unchanged domain features and therefore do not have category encoders in the current transform. Their stable domain cardinalities are 5 and 13. All encoded categorical feature sizes must be read from the runtime transform.

## Required trainer construction

```python
class RiskAlertMultiLabelTrainer:
    def __init__(
        self,
        params: dict,
        transform_path: Path,
        checkpoint_path: Path,
        output_dir: Path,
        feature_columns: Sequence[str],
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.params = params
        self.transform_path = Path(transform_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir)
        self.feature_columns = tuple(feature_columns)
        self.transform = None
        self.feature_layout = None
        self.model_params = None
        self.model = None

    def build_model(self):
        self.transform = load_transform(self.transform_path)
        self.feature_layout = build_feature_layout(self.feature_columns)
        self.model_params = {
            "dropout": self.params.get("dropout", 0.2),
            "embedding_size": self.params.get("embedding_size", 128),
            "output_size": 3,
            "org_inv_dk": category_cardinality(self.transform, "org_inv_dk"),
            "city": category_cardinality(self.transform, "city"),
            "season": 5,
            "l3_org_inv_dk": category_cardinality(self.transform, "l3_org_inv_dk"),
            "month": 13,
            "feature_layout": self.feature_layout,
        }
        self.model = RiskAlertMultiLabelModel(self.model_params).to(self.device)
```

The model reads indices from `params["feature_layout"]` instead of a global numeric map:

```python
class RiskAlertMultiLabelModel(torch.nn.Module):
    def __init__(self, params: dict):
        super().__init__()
        self.feature_layout = params["feature_layout"]
        self.embeddings = torch.nn.ModuleDict({
            name: torch.nn.Embedding(params[name], params["embedding_size"], padding_idx=0)
            for name in DISCRETE_FEATURES
        })
        # Define the remaining architecture here.

    def _value(self, x: torch.Tensor, name: str) -> torch.Tensor:
        return x[:, self.feature_layout[name]["value_idx"]]

    def _mask(self, x: torch.Tensor, name: str) -> torch.Tensor:
        return x[:, self.feature_layout[name]["mask_idx"]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = []
        for name in DISCRETE_FEATURES:
            ids = self._value(x, name).long()
            mask = self._mask(x, name).unsqueeze(1)
            embedded.append(self.embeddings[name](ids) * mask)
        # Process continuous and temporal features by name through _value/_mask.
        # Return three logits; do not apply sigmoid here.
        ...
```

## Load order in `main()`

Load the relevant feature CSV before calling `build_model()` so its columns can define the runtime layout:

```python
if args.mode in ("train", "train-predict"):
    train_x = pd.read_csv(data_dir / DATA_FILES["train_x"]).fillna(0)
    feature_columns = train_x.columns
else:
    predict_x = pd.read_csv(data_dir / DATA_FILES["predict_x"]).fillna(0)
    feature_columns = predict_x.columns

trainer = RiskAlertMultiLabelTrainer(
    params=params,
    transform_path=Path(args.transform_path),
    checkpoint_path=checkpoint,
    output_dir=output_dir,
    feature_columns=feature_columns,
)
trainer.build_model()
```

## Prohibited example

Never generate this pattern, even if the current JSON happens to contain these values:

```python
# WRONG: copied metadata and positions become stale when transform/data changes.
FEATURES = {
    "org_inv_dk": (0, 1, "discrete", 304),
    "city": (2, 3, "discrete", 75),
    "l3_org_inv_dk": (6, 7, "discrete", 24),
}
```

Fail fast on incompatible transform/data rather than silently using guessed or copied constants.
