"""
RiskAlertMultiLabelModel — PRRS Abortion Abnormal Risk Alert Model

Architecture:
  - Embedding layers for 5 discrete features (mask-aware)
  - Linear projection for 3 non-temporal continuous features (mask-aware)
  - 1D Conv temporal encoder for 7 abortion_rate_past_* features
  - Shared fusion MLP + 3 horizon-specific output heads

Input:  (batch_size, 30) — 15 features × 2 (feature + mask), interleaved
Output: (batch_size, 3)  — logits for BCEWithLogitsLoss
         indices: 0→abort_1_7, 1→abort_8_14, 2→abort_15_21
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
)

# ── Device ───────────────────────────────────────────────────────────────────
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ── Standard data filenames (resolved under --data-dir) ──────────────────────
DATA_FILES = {
    "train_x": "risk_alert_transformed_train_X_mask_null.csv",
    "train_y": "risk_alert_train_y.csv",
    "test_x": "risk_alert_transformed_test_X_mask_null.csv",
    "test_y": "risk_alert_test_y.csv",
    "predict_x": "risk_alert_predict_feature_transformed_masknull.csv",
    "predict_index": "risk_alert_predict_index_sample_data.csv",
}

LABEL_COLUMNS = ["abort_1_7", "abort_8_14", "abort_15_21"]
PRED_PROB_COLUMNS = [
    "abort_1_7_probability", "abort_8_14_probability", "abort_15_21_probability",
]
PRED_DEC_COLUMNS = [
    "abort_1_7_decision", "abort_8_14_decision", "abort_15_21_decision",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════════

class RiskAlertMultiLabelModel(nn.Module):
    """Multi-label risk prediction model with discrete embeddings,
    continuous mask-aware projections, and temporal encoding."""

    def __init__(self, params: dict):
        super().__init__()
        self.output_size = params.get("output_size", 3)
        dropout = params.get("dropout", 0.2)
        emb_dim = params.get("embedding_size", 128)

        # Discrete feature cardinalities (from transform metadata)
        self.discrete_cardinalities = [
            params.get("org_inv_dk", 2000),
            params.get("city", 200),
            params.get("season", 5),
            params.get("l3_org_inv_dk", 200),
            params.get("month", 13),
        ]
        self.discrete_embeddings = nn.ModuleList([
            nn.Embedding(card, emb_dim, padding_idx=0)
            for card in self.discrete_cardinalities
        ])

        # Continuous non-temporal: check_out_ratio_7d, reserve_sow_sqty, abortion_rate_ma_diff
        self.cont_norm = nn.BatchNorm1d(3)
        self.cont_proj = nn.Linear(3, emb_dim)

        # Temporal encoder: 7 abortion_rate_past_* → Conv1d + pooling
        self.temp_conv = nn.Conv1d(1, emb_dim, kernel_size=3, padding=1, bias=False)
        self.temp_bn = nn.BatchNorm1d(emb_dim)

        # Learnable missing embedding for temporal features
        self.temp_missing_emb = nn.Parameter(torch.zeros(1, emb_dim))
        nn.init.normal_(self.temp_missing_emb, std=0.01)

        # Fusion MLP
        fusion_dim = 5 * emb_dim + emb_dim + emb_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, emb_dim * 2),
            nn.BatchNorm1d(emb_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Shared head representation + 3 task-specific heads
        self.shared_head = nn.Linear(emb_dim, emb_dim)
        self.task_heads = nn.ModuleList([
            nn.Linear(emb_dim, 1) for _ in range(self.output_size)
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.padding_idx is not None:
                    with torch.no_grad():
                        m.weight[m.padding_idx] = 0.0
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch_size, 30) tensor — features at even indices, masks at odd indices
        Returns:
            logits: (batch_size, 3)
        """
        features = x[:, 0::2]   # (B, 15)
        masks = x[:, 1::2]      # (B, 15)

        # ── Discrete features (cols 0-4) ────────────────────────────
        discrete_ids = features[:, :5].long()
        discrete_msk = masks[:, :5]  # (B, 5)

        discrete_embs = []
        for i, emb in enumerate(self.discrete_embeddings):
            idx = discrete_ids[:, i].clamp(0, self.discrete_cardinalities[i] - 1)
            e = emb(idx)                           # (B, emb_dim)
            e = e * discrete_msk[:, i:i+1]         # mask-aware gating
            discrete_embs.append(e)
        discrete_out = torch.cat(discrete_embs, dim=1)  # (B, 5*emb_dim)

        # ── Continuous non-temporal (cols 5-7) ──────────────────────
        cont_feat = features[:, 5:8]   # (B, 3)
        cont_msk = masks[:, 5:8]       # (B, 3)
        cont_feat = cont_feat * cont_msk
        cont_feat = self.cont_norm(cont_feat)
        cont_out = self.cont_proj(cont_feat)                     # (B, emb_dim)
        cont_out = cont_out * cont_msk.mean(dim=1, keepdim=True)

        # ── Temporal features (cols 8-14) ──────────────────────────
        temp_feat = features[:, 8:15]  # (B, 7)
        temp_msk = masks[:, 8:15]      # (B, 7)
        temp_masked = temp_feat * temp_msk

        # Conv1d: (B, 1, 7) → (B, emb_dim, 7)
        temp_out = temp_masked.unsqueeze(1)
        temp_out = self.temp_conv(temp_out)
        temp_out = self.temp_bn(temp_out)
        temp_out = F.relu(temp_out)
        # Adaptive pooling over the 7 time steps
        temp_out = F.adaptive_avg_pool1d(temp_out, 1).squeeze(-1)  # (B, emb_dim)

        # Where no temporal features observed, use learned missing embedding
        temp_any_observed = temp_msk.sum(dim=1, keepdim=True) > 0  # (B, 1)
        temp_out = torch.where(temp_any_observed, temp_out, self.temp_missing_emb)

        # ── Fusion ──────────────────────────────────────────────────
        fusion_in = torch.cat([discrete_out, cont_out, temp_out], dim=1)
        shared = self.fusion(fusion_in)                            # (B, emb_dim)

        # ── Output heads ────────────────────────────────────────────
        shared_repr = F.relu(self.shared_head(shared))
        logits = [head(shared_repr) for head in self.task_heads]
        logits = torch.cat(logits, dim=1)                          # (B, 3)

        return logits


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset & Collate
# ═══════════════════════════════════════════════════════════════════════════════

def _collate_fn(batch):
    """Collate training/validation (features, labels) tuples."""
    xs, ys = zip(*batch)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


def _predict_collate_fn(batch):
    """Collate prediction features only."""
    return torch.stack(batch, dim=0)


class RiskAlertMultiLabelDataset(Dataset):
    """Dataset for training and validation."""

    def __init__(self, df: pd.DataFrame, label: pd.DataFrame):
        self.data = df.values.astype(np.float32)
        self.labels = label.values.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx])
        y = torch.from_numpy(self.labels[idx])
        return x, y


class RiskAlertMultiLabelPredictDataset(Dataset):
    """Dataset for prediction (features only)."""

    def __init__(self, df: pd.DataFrame):
        self.data = df.values.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx])


# ═══════════════════════════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════════════════════════

class RiskAlertMultiLabelTrainer:
    """Train, validate, and predict with RiskAlertMultiLabelModel."""

    def __init__(self, params: dict, transform_path: Path,
                 checkpoint_path: Path, output_dir: Path):
        self.params = params
        self.device = device
        self.batch_size = params.get("batch_size", 256)
        self.learning_rate = params.get("learning_rate", 0.001)
        self.num_workers = params.get("num_workers", 0)
        self.decision_threshold = params.get("decision_threshold", 0.5)
        self.output_size = params.get("output_size", 3)

        self.transform_path = transform_path
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir

        self.model = None
        self.model_params = None

    def _load_transform(self):
        """Load RiskAlertTransformPipeline from JSON and extract cardinalities."""
        sys.path.insert(0, str(Path.cwd()))
        try:
            from lorcy_code.dependencies.feature.risk_alert_transformer import (
                RiskAlertTransformPipeline,
            )
        except ImportError:
            raise ImportError(
                "Cannot import RiskAlertTransformPipeline. "
                "Make sure PYTHONPATH includes the project root."
            )

        with open(self.transform_path, "r", encoding="utf-8") as f:
            transform = RiskAlertTransformPipeline.from_json(f.read())

        fd = transform.trans.features.features
        self.model_params = {
            "dropout": self.params.get("dropout", 0.2),
            "embedding_size": self.params.get("embedding_size", 128),
            "output_size": self.output_size,
            "org_inv_dk": fd["org_inv_dk"].category_encode.size + 1,
            "city": fd["city"].category_encode.size + 1,
            "l3_org_inv_dk": fd["l3_org_inv_dk"].category_encode.size + 1,
            "season": 5,
            "month": 13,
        }
        return self.model_params

    def build_model(self):
        """Build the model from transform metadata."""
        if self.model_params is None:
            self._load_transform()
        self.model = RiskAlertMultiLabelModel(params=self.model_params).to(self.device)

    def train(self, num_epochs: int, train_X: pd.DataFrame, train_y: pd.DataFrame,
              test_X: pd.DataFrame, test_y: pd.DataFrame):
        """Full training loop with validation and history tracking."""

        # Datasets & DataLoaders
        train_dataset = RiskAlertMultiLabelDataset(train_X, train_y)
        test_dataset = RiskAlertMultiLabelDataset(test_X, test_y)

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, collate_fn=_collate_fn,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=_collate_fn,
        )

        # Optimizer & Loss
        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, verbose=False,
        )
        criterion = nn.BCEWithLogitsLoss()

        # Training history
        history = []
        best_val_loss = float("inf")

        for epoch in range(1, num_epochs + 1):
            # ── Training ────────────────────────────────────────────
            self.model.train()
            train_losses = []
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)

            # ── Validation ──────────────────────────────────────────
            val_loss, metrics = self.validate(test_loader, criterion)

            # LR scheduling
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.checkpoint_path)

            # Record history
            epoch_record = {
                "epoch": epoch,
                "train_loss": round(avg_train_loss, 6),
                "val_loss": round(val_loss, 6),
                "precision": round(metrics["precision"], 6),
                "recall": round(metrics["recall"], 6),
                "f1": round(metrics["f1"], 6),
                "auc": round(metrics["auc"], 6),
                "learning_rate": round(current_lr, 8),
            }
            # Add per-horizon metrics
            for h in range(self.output_size):
                epoch_record[f"precision_h{h}"] = round(metrics[f"precision_h{h}"], 6)
                epoch_record[f"recall_h{h}"] = round(metrics[f"recall_h{h}"], 6)
                epoch_record[f"f1_h{h}"] = round(metrics[f"f1_h{h}"], 6)
                epoch_record[f"auc_h{h}"] = round(metrics[f"auc_h{h}"], 6)

            history.append(epoch_record)

            print(
                f"Epoch {epoch:2d}/{num_epochs}  "
                f"Train Loss: {avg_train_loss:.4f}  "
                f"Val Loss: {val_loss:.4f}  "
                f"P: {metrics['precision']:.4f}  "
                f"R: {metrics['recall']:.4f}  "
                f"F1: {metrics['f1']:.4f}  "
                f"AUC: {metrics['auc']:.4f}"
            )

        # Save history
        history_df = pd.DataFrame(history)
        history_df.to_csv(self.output_dir / "train_history.csv", index=False)
        with open(self.output_dir / "train_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        print(f"Training complete. Best val loss: {best_val_loss:.4f}")
        print(f"Best model saved to: {self.checkpoint_path}")

    def validate(self, data_loader: DataLoader, criterion=None):
        """Validate and compute precision, recall, f1, auc for all labels."""
        self.model.eval()

        all_logits = []
        all_labels = []
        total_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                logits = self.model(batch_x)
                if criterion is not None:
                    total_loss += criterion(logits, batch_y).item() * batch_x.size(0)
                all_logits.append(logits.cpu())
                all_labels.append(batch_y.cpu())

        all_logits = torch.cat(all_logits, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()
        probabilities = 1.0 / (1.0 + np.exp(-all_logits))  # sigmoid
        decisions = (probabilities >= self.decision_threshold).astype(np.float32)

        avg_loss = total_loss / len(all_labels) if criterion is not None else 0.0

        # Aggregate metrics (micro)
        metrics = {
            "precision": precision_score(all_labels, decisions, average="micro", zero_division=0),
            "recall": recall_score(all_labels, decisions, average="micro", zero_division=0),
            "f1": f1_score(all_labels, decisions, average="micro", zero_division=0),
        }
        try:
            metrics["auc"] = roc_auc_score(all_labels, probabilities, average="micro")
        except ValueError:
            metrics["auc"] = 0.0

        # Per-horizon metrics
        for h in range(self.output_size):
            try:
                metrics[f"precision_h{h}"] = precision_score(
                    all_labels[:, h], decisions[:, h], zero_division=0,
                )
                metrics[f"recall_h{h}"] = recall_score(
                    all_labels[:, h], decisions[:, h], zero_division=0,
                )
                metrics[f"f1_h{h}"] = f1_score(
                    all_labels[:, h], decisions[:, h], zero_division=0,
                )
                metrics[f"auc_h{h}"] = roc_auc_score(
                    all_labels[:, h], probabilities[:, h],
                )
            except ValueError:
                metrics[f"precision_h{h}"] = 0.0
                metrics[f"recall_h{h}"] = 0.0
                metrics[f"f1_h{h}"] = 0.0
                metrics[f"auc_h{h}"] = 0.0

        return avg_loss, metrics

    def predict(self, X: pd.DataFrame, index_label_data: pd.DataFrame):
        """Run prediction and save results to predict_result.csv."""
        # Load best checkpoint
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        self.model.load_state_dict(
            torch.load(self.checkpoint_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

        # Dataset & DataLoader
        predict_dataset = RiskAlertMultiLabelPredictDataset(X)
        predict_loader = DataLoader(
            predict_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=_predict_collate_fn,
        )

        # Prediction loop
        all_probs = []
        with torch.no_grad():
            for batch_x in predict_loader:
                batch_x = batch_x.to(self.device)
                logits = self.model(batch_x)
                probs = torch.sigmoid(logits)  # (B, 3)
                all_probs.append(probs.cpu())

        all_probs = torch.cat(all_probs, dim=0).numpy()  # (N, 3)

        # Threshold → decisions
        all_decisions = (all_probs >= self.decision_threshold).astype(int)

        # Build result DataFrame (preserve row order)
        index_reset = index_label_data.reset_index(drop=True)

        # Add probability columns
        for h, col in enumerate(PRED_PROB_COLUMNS):
            index_reset[col] = all_probs[:, h]

        # Add decision columns
        for h, col in enumerate(PRED_DEC_COLUMNS):
            index_reset[col] = all_decisions[:, h]

        # Save
        output_path = self.output_dir / "predict_result.csv"
        index_reset.to_csv(output_path, index=False)
        print(f"Predictions saved to: {output_path}")
        print(f"Prediction shape: {index_reset.shape}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="PRRS Abortion Abnormal Risk Alert Model")
    parser.add_argument("--mode", choices=("train", "predict", "train-predict"),
                        default="train-predict")
    parser.add_argument("--data-dir",
                        default="lorcy_code/data/interim/PRRS_Abortion_Abnormal_Alert/risk_alert")
    parser.add_argument("--transform-path",
                        default="lorcy_code/data/model/PRRS_Abortion_Abnormal_Alert/risk_alert/risk_alert_nfm_transform.json")
    parser.add_argument("--output-dir", default="lorcy_code/data/result")
    parser.add_argument("--checkpoint", default=None)
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
    transform_path = Path(args.transform_path).resolve()
    checkpoint_path = (
        Path(args.checkpoint).resolve()
        if args.checkpoint
        else output_dir / "model_best.pth"
    )

    if args.mode == "predict" and not args.checkpoint:
        raise ValueError("--checkpoint is required in predict mode")

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Build trainer
    params = vars(args)
    params["output_size"] = 3
    trainer = RiskAlertMultiLabelTrainer(
        params=params,
        transform_path=transform_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
    )
    trainer.build_model()

    # Train
    if args.mode in ("train", "train-predict"):
        print("Loading training data...")
        train_X = pd.read_csv(data_dir / DATA_FILES["train_x"]).fillna(0)
        train_y = pd.read_csv(data_dir / DATA_FILES["train_y"])
        test_X = pd.read_csv(data_dir / DATA_FILES["test_x"]).fillna(0)
        test_y = pd.read_csv(data_dir / DATA_FILES["test_y"])
        print(f"Train: {train_X.shape}, Test: {test_X.shape}")
        trainer.train(args.epochs, train_X, train_y, test_X, test_y)

    # Predict
    if args.mode in ("predict", "train-predict"):
        print("Loading prediction data...")
        predict_X = pd.read_csv(data_dir / DATA_FILES["predict_x"]).fillna(0)
        predict_index = pd.read_csv(data_dir / DATA_FILES["predict_index"])
        print(f"Predict features: {predict_X.shape}, Index: {predict_index.shape}")
        trainer.predict(predict_X, predict_index)

    print("Done.")


if __name__ == "__main__":
    main()
