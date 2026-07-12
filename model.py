"""
RiskAlertMultiLabelModel — PRRS Abortion Abnormal Risk Alert Model (Enhanced)

Architecture:
  - Embedding layers for 5 discrete features (mask-aware, dynamically loaded cardinalities)
  - BatchNorm + Linear projection for 3 continuous features (mask-aware)
  - Bidirectional GRU temporal encoder for 7 abortion_rate_past_* features
  - Cross-attention: discrete features attend to continuous+temporal context
  - Shared fusion MLP (LayerNorm, Dropout)
  - 3 horizon-specific output heads with per-head temporal attention pooling

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
from collections.abc import Sequence
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

from lorcy_code.dependencies.feature.risk_alert_transformer import (
    RiskAlertTransformPipeline,
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

# ── Feature name groups (must match model_reference.md exactly) ──────────────
DISCRETE_FEATURES = ("org_inv_dk", "city", "season", "l3_org_inv_dk", "month")
CONTINUOUS_FEATURES = (
    "check_out_ratio_7d",
    "reserve_sow_sqty",
    "abortion_rate_ma_diff",
)
TEMPORAL_FEATURES = tuple(f"abortion_rate_past_{day}d" for day in range(1, 8))
MODEL_FEATURES = DISCRETE_FEATURES + CONTINUOUS_FEATURES + TEMPORAL_FEATURES


# ═══════════════════════════════════════════════════════════════════════════════
# Runtime transform / layout loading (per coding-contract.md)
# ═══════════════════════════════════════════════════════════════════════════════

def load_transform(transform_path: Path) -> RiskAlertTransformPipeline:
    """Load transform pipeline from JSON file."""
    if not transform_path.is_file():
        raise FileNotFoundError(f"transform file not found: {transform_path}")
    try:
        return RiskAlertTransformPipeline.from_json(
            transform_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise ValueError(f"invalid transform file {transform_path}: {exc}") from exc


def category_cardinality(transform: RiskAlertTransformPipeline, name: str) -> int:
    """Read categorical embedding cardinality from transform at runtime."""
    feature_dict = transform.trans.features.features
    if name not in feature_dict:
        raise KeyError(f"transform is missing categorical feature: {name}")
    if feature_dict[name].category_encode is None:
        raise ValueError(f"transform feature has no category encoder: {name}")
    return feature_dict[name].category_encode.size + 1  # id 0 reserved for unknown/missing


def build_feature_layout(columns: Sequence[str]) -> dict[str, dict[str, int]]:
    """Derive value/mask column indices from the DataFrame columns at runtime."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════════

class RiskAlertMultiLabelModel(nn.Module):
    """Multi-label risk prediction model with enhanced temporal and cross-attention."""

    def __init__(self, params: dict):
        super().__init__()
        self.feature_layout = params["feature_layout"]
        self.output_size = params.get("output_size", 3)
        emb_dim = params.get("embedding_size", 128)
        dropout = params.get("dropout", 0.3)
        num_heads = 4

        # ── Discrete feature embeddings ──────────────────────────────
        # Dynamically loaded cardinalities: org_inv_dk, city, season, l3_org_inv_dk, month
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(params[name], emb_dim, padding_idx=0)
            for name in DISCRETE_FEATURES
        })

        # ── Continuous non-temporal feature encoder ──────────────────
        # 3 features: check_out_ratio_7d, reserve_sow_sqty, abortion_rate_ma_diff
        self.cont_norm = nn.BatchNorm1d(3)
        self.cont_proj = nn.Sequential(
            nn.Linear(3, emb_dim),
            nn.ReLU(inplace=True),
        )

        # ── Temporal encoder (abortion_rate_past_1d ~ past_7d) ──────
        # Bidirectional GRU over 7 steps, hidden = emb_dim//2 per direction
        self.temp_gru = nn.GRU(
            input_size=1, hidden_size=emb_dim // 2,
            num_layers=1, batch_first=True, bidirectional=True,
        )
        # After GRU: (B, 7, emb_dim) — concatenated forward+backward

        # Learned missing embedding for when no temporal features observed
        self.temp_missing_emb = nn.Parameter(torch.zeros(1, emb_dim))
        nn.init.normal_(self.temp_missing_emb, std=0.01)

        # ── Cross-attention: discrete features attend to context ─────
        # Discrete embeddings (5, B, emb_dim) → Q
        # Continuous + temporal pooled (2, B, emb_dim) → K, V
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=emb_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )

        # ── Fusion MLP ────────────────────────────────────────────────
        # Input: 5*emb_dim (discrete after attention) + emb_dim (cont) + emb_dim (temporal)
        fusion_input_dim = 5 * emb_dim + emb_dim + emb_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, emb_dim * 2),
            nn.LayerNorm(emb_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ── Shared head representation ──────────────────────────────
        self.shared_head = nn.Linear(emb_dim, emb_dim)

        # ── Horizon-specific temporal attention weights ─────────────
        # Each horizon head learns a weighted combination of GRU time steps
        # Shape: (3, 7) — 3 horizons × 7 time steps
        self.horizon_temp_attn = nn.Parameter(torch.ones(self.output_size, 7) / 7.0)

        # ── Task-specific heads ──────────────────────────────────────
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
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _value(self, x: torch.Tensor, name: str) -> torch.Tensor:
        """Extract feature values by name from the interleaved tensor."""
        return x[:, self.feature_layout[name]["value_idx"]]

    def _mask(self, x: torch.Tensor, name: str) -> torch.Tensor:
        """Extract feature mask by name from the interleaved tensor."""
        return x[:, self.feature_layout[name]["mask_idx"]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, 30) — features at even indices, masks at odd indices
        Returns:
            logits: (batch_size, 3) — no sigmoid applied
        """
        batch_size = x.size(0)

        # ════════════════════════════════════════════════════════════════
        # Discrete features — embed and mask
        # ════════════════════════════════════════════════════════════════
        discrete_embs = []  # list of (B, emb_dim)
        for name in DISCRETE_FEATURES:
            ids = self._value(x, name).long()
            msk = self._mask(x, name).unsqueeze(1)  # (B, 1)
            emb = self.embeddings[name](ids)         # (B, emb_dim)
            emb = emb * msk
            discrete_embs.append(emb)

        discrete_out = torch.cat(discrete_embs, dim=1)  # (B, 5*emb_dim)

        # ════════════════════════════════════════════════════════════════
        # Continuous non-temporal features (3 features)
        # ════════════════════════════════════════════════════════════════
        cont_feats_list = []
        cont_masks_list = []
        for name in CONTINUOUS_FEATURES:
            cont_feats_list.append(self._value(x, name).unsqueeze(1))
            cont_masks_list.append(self._mask(x, name).unsqueeze(1))

        cont_feat = torch.cat(cont_feats_list, dim=1)  # (B, 3)
        cont_mask = torch.cat(cont_masks_list, dim=1)  # (B, 3)
        cont_feat = cont_feat * cont_mask
        cont_feat = self.cont_norm(cont_feat)
        cont_out = self.cont_proj(cont_feat)                       # (B, emb_dim)
        cont_out = cont_out * cont_mask.mean(dim=1, keepdim=True)   # mask gating

        # ════════════════════════════════════════════════════════════════
        # Temporal features (7 abortion_rate_past_*d)
        # ════════════════════════════════════════════════════════════════
        temp_feats_list = []
        temp_masks_list = []
        for name in TEMPORAL_FEATURES:
            temp_feats_list.append(self._value(x, name).unsqueeze(1))
            temp_masks_list.append(self._mask(x, name).unsqueeze(1))

        temp_feat = torch.cat(temp_feats_list, dim=1)   # (B, 7)
        temp_mask = torch.cat(temp_masks_list, dim=1)   # (B, 7)

        # Zero out missing values
        temp_masked = temp_feat * temp_mask              # (B, 7)
        temp_input = temp_masked.unsqueeze(-1)           # (B, 7, 1)

        # Bidirectional GRU
        gru_out, _ = self.temp_gru(temp_input)           # (B, 7, emb_dim)

        # Where no temporal features are observed, replace with learned embedding
        temp_any_observed = temp_mask.sum(dim=1, keepdim=True) > 0  # (B, 1)
        # Global average pooling over time (only over observed steps)
        temp_pooled = gru_out.mean(dim=1)                # (B, emb_dim)
        temp_out = torch.where(
            temp_any_observed.expand(-1, gru_out.size(-1)),
            temp_pooled,
            self.temp_missing_emb.expand(batch_size, -1),
        )                                                # (B, emb_dim)

        # ════════════════════════════════════════════════════════════════
        # Cross-attention: discrete features attend to continuous+temporal
        # ════════════════════════════════════════════════════════════════
        # Reshape discrete embeddings to (B, 5, emb_dim) for multi-head attention
        discrete_seq = torch.stack(discrete_embs, dim=1)  # (B, 5, emb_dim)

        # Context: continuous + temporal pooled → (B, 2, emb_dim)
        context_seq = torch.stack([cont_out, temp_out], dim=1)  # (B, 2, emb_dim)

        # Cross-attention: discrete → Q, context → K, V
        attn_out, _ = self.cross_attn(
            query=discrete_seq,
            key=context_seq,
            value=context_seq,
            key_padding_mask=None,
        )  # (B, 5, emb_dim)

        # Flatten attended discrete features back
        discrete_attended = attn_out.reshape(batch_size, -1)  # (B, 5*emb_dim)

        # ════════════════════════════════════════════════════════════════
        # Fusion
        # ════════════════════════════════════════════════════════════════
        fusion_in = torch.cat([discrete_attended, cont_out, temp_out], dim=1)
        shared = self.fusion(fusion_in)                    # (B, emb_dim)

        # ════════════════════════════════════════════════════════════════
        # Task-specific heads with per-horizon temporal attention
        # ════════════════════════════════════════════════════════════════
        shared_repr = F.relu(self.shared_head(shared))     # (B, emb_dim)

        # Per-horizon temporal attention: weighted sum of GRU outputs
        # temp_attn_weights: (3, 7) — softmax over 7 time steps per horizon
        temp_attn_weights = F.softmax(self.horizon_temp_attn, dim=1)  # (3, 7)

        logits_list = []
        for h in range(self.output_size):
            # Weighted temporal context for this horizon
            w = temp_attn_weights[h, :].view(1, 7, 1)      # (1, 7, 1)
            temp_context = (gru_out * w).sum(dim=1)         # (B, emb_dim)

            # Combine shared representation with horizon-specific temporal context
            horizon_feat = shared_repr + temp_context        # (B, emb_dim)
            logits_list.append(self.task_heads[h](horizon_feat))

        logits = torch.cat(logits_list, dim=1)              # (B, 3)
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
                 checkpoint_path: Path, output_dir: Path,
                 feature_columns: Sequence[str]):
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
        self.feature_columns = tuple(feature_columns)

        self.transform = None
        self.feature_layout = None
        self.model_params = None
        self.model = None

    def build_model(self):
        """Build the model from transform metadata and feature columns."""
        self.transform = load_transform(self.transform_path)
        self.feature_layout = build_feature_layout(self.feature_columns)

        # Read discrete cardinalities at runtime
        discrete_cards = {}
        for name in DISCRETE_FEATURES:
            if name in ("season", "month"):
                # season and month are unchanged; stable domain cardinalities
                discrete_cards[name] = 5 if name == "season" else 13
            else:
                discrete_cards[name] = category_cardinality(self.transform, name)

        self.model_params = {
            "dropout": self.params.get("dropout", 0.3),
            "embedding_size": self.params.get("embedding_size", 128),
            "output_size": self.output_size,
            "feature_layout": self.feature_layout,
            **discrete_cards,
        }

        self.model = RiskAlertMultiLabelModel(params=self.model_params).to(self.device)

        # Log model info
        print(f"Model built on {self.device}")
        card_str = ", ".join(f"{k}={v}" for k, v in discrete_cards.items())
        print(f"Discrete cardinalities: {card_str}")

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

        # ── Class imbalance: compute pos_weight per horizon ──────────
        train_labels = train_y.values  # (N, 3)
        pos_counts = train_labels.sum(axis=0)  # (3,)
        neg_counts = len(train_labels) - pos_counts
        pos_weight = torch.tensor(
            [neg_counts[h] / max(pos_counts[h], 1) for h in range(self.output_size)],
            dtype=torch.float32,
        ).to(self.device)
        print(f"pos_weight per horizon: {pos_weight.cpu().tolist()}")

        # Optimizer & Loss
        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3,
            min_lr=1e-6,
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # AMP scaler
        use_amp = self.device.type == "cuda"
        scaler = torch.amp.GradScaler(device=self.device.type) if use_amp else None

        # Training history
        history = []
        best_val_loss = float("inf")
        best_epoch = 0

        for epoch in range(1, num_epochs + 1):
            # ── Training ────────────────────────────────────────────
            self.model.train()
            train_losses = []

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()

                if use_amp:
                    with torch.amp.autocast(device_type=self.device.type):
                        logits = self.model(batch_x)
                        loss = criterion(logits, batch_y)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
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
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_epoch = epoch
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
                epoch_record[f"precision_h{h}"] = round(metrics.get(f"precision_h{h}", 0.0), 6)
                epoch_record[f"recall_h{h}"] = round(metrics.get(f"recall_h{h}", 0.0), 6)
                epoch_record[f"f1_h{h}"] = round(metrics.get(f"f1_h{h}", 0.0), 6)
                epoch_record[f"auc_h{h}"] = round(metrics.get(f"auc_h{h}", 0.0), 6)

            history.append(epoch_record)

            # Print progress
            best_mark = "  ← Best (saved)" if is_best else ""
            print(
                f"Epoch {epoch:2d}/{num_epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"P: {metrics['precision']:.4f} | "
                f"R: {metrics['recall']:.4f} | "
                f"F1: {metrics['f1']:.4f} | "
                f"AUC: {metrics['auc']:.4f} | "
                f"LR: {current_lr:.6f}{best_mark}"
            )

        # Save history
        history_df = pd.DataFrame(history)
        history_df.to_csv(self.output_dir / "train_history.csv", index=False)
        with open(self.output_dir / "train_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        print(f"\nTraining complete. Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
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
            torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"Loaded checkpoint from {self.checkpoint_path}")

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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.3)
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

    # Load data before building model (to get feature columns)
    if args.mode in ("train", "train-predict"):
        print("Loading training data...")
        train_X = pd.read_csv(data_dir / DATA_FILES["train_x"]).fillna(0)
        train_y = pd.read_csv(data_dir / DATA_FILES["train_y"])
        test_X = pd.read_csv(data_dir / DATA_FILES["test_x"]).fillna(0)
        test_y = pd.read_csv(data_dir / DATA_FILES["test_y"])
        print(f"Train: {train_X.shape}, Test: {test_X.shape}")
        feature_columns = train_X.columns
    else:
        print("Loading prediction data...")
        predict_X = pd.read_csv(data_dir / DATA_FILES["predict_x"]).fillna(0)
        predict_index = pd.read_csv(data_dir / DATA_FILES["predict_index"])
        print(f"Predict features: {predict_X.shape}, Index: {predict_index.shape}")
        feature_columns = predict_X.columns

    # Build trainer
    params = vars(args)
    params["output_size"] = 3
    trainer = RiskAlertMultiLabelTrainer(
        params=params,
        transform_path=transform_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        feature_columns=feature_columns,
    )
    trainer.build_model()

    # Train
    if args.mode in ("train", "train-predict"):
        print(f"\nStarting training for {args.epochs} epochs...")
        trainer.train(args.epochs, train_X, train_y, test_X, test_y)

    # Predict
    if args.mode in ("predict", "train-predict"):
        if args.mode == "predict":
            # Already loaded above
            pass
        else:
            print("\nLoading prediction data...")
            predict_X = pd.read_csv(data_dir / DATA_FILES["predict_x"]).fillna(0)
            predict_index = pd.read_csv(data_dir / DATA_FILES["predict_index"])
            print(f"Predict features: {predict_X.shape}, Index: {predict_index.shape}")
        trainer.predict(predict_X, predict_index)

    print("\nDone.")


if __name__ == "__main__":
    main()
