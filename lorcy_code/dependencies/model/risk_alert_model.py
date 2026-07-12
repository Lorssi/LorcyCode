"""
RiskAlertMultiLabelModel — PRRS Abortion Abnormal Risk Alert Model

Architecture:
  - Separate embedding layers for 5 discrete features (mask-aware)
  - Linear projections for 3 non-temporal continuous features (mask-aware)
  - Temporal encoder (1D conv + pooling) for 7 abortion_rate_past_* features
  - Shared fusion MLP with BatchNorm + Dropout
  - Three output heads (one per horizon: 1-7, 8-14, 15-21 days)

Input shape: (batch_size, 30) = 15 features + 15 masks interleaved
Output shape: (batch_size, 3) — logits for BCEWithLogitsLoss
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RiskAlertMultiLabelModel(nn.Module):
    """Multi-label risk prediction model with discrete embeddings,
    continuous mask-aware projections, and temporal encoding."""

    def __init__(self, params: dict):
        super().__init__()
        self.output_size = params.get("output_size", 3)
        dropout = params.get("dropout", 0.2)
        emb_dim = params.get("embedding_size", 128)

        # ── Discrete feature embeddings ──────────────────────────────
        # 5 discrete features: org_inv_dk, city, season, l3_org_inv_dk, month
        self.discrete_cardinalities = [
            params.get("org_inv_dk", 1000),
            params.get("city", 100),
            params.get("season", 5),
            params.get("l3_org_inv_dk", 100),
            params.get("month", 13),
        ]
        self.discrete_embeddings = nn.ModuleList([
            nn.Embedding(card, emb_dim, padding_idx=0)
            for card in self.discrete_cardinalities
        ])

        # ── Continuous non-temporal feature projections ──────────────
        # 3 features: check_out_ratio_7d, reserve_sow_sqty, abortion_rate_ma_diff
        self.cont_proj = nn.Linear(3, emb_dim)

        # ── Temporal encoder (abortion_rate_past_1d ~ past_7d) ───────
        # Treat as 7-step 1D signal → conv1d + adaptive avg pool
        self.temp_conv = nn.Conv1d(
            in_channels=1, out_channels=emb_dim, kernel_size=3,
            padding=1, bias=False
        )
        self.temp_bn = nn.BatchNorm1d(emb_dim)
        # After adaptive pooling: (emb_dim, 1) → squeeze to emb_dim

        # ── Shared fusion layers ─────────────────────────────────────
        fusion_input_dim = (
            5 * emb_dim      # discrete embeddings
            + emb_dim        # continuous projection
            + emb_dim        # temporal encoding
        )
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, emb_dim * 2),
            nn.BatchNorm1d(emb_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ── Task-specific heads ──────────────────────────────────────
        # One shared head + 3 lightweight horizon-specific adapters
        self.shared_head = nn.Linear(emb_dim, emb_dim)
        self.task_heads = nn.ModuleList([
            nn.Linear(emb_dim, 1) for _ in range(self.output_size)
        ])

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx] = 0.0
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, 30) — 15 features interleaved with 15 masks
        Returns:
            logits: (batch_size, 3)
        """
        # ── Split features and masks ────────────────────────────────
        # Features at even indices, masks at odd indices
        features = x[:, 0::2]   # (B, 15)
        masks = x[:, 1::2]      # (B, 15)

        # ── Discrete features (indices 0..4) ────────────────────────
        discrete_feats = features[:, :5].long()  # (B, 5)
        discrete_masks = masks[:, :5]             # (B, 5)

        discrete_embs = []
        for i, emb in enumerate(self.discrete_embeddings):
            # Clamp indices to valid range (handle OOV)
            idx = discrete_feats[:, i].clamp(0, self.discrete_cardinalities[i] - 1)
            emb_out = emb(idx)                       # (B, emb_dim)
            # Mask: if mask=0, zero out the embedding
            emb_out = emb_out * discrete_masks[:, i:i+1]
            discrete_embs.append(emb_out)
        discrete_out = torch.cat(discrete_embs, dim=1)  # (B, 5*emb_dim) — actually keep separately

        # ── Continuous non-temporal (indices 5, 6, 7) ───────────────
        cont_feats = features[:, 5:8]        # (B, 3)
        cont_masks = masks[:, 5:8]           # (B, 3)
        # Mask-aware: replace missing with 0, but also pass mask signal
        cont_feats = cont_feats * cont_masks
        cont_out = self.cont_proj(cont_feats)           # (B, emb_dim)
        cont_out = cont_out * cont_masks.mean(dim=1, keepdim=True)  # overall mask signal

        # ── Temporal features (indices 8..14) ───────────────────────
        temp_feats = features[:, 8:15]       # (B, 7)
        temp_masks = masks[:, 8:15]          # (B, 7)
        temp_feats = temp_feats * temp_masks

        # Conv1d expects (B, C, L) → (B, 1, 7)
        temp_out = temp_feats.unsqueeze(1)   # (B, 1, 7)
        temp_out = self.temp_conv(temp_out)  # (B, emb_dim, 7)
        temp_out = self.temp_bn(temp_out)
        temp_out = F.relu(temp_out)
        # Adaptive average pool over the 7 steps → (B, emb_dim, 1)
        temp_out = F.adaptive_avg_pool1d(temp_out, 1).squeeze(-1)  # (B, emb_dim)
        # Mask signal: fraction of temporal features observed
        temp_mask_frac = temp_masks.mean(dim=1, keepdim=True)       # (B, 1)
        temp_out = temp_out * temp_mask_frac

        # ── Fusion ──────────────────────────────────────────────────
        # Concatenate all encoded representations
        fusion_in = torch.cat([
            discrete_out,      # (B, 5*emb_dim)
            cont_out,          # (B, emb_dim)
            temp_out,          # (B, emb_dim)
        ], dim=1)

        shared = self.fusion(fusion_in)          # (B, emb_dim)

        # ── Heads ───────────────────────────────────────────────────
        shared_repr = F.relu(self.shared_head(shared))
        logits = [head(shared_repr) for head in self.task_heads]
        logits = torch.cat(logits, dim=1)        # (B, 3)

        return logits
