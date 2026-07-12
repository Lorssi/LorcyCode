# Design: RiskAlertMultiLabelModel — Enhanced Multi-Horizon Abortion Risk Alert

## Optimization Goal

Improve multi-label binary prediction for three forward-looking horizons (1-7d, 8-14d, 15-21d).  
The previous baseline achieved F1=0.8036 / AUC=0.9605 on validation, but evaluation on the predict set showed significant degradation: abort_1_7 (F1=0.6633, AUC=0.7984), abort_8_14 (F1=0.5332, AUC=0.6610), abort_15_21 (F1=0.5030, AUC=0.6009).  
The drop from validation to prediction metrics and the decay across horizons suggests improvements are needed in temporal modeling, label imbalance handling, and generalization.

## Architecture

```
Input (batch_size, 30) — 15 features × (value + mask)
│
├── Discrete Encoder (5 features: org_inv_dk, city, season, l3_org_inv_dk, month)
│   └── Embedding(vocab_size → emb_dim) × mask.unsqueeze(1)
│
├── Continuous Encoder (3 features: check_out_ratio_7d, reserve_sow_sqty, abortion_rate_ma_diff)
│   ├── BatchNorm1d(3)
│   ├── Linear(3 → emb_dim) + ReLU
│   └── × mask_fraction (mean of continuous masks)
│
├── Temporal Encoder (7 abortion_rate_past_*d features)
│   ├── GRU(input_size=1, hidden_size=emb_dim//2, bidirectional=True, batch_first=True)
│   │   → (batch, 7, emb_dim)  ← concatenated forward+backward
│   ├── Mask-aware: zero-out unobserved steps before GRU
│   ├── GlobalAvgPool over time → (batch, emb_dim)
│   └── Residual connection: skip-connection from raw temporal projection (if observed)
│
├── Cross-Attention Fusion
│   ├── Discrete features → Q, Continuous+Temporal → K,V (multi-head attention, 4 heads)
│   │   → Attended discrete context (batch, 5*emb_dim)
│   ├── Concatenate discrete, continuous, temporal representations
│   │   → (batch, 7*emb_dim)
│
├── Shared Fusion MLP
│   ├── Linear(7*emb_dim → emb_dim*2) → LayerNorm → ReLU → Dropout(0.3)
│   ├── Linear(emb_dim*2 → emb_dim) → LayerNorm → ReLU → Dropout(0.3)
│
├── Task-Specific Multi-Head Outputs
│   ├── Shared head: Linear(emb_dim → emb_dim) → ReLU
│   ├── Horizon-specific head 0 (abort_1_7):  Linear(emb_dim → 1)
│   ├── Horizon-specific head 1 (abort_8_14): Linear(emb_dim → 1)
│   ├── Horizon-specific head 2 (abort_15_21): Linear(emb_dim → 1)
│   └── Horizon-specific temporal attention: each head can attend to GRU outputs differently
│       via a learned weighted sum over the 7 time steps
│
└── Output: (batch_size, 3) logits — no sigmoid
```

### Key Dimensions
- `embedding_size`: 128 (default, configurable via `--embedding-size`)
- GRU hidden: `embedding_size // 2` each direction → total `embedding_size`
- Fusion MLP hidden: `max(embedding_size * 2, 256)`
- Cross-attention heads: 4
- Dropout: 0.3 (increased from 0.2 for better generalization)

## Mask Strategy
- **Discrete**: `embedding(idx) × mask.unsqueeze(1)` — zero out missing embedding vectors
- **Continuous**: `value × mask` before BatchNorm, then output × `mask.mean(dim=1, keepdim=True)`
- **Temporal**: zero missing steps before GRU; when no temporal feature is observed, use a learned missing embedding rather than zeros
- **All masks** derived from interleaved feature/mask columns via `feature_layout` (runtime-resolved)

## Key Improvements Over Baseline

| Aspect | Previous (Baseline) | Current (This Design) |
|--------|---------------------|----------------------|
| Temporal encoder | Conv1d + AvgPool | Bidirectional GRU + GlobalAvgPool |
| Temporal missing handling | Zero + mask fraction | Learned missing embedding |
| Feature interaction | Simple concat + MLP | Cross-attention between discrete and continuous/temporal |
| Task heads | Shared + 3×Linear | Shared + Temporal attention per head |
| Normalization | BatchNorm1d | LayerNorm (more stable) |
| Dropout | 0.2 | 0.3 |
| Loss | BCEWithLogitsLoss | BCEWithLogitsLoss + pos_weight (inverse frequency) |
| Epochs | 10 | 20 |

## Training Notes
- **Loss**: `BCEWithLogitsLoss` with `pos_weight` computed from training label frequency per horizon (to handle label imbalance)
- **Optimizer**: AdamW with `weight_decay=1e-4`
- **Scheduler**: `ReduceLROnPlateau` (factor=0.5, patience=3, min_lr=1e-6)
- **Gradient clipping**: `max_norm=1.0`
- **Mixed precision (AMP)**: when CUDA available
- **Seed**: reproducibility via `--seed`
- **Feature columns**: loaded from CSV at runtime; no hardcoded indices
- **Transform**: loaded dynamically from `--transform-path` using `RiskAlertTransformPipeline.from_json()`
- **Cardinalities**: `org_inv_dk`, `city`, `l3_org_inv_dk` from `category_encode.size + 1`; `season=5`, `month=13`

## Risks
1. **GRU vs Conv1d**: GRU has more parameters and may overfit on only 7 time steps; dropout=0.3 and weight_decay mitigate
2. **Cross-attention**: May add complexity; if validation metrics don't improve, can be simplified to direct concat
3. **Horizon-specific attention**: each head has an additional small parameter set; properly regularized by shared backbone
4. **Label imbalance**: Some horizons may have very few positive samples; pos_weight helps but AUC may still be undefined for single-class batches
5. **Overfitting**: If train loss decreases but val metrics plateau, increase dropout or reduce embedding_size
