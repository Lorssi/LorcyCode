# Architecture Patterns

## Strong Default

Use separate encoders for discrete and continuous features:

- Embedding layers for discrete IDs.
- Mask-aware projections for continuous features.
- A compact temporal encoder for `abortion_rate_past_1d` to `abortion_rate_past_7d`.
- A shared fusion MLP.
- Either one 3-logit head or lightweight task-specific heads.

## Mask-Aware Feature Handling

Use masks directly. Common options:

- Multiply continuous values or embeddings by masks.
- Concatenate mask values as additional signals.
- Use a learned missing-value embedding or projection only when implementation remains simple.

Avoid treating missing values as ordinary zeros without also using the mask.

## Multi-Task Head

Prefer a shared backbone plus three outputs. If one horizon is much weaker, use small horizon-specific heads after the shared representation.

## Temporal Signal

The seven abortion-rate history features can be treated as:

- a 7-step sequence passed through a shared linear projection plus pooling,
- a small 1D convolution,
- a GRU if training time permits,
- or summary statistics plus raw projected values.

Avoid overly complex models in early iterations; this dataset and code loop benefit from reliable runnable improvements.

## Regularization

Use dropout, weight decay, batch normalization, or layer normalization when logs suggest overfitting. Increase complexity only if validation metrics stagnate and code remains stable.
