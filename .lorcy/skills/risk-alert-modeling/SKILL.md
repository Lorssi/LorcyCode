---
name: risk-alert-modeling
description: Design model architectures for PRRS abortion abnormal risk alert code-iteration tasks. Use when Codex must understand the risk-alert prediction task, feature/mask schema, model constraints, and produce or revise a feasible PyTorch architecture plan before code implementation.
---

# Risk Alert Modeling

## Purpose

Use this skill to design or revise the model architecture for the PRRS abortion abnormal alert task. Keep this skill focused on task semantics, feature handling, model constraints, and architecture choices; use `risk-alert-training-debugging` when generating/running code.

## Required Context

Read these references before producing a design plan:

- `references/task-and-labels.md` for task objective and labels.
- `references/feature-schema.md` for feature order, discrete/continuous groups, and mask layout.
- `references/model-constraints.md` for non-negotiable code and output constraints.
- `references/architecture-patterns.md` for recommended architecture families.

## Design Procedure

1. Restate the current optimization goal: improve multi-label prediction for 1-7, 8-14, and 15-21 day windows.
2. Identify the likely bottleneck from the latest evaluation or user request.
3. Choose an architecture pattern that respects the feature schema and code constraints.
4. Specify the model components concretely enough for code generation: embeddings, continuous feature transforms, mask usage, temporal handling, shared layers, task heads, dropout, and loss compatibility.
5. Avoid proposing changes to file loading, saving, label names, prediction column names, or `RiskAlertMultiLabelModel` class naming.

## Output Contract

Return a concise design plan with:

- `architecture`: model structure and data flow.
- `mask_strategy`: how valid/invalid feature masks are used.
- `training_notes`: only training suggestions needed by implementation.
- `risks`: possible implementation or overfitting risks.

Do not output full code from this skill. Hand the plan to `risk-alert-training-debugging` for implementation.
