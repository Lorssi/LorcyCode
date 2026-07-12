---
name: risk-alert-training-debugging
description: Implement, parameterize, execute, archive, validate, and debug PyTorch code for the PRRS abortion abnormal risk-alert model. Use when turning a design into runnable training/prediction code, overriding data/transform/output paths or hyperparameters, creating model_v_* experiment runs, repairing failures, or checking run artifacts.
---

# Risk Alert Training Debugging

## Workflow

1. Read `assets/model_reference.md`, `references/model_template.md`, `references/project-template-contract.md`, and `references/coding-contract.md` before generating code. Treat the dynamic transform-loading block in the asset as mandatory, not illustrative.
2. Generate one self-contained parameterized Python model and one `design.md`.
3. Run `scripts/execute_model.py`; let it create the run directory before model execution.
4. On failure, inspect `<run-dir>/stderr.txt`, `train_log.txt`, `resolved_config.json`, and `run_manifest.json`; then read `references/common-errors.md` and create a new run for the repair.
5. Run `scripts/validate_artifacts.py <run-dir>` using the recorded mode.
6. Return `run_id`, `run_dir`, status, and concrete repair summary.

Never overwrite a prior run or copy generated code into the project source tree.
Never copy category counts or feature positions from a particular transform JSON into generated source code. Resolve them at runtime from `--transform-path` and the loaded CSV columns.

## Execute

```powershell
python .lorcy/skills/risk-alert-training-debugging/scripts/execute_model.py `
  --code-file generated_model.py `
  --design-file design.md
```

The default is `train-predict` with project paths and current hyperparameters. Optional inputs include `--mode`, `--data-dir`, `--transform-path`, `--result-root`, `--checkpoint`, `--config`, `--epochs`, `--batch-size`, `--learning-rate`, `--num-workers`, `--dropout`, `--embedding-size`, `--decision-threshold`, and `--seed`.

Configuration priority is CLI > JSON config > defaults. `predict` mode requires `--checkpoint`. The executor prints `run_id` and `run_dir` even when model execution fails.

## Validate

```powershell
python .lorcy/skills/risk-alert-training-debugging/scripts/validate_artifacts.py <run-dir>
```

Use `--strict` after the evaluation skill writes its results into the same directory.
