#!/usr/bin/env python
"""Validate artifacts according to the recorded run mode."""
import argparse
import json
from pathlib import Path

COMMON = {"model.py", "design.md", "resolved_config.json", "run_manifest.json", "transform.json", "train_log.txt", "stderr.txt"}
TRAIN = {"model_best.pth", "train_history.csv", "train_history.json"}
PREDICT = {"predict_result.csv"}
EVALUATION = {"evaluate_result.txt", "evaluate_result.csv"}


def validate_dynamic_source(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    required = {
        "RiskAlertTransformPipeline.from_json": "runtime transform deserialization",
        "category_encode.size": "runtime category cardinality lookup",
        "get_loc(": "runtime DataFrame column lookup",
    }
    errors = [f"model.py lacks {description}" for token, description in required.items() if token not in source]
    fixed_index_tokens = ("(0, 1", "(2, 3", "(4, 5")
    if "FEATURES = {" in source and any(token in source for token in fixed_index_tokens):
        errors.append("model.py contains a hard-coded FEATURES index map")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("version_dir")
    parser.add_argument("--mode", choices=("train", "predict", "train-predict"))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    root = Path(args.version_dir)
    config = json.loads((root / "resolved_config.json").read_text(encoding="utf-8")) if (root / "resolved_config.json").is_file() else {}
    mode = args.mode or config.get("mode", "train-predict")
    required = set(COMMON)
    if mode in ("train", "train-predict"):
        required |= TRAIN
    if mode in ("predict", "train-predict"):
        required |= PREDICT
    if args.strict:
        required |= EVALUATION
    missing = sorted(name for name in required if not (root / name).is_file())
    source_errors = validate_dynamic_source(root / "model.py") if (root / "model.py").is_file() else []
    if missing or source_errors:
        if missing:
            print("missing artifacts:\n" + "\n".join(f"- {name}" for name in missing))
        if source_errors:
            print("source contract violations:\n" + "\n".join(f"- {item}" for item in source_errors))
        return 1
    print(f"artifact validation passed ({mode})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
