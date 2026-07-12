#!/usr/bin/env python
"""Create an immutable run directory, execute model code, and archive provenance."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

DATA_FILES = {
    "train_x": "risk_alert_transformed_train_X_mask_null.csv",
    "train_y": "risk_alert_train_y.csv",
    "test_x": "risk_alert_transformed_test_X_mask_null.csv",
    "test_y": "risk_alert_test_y.csv",
    "predict_x": "risk_alert_predict_feature_transformed_masknull.csv",
    "predict_index": "risk_alert_predict_index_sample_data.csv",
}
DEFAULTS = {
    "mode": "train-predict", "epochs": 10, "batch_size": 256,
    "learning_rate": 0.001, "num_workers": 0, "dropout": 0.2,
    "embedding_size": 128, "decision_threshold": 0.5, "seed": 42,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(path: Path) -> dict:
    item = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        stat = path.stat()
        item.update(size=stat.st_size, modified_at=dt.datetime.fromtimestamp(
            stat.st_mtime, dt.timezone.utc).isoformat(), sha256=sha256(path))
    return item


def load_config(path: str | None) -> dict:
    if not path:
        return {}
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("--config must contain a JSON object")
    unknown = set(value) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown config keys: {', '.join(sorted(unknown))}")
    return value


def source_contract_errors(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    required = {
        "RiskAlertTransformPipeline.from_json": "deserialize --transform-path at runtime",
        "category_encode.size": "derive categorical cardinalities from the transform",
        "get_loc(": "derive feature and mask indices from DataFrame columns",
    }
    errors = [description for token, description in required.items() if token not in source]
    fixed_index_tokens = ("(0, 1", "(2, 3", "(4, 5")
    if "FEATURES = {" in source and any(token in source for token in fixed_index_tokens):
        errors.append("remove the hard-coded FEATURES numeric index map")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-file", required=True)
    parser.add_argument("--design-file")
    parser.add_argument("--project-root", default=os.getcwd())
    parser.add_argument("--result-root")
    parser.add_argument("--data-dir")
    parser.add_argument("--transform-path")
    parser.add_argument("--checkpoint")
    parser.add_argument("--config")
    parser.add_argument("--mode", choices=("train", "predict", "train-predict"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--embedding-size", type=int)
    parser.add_argument("--decision-threshold", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    code_file = Path(args.code_file).resolve()
    if not code_file.is_file():
        raise FileNotFoundError(code_file)
    result_root = Path(args.result_root).resolve() if args.result_root else project_root / "lorcy_code" / "data" / "result"
    data_dir = Path(args.data_dir).resolve() if args.data_dir else project_root / "lorcy_code" / "data" / "interim" / "PRRS_Abortion_Abnormal_Alert" / "risk_alert"
    transform_path = Path(args.transform_path).resolve() if args.transform_path else project_root / "lorcy_code" / "data" / "model" / "PRRS_Abortion_Abnormal_Alert" / "risk_alert" / "risk_alert_nfm_transform.json"

    resolved = {**DEFAULTS, **load_config(args.config)}
    for key in DEFAULTS:
        value = getattr(args, key, None)
        if value is not None:
            resolved[key] = value
    resolved.update(data_dir=str(data_dir), transform_path=str(transform_path),
                    checkpoint=str(Path(args.checkpoint).resolve()) if args.checkpoint else None)
    if resolved["mode"] == "predict" and not resolved["checkpoint"]:
        parser.error("--checkpoint is required in predict mode")

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"model_v_{timestamp}_{uuid.uuid4().hex[:8]}"
    run_dir = result_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(code_file, run_dir / "model.py")
    if args.design_file:
        shutil.copy2(Path(args.design_file).resolve(), run_dir / "design.md")
    else:
        (run_dir / "design.md").write_text("# Model design\n\nNot supplied.\n", encoding="utf-8")

    resolved["output_dir"] = str(run_dir)
    (run_dir / "resolved_config.json").write_text(json.dumps(resolved, indent=2, ensure_ascii=False), encoding="utf-8")
    if transform_path.is_file():
        shutil.copy2(transform_path, run_dir / "transform.json")

    inputs = {name: describe(data_dir / filename) for name, filename in DATA_FILES.items()}
    inputs["transform"] = describe(transform_path)
    if resolved["checkpoint"]:
        inputs["checkpoint"] = describe(Path(resolved["checkpoint"]))
    manifest = {
        "run_id": run_id, "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "running", "mode": resolved["mode"], "python": platform.python_version(),
        "platform": platform.platform(), "inputs": inputs,
    }
    try:
        import torch
        manifest.update(torch_version=torch.__version__, cuda_available=torch.cuda.is_available())
    except ImportError:
        manifest.update(torch_version=None, cuda_available=None)
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    contract_errors = source_contract_errors(run_dir / "model.py")
    if contract_errors:
        message = "Generated model violates the dynamic transform contract:\n" + "\n".join(
            f"- {item}" for item in contract_errors
        )
        (run_dir / "train_log.txt").write_text("", encoding="utf-8")
        (run_dir / "stderr.txt").write_text(message, encoding="utf-8")
        manifest.update(status="source_contract_failed", return_code=2,
                        finished_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                        source_contract_errors=contract_errors)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(run_id)
        print(run_dir)
        print(message, file=sys.stderr)
        return 2

    command = [args.python_executable, str(run_dir / "model.py"), "--mode", str(resolved["mode"]),
               "--data-dir", str(data_dir), "--transform-path", str(transform_path),
               "--output-dir", str(run_dir), "--epochs", str(resolved["epochs"]),
               "--batch-size", str(resolved["batch_size"]), "--learning-rate", str(resolved["learning_rate"]),
               "--num-workers", str(resolved["num_workers"]), "--dropout", str(resolved["dropout"]),
               "--embedding-size", str(resolved["embedding_size"]),
               "--decision-threshold", str(resolved["decision_threshold"]), "--seed", str(resolved["seed"])]
    if resolved["checkpoint"]:
        command += ["--checkpoint", resolved["checkpoint"]]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")
    try:
        result = subprocess.run(command, cwd=project_root, env=env, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=args.timeout)
        (run_dir / "train_log.txt").write_text(result.stdout, encoding="utf-8")
        (run_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")
        manifest.update(status="success" if result.returncode == 0 else "failed", return_code=result.returncode,
                        command=command, finished_at=dt.datetime.now(dt.timezone.utc).isoformat())
        return_code = result.returncode
    except subprocess.TimeoutExpired as error:
        (run_dir / "train_log.txt").write_text(error.stdout or "", encoding="utf-8")
        (run_dir / "stderr.txt").write_text(error.stderr or f"Timed out after {args.timeout}s", encoding="utf-8")
        manifest.update(status="timeout", return_code=124, command=command,
                        finished_at=dt.datetime.now(dt.timezone.utc).isoformat())
        return_code = 124
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(run_id)
    print(run_dir)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
