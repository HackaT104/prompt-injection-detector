"""Run full v3 Transformer fine-tuning jobs sequentially.

This helper is intentionally small: it reuses src/train_transformers.py for all
model logic, writes one log per attempt, and retries with smaller settings when
CUDA memory is tight.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "datasets" / "unified" / "prompt_injection_transformer_ready_v3.csv"
LOG_ROOT = PROJECT_ROOT / "logs" / f"transformer_v3_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
STATUS_PATH = LOG_ROOT / "training_status.json"

MODELS = [
    {"model": "distilbert", "checkpoint": "distilbert_v3"},
    {"model": "roberta", "checkpoint": "roberta_v3"},
    {"model": "xlm_roberta", "checkpoint": "xlm_roberta_v3"},
]

ATTEMPTS = [
    {"batch_size": 4, "max_length": 128},
    {"batch_size": 2, "max_length": 128},
    {"batch_size": 2, "max_length": 64},
]


def checkpoint_ready(name: str) -> bool:
    checkpoint_dir = PROJECT_ROOT / "models" / "transformers" / name
    required = [
        checkpoint_dir / "config.json",
        checkpoint_dir / "training_metadata.json",
        checkpoint_dir / "metrics.json",
    ]
    weights = [checkpoint_dir / "model.safetensors", checkpoint_dir / "pytorch_model.bin"]
    return all(path.exists() for path in required) and any(path.exists() for path in weights)


def save_status(status: dict) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def run_command(command: list[str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env={**dict(**__import__("os").environ), "PYTHONUNBUFFERED": "1"},
        )
        return process.wait()


def main() -> int:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    status: dict = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(DATASET_PATH),
        "log_root": str(LOG_ROOT),
        "models": {},
        "calibration": {"status": "pending"},
    }
    save_status(status)

    if not DATASET_PATH.exists():
        status["error"] = f"Dataset not found: {DATASET_PATH}"
        save_status(status)
        return 1

    for item in MODELS:
        model = item["model"]
        checkpoint = item["checkpoint"]
        status["models"][checkpoint] = {
            "base_command_model": model,
            "status": "running",
            "attempts": [],
        }
        save_status(status)

        if checkpoint_ready(checkpoint):
            status["models"][checkpoint]["status"] = "skipped_already_ready"
            save_status(status)
            continue

        success = False
        for attempt_index, attempt in enumerate(ATTEMPTS, start=1):
            log_path = LOG_ROOT / f"{checkpoint}_attempt_{attempt_index}.log"
            command = [
                sys.executable,
                "src/train_transformers.py",
                "--model",
                model,
                "--dataset",
                str(DATASET_PATH),
                "--epochs",
                "3",
                "--batch-size",
                str(attempt["batch_size"]),
                "--max-length",
                str(attempt["max_length"]),
                "--gradient-accumulation-steps",
                "2",
            ]
            status["models"][checkpoint]["attempts"].append(
                {
                    "attempt": attempt_index,
                    "batch_size": attempt["batch_size"],
                    "max_length": attempt["max_length"],
                    "log_path": str(log_path),
                    "status": "running",
                }
            )
            save_status(status)

            return_code = run_command(command, log_path)
            status["models"][checkpoint]["attempts"][-1]["return_code"] = return_code
            status["models"][checkpoint]["attempts"][-1]["status"] = (
                "success" if return_code == 0 else "failed"
            )
            save_status(status)

            if return_code == 0 and checkpoint_ready(checkpoint):
                success = True
                status["models"][checkpoint]["status"] = "success"
                save_status(status)
                break

        if not success:
            status["models"][checkpoint]["status"] = "failed"
            status["finished_at"] = datetime.now().isoformat(timespec="seconds")
            save_status(status)
            return 1

    calibration_log = LOG_ROOT / "threshold_calibration.log"
    calibration_command = [
        sys.executable,
        "scripts/calibrate_thresholds.py",
        "--dataset",
        str(DATASET_PATH),
        "--models",
        "distilbert_v3",
        "roberta_v3",
        "xlm_roberta_v3",
    ]
    status["calibration"] = {"status": "running", "log_path": str(calibration_log)}
    save_status(status)
    calibration_return_code = run_command(calibration_command, calibration_log)
    status["calibration"]["return_code"] = calibration_return_code
    status["calibration"]["status"] = "success" if calibration_return_code == 0 else "failed"
    status["finished_at"] = datetime.now().isoformat(timespec="seconds")
    save_status(status)
    return calibration_return_code


if __name__ == "__main__":
    raise SystemExit(main())
