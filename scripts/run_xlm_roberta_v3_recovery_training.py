"""Recover XLM-RoBERTa v3 training on a 4GB GPU.

Full XLM-RoBERTa fine-tuning on the full v3 dataset can exceed 4GB VRAM. This
runner keeps the same dataset/checkpoint target but retries with progressively
more conservative partial fine-tuning settings.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "datasets" / "unified" / "prompt_injection_transformer_ready_v3.csv"
LOG_ROOT = PROJECT_ROOT / "logs" / f"xlm_roberta_v3_recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
STATUS_PATH = LOG_ROOT / "training_status.json"
CHECKPOINT_NAME = "xlm_roberta_v3"

ATTEMPTS = [
    {
        "name": "partial_last_2_layers",
        "batch_size": 8,
        "max_length": 64,
        "gradient_accumulation_steps": 4,
        "freeze_encoder_layers": 10,
        "optim": "adafactor",
        "gradient_checkpointing": True,
    },
    {
        "name": "partial_last_1_layer",
        "batch_size": 8,
        "max_length": 64,
        "gradient_accumulation_steps": 4,
        "freeze_encoder_layers": 11,
        "optim": "adafactor",
        "gradient_checkpointing": True,
    },
    {
        "name": "classifier_head_only",
        "batch_size": 16,
        "max_length": 64,
        "gradient_accumulation_steps": 2,
        "freeze_base_model": True,
        "optim": "adafactor",
        "gradient_checkpointing": False,
    },
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
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        return process.wait()


def build_command(attempt: dict) -> list[str]:
    command = [
        sys.executable,
        "src/train_transformers.py",
        "--model",
        "xlm_roberta",
        "--dataset",
        str(DATASET_PATH),
        "--checkpoint-name",
        CHECKPOINT_NAME,
        "--epochs",
        "3",
        "--batch-size",
        str(attempt["batch_size"]),
        "--max-length",
        str(attempt["max_length"]),
        "--gradient-accumulation-steps",
        str(attempt["gradient_accumulation_steps"]),
        "--optim",
        str(attempt["optim"]),
    ]
    if attempt.get("gradient_checkpointing"):
        command.append("--gradient-checkpointing")
    if attempt.get("freeze_base_model"):
        command.append("--freeze-base-model")
    if int(attempt.get("freeze_encoder_layers", 0)) > 0:
        command.extend(["--freeze-encoder-layers", str(attempt["freeze_encoder_layers"])])
    return command


def main() -> int:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    status = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(DATASET_PATH),
        "checkpoint": str(PROJECT_ROOT / "models" / "transformers" / CHECKPOINT_NAME),
        "log_root": str(LOG_ROOT),
        "status": "running",
        "attempts": [],
    }
    save_status(status)

    if checkpoint_ready(CHECKPOINT_NAME):
        status["status"] = "skipped_already_ready"
        status["finished_at"] = datetime.now().isoformat(timespec="seconds")
        save_status(status)
        return 0

    for index, attempt in enumerate(ATTEMPTS, start=1):
        log_path = LOG_ROOT / f"xlm_roberta_v3_{index}_{attempt['name']}.log"
        status["attempts"].append(
            {
                **attempt,
                "attempt": index,
                "log_path": str(log_path),
                "status": "running",
            }
        )
        save_status(status)

        return_code = run_command(build_command(attempt), log_path)
        status["attempts"][-1]["return_code"] = return_code
        status["attempts"][-1]["status"] = "success" if return_code == 0 else "failed"
        save_status(status)
        if return_code == 0 and checkpoint_ready(CHECKPOINT_NAME):
            status["status"] = "success"
            status["finished_at"] = datetime.now().isoformat(timespec="seconds")
            save_status(status)
            return 0

    status["status"] = "failed"
    status["finished_at"] = datetime.now().isoformat(timespec="seconds")
    save_status(status)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
