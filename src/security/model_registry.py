"""Read-only runtime model registry backed by the centralized YAML config."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from src.runtime_config import project_path, security_model_config


def _file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def active_model_snapshot() -> dict[str, Any]:
    config = security_model_config()
    model_path = project_path(config.get("modelPath"))
    tokenizer_path = project_path(config.get("tokenizerPath"))
    weight_path = next(
        (model_path / name for name in ("model.safetensors", "pytorch_model.bin") if (model_path / name).exists()),
        None,
    )
    return {
        "modelName": config.get("modelName"),
        "modelVersion": config.get("modelVersion"),
        "modelPath": str(model_path),
        "tokenizerPath": str(tokenizer_path),
        "trainingDatasetVersion": config.get("trainingDatasetVersion"),
        "evaluationDatasetVersion": config.get("evaluationDatasetVersion"),
        "thresholdVersion": config.get("thresholdVersion"),
        "evaluationThreshold": config.get("evaluationThreshold"),
        "inputWarnThreshold": config.get("inputWarnThreshold"),
        "inputBlockThreshold": config.get("inputBlockThreshold"),
        "outputWarnThreshold": config.get("outputWarnThreshold"),
        "outputBlockThreshold": config.get("outputBlockThreshold"),
        "outputThresholdStatus": config.get("outputThresholdStatus"),
        "calibratorEnabled": bool(config.get("calibratorEnabled")),
        "calibratorPath": config.get("calibratorPath"),
        "calibratorVersion": config.get("calibratorVersion"),
        "maxLength": config.get("maxLength"),
        "batchSize": config.get("batchSize"),
        "device": config.get("device"),
        "activeInput": bool(config.get("activeInput", True)),
        "activeOutput": bool(config.get("activeOutput", True)),
        "artifactSha256": _file_sha256(weight_path) if weight_path else None,
    }

