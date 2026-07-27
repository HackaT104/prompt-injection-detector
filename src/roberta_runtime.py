"""RoBERTa runtime service for the official detection pipeline."""

from __future__ import annotations

import os
from time import perf_counter
from typing import Any

from src.runtime_config import load_runtime_config
from src.runtime_config import project_path, security_model_config
from src.transformer_utils import (
    is_finetuned_transformer_checkpoint,
    predict_transformer,
    predict_transformer_batch,
    resolve_transformer_model_dir,
)


def _score(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


class RoBERTaRuntimeService:
    """Cached RoBERTa inference wrapper.

    `predict_transformer` uses cached model/tokenizer artifacts internally, so
    the model is not reloaded on every request.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name

    @property
    def configuration(self) -> dict[str, Any]:
        return security_model_config()

    @property
    def resolved_model_name(self) -> str:
        return str(self.model_name or self.configuration.get("modelName") or "roberta_v4")

    @property
    def model_path(self):
        configured = self.configuration.get("modelPath")
        if configured:
            return project_path(configured)
        return resolve_transformer_model_dir(self.resolved_model_name)

    @property
    def model_version(self) -> str:
        return str(self.configuration.get("modelVersion") or load_runtime_config().get("modelVersion"))

    def health(self) -> dict[str, Any]:
        model_dir = self.model_path
        tokenizer_dir = project_path(self.configuration.get("tokenizerPath"))
        calibrator_path = self.configuration.get("calibratorPath")
        calibrator_enabled = bool(self.configuration.get("calibratorEnabled"))
        tokenizer_ready = tokenizer_dir.exists() and any(
            (tokenizer_dir / name).exists()
            for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json")
        )
        return {
            "model": self.resolved_model_name,
            "modelVersion": self.model_version,
            "modelPath": str(model_dir),
            "tokenizerPath": str(tokenizer_dir),
            "ready": is_finetuned_transformer_checkpoint(model_dir) and tokenizer_ready,
            "modelReady": is_finetuned_transformer_checkpoint(model_dir),
            "tokenizerReady": tokenizer_ready,
            "cacheMode": "process_lru_cache",
            "maxLength": int(self.configuration.get("maxLength", 128)),
            "thresholdVersion": self.configuration.get("thresholdVersion"),
            "inputThreshold": self.configuration.get("inputWarnThreshold"),
            "outputThreshold": self.configuration.get("outputWarnThreshold"),
            "outputThresholdStatus": self.configuration.get("outputThresholdStatus"),
            "calibratorEnabled": calibrator_enabled,
            "calibratorReady": (
                bool(calibrator_path and project_path(calibrator_path).exists())
                if calibrator_enabled
                else True
            ),
            "calibratorVersion": self.configuration.get("calibratorVersion"),
        }

    def predict(
        self,
        text: str,
        *,
        use_cuda: bool = False,
        max_length: int | None = None,
        stage: str = "input",
    ) -> dict[str, Any]:
        timeout_seconds = float(os.getenv("ROBERTA_TIMEOUT_SECONDS", "30") or 30)
        start = perf_counter()
        try:
            model_dir = self.model_path
            normalized_stage = str(stage or "input").strip().lower()
            if normalized_stage not in {"input", "output"}:
                raise ValueError("RoBERTa stage must be 'input' or 'output'.")
            stage_warn = float(
                self.configuration.get(
                    "outputWarnThreshold" if normalized_stage == "output" else "inputWarnThreshold",
                    0.30,
                )
            )
            stage_block = float(
                self.configuration.get(
                    "outputBlockThreshold" if normalized_stage == "output" else "inputBlockThreshold",
                    0.85 if normalized_stage == "output" else 0.45,
                )
            )
            thresholds = {
                "evaluation_threshold": float(self.configuration.get("evaluationThreshold", 0.02)),
                "runtime_warn_threshold": stage_warn,
                "runtime_block_threshold": stage_block,
            }
            result = predict_transformer(
                text=text,
                model_path=model_dir,
                model_name=self.resolved_model_name,
                max_length=int(max_length or self.configuration.get("maxLength", 128)),
                thresholds=thresholds,
                use_cuda=use_cuda,
                use_runtime_calibration=bool(self.configuration.get("calibratorEnabled", False)),
            )
            latency_ms = round((perf_counter() - start) * 1000, 3)
            score = _score(result.get("risk_score"))
            label = "injection" if score >= stage_warn else "safe"
            warnings = list(result.get("warnings", []) or [])
            if latency_ms > timeout_seconds * 1000:
                warnings.append("ROBERTA_TIMEOUT_EXCEEDED")
            return {
                "score": score,
                "label": label,
                "stage": normalized_stage,
                "modelVersion": self.model_version,
                "modelName": self.resolved_model_name,
                "modelPath": str(model_dir),
                "latencyMs": latency_ms,
                "available": True,
                "error": None,
                "rawScore": result.get("raw_score"),
                "primaryRawScore": result.get("primary_raw_score"),
                "calibratedScore": result.get("calibrated_score"),
                "intentAdjustedScore": result.get("intent_adjusted_score"),
                "scoreUsed": result.get("score_used"),
                "calibrationEnabled": result.get("calibration_enabled"),
                "calibrationSource": result.get("calibration_source"),
                "runtimeBenignIntent": result.get("runtime_benign_intent"),
                "benignGuard": result.get("benign_guard"),
                "inputPreprocessing": result.get("input_preprocessing"),
                "inputVariants": result.get("input_variants"),
                "thresholdUsed": result.get("threshold_used"),
                "thresholdSource": result.get("threshold_source"),
                "thresholdVersion": self.configuration.get("thresholdVersion"),
                "calibratorVersion": self.configuration.get("calibratorVersion"),
                "warnings": warnings,
            }
        except Exception as exc:  # pragma: no cover - defensive runtime path
            latency_ms = round((perf_counter() - start) * 1000, 3)
            return {
                "score": None,
                "label": "error",
                "stage": str(stage or "input"),
                "modelVersion": self.model_version,
                "modelName": self.resolved_model_name,
                "modelPath": str(self.model_path),
                "latencyMs": latency_ms,
                "available": False,
                "error": exc.__class__.__name__,
                "scoreUsed": "error",
                "thresholdUsed": None,
                "warnings": ["ROBERTA_INFERENCE_ERROR"],
            }

    def predict_many(
        self,
        texts: list[str],
        *,
        use_cuda: bool = False,
        max_length: int | None = None,
        stage: str = "input",
    ) -> list[dict[str, Any]]:
        """Score variants in batches while reusing the singleton checkpoint."""
        if not texts:
            return []
        started = perf_counter()
        normalized_stage = str(stage or "input").strip().lower()
        if normalized_stage not in {"input", "output"}:
            raise ValueError("RoBERTa stage must be 'input' or 'output'.")
        warn = float(self.configuration.get("outputWarnThreshold" if normalized_stage == "output" else "inputWarnThreshold", 0.30))
        block = float(self.configuration.get("outputBlockThreshold" if normalized_stage == "output" else "inputBlockThreshold", 0.85 if normalized_stage == "output" else 0.45))
        thresholds = {
            "evaluation_threshold": float(self.configuration.get("evaluationThreshold", 0.02)),
            "runtime_warn_threshold": warn,
            "runtime_block_threshold": block,
        }
        try:
            raw_results = predict_transformer_batch(
                texts=texts,
                model_path=self.model_path,
                model_name=self.resolved_model_name,
                max_length=int(max_length or self.configuration.get("maxLength", 128)),
                thresholds=thresholds,
                use_cuda=use_cuda,
                use_runtime_calibration=bool(self.configuration.get("calibratorEnabled", False)),
                batch_size=int(self.configuration.get("batchSize", 8)),
            )
        except Exception:
            return [self.predict(text, use_cuda=use_cuda, max_length=max_length, stage=normalized_stage) for text in texts]
        elapsed_ms = round((perf_counter() - started) * 1000, 3)
        per_item_latency = round(elapsed_ms / max(1, len(raw_results)), 3)
        return [
            {
                "score": _score(result.get("risk_score")),
                "label": "injection" if _score(result.get("risk_score")) >= warn else "safe",
                "stage": normalized_stage,
                "modelVersion": self.model_version,
                "modelName": self.resolved_model_name,
                "modelPath": str(self.model_path),
                "latencyMs": per_item_latency,
                "batchLatencyMs": elapsed_ms,
                "available": True,
                "error": None,
                "rawScore": result.get("raw_score"),
                "primaryRawScore": result.get("primary_raw_score"),
                "calibratedScore": result.get("calibrated_score"),
                "intentAdjustedScore": result.get("intent_adjusted_score"),
                "scoreUsed": result.get("score_used"),
                "calibrationEnabled": result.get("calibration_enabled"),
                "calibrationSource": result.get("calibration_source"),
                "runtimeBenignIntent": result.get("runtime_benign_intent"),
                "benignGuard": result.get("benign_guard"),
                "inputPreprocessing": result.get("input_preprocessing"),
                "inputVariants": result.get("input_variants"),
                "thresholdUsed": result.get("threshold_used"),
                "thresholdSource": result.get("threshold_source"),
                "thresholdVersion": self.configuration.get("thresholdVersion"),
                "calibratorVersion": self.configuration.get("calibratorVersion"),
                "warnings": result.get("warnings", []),
            }
            for result in raw_results
        ]

    def warmup(self, *, use_cuda: bool = False) -> dict[str, Any]:
        """Load and cache the selected checkpoint once, returning a safe health result."""
        prediction = self.predict("Model health check.", use_cuda=use_cuda, stage="input")
        return {**self.health(), "warmupAvailable": prediction.get("available", False)}


roberta_service = RoBERTaRuntimeService()
