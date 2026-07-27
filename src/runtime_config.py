"""Central runtime configuration for the Hybrid Sandwich pipeline."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_CONFIG_PATH = PROJECT_ROOT / "configs" / "security_runtime.yaml"
LEGACY_RUNTIME_CONFIG_PATH = PROJECT_ROOT / "configs" / "runtime_policy.json"

DEFAULT_RUNTIME_CONFIG: dict[str, Any] = {
    "schemaVersion": "hybrid-sandwich-security-v1",
    "policyVersion": "hybrid-sandwich-policy-v1",
    "ruleVersion": "rule-runtime-v1",
    "runtimeModel": "roberta_v4",
    "modelVersion": "roberta-v4-neuralchemy",
    "weights": {"rule": 0.30, "roberta": 0.50, "context": 0.20},
    "thresholds": {"warn": 0.30, "block": 0.70},
    "hardBlockRuleCodes": [
        "PI_DATA_EXFILTRATION",
        "PI_TOOL_ABUSE_CRITICAL",
        "PI_CODE_EXECUTION_CRITICAL",
        "PI_CYBER_ABUSE_CRITICAL",
    ],
    "llm": {"callOnSafe": True, "callOnWarning": False, "callOnBlocked": False},
    "securityModel": {
        "provider": "roberta",
        "modelName": "roberta_v4",
        "modelPath": "models/transformers/roberta_v4",
        "tokenizerPath": "models/transformers/roberta_v4",
        "modelVersion": "roberta-v4-neuralchemy",
        "maxLength": 128,
        "batchSize": 8,
        "device": "auto",
        "eagerLoad": True,
        "thresholdVersion": "roberta-v4-artifact-v1",
        "evaluationThreshold": 0.02,
        "inputWarnThreshold": 0.30,
        "inputBlockThreshold": 0.45,
        "outputWarnThreshold": 0.30,
        "outputBlockThreshold": 0.85,
        "outputThresholdStatus": "baseline_not_optimized",
        "calibratorEnabled": False,
        "calibratorPath": None,
        "calibratorVersion": "disabled_incompatible_provenance",
    },
}


def _camelize_security_model(raw: dict[str, Any]) -> dict[str, Any]:
    mapping = {
        "model_name": "modelName",
        "model_path": "modelPath",
        "tokenizer_path": "tokenizerPath",
        "model_version": "modelVersion",
        "training_dataset_version": "trainingDatasetVersion",
        "evaluation_dataset_version": "evaluationDatasetVersion",
        "max_length": "maxLength",
        "batch_size": "batchSize",
        "eager_load": "eagerLoad",
        "threshold_version": "thresholdVersion",
        "evaluation_threshold": "evaluationThreshold",
        "input_warn_threshold": "inputWarnThreshold",
        "input_block_threshold": "inputBlockThreshold",
        "output_warn_threshold": "outputWarnThreshold",
        "output_block_threshold": "outputBlockThreshold",
        "output_threshold_status": "outputThresholdStatus",
        "calibrator_enabled": "calibratorEnabled",
        "calibrator_path": "calibratorPath",
        "calibrator_version": "calibratorVersion",
        "active_input": "activeInput",
        "active_output": "activeOutput",
    }
    return {mapping.get(key, key): value for key, value in raw.items()}


def _load_payload(config_path: Path) -> dict[str, Any]:
    text = config_path.read_text(encoding="utf-8-sig")
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError(f"Runtime config must be an object: {config_path}")
    return payload


def _normalize_yaml_config(raw: dict[str, Any]) -> dict[str, Any]:
    if "security_model" not in raw:
        return raw
    model = _camelize_security_model(dict(raw.get("security_model") or {}))
    versions = dict(raw.get("versions") or {})
    input_security = dict(raw.get("input_security") or {})
    thresholds = dict(input_security.get("thresholds") or {})
    weight_sets = dict(input_security.get("weights") or {})
    chat_weights = dict(weight_sets.get("chat") or {})
    llm_raw = dict(raw.get("llm") or {})
    return {
        **raw,
        "schemaVersion": raw.get("schema_version", DEFAULT_RUNTIME_CONFIG["schemaVersion"]),
        "policyVersion": versions.get("policy", DEFAULT_RUNTIME_CONFIG["policyVersion"]),
        "ruleVersion": versions.get("rules", DEFAULT_RUNTIME_CONFIG["ruleVersion"]),
        "runtimeModel": model.get("modelName", DEFAULT_RUNTIME_CONFIG["runtimeModel"]),
        "modelVersion": model.get("modelVersion", DEFAULT_RUNTIME_CONFIG["modelVersion"]),
        "securityModel": model,
        "weights": {
            "rule": chat_weights.get("rule", DEFAULT_RUNTIME_CONFIG["weights"]["rule"]),
            "roberta": chat_weights.get("roberta", DEFAULT_RUNTIME_CONFIG["weights"]["roberta"]),
            "context": chat_weights.get("context", DEFAULT_RUNTIME_CONFIG["weights"]["context"]),
        },
        "thresholds": {
            "warn": thresholds.get("warn", DEFAULT_RUNTIME_CONFIG["thresholds"]["warn"]),
            "block": thresholds.get("block", DEFAULT_RUNTIME_CONFIG["thresholds"]["block"]),
        },
        "hardBlockRuleCodes": input_security.get(
            "hard_block_rule_codes", DEFAULT_RUNTIME_CONFIG["hardBlockRuleCodes"]
        ),
        "llm": {
            "callOnSafe": llm_raw.get("call_on_safe", True),
            "callOnWarning": llm_raw.get("call_on_warning", False),
            "callOnBlocked": llm_raw.get("call_on_blocked", False),
            "maxRegenerations": int((raw.get("output_security") or {}).get("max_regenerations", 1)),
        },
    }


def _merge_config(raw: dict[str, Any]) -> dict[str, Any]:
    raw = _normalize_yaml_config(raw)
    merged = {**DEFAULT_RUNTIME_CONFIG, **raw}
    merged["weights"] = {**DEFAULT_RUNTIME_CONFIG["weights"], **raw.get("weights", {})}
    merged["thresholds"] = {**DEFAULT_RUNTIME_CONFIG["thresholds"], **raw.get("thresholds", {})}
    merged["llm"] = {**DEFAULT_RUNTIME_CONFIG["llm"], **raw.get("llm", {})}
    merged["securityModel"] = {
        **DEFAULT_RUNTIME_CONFIG["securityModel"],
        **raw.get("securityModel", {}),
    }
    validate_runtime_config(merged)
    return merged


def validate_runtime_config(config: dict[str, Any]) -> None:
    weights = config.get("weights", {})
    thresholds = config.get("thresholds", {})
    for key in ["rule", "roberta", "context"]:
        value = float(weights.get(key, 0.0))
        if value < 0.0:
            raise ValueError(f"Runtime weight '{key}' must be >= 0.")
    total = sum(float(weights.get(key, 0.0)) for key in ["rule", "roberta", "context"])
    if total <= 0:
        raise ValueError("At least one runtime weight must be positive.")

    warn = float(thresholds.get("warn", 0.30))
    block = float(thresholds.get("block", 0.70))
    if not 0.0 <= warn <= 1.0:
        raise ValueError("Warn threshold must be within 0..1.")
    if not 0.0 <= block <= 1.0:
        raise ValueError("Block threshold must be within 0..1.")
    if warn >= block:
        raise ValueError("Warn threshold must be lower than block threshold.")

    model = config.get("securityModel", {})
    for key in [
        "evaluationThreshold",
        "inputWarnThreshold",
        "inputBlockThreshold",
        "outputWarnThreshold",
        "outputBlockThreshold",
    ]:
        value = float(model.get(key, 0.0))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Security model '{key}' must be within 0..1.")
    model_path = str(model.get("modelPath", "")).strip()
    tokenizer_path = str(model.get("tokenizerPath", "")).strip()
    if not model_path or not tokenizer_path:
        raise ValueError("Security model path and tokenizer path are required.")


@lru_cache(maxsize=1)
def load_runtime_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_RUNTIME_CONFIG_PATH
    if not config_path.exists():
        config_path = LEGACY_RUNTIME_CONFIG_PATH
    if not config_path.exists():
        return _merge_config({})
    raw = _load_payload(config_path)
    return _merge_config(raw)


def project_path(value: str | Path | None) -> Path:
    """Resolve a configured path without silently searching another checkpoint."""
    path = Path(str(value or ""))
    return path if path.is_absolute() else PROJECT_ROOT / path


def security_model_config() -> dict[str, Any]:
    return dict(load_runtime_config().get("securityModel", {}))


def clear_runtime_config_cache() -> None:
    load_runtime_config.cache_clear()
