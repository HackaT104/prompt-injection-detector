"""Runtime helpers for post-hoc calibrated probabilities and thresholds.

The calibration artifacts in this module are separate from training checkpoints.
They are produced from strict validation/test direct external benchmark runs:

- Fit calibrator on validation split only.
- Select threshold on validation split only.
- Apply both to held-out test split.

Runtime may use these artifacts for clearer probability/threshold semantics, but
evaluation scripts can still operate on raw scores when needed.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
CALIBRATED_THRESHOLDS_PATH = MODELS_DIR / "calibrated_thresholds.json"
RUNTIME_CALIBRATION_DIR = MODELS_DIR / "calibration"


MODEL_ALIASES = {
    "logistic-regression": "logistic_regression",
    "logistic_regression": "logistic_regression",
    "random-forest": "random_forest",
    "random_forest": "random_forest",
    "roberta": "roberta",
    "roberta-base": "roberta",
    "roberta_v3": "roberta",
    "roberta_v4": "roberta",
    "roberta_v5_vi": "roberta",
    "xlm-roberta": "xlm_roberta",
    "xlm_roberta": "xlm_roberta",
    "xlm-roberta-base": "xlm_roberta",
    "xlm_roberta_v3": "xlm_roberta",
    "xlm_roberta_v4": "xlm_roberta",
    "xlm_roberta_v5_vi": "xlm_roberta",
}


def canonical_model_key(model_key: str | None) -> str:
    value = str(model_key or "").strip().lower()
    return MODEL_ALIASES.get(value, value)


@lru_cache(maxsize=1)
def load_calibrated_thresholds() -> dict[str, Any]:
    if not CALIBRATED_THRESHOLDS_PATH.exists():
        return {}
    try:
        return json.loads(CALIBRATED_THRESHOLDS_PATH.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}


def get_calibrated_threshold_entry(model_key: str | None) -> dict[str, Any] | None:
    canonical = canonical_model_key(model_key)
    payload = load_calibrated_thresholds()
    entry = payload.get(canonical)
    return entry if isinstance(entry, dict) else None


def runtime_calibrator_path(model_key: str | None, dataset: str = "direct_all") -> Path:
    canonical = canonical_model_key(model_key)
    return RUNTIME_CALIBRATION_DIR / dataset / canonical / "probability_calibrator.joblib"


@lru_cache(maxsize=16)
def load_runtime_calibrator(model_key: str | None, dataset: str = "direct_all") -> Any | None:
    path = runtime_calibrator_path(model_key, dataset)
    if not path.exists():
        return None
    return joblib.load(path)


def apply_probability_calibrator(calibrator: Any, raw_score: float) -> float:
    if hasattr(calibrator, "predict_proba"):
        value = calibrator.predict_proba([[float(raw_score)]])[0][1]
    elif hasattr(calibrator, "predict"):
        value = calibrator.predict([float(raw_score)])[0]
    else:
        raise TypeError(f"Unsupported calibrator type: {type(calibrator).__name__}")
    return float(min(1.0, max(0.0, value)))


def clear_runtime_calibration_cache() -> None:
    load_calibrated_thresholds.cache_clear()
    load_runtime_calibrator.cache_clear()
