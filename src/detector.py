"""Hybrid runtime detector for prompt injection prompts."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import joblib

from src.benign_intent import detect_benign_reference_intent
from src.indirect_rule_detector import detect_indirect_by_rules
from src.preprocessing import prepare_text_for_detection
from src.rule_based import detect_by_rules


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
INDIRECT_MODELS_DIR = MODELS_DIR / "indirect"
THRESHOLDS_FILE = MODELS_DIR / "model_thresholds.json"
RUNTIME_WARN_THRESHOLD = 0.50
RUNTIME_BLOCK_THRESHOLD = 0.80
DEFAULT_THRESHOLDS = {
    "evaluation_threshold": 0.50,
    "runtime_warn_threshold": RUNTIME_WARN_THRESHOLD,
    "runtime_block_threshold": RUNTIME_BLOCK_THRESHOLD,
}

MODEL_FILES: dict[str, dict[str, Path]] = {
    "logistic_regression": {
        "model": MODELS_DIR / "logistic_regression_model.joblib",
        "vectorizer": MODELS_DIR / "logistic_regression_vectorizer.joblib",
    },
    "linear_svm": {
        "model": MODELS_DIR / "linear_svm_model.joblib",
        "vectorizer": MODELS_DIR / "linear_svm_vectorizer.joblib",
    },
    "random_forest": {
        "model": MODELS_DIR / "random_forest_model.joblib",
        "vectorizer": MODELS_DIR / "random_forest_vectorizer.joblib",
    },
}

INDIRECT_MODEL_FILES: dict[str, dict[str, Path]] = {
    "logistic_regression": {
        "model": INDIRECT_MODELS_DIR / "logistic_regression_model.joblib",
        "vectorizer": INDIRECT_MODELS_DIR / "logistic_regression_vectorizer.joblib",
    },
    "linear_svm": {
        "model": INDIRECT_MODELS_DIR / "linear_svm_model.joblib",
        "vectorizer": INDIRECT_MODELS_DIR / "linear_svm_vectorizer.joblib",
    },
}

VALID_MODEL_TYPES = {"logistic_regression", "linear_svm", "random_forest", "hybrid"}


def action_from_score(
    risk_score: float,
    warn_threshold: float = RUNTIME_WARN_THRESHOLD,
    block_threshold: float = RUNTIME_BLOCK_THRESHOLD,
) -> str:
    if risk_score >= block_threshold:
        return "block"
    if risk_score >= warn_threshold:
        return "warn"
    return "allow"


def model_files_status() -> dict[str, dict[str, bool]]:
    status = {
        model_type: {
            "model_found": paths["model"].exists(),
            "vectorizer_found": paths["vectorizer"].exists(),
            "thresholds_found": THRESHOLDS_FILE.exists(),
        }
        for model_type, paths in MODEL_FILES.items()
    }
    status.update(
        {
            f"indirect_{model_type}": {
                "model_found": paths["model"].exists(),
                "vectorizer_found": paths["vectorizer"].exists(),
                "thresholds_found": True,
            }
            for model_type, paths in INDIRECT_MODEL_FILES.items()
        }
    )
    return status


@lru_cache(maxsize=1)
def load_thresholds() -> dict[str, dict[str, float]]:
    if not THRESHOLDS_FILE.exists():
        return {
            model_type: DEFAULT_THRESHOLDS.copy()
            for model_type in MODEL_FILES
        }

    raw_thresholds = json.loads(THRESHOLDS_FILE.read_text(encoding="utf-8"))
    thresholds: dict[str, dict[str, float]] = {}
    for model_type in MODEL_FILES:
        model_thresholds = raw_thresholds.get(model_type, {})
        thresholds[model_type] = {
            "evaluation_threshold": float(
                model_thresholds.get("evaluation_threshold", DEFAULT_THRESHOLDS["evaluation_threshold"])
            ),
            "runtime_warn_threshold": float(
                model_thresholds.get("runtime_warn_threshold", DEFAULT_THRESHOLDS["runtime_warn_threshold"])
            ),
            "runtime_block_threshold": float(
                model_thresholds.get("runtime_block_threshold", DEFAULT_THRESHOLDS["runtime_block_threshold"])
            ),
        }
    return thresholds


def _missing_model_response(model_type: str) -> dict[str, Any]:
    return {
        "error": True,
        "message": (
            f"Model '{model_type}' chưa được train hoặc thiếu file. "
            "Hãy chạy: python -m src.train_models"
        ),
        "available": False,
        "action": "model_not_ready",
        "risk_score": None,
        "model_path": str(MODEL_FILES.get(model_type, {}).get("model", "")),
        "vectorizer_path": str(MODEL_FILES.get(model_type, {}).get("vectorizer", "")),
        "missing_files": {
            name: str(path)
            for name, path in MODEL_FILES.get(model_type, {}).items()
            if not path.exists()
        },
    }


@lru_cache(maxsize=4)
def load_model_artifacts(model_type: str) -> tuple[Any, Any]:
    if model_type not in MODEL_FILES:
        raise ValueError(f"model_type không hợp lệ: {model_type}")

    paths = MODEL_FILES[model_type]
    if not paths["model"].exists() or not paths["vectorizer"].exists():
        missing = ", ".join(str(path) for path in paths.values() if not path.exists())
        raise FileNotFoundError(missing)

    model = joblib.load(paths["model"])
    vectorizer = joblib.load(paths["vectorizer"])
    return model, vectorizer


@lru_cache(maxsize=4)
def load_indirect_model_artifacts(model_type: str) -> tuple[Any, Any]:
    if model_type not in INDIRECT_MODEL_FILES:
        raise ValueError(f"indirect model_type không hợp lệ: {model_type}")

    paths = INDIRECT_MODEL_FILES[model_type]
    if not paths["model"].exists() or not paths["vectorizer"].exists():
        missing = ", ".join(str(path) for path in paths.values() if not path.exists())
        raise FileNotFoundError(missing)

    model = joblib.load(paths["model"])
    vectorizer = joblib.load(paths["vectorizer"])
    return model, vectorizer


def _probability_for_positive_class(model: Any, vectorized_text: Any) -> float:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vectorized_text)[0]
        classes = list(getattr(model, "classes_", [0, 1]))
        positive_index = classes.index(1) if 1 in classes else 1
        return float(probabilities[positive_index])

    if hasattr(model, "decision_function"):
        import math

        score = float(model.decision_function(vectorized_text)[0])
        return 1.0 / (1.0 + math.exp(-score))

    prediction = int(model.predict(vectorized_text)[0])
    return float(prediction)


def _probabilities_for_binary_model(model: Any, vectorized_text: Any) -> tuple[float, float]:
    risk_score = max(0.0, min(1.0, _probability_for_positive_class(model, vectorized_text)))
    return 1.0 - risk_score, risk_score


def _raw_model_score(model: Any, vectorized_text: Any) -> float | None:
    if not hasattr(model, "decision_function"):
        return None
    raw_score = model.decision_function(vectorized_text)
    try:
        return float(raw_score[0])
    except (TypeError, IndexError, ValueError):
        return float(raw_score)


def detect_by_ml(
    text: str,
    model_type: str = "logistic_regression",
    decision_mode: str = "runtime",
) -> dict[str, Any]:
    """Run a trained ML model and return label, risk score and action."""
    if model_type not in MODEL_FILES:
        return {
            "error": True,
            "message": "model_type phải là logistic_regression, linear_svm hoặc random_forest.",
        }
    if decision_mode not in {"runtime", "evaluation"}:
        return {
            "error": True,
            "message": "decision_mode phải là runtime hoặc evaluation.",
        }

    try:
        model, vectorizer = load_model_artifacts(model_type)
    except FileNotFoundError:
        return _missing_model_response(model_type)

    prepared_text = prepare_text_for_detection(text)
    cleaned = prepared_text["cleaned_text"]
    vectorized = vectorizer.transform([cleaned])
    safe_probability, risk_score = _probabilities_for_binary_model(model, vectorized)
    raw_risk_score = risk_score
    benign_guard = detect_benign_reference_intent(text)
    # Benign guard is an explanatory signal only; compare/inference must expose the true model score.
    confidence = max(safe_probability, risk_score)
    raw_score = _raw_model_score(model, vectorized)
    thresholds = load_thresholds().get(model_type, DEFAULT_THRESHOLDS)
    evaluation_threshold = float(thresholds["evaluation_threshold"])
    warn_threshold = float(thresholds.get("runtime_warn_threshold", DEFAULT_THRESHOLDS["runtime_warn_threshold"]))
    block_threshold = float(thresholds.get("runtime_block_threshold", DEFAULT_THRESHOLDS["runtime_block_threshold"]))
    evaluation_label = 1 if risk_score >= evaluation_threshold else 0
    action = action_from_score(risk_score, warn_threshold, block_threshold)
    ml_label = 1 if action in {"warn", "block"} else 0

    return {
        "label": ml_label,
        "evaluation_label": evaluation_label,
        "risk_score": round(float(risk_score), 4),
        "raw_risk_score": round(float(raw_risk_score), 4),
        "confidence": round(float(confidence), 4),
        "probabilities": {
            "safe": round(float(safe_probability), 6),
            "injection": round(float(risk_score), 6),
        },
        "raw_score": None if raw_score is None else round(float(raw_score), 6),
        "model_path": str(MODEL_FILES[model_type]["model"]),
        "vectorizer_path": str(MODEL_FILES[model_type]["vectorizer"]),
        "benign_guard": benign_guard,
        "action": action,
        "method": f"tfidf_{model_type}",
        "decision_mode": decision_mode,
        "original_text": prepared_text["original_text"],
        "detected_language": prepared_text["detected_language"],
        "canonical_text": prepared_text["cleaned_text"],
        "thresholds": {
            "evaluation_threshold": evaluation_threshold,
            "runtime_warn_threshold": warn_threshold,
            "runtime_block_threshold": block_threshold,
        },
        "explanation": (
            f"ML model '{model_type}' dự đoán risk_score={risk_score:.4f}. "
            f"Evaluation threshold={evaluation_threshold:.4f}; "
            f"Runtime thresholds: warn>={warn_threshold:.4f}, block>={block_threshold:.4f}; "
            f"confidence={confidence:.4f}; "
            f"decision_mode={decision_mode}; "
            f"hành động đề xuất: {action}."
        ),
    }


def _combined_context_text(user_prompt: str, context: str | None) -> tuple[str, dict[str, str], dict[str, str]]:
    prepared_user = prepare_text_for_detection(user_prompt)
    prepared_context = prepare_text_for_detection("" if context is None else context)
    combined = (
        f"USER_INTENT: {prepared_user['cleaned_text']}\n"
        f"CONTEXT: {prepared_context['cleaned_text']}"
    )
    return combined, prepared_user, prepared_context


def detect_indirect_by_ml(
    user_prompt: str,
    context: str | None = None,
    model_type: str = "logistic_regression",
) -> dict[str, Any]:
    """Run the optional indirect ML detector if models/indirect exists."""
    if model_type not in INDIRECT_MODEL_FILES:
        return {
            "available": False,
            "message": "indirect model_type phải là logistic_regression hoặc linear_svm.",
            "risk_score": 0.0,
            "action": "allow",
        }

    try:
        model, vectorizer = load_indirect_model_artifacts(model_type)
    except FileNotFoundError:
        return {
            "available": False,
            "message": (
                f"Indirect model '{model_type}' chưa được train. "
                "Hãy chạy: python training/train_indirect_detector.py"
            ),
            "risk_score": 0.0,
            "action": "allow",
            "missing_files": {
                name: str(path)
                for name, path in INDIRECT_MODEL_FILES[model_type].items()
                if not path.exists()
            },
        }

    combined_text, prepared_user, prepared_context = _combined_context_text(user_prompt, context)
    vectorized = vectorizer.transform([combined_text])
    risk_score = _probability_for_positive_class(model, vectorized)
    action = action_from_score(risk_score)

    return {
        "available": True,
        "label": 1 if action in {"warn", "block"} else 0,
        "risk_score": round(float(risk_score), 4),
        "action": action,
        "method": f"tfidf_indirect_{model_type}",
        "canonical_combined_text": combined_text,
        "user_detected_language": prepared_user["detected_language"],
        "context_detected_language": prepared_context["detected_language"],
        "explanation": (
            f"Indirect ML model '{model_type}' scored context risk={risk_score:.4f}; "
            f"thresholds: warn>={RUNTIME_WARN_THRESHOLD:.4f}, block>={RUNTIME_BLOCK_THRESHOLD:.4f}."
        ),
    }


def _final_response(
    text: str,
    prepared_text: dict[str, str],
    risk_score: float,
    action: str,
    method: str,
    rule_based_result: dict[str, Any],
    ml_result: dict[str, Any] | None,
    explanation: str,
    label: int | None = None,
) -> dict[str, Any]:
    return {
        "input": text,
        "original_text": prepared_text["original_text"],
        "detected_language": prepared_text["detected_language"],
        "canonical_text": prepared_text["cleaned_text"],
        "label": int(label) if label is not None else (1 if action in {"warn", "block"} else 0),
        "risk_score": round(float(risk_score), 4),
        "action": action,
        "method": method,
        "rule_based_result": rule_based_result,
        "ml_result": ml_result,
        "explanation": explanation,
    }


def detect_prompt(text: str, model_type: str = "hybrid") -> dict[str, Any]:
    """Detect prompt injection using ML-only modes or the hybrid detector."""
    if text is None or not str(text).strip():
        return {
            "error": True,
            "message": "Input text không được rỗng.",
        }

    if model_type not in VALID_MODEL_TYPES:
        return {
            "error": True,
            "message": "model_type phải là logistic_regression, linear_svm, random_forest hoặc hybrid.",
        }

    prepared_text = prepare_text_for_detection(text)
    rule_result = detect_by_rules(text)

    if model_type in {"logistic_regression", "linear_svm", "random_forest"}:
        ml_result = detect_by_ml(text, model_type=model_type, decision_mode="evaluation")
        if ml_result.get("error"):
            return ml_result
        return _final_response(
            text=text,
            prepared_text=prepared_text,
            risk_score=float(ml_result["risk_score"]),
            action=str(ml_result["action"]),
            method=str(ml_result["method"]),
            rule_based_result=rule_result,
            ml_result=ml_result,
            explanation=str(ml_result["explanation"]),
            label=int(ml_result["label"]),
        )

    rule_score = float(rule_result["risk_score"])
    if rule_score >= RUNTIME_BLOCK_THRESHOLD:
        return _final_response(
            text=text,
            prepared_text=prepared_text,
            risk_score=rule_score,
            action="block",
            method="hybrid_rule_based",
            rule_based_result=rule_result,
            ml_result=None,
            explanation=(
                "Prompt bị block ngay vì rule-based phát hiện dấu hiệu nguy hiểm rõ ràng. "
                "Không cần gọi ML trong trường hợp này."
            ),
        )

    ml_result = detect_by_ml(text, model_type="logistic_regression")
    if ml_result.get("error"):
        return ml_result

    ml_score = float(ml_result["risk_score"])
    thresholds = ml_result.get("thresholds", DEFAULT_THRESHOLDS)
    warn_threshold = float(thresholds.get("runtime_warn_threshold", 0.80))
    block_threshold = float(thresholds.get("runtime_block_threshold", 0.90))

    if rule_score >= RUNTIME_WARN_THRESHOLD:
        final_score = max(rule_score, ml_score)
        final_action = "block" if ml_score >= block_threshold else "warn"
    else:
        final_score = ml_score
        final_action = action_from_score(ml_score, warn_threshold, block_threshold)

    if rule_score == 0.0 and ml_score < warn_threshold:
        final_action = "allow"
        final_score = ml_score
    elif 0.0 < rule_score < RUNTIME_WARN_THRESHOLD and ml_score < warn_threshold:
        final_action = "allow"
        final_score = max(rule_score, ml_score)

    if rule_score >= RUNTIME_WARN_THRESHOLD and final_action == "allow":
        final_action = "warn"
        final_score = max(final_score, RUNTIME_WARN_THRESHOLD)

    return _final_response(
        text=text,
        prepared_text=prepared_text,
        risk_score=final_score,
        action=final_action,
        method="hybrid_rule_based_logistic_regression",
        rule_based_result=rule_result,
        ml_result=ml_result,
        explanation=(
            "Hybrid detector kết hợp rule-based và Logistic Regression. "
            f"Rule score={rule_score:.4f}, ML score={ml_score:.4f}; "
            f"ML warn threshold={warn_threshold:.4f}, block threshold={block_threshold:.4f}; "
            f"hành động cuối cùng: {final_action}."
        ),
    )


def _risk_from_result(result: dict[str, Any]) -> float:
    try:
        return float(result.get("risk_score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _is_positive_result(result: dict[str, Any], warn_threshold: float = RUNTIME_WARN_THRESHOLD) -> bool:
    action = str(result.get("action", "allow"))
    return action in {"warn", "block"} or _risk_from_result(result) >= warn_threshold


def _combined_attack_type(
    direct_rule_result: dict[str, Any],
    direct_ml_result: dict[str, Any],
    indirect_rule_result: dict[str, Any],
    indirect_ml_result: dict[str, Any],
) -> str:
    direct_positive = _is_positive_result(direct_rule_result) or str(direct_ml_result.get("action")) in {"warn", "block"}
    indirect_positive = _is_positive_result(indirect_rule_result) or str(indirect_ml_result.get("action")) in {"warn", "block"}
    if direct_positive and indirect_positive:
        return "mixed"
    if direct_positive:
        return "direct"
    if indirect_positive:
        return "indirect"
    return "none"


def detect_prompt_with_context(
    user_prompt: str,
    context: str | None = None,
    model_type: str = "hybrid",
) -> dict[str, Any]:
    """Context-aware detector for direct and indirect prompt injection."""
    if user_prompt is None or not str(user_prompt).strip():
        return {
            "error": True,
            "message": "user_prompt không được rỗng.",
        }
    if model_type not in VALID_MODEL_TYPES:
        return {
            "error": True,
            "message": "model_type phải là logistic_regression, linear_svm, random_forest hoặc hybrid.",
        }

    context_text = "" if context is None else str(context)
    combined_text, prepared_user, prepared_context = _combined_context_text(str(user_prompt), context_text)

    direct_rule_result = detect_by_rules(str(user_prompt))
    direct_ml_result = detect_by_ml(str(user_prompt), model_type="logistic_regression")
    if direct_ml_result.get("error"):
        direct_ml_result = {"available": False, "risk_score": 0.0, "action": "allow", **direct_ml_result}

    indirect_rule_result = detect_indirect_by_rules(str(user_prompt), context_text)
    indirect_ml_result = detect_indirect_by_ml(str(user_prompt), context_text, model_type="logistic_regression")

    direct_ml_decision_score = (
        _risk_from_result(direct_ml_result)
        if str(direct_ml_result.get("action")) in {"warn", "block"}
        else 0.0
    )
    indirect_ml_decision_score = (
        _risk_from_result(indirect_ml_result)
        if str(indirect_ml_result.get("action")) in {"warn", "block"}
        else 0.0
    )
    scores = [
        _risk_from_result(direct_rule_result),
        direct_ml_decision_score,
        _risk_from_result(indirect_rule_result),
        indirect_ml_decision_score,
    ]
    final_score = max(scores)
    strong_rule = (
        _risk_from_result(direct_rule_result) >= RUNTIME_BLOCK_THRESHOLD
        or _risk_from_result(indirect_rule_result) >= RUNTIME_BLOCK_THRESHOLD
    )

    if strong_rule or final_score >= RUNTIME_BLOCK_THRESHOLD:
        action = "block"
    elif final_score >= RUNTIME_WARN_THRESHOLD:
        action = "warn"
    else:
        action = "allow"

    attack_type = _combined_attack_type(
        direct_rule_result,
        direct_ml_result,
        indirect_rule_result,
        indirect_ml_result,
    )
    if action == "allow":
        attack_type = "none"

    matched_rules = []
    for source, result in [
        ("direct", direct_rule_result),
        ("indirect", indirect_rule_result),
    ]:
        for rule in result.get("matched_rules", []):
            matched_rules.append({"source": source, **rule})

    if attack_type == "indirect":
        explanation = "Suspicious instructions were found in the external context."
    elif attack_type == "direct":
        explanation = "Suspicious instructions were found in the user prompt."
    elif attack_type == "mixed":
        explanation = "Suspicious instructions were found in both user prompt and external context."
    else:
        explanation = "No direct or indirect prompt injection signal exceeded the decision threshold."

    return {
        "action": action,
        "attack_type": attack_type,
        "risk_score": round(float(final_score), 4),
        "detected_language": prepared_context["detected_language"] or prepared_user["detected_language"],
        "language_result": {
            "user_prompt": prepared_user["detected_language"],
            "context": prepared_context["detected_language"],
        },
        "canonical_user_prompt": prepared_user["cleaned_text"],
        "canonical_context": prepared_context["cleaned_text"],
        "canonical_combined_text": combined_text,
        "matched_rules": matched_rules,
        "rule_result": {
            "direct": direct_rule_result,
            "indirect": indirect_rule_result,
        },
        "ml_result": {
            "direct": direct_ml_result,
            "indirect": indirect_ml_result,
        },
        "explanation": explanation,
    }
