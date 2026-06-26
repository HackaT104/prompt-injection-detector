"""Advanced configurable detector used by the demo UI."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.detector import (
    MODEL_FILES,
    RUNTIME_BLOCK_THRESHOLD,
    RUNTIME_WARN_THRESHOLD,
    action_from_score,
    detect_by_ml,
    load_model_artifacts,
)
from src.preprocessing import prepare_text_for_detection
from src.rule_based import detect_by_rules
from src.transformer_utils import (
    DEFAULT_TRANSFORMER_DATASET_CONFIG,
    clear_transformer_runtime_cache,
    diagnose_transformer,
    is_finetuned_transformer_checkpoint,
    predict_transformer,
    resolve_transformer_model_dir,
    resolve_transformer_model_name,
    safe_model_dir_name,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

TRADITIONAL_MODEL_ALIASES = {
    "all": "all",
    "all_traditional": "all",
    "logistic": "logistic_regression",
    "logistic_regression": "logistic_regression",
    "svm": "linear_svm",
    "linear_svm": "linear_svm",
    "random_forest": "random_forest",
}
TRADITIONAL_MODEL_TYPES = {"logistic_regression", "linear_svm", "random_forest"}

TRANSFORMER_MODEL_ALIASES = {
    "distilbert": "distilbert-base-uncased",
    "distilbert-base-uncased": "distilbert-base-uncased",
    "distilbert_v2": "distilbert_v2",
    "distilbert-v2": "distilbert_v2",
    "distilbert_v3": "distilbert_v3",
    "distilbert-v3": "distilbert_v3",
    "roberta": "roberta-base",
    "roberta-base": "roberta-base",
    "roberta_v2": "roberta_v2",
    "roberta-v2": "roberta_v2",
    "roberta_v3": "roberta_v3",
    "roberta-v3": "roberta_v3",
    "roberta_v4": "roberta_v4",
    "roberta-v4": "roberta_v4",
    "roberta_v5_vi": "roberta_v5_vi",
    "roberta-v5-vi": "roberta_v5_vi",
    "xlm_roberta": "xlm-roberta-base",
    "xlm-roberta": "xlm-roberta-base",
    "xlm-roberta-base": "xlm-roberta-base",
    "xlm_roberta_v3": "xlm_roberta_v3",
    "xlm-roberta-v3": "xlm_roberta_v3",
    "xlm_roberta_v4": "xlm_roberta_v4",
    "xlm-roberta-v4": "xlm_roberta_v4",
    "xlm_roberta_v5_vi": "xlm_roberta_v5_vi",
    "xlm-roberta-v5-vi": "xlm_roberta_v5_vi",
}

ADVANCED_MODEL_ALIASES = {
    **TRADITIONAL_MODEL_ALIASES,
    **TRANSFORMER_MODEL_ALIASES,
    "hybrid": "hybrid",
}

TEXT_FILE_EXTENSIONS = {".txt", ".csv", ".json", ".md"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
COMPARISON_MODELS = [
    "logistic_regression",
    "linear_svm",
    "random_forest",
    "roberta",
    "xlm_roberta",
    "hybrid",
]
SUSPICIOUS_PHRASES = {
    "instruction_override": [
        "ignore previous instructions",
        "disregard previous instructions",
        "forget all prior instructions",
        "override instruction",
        "ignore system prompt",
    ],
    "prompt_leakage": [
        "reveal system prompt",
        "show system prompt",
        "print hidden instructions",
        "disclose internal instructions",
    ],
    "jailbreak": [
        "act as dan",
        "developer mode",
        "bypass safety",
        "unrestricted mode",
        "no ethical restrictions",
    ],
    "secret_leakage": [
        "api key",
        "print token",
        "expose credentials",
        "reveal secret",
        "leak data",
    ],
    "code_execution": [
        "os.system",
        "subprocess",
        "eval(",
        "exec(",
        "rm -rf",
        "whoami",
    ],
}


def _normalize_model_name(model_name: str | None) -> str:
    value = (model_name or "hybrid").strip().lower()
    return ADVANCED_MODEL_ALIASES.get(value, value)


def _normalize_traditional_model(model_name: str | None) -> str:
    value = (model_name or "all").strip().lower()
    return TRADITIONAL_MODEL_ALIASES.get(value, value)


def _normalize_transformer_model(model_name: str | None) -> str:
    value = (model_name or "roberta").strip().lower()
    return TRANSFORMER_MODEL_ALIASES.get(value, value)


def _is_transformer_model_name(model_name: str) -> bool:
    value = str(model_name or "").strip().lower()
    if value in TRANSFORMER_MODEL_ALIASES.values():
        return True
    try:
        return is_finetuned_transformer_checkpoint(resolve_transformer_model_dir(value))
    except Exception:
        return False


def _compact_rule_signal(result: dict[str, Any] | None, enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {
            "enabled": False,
            "triggered": False,
            "score": 0.0,
            "action": "skipped",
            "matched_rules": [],
        }
    if not result:
        return {
            "enabled": True,
            "triggered": False,
            "score": 0.0,
            "action": "allow",
            "matched_rules": [],
        }
    score = float(result.get("risk_score", 0.0) or 0.0)
    return {
        "enabled": True,
        "triggered": bool(result.get("label") == 1 or score > 0),
        "score": round(score, 4),
        "action": result.get("action", "allow"),
        "matched_rules": result.get("matched_rules", []),
        "benign_guard": result.get("benign_guard"),
    }


def _compact_ml_signal(result: dict[str, Any] | None, enabled: bool, model_name: str) -> dict[str, Any]:
    if not enabled:
        return {
            "enabled": False,
            "available": False,
            "model": model_name,
            "score": 0.0,
            "action": "skipped",
        }
    if not result:
        return {
            "enabled": True,
            "available": False,
            "model": model_name,
            "score": 0.0,
            "action": "unavailable",
        }

    raw_score = result.get("risk_score")
    score = None if raw_score is None else float(raw_score or 0.0)
    return {
        "enabled": True,
        "available": not bool(result.get("error")) and result.get("available", True) is not False,
        "model": result.get("method", model_name),
        "predicted_label": result.get("label", result.get("evaluation_label")),
        "score": None if score is None else round(score, 4),
        "confidence": result.get("confidence"),
        "probabilities": result.get("probabilities"),
        "raw_score": result.get("raw_score"),
        "raw_risk_score": result.get("raw_risk_score"),
        "raw_probabilities": result.get("raw_probabilities"),
        "benign_guard": result.get("benign_guard"),
        "action": result.get("action", result.get("runtime_action", "allow")),
        "thresholds": result.get("thresholds"),
        "message": result.get("message"),
        "error": result.get("message") if result.get("error") else None,
        "model_path": result.get("model_path") or result.get("missing_path"),
    }


def _missing_transformer_response(model_name: str) -> dict[str, Any]:
    model_dir = resolve_transformer_model_dir(model_name)
    safe_name = model_dir.name
    return {
        "available": False,
        "error": True,
        "method": f"transformer_{safe_name}",
        "label": None,
        "risk_score": None,
        "confidence": None,
        "action": "model_not_ready",
        "message": (
            f"Transformer model '{model_name}' chÆ°a Ä‘Æ°á»£c train hoáº·c thiáº¿u file. "
            f"HÃ£y cháº¡y: python src/train_transformers.py --model {model_name} --dataset-config core"
        ),
        "missing_path": str(model_dir),
        "model_path": str(model_dir),
    }


def _run_transformer(text: str, transformer_model: str, use_cuda: bool = True) -> dict[str, Any]:
    try:
        model_name = resolve_transformer_model_name(transformer_model)
    except ValueError:
        model_name = transformer_model
    model_dir = resolve_transformer_model_dir(transformer_model)
    runtime_name = model_dir.name
    if not is_finetuned_transformer_checkpoint(model_dir):
        return _missing_transformer_response(model_name)

    try:
        result = predict_transformer(
            text=text,
            model_path=model_dir,
            model_name=model_name,
            max_length=128,
            use_cuda=use_cuda,
        )
    except Exception as exc:  # keep API stable when optional deep-learning stack fails
        return {
            "available": False,
            "error": True,
            "method": f"transformer_{runtime_name}",
            "label": None,
            "risk_score": None,
            "confidence": None,
            "action": "model_not_ready",
            "message": str(exc),
            "model_path": str(model_dir),
        }

    return {
        "available": True,
        "label": result["evaluation_label"],
        "risk_score": result["risk_score"],
        "raw_risk_score": result.get("raw_risk_score"),
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
        "raw_probabilities": result.get("raw_probabilities"),
        "benign_guard": result.get("benign_guard"),
        "logits": result["logits"],
        "predicted_label": result["predicted_label"],
        "action": result["runtime_action"],
        "method": f"transformer_{runtime_name}",
        "model_path": str(model_dir),
        "thresholds": result["thresholds"],
        "runtime_device": result.get("runtime_device"),
        "runtime_warnings": result.get("warnings", []),
        "canonical_text": result.get("canonical_text"),
        "detected_language": result.get("detected_language"),
    }

def _unsupported_input_response(
    input_type: str,
    text: str,
    file_name: str | None,
    mime_type: str | None,
    started_at: float,
) -> dict[str, Any]:
    message = (
        "Image uploaded, OCR/text extraction not implemented yet"
        if input_type == "image"
        else "File uploaded, text extraction for this file type is not implemented yet"
    )
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    return {
        "input": {
            "input_type": input_type,
            "original_text": text,
            "detected_language": None,
            "canonical_text": "",
            "file_name": file_name,
            "mime_type": mime_type,
        },
        "decision": {
            "label": 0,
            "risk_score": 0.0,
            "action": "allow",
            "model": "not_analyzed",
            "processing_time_ms": round(elapsed_ms, 2),
        },
        "signals": {
            "rule_based": _compact_rule_signal(None, enabled=False),
            "traditional_ml": _compact_ml_signal(None, enabled=False, model_name="none"),
            "transformer": _compact_ml_signal(None, enabled=False, model_name="none"),
        },
        "hybrid_config": None,
        "warnings": [message],
        "explanation": message,
    }


def _signal_score(signal: dict[str, Any] | None) -> float:
    if not signal:
        return 0.0
    try:
        return float(signal.get("risk_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _final_action_from_signals(
    signals: list[dict[str, Any]],
    rule_signal: dict[str, Any] | None,
    decision_strategy: str = "weighted_voting",
) -> tuple[float, str, dict[str, Any]]:
    strategy = (decision_strategy or "weighted_voting").strip().lower()
    active_signals: list[tuple[str, dict[str, Any]]] = []
    if rule_signal:
        active_signals.append(("rule_based", rule_signal))
    for index, signal in enumerate(signals):
        if signal:
            active_signals.append((f"model_{index + 1}", signal))

    vote_rows = []
    scores: list[float] = []
    for name, signal in active_signals:
        if signal.get("error") or str(signal.get("action")) in {"unavailable", "model_not_ready"}:
            continue
        score = _signal_score(signal)
        scores.append(score)
        action = action_from_score(score)
        vote_rows.append(
            {
                "source": name,
                "method": signal.get("method", name),
                "score": round(score, 4),
                "vote": action,
                "positive": action in {"warn", "block"},
            }
        )

    strong_rule = bool(rule_signal and _signal_score(rule_signal) >= RUNTIME_BLOCK_THRESHOLD)
    positive_votes = sum(1 for row in vote_rows if row["positive"])
    block_votes = sum(1 for row in vote_rows if row["vote"] == "block")
    total_votes = len(vote_rows)

    if strategy == "majority_vote":
        final_score = max(scores) if scores else 0.0
        if strong_rule or block_votes >= 2:
            final_action = "block"
        elif total_votes and positive_votes > total_votes / 2:
            final_action = "warn"
        else:
            final_action = "allow"

    elif strategy == "weighted_voting":
        weighted_items = []
        if rule_signal:
            weighted_items.append((_signal_score(rule_signal), 0.30))
        for signal in signals:
            if not signal or signal.get("error") or str(signal.get("action")) in {"unavailable", "model_not_ready"}:
                continue
            method = str(signal.get("method", "")).lower()
            if "xlm_roberta" in method or "xlm-roberta" in method:
                weight = 0.20
            elif "roberta" in method:
                weight = 0.24
            elif "distilbert" in method:
                weight = 0.18
            elif "random_forest" in method:
                weight = 0.18
            else:
                weight = 0.22
            weighted_items.append((_signal_score(signal), weight))
        weight_sum = sum(weight for _, weight in weighted_items) or 1.0
        final_score = sum(score * weight for score, weight in weighted_items) / weight_sum
        if strong_rule or block_votes >= 2:
            final_action = "block"
        else:
            final_action = action_from_score(final_score)
    else:
        final_score = max(scores) if scores else 0.0
        final_action = "block" if strong_rule else action_from_score(final_score)

    breakdown = {
        "strategy": strategy,
        "votes": vote_rows,
        "positive_votes": positive_votes,
        "block_votes": block_votes,
        "total_votes": total_votes,
        "strong_rule": strong_rule,
        "runtime_policy": {
            "allow": f"risk < {RUNTIME_WARN_THRESHOLD:.2f}",
            "warn": f"{RUNTIME_WARN_THRESHOLD:.2f} <= risk < {RUNTIME_BLOCK_THRESHOLD:.2f}",
            "block": f"risk >= {RUNTIME_BLOCK_THRESHOLD:.2f}",
        },
    }
    return final_score, final_action, breakdown


def _confidence_from_probabilities(score: float) -> float:
    normalized_score = min(1.0, max(0.0, float(score)))
    return round(max(1.0 - normalized_score, normalized_score), 4)


def _detect_suspicious_phrases(text: str) -> list[dict[str, str]]:
    prepared = prepare_text_for_detection(text)
    haystack = f"{text.lower()} {prepared['cleaned_text'].lower()}"
    matches: list[dict[str, str]] = []
    for attack_pattern, phrases in SUSPICIOUS_PHRASES.items():
        for phrase in phrases:
            if phrase in haystack:
                matches.append({"pattern": attack_pattern, "phrase": phrase})
    return matches


def _explain_traditional_features(text: str, model_type: str, top_k: int = 8) -> list[dict[str, Any]]:
    try:
        model, vectorizer = load_model_artifacts(model_type)
        cleaned = prepare_text_for_detection(text)["cleaned_text"]
        vectorized = vectorizer.transform([cleaned])
        feature_names = vectorizer.get_feature_names_out()
        active_indices = vectorized.nonzero()[1]
        if len(active_indices) == 0:
            return []

        rows: list[dict[str, Any]] = []
        if hasattr(model, "coef_"):
            weights = model.coef_[0]
            for index in active_indices:
                contribution = float(vectorized[0, index] * weights[index])
                rows.append(
                    {
                        "feature": str(feature_names[index]),
                        "contribution": round(contribution, 4),
                        "direction": "malicious" if contribution >= 0 else "benign",
                    }
                )
            return sorted(rows, key=lambda row: abs(float(row["contribution"])), reverse=True)[:top_k]

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            for index in active_indices:
                contribution = float(vectorized[0, index] * importances[index])
                rows.append(
                    {
                        "feature": str(feature_names[index]),
                        "contribution": round(contribution, 4),
                        "direction": "importance_only",
                    }
                )
            return sorted(rows, key=lambda row: float(row["contribution"]), reverse=True)[:top_k]
    except Exception:
        return []
    return []


def _build_explainability(
    text: str,
    selected_model: str,
    traditional_model: str,
    transformer_model: str,
    rule_result: dict[str, Any] | None,
) -> dict[str, Any]:
    suspicious_matches = _detect_suspicious_phrases(text)
    rule_matches = [] if not rule_result else rule_result.get("matched_rules", [])
    traditional_features = []
    transformer_patterns = []

    if selected_model in TRADITIONAL_MODEL_TYPES:
        traditional_features = _explain_traditional_features(text, selected_model)
    elif selected_model == "hybrid":
        feature_model = "logistic_regression" if traditional_model == "all" else traditional_model
        traditional_features = _explain_traditional_features(text, feature_model)

    if _is_transformer_model_name(selected_model) or selected_model == "hybrid":
        transformer_patterns = suspicious_matches

    return {
        "traditional_features": traditional_features,
        "transformer_attention_keywords": sorted(
            {match["phrase"] for match in suspicious_matches}
        )[:10],
        "detected_attack_patterns": transformer_patterns,
        "rule_matches": rule_matches,
        "note": (
            "Traditional ML contributions use TF-IDF feature weights. "
            "Transformer explanations are heuristic phrase/pattern indicators, not true attention attribution."
        ),
    }


def detect_prompt_advanced(
    text: str,
    input_type: str = "text",
    model: str = "hybrid",
    hybrid_config: dict[str, Any] | None = None,
    file_name: str | None = None,
    mime_type: str | None = None,
    transformer_use_cuda: bool = True,
) -> dict[str, Any]:
    """Detect prompt injection with configurable model selection for the advanced UI."""
    started_at = time.perf_counter()
    normalized_input_type = (input_type or "text").strip().lower()
    text_value = "" if text is None else str(text)

    if normalized_input_type not in {"text", "file", "image"}:
        raise ValueError("input_type pháº£i lÃ  text, file hoáº·c image.")

    if normalized_input_type == "image" and not text_value.strip():
        return _unsupported_input_response(normalized_input_type, text_value, file_name, mime_type, started_at)

    if normalized_input_type == "file" and not text_value.strip():
        suffix = Path(file_name or "").suffix.lower()
        if suffix not in TEXT_FILE_EXTENSIONS:
            return _unsupported_input_response(normalized_input_type, text_value, file_name, mime_type, started_at)

    if not text_value.strip():
        raise ValueError("Field 'text' khÃ´ng Ä‘Æ°á»£c rá»—ng vá»›i input_type text hoáº·c file Ä‘Ã£ trÃ­ch xuáº¥t text.")

    prepared_text = prepare_text_for_detection(text_value)
    selected_model = _normalize_model_name(model)
    config = hybrid_config or {}
    traditional_model = _normalize_traditional_model(config.get("traditional_model") or "all")
    transformer_model = _normalize_transformer_model(config.get("transformer_model"))
    use_rule_based = bool(config.get("use_rule_based", True))
    decision_strategy = str(config.get("decision_strategy", "weighted_voting")).strip().lower()
    if decision_strategy not in {"majority_vote", "maximum_risk", "weighted_voting"}:
        decision_strategy = "weighted_voting"

    rule_result: dict[str, Any] | None = None
    traditional_result: dict[str, Any] | None = None
    traditional_results: list[dict[str, Any]] = []
    transformer_result: dict[str, Any] | None = None
    hybrid_breakdown: dict[str, Any] | None = None

    if selected_model == "hybrid":
        if use_rule_based:
            rule_result = detect_by_rules(text_value)
        traditional_model_names = (
            ["logistic_regression", "linear_svm", "random_forest"]
            if traditional_model == "all"
            else [traditional_model]
        )
        traditional_results = [
            detect_by_ml(text_value, model_name, decision_mode="runtime")
            for model_name in traditional_model_names
        ]
        traditional_result = (
            max(traditional_results, key=lambda item: _signal_score(item))
            if traditional_results
            else None
        )
        transformer_result = _run_transformer(text_value, transformer_model, use_cuda=transformer_use_cuda)
        final_score, final_action, hybrid_breakdown = _final_action_from_signals(
            [*traditional_results, transformer_result],
            rule_result,
            decision_strategy=decision_strategy,
        )
        method = f"advanced_hybrid_{traditional_model}_{safe_model_dir_name(transformer_model)}"
    elif selected_model in TRADITIONAL_MODEL_TYPES:
        traditional_result = detect_by_ml(text_value, selected_model, decision_mode="runtime")
        final_score = float(traditional_result.get("risk_score", 0.0) or 0.0)
        final_action = str(traditional_result.get("action", "allow"))
        method = str(traditional_result.get("method", f"tfidf_{selected_model}"))
    elif _is_transformer_model_name(selected_model):
        transformer_result = _run_transformer(text_value, selected_model, use_cuda=transformer_use_cuda)
        final_score = float(transformer_result.get("risk_score", 0.0) or 0.0)
        final_action = str(transformer_result.get("action", "unavailable"))
        method = str(transformer_result.get("method", f"transformer_{safe_model_dir_name(selected_model)}"))
    else:
        raise ValueError("model khÃ´ng há»£p lá»‡.")

    elapsed_ms = (time.perf_counter() - started_at) * 1000
    available_errors = [
        signal.get("message")
        for signal in [traditional_result, transformer_result]
        if signal and signal.get("error")
    ]
    label = None if final_action == "model_not_ready" else (1 if final_action in {"warn", "block"} else 0)

    if final_action == "model_not_ready":
        explanation = "Model chÆ°a sáºµn sÃ ng. Checkpoint Transformer chÆ°a tá»“n táº¡i hoáº·c chÆ°a Ä‘Æ°á»£c fine-tune."
    elif final_action == "unavailable":
        explanation = "Model Ä‘Æ°á»£c chá»n chÆ°a sáºµn sÃ ng. Kiá»ƒm tra message trong signals."
    elif final_action == "block":
        explanation = "CÃ³ tÃ­n hiá»‡u prompt injection máº¡nh vÆ°á»£t ngÆ°á»¡ng block."
    elif final_action == "warn":
        explanation = "CÃ³ tÃ­n hiá»‡u rá»§i ro vÆ°á»£t ngÆ°á»¡ng cáº£nh bÃ¡o nhÆ°ng chÆ°a Ä‘áº¡t ngÆ°á»¡ng block."
    else:
        explanation = "KhÃ´ng cÃ³ tÃ­n hiá»‡u vÆ°á»£t ngÆ°á»¡ng cáº£nh bÃ¡o runtime."

    return {
        "input": {
            "input_type": normalized_input_type,
            "original_text": prepared_text["original_text"],
            "detected_language": prepared_text["detected_language"],
            "canonical_text": prepared_text["cleaned_text"],
            "file_name": file_name,
            "mime_type": mime_type,
        },
        "decision": {
            "label": label,
            "risk_score": None if final_action == "model_not_ready" else round(float(final_score), 8),
            "action": final_action,
            "model": method,
            "processing_time_ms": round(elapsed_ms, 2),
            "confidence": None if final_action == "model_not_ready" else _confidence_from_probabilities(final_score),
        },
        "signals": {
            "rule_based": _compact_rule_signal(rule_result, enabled=selected_model == "hybrid" and use_rule_based),
            "traditional_ml": _compact_ml_signal(
                traditional_result,
                enabled=selected_model == "hybrid" or selected_model in TRADITIONAL_MODEL_TYPES,
                model_name=traditional_model if selected_model == "hybrid" else selected_model,
            ),
            "traditional_ml_members": [
                _compact_ml_signal(item, enabled=True, model_name=str(item.get("method", "traditional_ml")))
                for item in traditional_results
            ],
            "transformer": _compact_ml_signal(
                transformer_result,
                enabled=selected_model == "hybrid" or _is_transformer_model_name(selected_model),
                model_name=transformer_model if selected_model == "hybrid" else selected_model,
            ),
        },
        "hybrid_config": {
            "traditional_model": traditional_model,
            "transformer_model": transformer_model,
            "use_rule_based": use_rule_based,
            "decision_strategy": decision_strategy,
        }
        if selected_model == "hybrid"
        else None,
        "hybrid_breakdown": hybrid_breakdown,
        "explainability": _build_explainability(
            text_value,
            selected_model,
            traditional_model,
            transformer_model,
            rule_result,
        ),
        "warnings": [message for message in available_errors if message],
        "explanation": explanation,
    }


def _model_path_for_compare(model_name: str, signal: dict[str, Any] | None = None) -> str | None:
    if signal and signal.get("model_path"):
        return str(signal["model_path"])

    normalized = _normalize_model_name(model_name)
    if normalized in TRADITIONAL_MODEL_TYPES:
        paths = MODEL_FILES.get(normalized, {})
        model_path = paths.get("model")
        return str(model_path) if model_path else None
    if _is_transformer_model_name(normalized):
        return str(resolve_transformer_model_dir(normalized))
    if normalized == "hybrid":
        return "ensemble"
    return None


def _primary_signal_for_compare(model_name: str, result: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_model_name(model_name)
    signals = result.get("signals", {}) or {}
    if normalized in TRADITIONAL_MODEL_TYPES:
        return signals.get("traditional_ml", {}) or {}
    if _is_transformer_model_name(normalized):
        return signals.get("transformer", {}) or {}
    return {}


def _hybrid_selected_models(result: dict[str, Any]) -> list[str]:
    config = result.get("hybrid_config") or {}
    selected: list[str] = []
    if config.get("use_rule_based"):
        selected.append("rule_based")

    traditional = config.get("traditional_model") or "all"
    if traditional == "all":
        selected.extend(["logistic_regression", "linear_svm", "random_forest"])
    else:
        selected.append(str(traditional))

    transformer = config.get("transformer_model")
    if transformer:
        selected.append(str(transformer))
    return selected


def _compare_row_from_result(model_name: str, result: dict[str, Any]) -> dict[str, Any]:
    decision = result.get("decision", {}) or {}
    signal = _primary_signal_for_compare(model_name, result)
    normalized = _normalize_model_name(model_name)
    action = decision.get("action")
    risk_score = decision.get("risk_score")
    raw_score = signal.get("raw_score")
    if raw_score is None:
        raw_score = signal.get("raw_risk_score")
    if raw_score is None:
        raw_score = risk_score

    available = (
        risk_score is not None
        and action not in {"model_not_ready", "unavailable"}
        and signal.get("available", True) is not False
    )
    error = None
    if not available:
        warnings = result.get("warnings", []) or []
        error = signal.get("error") or signal.get("message") or "; ".join(str(item) for item in warnings) or "Model checkpoint not found or inference failed."

    row: dict[str, Any] = {
        "model": model_name,
        "requested_model": model_name,
        "resolved_model": decision.get("model"),
        "loaded_model": signal.get("model") or decision.get("model"),
        "model_path": _model_path_for_compare(model_name, signal),
        "available": bool(available),
        "raw_score": raw_score,
        "prediction": decision.get("label"),
        "predicted_label": signal.get("predicted_label", decision.get("label")),
        "action": action if available else "model_not_ready",
        "risk_score": risk_score if available else None,
        "latency_ms": decision.get("processing_time_ms", 0.0),
        "confidence": decision.get("confidence") if available else None,
        "error": error,
        "warnings": result.get("warnings", []),
    }

    if normalized == "hybrid":
        breakdown = result.get("hybrid_breakdown") or {}
        individual_scores = [
            {
                "source": vote.get("source"),
                "model": vote.get("method"),
                "risk_score": vote.get("score"),
                "action": vote.get("vote"),
                "positive": vote.get("positive"),
            }
            for vote in breakdown.get("votes", [])
        ]
        row.update(
            {
                "individual_scores": individual_scores,
                "selected_models": _hybrid_selected_models(result),
                "voting_strategy": breakdown.get("strategy"),
                "final_score": risk_score if available else None,
                "final_action": action if available else "model_not_ready",
            }
        )
    return row


def _compare_error_row(model_name: str, exc: Exception) -> dict[str, Any]:
    return {
        "model": model_name,
        "requested_model": model_name,
        "resolved_model": None,
        "loaded_model": _normalize_model_name(model_name),
        "model_path": _model_path_for_compare(model_name),
        "available": False,
        "raw_score": None,
        "prediction": None,
        "predicted_label": None,
        "action": "model_not_ready",
        "risk_score": None,
        "latency_ms": 0.0,
        "confidence": None,
        "error": str(exc) or "Model checkpoint not found or inference failed.",
        "warnings": ["Model checkpoint not found or inference failed."],
    }


def compare_all_models(
    text: str,
    input_type: str = "text",
    hybrid_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the same input through all supported advanced models with per-model debug data."""
    if not text or not str(text).strip():
        raise ValueError("text khong duoc rong khi compare models.")

    rows: list[dict[str, Any]] = []
    for model_name in COMPARISON_MODELS:
        try:
            result = detect_prompt_advanced(
                text=text,
                input_type=input_type,
                model=model_name,
                hybrid_config=hybrid_config,
                transformer_use_cuda=False,
            )
            rows.append(_compare_row_from_result(model_name, result))
        except Exception as exc:  # compare must not fabricate a score when a model fails
            rows.append(_compare_error_row(model_name, exc))
        finally:
            if model_name in {"roberta", "xlm_roberta", "hybrid"}:
                clear_transformer_runtime_cache()

    ready_rows = [row for row in rows if row.get("risk_score") is not None]
    highest_risk = max(ready_rows, key=lambda row: float(row["risk_score"])) if ready_rows else None
    fastest_candidates = [row for row in rows if row.get("latency_ms") is not None]
    fastest = min(fastest_candidates, key=lambda row: float(row["latency_ms"])) if fastest_candidates else None
    return {
        "input_preview": str(text)[:220],
        "results": rows,
        "highest_risk_model": highest_risk["model"] if highest_risk else None,
        "fastest_model": fastest["model"] if fastest else None,
    }


def get_project_statistics() -> dict[str, Any]:
    """Read dataset/model metrics from generated reports when available."""
    metrics_path = REPORTS_DIR / "metrics.json"
    transformer_path = OUTPUTS_DIR / "transformer_results.json"
    payload: dict[str, Any] = {
        "dataset": {
            "num_rows": None,
            "train_size": None,
            "validation_size": None,
            "test_size": None,
            "label_distribution": {},
        },
        "models": {},
        "sources": [],
    }

    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
        dataset = metrics.get("dataset", {})
        split = metrics.get("split", {})
        payload["dataset"].update(
            {
                "num_rows": dataset.get("num_rows"),
                "train_size": split.get("train_size"),
                "validation_size": split.get("validation_size"),
                "test_size": split.get("test_size"),
                "label_distribution": dataset.get("label_distribution", {}),
            }
        )
        for model_name, model_metrics in metrics.get("models", {}).items():
            payload["models"][model_name] = {
                "accuracy": model_metrics.get("accuracy"),
                "precision": model_metrics.get("precision"),
                "recall": model_metrics.get("recall"),
                "f1": model_metrics.get("f1"),
            }
        payload["sources"].append(str(metrics_path))

    if transformer_path.exists():
        transformer_results = json.loads(transformer_path.read_text(encoding="utf-8-sig"))
        for model_name, result in transformer_results.get("models", {}).items():
            test_metrics = result.get("test_metrics", {})
            payload["models"][model_name] = {
                "accuracy": test_metrics.get("accuracy"),
                "precision": test_metrics.get("precision"),
                "recall": test_metrics.get("recall"),
                "f1": test_metrics.get("f1"),
            }
        payload["sources"].append(str(transformer_path))

    return payload


def simulate_chat_detection(
    user_prompt: str,
    model: str = "hybrid",
    hybrid_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Simulate detector-gated LLM deployment without calling a real LLM."""
    detection = detect_prompt_advanced(
        text=user_prompt,
        input_type="text",
        model=model,
        hybrid_config=hybrid_config,
    )
    action = detection.get("decision", {}).get("action")
    forwarded_to_mock_llm = False
    if action == "model_not_ready":
        llm_response = "Model not ready. Request was not forwarded to the mock LLM."
    elif action == "block":
        llm_response = "Prompt blocked by Prompt Injection Detection Engine"
    elif action == "warn":
        llm_response = (
            "Mock LLM response withheld for review because the detector returned WARN. "
            "In production this could require human approval or stricter policy."
        )
    else:
        forwarded_to_mock_llm = True
        llm_response = (
            "Mock LLM response: request accepted by the detector. "
            "No real external LLM API is called in this project."
        )

    return {
        "user_prompt": user_prompt,
        "detector": detection,
        "forwarded_to_mock_llm": forwarded_to_mock_llm,
        "llm_response": llm_response,
    }



