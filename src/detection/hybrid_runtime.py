"""Runtime Adaptive Risk Fusion + Decision Policy Engine.

The runtime hybrid path in this module uses exactly three detector signals:

- rule-based detector
- fine-tuned RoBERTa calibrated probability
- fine-tuned XLM-RoBERTa calibrated probability

Logistic Regression, Linear SVM and Random Forest are intentionally excluded
from runtime fusion. They remain research/evaluation baselines elsewhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.detection.adaptive_risk_fusion import AdaptiveRiskFusion
from src.detection.policy_engine import DecisionPolicyEngine
from src.language_utils import detect_language
from src.preprocessing import prepare_text_for_detection
from src.rule_based import detect_by_rules
from src.transformer_utils import (
    is_finetuned_transformer_checkpoint,
    predict_transformer,
    resolve_transformer_model_dir,
)


def _normalize_runtime_language(language: str | None, text: str) -> str:
    if language and str(language).strip():
        normalized = str(language).strip().lower()
    else:
        normalized = detect_language(text)
    if normalized in {"en", "eng", "english"}:
        return "en"
    if normalized in {"vi", "vie", "vietnamese", "vn"}:
        return "vi"
    if normalized in {"mixed", "multi", "multilingual"}:
        return "mixed"
    return "unknown"


def _normalize_source_type(source_type: str | None) -> str:
    normalized = (source_type or "user_prompt").strip().lower()
    aliases = {
        "raw_text": "user_prompt",
        "prompt": "user_prompt",
        "user": "user_prompt",
        "external": "external_content",
        "content": "external_content",
        "retrieval": "rag",
    }
    return aliases.get(normalized, normalized)


def _model_warning(model_name: str, message: str) -> str:
    return f"{model_name}: {message}"


def _predict_transformer_signal(
    *,
    text: str,
    model_name: str,
    use_cuda: bool,
    max_length: int,
) -> dict[str, Any]:
    """Run one Transformer and return a compact, warning-rich signal dict."""
    model_dir: Path = resolve_transformer_model_dir(model_name)
    warnings: list[str] = []
    if not is_finetuned_transformer_checkpoint(model_dir):
        warnings.append(
            _model_warning(
                model_name,
                f"missing_or_invalid_finetuned_checkpoint at {model_dir}",
            )
        )
        return {
            "model": model_name,
            "available": False,
            "model_path": str(model_dir),
            "score": 0.0,
            "raw_score": None,
            "calibrated_score": None,
            "score_used": "unavailable",
            "calibration_method": None,
            "calibration_source": None,
            "calibration_warning": warnings[-1],
            "threshold_used": None,
            "threshold_source": None,
            "warnings": warnings,
        }

    try:
        result = predict_transformer(
            text=text,
            model_path=model_dir,
            model_name=model_name,
            max_length=max_length,
            use_cuda=use_cuda,
        )
    except Exception as exc:  # pragma: no cover - defensive runtime path
        warnings.append(_model_warning(model_name, f"inference_failed: {exc}"))
        return {
            "model": model_name,
            "available": False,
            "model_path": str(model_dir),
            "score": 0.0,
            "raw_score": None,
            "calibrated_score": None,
            "score_used": "error",
            "calibration_method": None,
            "calibration_source": None,
            "calibration_warning": warnings[-1],
            "threshold_used": None,
            "threshold_source": None,
            "warnings": warnings,
        }

    calibration_warning = result.get("calibration_warning")
    if calibration_warning:
        warnings.append(_model_warning(model_name, str(calibration_warning)))
    warnings.extend(str(item) for item in result.get("warnings", []))

    return {
        "model": model_name,
        "available": True,
        "model_path": str(model_dir),
        "score": float(result.get("risk_score", 0.0) or 0.0),
        "raw_score": result.get("raw_score", result.get("raw_risk_score")),
        "calibrated_score": result.get("calibrated_score"),
        "score_used": result.get("score_used", "raw_softmax_probability"),
        "calibration_method": result.get("calibration_method"),
        "calibration_source": result.get("calibration_source"),
        "calibration_warning": calibration_warning,
        "threshold_used": result.get("threshold_used"),
        "threshold_source": result.get("threshold_source"),
        "runtime_device": result.get("runtime_device"),
        "warnings": warnings,
    }


def detect_hybrid_adaptive(
    text: str,
    language: str | None = None,
    source_type: str = "user_prompt",
    user_task: str | None = None,
    external_content: str | None = None,
    use_cuda: bool = False,
    max_length: int = 128,
) -> dict[str, Any]:
    """Detect prompt injection with adaptive fusion and policy decision."""
    if text is None and external_content is None:
        raise ValueError("Field 'text' hoặc 'external_content' không được rỗng.")

    analyzed_text = external_content if external_content is not None else text
    analyzed_text = "" if analyzed_text is None else str(analyzed_text)
    if not analyzed_text.strip():
        raise ValueError("Nội dung cần phân tích không được rỗng.")

    normalized_source = _normalize_source_type(source_type)
    normalized_language = _normalize_runtime_language(language, analyzed_text)
    prepared = prepare_text_for_detection(analyzed_text)
    warnings: list[str] = []

    rule_result = detect_by_rules(analyzed_text, source_type=normalized_source)
    rule_score = float(rule_result.get("rule_score", rule_result.get("risk_score", 0.0)) or 0.0)

    roberta_signal = _predict_transformer_signal(
        text=analyzed_text,
        model_name="roberta",
        use_cuda=use_cuda,
        max_length=max_length,
    )
    xlm_signal = _predict_transformer_signal(
        text=analyzed_text,
        model_name="xlm_roberta",
        use_cuda=use_cuda,
        max_length=max_length,
    )
    warnings.extend(roberta_signal.get("warnings", []))
    warnings.extend(xlm_signal.get("warnings", []))

    scores_are_calibrated = (
        roberta_signal.get("available") is True
        and xlm_signal.get("available") is True
        and roberta_signal.get("score_used") == "calibrated_probability"
        and xlm_signal.get("score_used") == "calibrated_probability"
    )

    if not scores_are_calibrated:
        warnings.append(
            "Không đủ calibrated probability cho cả RoBERTa và XLM-R; output vẫn chạy nhưng cần kiểm tra calibrator/checkpoint."
        )

    fusion = AdaptiveRiskFusion().fuse(
        text=analyzed_text,
        language=normalized_language,
        source_type=normalized_source,
        rule_score=rule_score,
        rule_matches=rule_result.get("matched_rules", []),  # type: ignore[arg-type]
        roberta_score=float(roberta_signal.get("score", 0.0) or 0.0),
        xlm_score=float(xlm_signal.get("score", 0.0) or 0.0),
        scores_are_calibrated=scores_are_calibrated,
        highest_severity=str(rule_result.get("highest_severity", "none")),
        has_high_severity_rule=bool(rule_result.get("has_high_severity_rule", False)),
        has_critical_rule=bool(rule_result.get("has_critical_rule", False)),
    )

    policy = DecisionPolicyEngine().decide(
        final_risk=float(fusion["final_risk"]),
        model_risk=float(fusion["model_risk"]),
        rule_score=rule_score,
        roberta_score=float(roberta_signal.get("score", 0.0) or 0.0),
        xlm_score=float(xlm_signal.get("score", 0.0) or 0.0),
        highest_severity=str(rule_result.get("highest_severity", "none")),
        has_high_severity_rule=bool(rule_result.get("has_high_severity_rule", False)),
        has_critical_rule=bool(rule_result.get("has_critical_rule", False)),
        source_type=normalized_source,
        language=normalized_language,
        rule_matches=rule_result.get("matched_rules", []),  # type: ignore[arg-type]
        weights=fusion["weights"],
        fusion_method=str(fusion["fusion_method"]),
        scores_are_calibrated=scores_are_calibrated,
        benign_reference_intent=bool(
            isinstance(rule_result.get("benign_guard"), dict)
            and rule_result.get("benign_guard", {}).get("triggered")
        ),
    )

    final_risk = float(fusion["final_risk"])
    model_risk = float(fusion["model_risk"])
    decision = str(policy["decision"])
    risk_level = str(policy["risk_level"])

    return {
        "input": {
            "text": text,
            "analyzed_text": analyzed_text,
            "user_task": user_task,
            "external_content_used": external_content is not None,
            "language": normalized_language,
            "source_type": normalized_source,
            "detected_language": prepared.get("detected_language"),
            "canonical_text": prepared.get("cleaned_text"),
        },
        "decision": decision,
        "risk_level": risk_level,
        "label": 0 if decision == "SAFE" else 1,
        "action": decision.lower(),
        "final_risk": round(final_risk, 6),
        "model_risk": round(model_risk, 6),
        "rule_score": round(rule_score, 6),
        "roberta_score": round(float(roberta_signal.get("score", 0.0) or 0.0), 6),
        "xlm_score": round(float(xlm_signal.get("score", 0.0) or 0.0), 6),
        "scores": {
            "rule_based": {
                "rule_score": round(rule_score, 6),
                "risk_score": rule_result.get("risk_score"),
                "action": rule_result.get("action"),
                "highest_severity": rule_result.get("highest_severity", "none"),
                "has_high_severity_rule": rule_result.get("has_high_severity_rule", False),
                "has_critical_rule": rule_result.get("has_critical_rule", False),
                "matched_rules": rule_result.get("matched_rules", []),
            },
            "roberta": roberta_signal,
            "xlm_roberta": xlm_signal,
        },
        "weights": fusion["weights"],
        "fusion": fusion,
        "policy": policy,
        "decision_policy": policy["decision_policy"],
        "reasons": {
            "fusion": fusion["reasons"],
            "policy": policy["reasons"],
        },
        "recommendation": policy["recommendation"],
        "warnings": warnings,
    }
