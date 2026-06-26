"""Explainable hybrid runtime pipeline.

This module composes the existing project detectors with the new
context-aware layer. It is additive: legacy training and evaluation entrypoints
continue to use their original modules.
"""

from __future__ import annotations

from typing import Any

from src.benign_intent import detect_benign_reference_intent
from src.detection.context_aware_detector import detect_context_aware
from src.detector import DEFAULT_THRESHOLDS, detect_by_ml, detect_by_rules
from src.transformer_utils import (
    is_finetuned_transformer_checkpoint,
    predict_transformer,
    resolve_transformer_model_dir,
)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _score_ml(text: str, model_type: str, enabled: bool) -> tuple[float | None, dict[str, Any] | None, str | None]:
    if not enabled:
        return None, None, "ML model scoring disabled by runtime option."
    result = detect_by_ml(text, model_type=model_type)
    if result.get("error"):
        return None, result, str(result.get("message", "ML model unavailable."))
    return _as_float(result.get("risk_score")), result, None


def _score_transformer(
    text: str,
    *,
    model_name: str,
    enabled: bool,
    use_cuda: bool,
) -> tuple[float | None, dict[str, Any] | None, str | None]:
    if not enabled:
        return None, None, "Transformer scoring disabled by runtime option."

    model_dir = resolve_transformer_model_dir(model_name)
    if not is_finetuned_transformer_checkpoint(model_dir):
        return None, {"model_path": str(model_dir), "available": False}, "Transformer checkpoint unavailable."

    try:
        result = predict_transformer(
            text=text,
            model_path=model_dir,
            model_name=model_name,
            max_length=128,
            use_cuda=use_cuda,
        )
    except Exception as exc:  # pragma: no cover - defensive guard for optional runtime stacks.
        return None, {"available": False, "error": f"{type(exc).__name__}: {exc}"}, "Transformer scoring failed."
    return _as_float(result.get("risk_score")), result, None


def _collect_rule_reasons(rule_result: dict[str, Any], prefix: str = "") -> list[str]:
    reasons: list[str] = []
    for rule in rule_result.get("matched_rules", []):
        pattern = rule.get("pattern")
        if pattern:
            source = f"{prefix}: " if prefix else ""
            reasons.append(f"Matched keyword: {source}{pattern}")
    return reasons


def _risk_level(score: float, warn_threshold: float, block_threshold: float) -> str:
    if score >= block_threshold:
        return "block"
    if score >= warn_threshold:
        return "warn"
    return "safe"


def _recommendation(risk_level: str, context_result: dict[str, Any]) -> str:
    if risk_level == "block":
        if context_result.get("context_mismatch"):
            return "Block this input because external content contains an instruction unrelated to the user task."
        return "Block this input because one or more detector signals exceed the block threshold."
    if risk_level == "warn":
        return "Warn or sanitize before sending this content to an LLM."
    return "Allow this input; keep external content treated as untrusted data."


def run_hybrid_detection(
    *,
    user_prompt: str,
    user_task: str | None = None,
    external_content: str | None = None,
    ml_model_type: str = "logistic_regression",
    transformer_model: str = "roberta",
    use_ml: bool = True,
    use_transformer: bool = False,
    use_cuda: bool = False,
    warn_threshold: float | None = None,
    block_threshold: float | None = None,
) -> dict[str, Any]:
    """Run rule, ML, optional Transformer, context-aware scoring, and explanation."""
    task = str(user_task or user_prompt or "").strip()
    prompt = str(user_prompt or "").strip()
    context = "" if external_content is None else str(external_content)
    model_text = f"{prompt}\n\n{context}".strip() if context.strip() else prompt

    prompt_rule_result = detect_by_rules(prompt)
    context_rule_result = detect_by_rules(context) if context.strip() else {
        "risk_score": 0.0,
        "matched_rules": [],
        "action": "allow",
    }

    ml_score, ml_result, ml_warning = _score_ml(model_text, ml_model_type, enabled=use_ml)
    transformer_score, transformer_result, transformer_warning = _score_transformer(
        model_text,
        model_name=transformer_model,
        enabled=use_transformer,
        use_cuda=use_cuda,
    )

    model_scores_for_context = [score for score in [ml_score, transformer_score] if score is not None]
    prompt_context_result = detect_context_aware(
        user_task=prompt,
        external_content=prompt,
        model_score=max(model_scores_for_context, default=0.0),
        rule_hits=prompt_rule_result.get("matched_rules", []),
    )
    context_result = detect_context_aware(
        user_task=task,
        external_content=context,
        model_score=max(model_scores_for_context, default=0.0),
        rule_hits=context_rule_result.get("matched_rules", []),
    )
    context_score = max(
        _as_float(prompt_context_result.get("context_risk_score")),
        _as_float(context_result.get("context_risk_score")),
    )

    external_is_benign_reference = bool(
        context_result.get("benign_reference_guard", {}).get("triggered")
        and not context_result.get("context_mismatch")
    )
    effective_context_rule_score = 0.0 if external_is_benign_reference else _as_float(context_rule_result.get("risk_score"))
    rule_score = max(_as_float(prompt_rule_result.get("risk_score")), effective_context_rule_score)

    thresholds = DEFAULT_THRESHOLDS.copy()
    if isinstance(ml_result, dict):
        thresholds.update(ml_result.get("thresholds", {}) or {})
    warn = float(warn_threshold if warn_threshold is not None else thresholds.get("runtime_warn_threshold", 0.50))
    block = float(block_threshold if block_threshold is not None else thresholds.get("runtime_block_threshold", 0.80))

    component_scores = [
        rule_score,
        context_score,
        _as_float(ml_score, default=0.0),
        _as_float(transformer_score, default=0.0),
    ]
    final_score = max(component_scores)

    transformer_crosses_warn = transformer_score is not None and transformer_score >= warn
    if (context_result.get("context_mismatch") or prompt_context_result.get("context_mismatch")) and transformer_crosses_warn:
        final_score = max(final_score, block)

    benign_guard = detect_benign_reference_intent(model_text)
    if (
        benign_guard.get("triggered")
        and not context_result.get("context_mismatch")
        and rule_score < warn
    ):
        # Educational/quoted examples should not be blocked simply because a model
        # reacts to the quoted phrase.
        final_score = min(final_score, max(0.0, warn - 0.05))

    risk_level = _risk_level(final_score, warn, block)
    reasons: list[str] = []
    reasons.extend(_collect_rule_reasons(prompt_rule_result, prefix="user_prompt"))
    reasons.extend(_collect_rule_reasons(context_rule_result, prefix="external_content"))

    if ml_score is not None and ml_score >= warn:
        level = "block" if ml_score >= block else "warn"
        reasons.append(f"ML score: {ml_score:.4f} reaches {level} threshold.")
    elif ml_warning:
        reasons.append(ml_warning)

    if transformer_score is not None and transformer_score >= warn:
        level = "block" if transformer_score >= block else "warn"
        reasons.append(f"Transformer score: {transformer_score:.4f} reaches {level} threshold.")
    elif transformer_warning:
        reasons.append(transformer_warning)

    if prompt_context_result.get("context_mismatch"):
        reasons.append("Direct instruction signal: " + str(prompt_context_result.get("reason")))
    if context_result.get("context_mismatch"):
        reasons.append(str(context_result.get("reason")))
    elif _as_float(context_result.get("context_risk_score")) > 0:
        reasons.append(str(context_result.get("reason")))

    if not reasons or risk_level == "safe":
        safe_reason = "No detector signal exceeded the warn threshold."
        if context_score > 0 and not context_result.get("context_mismatch") and not prompt_context_result.get("context_mismatch"):
            safe_reason = str(context_result.get("reason"))
        reasons = [safe_reason]

    return {
        "risk_level": risk_level,
        "final_score": round(float(final_score), 4),
        "model_scores": {
            "rule_based": round(float(rule_score), 4),
            "ml_model": None if ml_score is None else round(float(ml_score), 4),
            "transformer": None if transformer_score is None else round(float(transformer_score), 4),
            "context_aware": round(float(context_score), 4),
        },
        "threshold_used": {
            "warn": round(float(warn), 4),
            "block": round(float(block), 4),
        },
        "reasons": reasons,
        "recommendation": _recommendation(risk_level, context_result),
        "label": 0 if risk_level == "safe" else 1,
        "context_analysis": {
            "user_prompt": prompt_context_result,
            "external_content": context_result,
        },
        "details": {
            "rule_based": {
                "user_prompt": prompt_rule_result,
                "external_content": context_rule_result,
            },
            "ml_model": ml_result,
            "transformer": transformer_result,
            "benign_reference_guard": benign_guard,
            "score_formula": "final_score = max(rule_score, ml_score, transformer_score, context_risk_score)",
            "context_escalation_policy": (
                "If context_mismatch is true and transformer_score exceeds warn threshold, "
                "risk_level is escalated to block."
            ),
        },
    }
