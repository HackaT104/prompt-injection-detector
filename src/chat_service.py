"""User-facing chat check service.

This module adapts the existing detection pipeline into a small public schema
for the end-user chatbot page. It intentionally hides admin-only internals while
keeping enough policy/model detail for a clear demo response.
"""

from __future__ import annotations

from typing import Any

from src.official_runtime import run_official_runtime


DECISION_RANK = {
    "safe": 0,
    "warning": 1,
    "blocked": 2,
}


def _decision_from_action(action: str | None) -> str:
    normalized = (action or "").strip().lower()
    if normalized in {"block", "blocked"}:
        return "blocked"
    if normalized in {"warn", "warning", "sanitize_or_warn"}:
        return "warning"
    return "safe"


def _label_for_decision(decision: str) -> str:
    if decision == "blocked":
        return "BLOCKED"
    if decision == "warning":
        return "WARNING"
    return "SAFE"


def _score(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _reasons_from_runtime(result: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    raw_reasons = result.get("reasons", {})
    if isinstance(raw_reasons, dict):
        for group in ("policy", "fusion"):
            values = raw_reasons.get(group, [])
            if isinstance(values, list):
                reasons.extend(str(item) for item in values if str(item).strip())
    elif isinstance(raw_reasons, list):
        reasons.extend(str(item) for item in raw_reasons if str(item).strip())

    for warning in result.get("warnings", []) or []:
        text = str(warning).strip()
        if text:
            reasons.append(text)

    if not reasons and result.get("recommendation"):
        reasons.append(str(result["recommendation"]))
    return _unique_preserving_order(reasons)


def _reasons_from_context(result: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    explanation = result.get("explanation")
    if explanation:
        reasons.append(str(explanation))
    for rule in result.get("matched_rules", []) or []:
        if not isinstance(rule, dict):
            continue
        group = rule.get("group") or rule.get("rule") or "matched_rule"
        phrase = rule.get("phrase") or rule.get("pattern")
        source = rule.get("source")
        reason = f"{source + ': ' if source else ''}{group}"
        if phrase:
            reason = f"{reason} matched '{phrase}'"
        reasons.append(reason)
    return _unique_preserving_order(reasons)


def _unique_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in seen:
            seen.add(text)
            unique.append(text)
    return unique


def _model_scores_from_runtime(result: dict[str, Any]) -> dict[str, Any]:
    scores = result.get("scores", {})
    rule_based = scores.get("rule_based", {}) if isinstance(scores, dict) else {}
    return {
        "ruleBased": {
            "score": _score(rule_based.get("rule_score", result.get("rule_score"))),
            "highestSeverity": rule_based.get("highest_severity"),
            "matchedRules": rule_based.get("matched_rules", []),
        },
        "roberta": {
            "score": _score(result.get("roberta_score")),
            "available": bool(scores.get("roberta", {}).get("available")) if isinstance(scores, dict) else False,
        },
        "modelRisk": _score(result.get("model_risk")),
        "finalRisk": _score(result.get("final_risk")),
    }


def _policy_result_from_runtime(result: dict[str, Any]) -> dict[str, Any]:
    policy = result.get("policy", {})
    if not isinstance(policy, dict):
        policy = {}
    return {
        "decisionPolicy": result.get("decision_policy") or policy.get("decision_policy"),
        "riskLevel": result.get("risk_level") or policy.get("risk_level"),
        "recommendation": result.get("recommendation") or policy.get("recommendation"),
        "inputs": policy.get("policy_inputs", {}),
        "fusion": {
            "method": result.get("fusion", {}).get("fusion_method")
            if isinstance(result.get("fusion"), dict)
            else None,
            "weights": result.get("weights", {}),
        },
    }


def _compact_runtime_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision": result.get("decision"),
        "riskLevel": result.get("risk_level"),
        "finalRisk": result.get("final_risk"),
        "modelRisk": result.get("model_risk"),
        "ruleScore": result.get("rule_score"),
        "policy": _policy_result_from_runtime(result),
    }


def _context_result_to_signal(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "action": result.get("action"),
        "attackType": result.get("attack_type"),
        "riskScore": result.get("risk_score"),
        "matchedRules": result.get("matched_rules", []),
        "explanation": result.get("explanation"),
    }


def _project_context_to_text(project_context: dict[str, Any] | None) -> str:
    if not project_context:
        return ""
    parts = [
        f"Project: {project_context.get('projectName', '')}",
        f"Description: {project_context.get('projectDescription', '')}",
        f"System instruction: {project_context.get('systemInstruction', '')}",
        f"Context summary: {project_context.get('contextSummary', '')}",
    ]
    for document in project_context.get("documents", []) or []:
        if not isinstance(document, dict):
            continue
        parts.append(
            f"Untrusted context item {document.get('title', 'Untitled')}: {document.get('content', '')}"
        )
    return "\n\n".join(part for part in parts if part.strip())


def _assistant_message(decision: str, risk_score: float, reasons: list[str]) -> str:
    first_reason = reasons[0] if reasons else "No risky signal exceeded the current policy threshold."
    if decision == "blocked":
        return (
            f"Blocked by policy. Risk score: {risk_score:.2f}. "
            f"Reason: {first_reason}"
        )
    if decision == "warning":
        return (
            f"Warning: this prompt may contain risky instructions. Risk score: {risk_score:.2f}. "
            f"Reason: {first_reason}"
        )
    return (
        f"Safe to continue. Risk score: {risk_score:.2f}. "
        f"Reason: {first_reason}"
    )


def _detection_types(source: str, context_signal: dict[str, Any] | None, model_scores: dict[str, Any]) -> list[str]:
    types: list[str] = []
    if source == "message":
        types.append("direct")
    if source in {"context", "context_aware"}:
        types.append("indirect")
    attack_type = None if context_signal is None else context_signal.get("attackType")
    if attack_type and attack_type != "none":
        if attack_type == "mixed":
            types.extend(["direct", "indirect"])
        else:
            types.append(str(attack_type))
    context_score = 0.0
    if context_signal is not None:
        context_score = _score(context_signal.get("riskScore"))
    if context_score >= 0.5:
        types.append("context_mismatch")
    matched_rules = model_scores.get("ruleBased", {}).get("matchedRules", [])
    if matched_rules:
        types.append("suspicious_rule_match")
    return _unique_preserving_order(types or ["direct"])


def _details(
    *,
    selected_runtime: dict[str, Any],
    primary_runtime: dict[str, Any],
    context_signal: dict[str, Any] | None,
    decision: str,
    risk_score: float,
) -> dict[str, Any]:
    context_score = _score(context_signal.get("riskScore")) if context_signal else 0.0
    policy = _policy_result_from_runtime(primary_runtime)
    if decision == "blocked":
        threshold = 0.7
    elif decision == "warning":
        threshold = 0.3
    else:
        threshold = 0.3
    return {
        "ruleScore": _score(primary_runtime.get("rule_score")),
        "robertaScore": _score(primary_runtime.get("roberta_score")),
        "contextAwareScore": context_score,
        "fusionScore": _score(selected_runtime.get("final_risk", risk_score)),
        "threshold": threshold,
        "policyAction": {"safe": "allow", "warning": "warn", "blocked": "block"}[decision],
        "policyDecision": policy.get("decisionPolicy"),
        "riskLevel": policy.get("riskLevel"),
    }


def check_chat_message(
    *,
    message: str,
    context: str | None = None,
    session_id: str | None = None,
    conversation_id: str | None = None,
    project_id: str | None = None,
    project_context: dict[str, Any] | None = None,
    document_signal: dict[str, Any] | None = None,
    user_id: str = "demo-user",
    request_id: str | None = None,
    use_cuda: bool = False,
    user_role: str = "user",
    requested_tools: list[dict[str, Any]] | None = None,
    input_type: str = "chat",
) -> dict[str, Any]:
    """Check one end-user chat message with the shared backend pipeline."""
    if message is None or not str(message).strip():
        raise ValueError("Field 'message' must not be empty.")

    result = run_official_runtime(
        message=message,
        user_id=user_id,
        project_id=project_id,
        conversation_id=conversation_id,
        project_context=project_context,
        explicit_context=context,
        document_signal=document_signal,
        session_id=session_id,
        request_id=request_id,
        use_cuda=use_cuda,
        user_role=user_role,
        requested_tools=requested_tools,
        input_type=input_type,
    )
    result["projectContext"] = {
        "projectId": None if not project_context else project_context.get("projectId"),
        "projectName": None if not project_context else project_context.get("projectName"),
        "documentCount": 0 if not project_context else len(project_context.get("documents", []) or []),
    }
    return result
