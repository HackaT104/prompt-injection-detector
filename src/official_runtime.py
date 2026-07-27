"""Official runtime pipeline for User Chat detection.

This path intentionally excludes XLM-R from runtime decisions. XLM-R remains
available elsewhere for benchmark/experimental evaluation.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any
from uuid import uuid4
from datetime import datetime, timezone

from src.context_runtime import detect_context_signal
from src.llm_service import call_llm
from src.roberta_runtime import roberta_service
from src.runtime_config import load_runtime_config
from src.runtime_rule_signal import detect_rule_signal
from src.security.pipeline import security_pipeline
from src.security.variant_analysis import analyze_security_variants


def _score(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _normalize_weights(weights: dict[str, Any]) -> dict[str, float]:
    raw = {
        "rule": _score(weights.get("rule", 0.30)),
        "roberta": _score(weights.get("roberta", 0.50)),
        "context": _score(weights.get("context", 0.20)),
    }
    total = sum(raw.values()) or 1.0
    return {key: value / total for key, value in raw.items()}


def fuse_runtime_scores(
    *,
    rule_score: float,
    roberta_score: float,
    context_score: float,
    weights: dict[str, Any],
    source_risk: float = 0.0,
    obfuscation_score: float = 0.0,
    tool_risk: float = 0.0,
    sensitive_target_score: float = 0.0,
    source_type: str = "chat",
    selected_variant_score: float = 0.0,
    decode_depth: int = 0,
    variant_confidence: float = 0.0,
    decoded_malicious: bool = False,
    execution_intent: bool = False,
    benign_reference_intent: bool = False,
    attack_category: str | None = None,
) -> dict[str, Any]:
    normalized_weights = _normalize_weights(weights)
    contributions = {
        "rule": round(normalized_weights["rule"] * _score(rule_score), 6),
        "roberta": round(normalized_weights["roberta"] * _score(roberta_score), 6),
        "context": round(normalized_weights["context"] * _score(context_score), 6),
    }
    adaptive_contributions: dict[str, float] = {}
    if any(_score(value) > 0 for value in [source_risk, obfuscation_score, tool_risk, sensitive_target_score]):
        source_factor = 0.10 if source_type in {"external_document", "web", "email", "rag"} else 0.03
        adaptive_contributions = {
            "source": round(source_factor * _score(source_risk), 6),
            "obfuscation": round(0.12 * _score(obfuscation_score), 6),
            "tool": round(0.20 * _score(tool_risk), 6),
            "sensitiveTarget": round(0.10 * _score(sensitive_target_score), 6),
        }
    if _score(selected_variant_score) > _score(roberta_score) and not benign_reference_intent:
        adaptive_contributions["selectedVariant"] = round(
            0.22 * _score(selected_variant_score) * max(0.4, _score(variant_confidence)), 6
        )
    contributions.update(adaptive_contributions)
    fusion_score = round(min(1.0, sum(contributions.values())), 6)
    overrides: list[str] = []
    if _score(tool_risk) >= 0.90 or _score(sensitive_target_score) >= 0.95:
        fusion_score = max(fusion_score, 0.90)
        overrides.append("CRITICAL_SIGNAL_FLOOR")
    if decoded_malicious and not benign_reference_intent:
        floor = 0.82
        code = "ENCODED_MALICIOUS_CONTENT_FLOOR"
        if source_type != "chat":
            floor, code = 0.88, "UNTRUSTED_ENCODED_INSTRUCTION_FLOOR"
        if execution_intent:
            floor, code = 0.92, "DECODE_AND_EXECUTE_FLOOR"
        if _score(selected_variant_score) >= 0.90:
            floor, code = max(floor, 0.90), "MALICIOUS_VARIANT_MODEL_FLOOR"
        if decode_depth >= 2:
            floor, code = max(floor, 0.93), "NESTED_ENCODING_FLOOR"
        if _score(sensitive_target_score) >= 0.90:
            floor, code = 0.97, "ENCODED_SENSITIVE_TARGET_FLOOR"
        if _score(tool_risk) >= 0.90:
            floor, code = 0.97, "ENCODED_TOOL_ACTIVATION_FLOOR"
        if attack_category == "encoded_system_prompt_extraction":
            floor, code = max(floor, 0.95), "ENCODED_SYSTEM_PROMPT_EXTRACTION"
        if attack_category in {"encoded_data_exfiltration", "encoded_sensitive_extraction"}:
            floor, code = max(floor, 0.97), "ENCODED_DATA_EXFILTRATION"
        if attack_category == "encoded_tool_activation":
            floor, code = max(floor, 0.97), "ENCODED_DESTRUCTIVE_TOOL_CALL"
        fusion_score = max(fusion_score, floor)
        overrides.append(code)
    if benign_reference_intent and not execution_intent:
        fusion_score = min(fusion_score, 0.24)
        overrides.append("BENIGN_REFERENCE_RISK_CAP")
    positive = {key: value for key, value in contributions.items() if value > 0}
    highest_risk_source = max(positive, key=positive.get) if positive else "none"
    return {
        "fusionScore": fusion_score,
        "highestRiskSource": highest_risk_source,
        "contributions": contributions,
        "weights": normalized_weights,
        "sourceType": source_type,
        "overridesApplied": overrides,
        "explanation": [
            f"{name} contributed {value:.3f} to final risk."
            for name, value in contributions.items()
            if value > 0
        ],
    }


def apply_policy(
    *,
    fusion_score: float,
    rule_signal: dict[str, Any],
    roberta_signal: dict[str, Any],
    context_signal: dict[str, Any],
    config: dict[str, Any],
    variant_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    thresholds = config["thresholds"]
    warn = float(thresholds["warn"])
    block = float(thresholds["block"])
    reason_codes: list[str] = []
    decision = "safe"
    action = "allow"
    context_score = _score(context_signal.get("score"))
    context_mismatch = bool(context_signal.get("mismatch"))
    roberta_unavailable = not roberta_signal.get("available", True)
    rule_score = _score(rule_signal.get("score"))
    rule_action = str(rule_signal.get("action", "allow")).lower()
    variant_analysis = variant_analysis or {}
    benign_reference = bool(
        variant_analysis.get("benignReferenceIntent") or variant_analysis.get("benignTransformedPayload")
    ) and not bool(variant_analysis.get("executionIntent"))
    decoded_malicious = bool(variant_analysis.get("decodedMaliciousContent"))

    hard_block_codes = set(config.get("hardBlockRuleCodes", []))
    matched_rule_codes = {
        str(rule.get("code"))
        for rule in rule_signal.get("matchedRules", [])
        if isinstance(rule, dict)
    }
    context_rule_signal = context_signal.get("contextRuleSignal", {})
    context_hard_block = isinstance(context_rule_signal, dict) and bool(context_rule_signal.get("hardBlock"))
    if benign_reference:
        decision = "safe"
        action = "allow"
        reason_codes.append("POLICY_BENIGN_ENCODED_OR_QUOTED_REFERENCE")
    elif decoded_malicious and fusion_score >= block:
        decision = "blocked"
        action = "block"
        reason_codes.append("POLICY_ENCODED_MALICIOUS_BLOCK")
    elif rule_signal.get("hardBlock") or matched_rule_codes.intersection(hard_block_codes):
        decision = "blocked"
        action = "block"
        reason_codes.append("POLICY_HARD_BLOCK_RULE")
    elif context_hard_block:
        decision = "blocked"
        action = "block"
        reason_codes.append("POLICY_CONTEXT_HARD_BLOCK_RULE")
    elif context_mismatch and context_score >= block:
        decision = "blocked"
        action = "block"
        reason_codes.append("POLICY_CONTEXT_BLOCK_THRESHOLD")
    elif fusion_score >= block:
        decision = "blocked"
        action = "block"
        reason_codes.append("POLICY_BLOCK_THRESHOLD")
    elif roberta_unavailable:
        decision = "warning"
        action = "warn"
        reason_codes.append("RUNTIME_MODEL_ERROR")
    elif context_mismatch and context_score >= warn:
        decision = "warning"
        action = "warn"
        reason_codes.append("POLICY_CONTEXT_WARN_THRESHOLD")
    elif rule_action in {"warn", "block"} or rule_score >= warn:
        decision = "warning"
        action = "warn"
        reason_codes.append("POLICY_RULE_WARN_SIGNAL")
    elif fusion_score >= warn:
        decision = "warning"
        action = "warn"
        reason_codes.append("POLICY_WARN_THRESHOLD")
    else:
        reason_codes.append("POLICY_SAFE_THRESHOLD")

    if roberta_unavailable and "RUNTIME_MODEL_ERROR" not in reason_codes:
        reason_codes.append("RUNTIME_MODEL_ERROR")
    if context_signal.get("mismatch"):
        reason_codes.extend(context_signal.get("reasonCodes", []))
    policy_id_by_reason = {
        "POLICY_HARD_BLOCK_RULE": "POL-INPUT-CRITICAL-RULE",
        "POLICY_CONTEXT_HARD_BLOCK_RULE": "POL-INDIRECT-CRITICAL",
        "POLICY_CONTEXT_BLOCK_THRESHOLD": "POL-INDIRECT-BLOCK",
        "POLICY_BLOCK_THRESHOLD": "POL-INPUT-RISK-BLOCK",
        "RUNTIME_MODEL_ERROR": "POL-INPUT-MODEL-FAIL-SAFE",
        "POLICY_CONTEXT_WARN_THRESHOLD": "POL-INDIRECT-QUARANTINE",
        "POLICY_RULE_WARN_SIGNAL": "POL-INPUT-RULE-WARN",
        "POLICY_WARN_THRESHOLD": "POL-INPUT-RISK-WARN",
        "POLICY_SAFE_THRESHOLD": "POL-INPUT-ALLOW",
        "POLICY_BENIGN_ENCODED_OR_QUOTED_REFERENCE": "POL-ENC-BENIGN-ALLOW",
        "POLICY_ENCODED_MALICIOUS_BLOCK": "POL-ENC-MALICIOUS-BLOCK",
    }
    primary_reason = reason_codes[0] if reason_codes else "POLICY_SAFE_THRESHOLD"
    policy_id = policy_id_by_reason.get(primary_reason, "POL-INPUT-RUNTIME")
    actions = {
        "allow": ["ALLOW", "ALLOW_WITH_LOG"],
        "warn": ["WARN", "RESTRICT_TOOLS"],
        "block": ["BLOCK"],
    }[action]
    attack_category = str(variant_analysis.get("attackCategory") or "none")
    if action == "block" and attack_category == "encoded_system_prompt_extraction":
        policy_id = "POL-ENC-SYSTEM-PROMPT-BLOCK"
        actions = ["BLOCK", "LOG_INCIDENT"]
    elif action == "block" and attack_category in {"encoded_data_exfiltration", "encoded_sensitive_extraction"}:
        policy_id = "POL-ENC-EXFILTRATION-BLOCK"
        actions = ["BLOCK", "LOG_INCIDENT", "ESCALATE"]
    elif action == "block" and attack_category == "encoded_tool_activation":
        policy_id = "POL-ENC-TOOL-ABUSE-BLOCK"
        actions = ["BLOCK", "RESTRICT_TOOLS", "LOG_INCIDENT"]
    elif policy_id == "POL-ENC-BENIGN-ALLOW":
        actions = ["ALLOW", "ALLOW_WITH_LOG"]
    return {
        "decision": decision,
        "label": {"safe": "SAFE", "warning": "WARNING", "blocked": "BLOCKED"}[decision],
        "action": action,
        "warnThreshold": warn,
        "blockThreshold": block,
        "policyVersion": config["policyVersion"],
        "policyId": policy_id,
        "actions": actions,
        "reasonCodes": sorted(set(reason_codes)),
        "selectedVariant": variant_analysis.get("selectedVariantId"),
        "transformationChain": variant_analysis.get("selectedTransformChain", []),
        "userSafeReason": (
            "The request contains unsafe hidden instructions and cannot be processed."
            if action == "block"
            else "The encoded or referenced content was analyzed without executing it."
            if policy_id == "POL-ENC-BENIGN-ALLOW"
            else "The request passed the configured input security policy."
        ),
        "adminTechnicalReason": sorted(set([*reason_codes, *(variant_analysis.get("reasonCodes", []) or [])])),
    }


def _detection_types(
    *,
    rule_signal: dict[str, Any],
    roberta_signal: dict[str, Any],
    context_signal: dict[str, Any],
    document_signal: dict[str, Any] | None = None,
    policy: dict[str, Any],
) -> list[str]:
    types: list[str] = []
    if rule_signal.get("score", 0.0) > 0 or roberta_signal.get("score", 0.0):
        types.append("direct")
    if context_signal.get("score", 0.0) > 0:
        types.append("indirect")
    if context_signal.get("mismatch"):
        types.append("context_mismatch")
    if rule_signal.get("matchedRules"):
        types.append("suspicious_rule_match")
    if document_signal:
        if _score(document_signal.get("score")) > 0:
            types.append("document")
        if document_signal.get("decision") != "safe":
            types.extend(["indirect", "document_indirect"])
    if policy.get("decision") == "safe" and not types:
        types.append("benign")
    if not roberta_signal.get("available", True):
        types.append("model_error")
    return sorted(set(types))


def _merge_document_signal(
    context_signal: dict[str, Any],
    document_signal: dict[str, Any] | None,
) -> dict[str, Any]:
    if not document_signal:
        return context_signal

    combined = {
        **context_signal,
        "reasonCodes": list(context_signal.get("reasonCodes", []) or []),
        "evidence": list(context_signal.get("evidence", []) or []),
        "contextRuleSignal": dict(context_signal.get("contextRuleSignal", {}) or {}),
    }
    document_score = _score(document_signal.get("score"))
    combined["score"] = max(_score(combined.get("score")), document_score)
    if document_signal.get("decision") != "safe":
        combined["mismatch"] = True
        combined["attackType"] = "indirect"
        combined["reasonCodes"].extend(document_signal.get("reasonCodes", []) or ["DOC_INDIRECT_INJECTION"])
    for item in document_signal.get("evidence", []) or []:
        if isinstance(item, dict):
            combined["evidence"].append({"source": "document", **item})

    context_rule_signal = combined["contextRuleSignal"]
    context_rule_signal["hardBlock"] = bool(context_rule_signal.get("hardBlock")) or bool(
        document_signal.get("hardBlock")
    )
    context_rule_signal["score"] = max(_score(context_rule_signal.get("score")), document_score)
    context_rule_signal["matchedRules"] = [
        *(context_rule_signal.get("matchedRules", []) or []),
        *(document_signal.get("matchedRules", []) or []),
    ]
    combined["reasonCodes"] = sorted({str(item) for item in combined["reasonCodes"] if str(item).strip()})
    return combined


def _public_document_signal(document_signal: dict[str, Any] | None) -> dict[str, Any] | None:
    if not document_signal:
        return None
    return {
        key: value
        for key, value in document_signal.items()
        if key not in {"safeContextText"}
    }


def _public_variant_analysis(variant_analysis: dict[str, Any]) -> dict[str, Any]:
    hidden = {
        "selectedRuleSignal", "selectedRoBERTaSignal", "originalRoBERTaSignal",
        "selectedContextSignal",
    }
    public = {key: value for key, value in variant_analysis.items() if key not in hidden}
    selected_rule = variant_analysis.get("selectedRuleSignal", {})
    selected_roberta = variant_analysis.get("selectedRoBERTaSignal", {})
    selected_context = variant_analysis.get("selectedContextSignal", {})
    public["selectedRuleSignal"] = {
        "score": selected_rule.get("score", 0.0),
        "matchedRules": selected_rule.get("matchedRules", []),
        "hardBlock": selected_rule.get("hardBlock", False),
        "highestSeverity": selected_rule.get("highestSeverity"),
    }
    public["selectedRoBERTaSignal"] = {
        "score": selected_roberta.get("score"),
        "rawScore": selected_roberta.get("rawScore"),
        "available": selected_roberta.get("available", False),
        "modelVersion": selected_roberta.get("modelVersion"),
        "latencyMs": selected_roberta.get("latencyMs"),
    }
    public["selectedContextSignal"] = {
        key: value for key, value in selected_context.items()
        if key not in {"decodedText", "text", "raw"}
    }
    return public


def _assistant_message(
    policy: dict[str, Any],
    risk_score: float,
    reason: str,
    llm: dict[str, Any],
) -> str:
    if policy["decision"] == "blocked":
        return (
            "This request cannot be processed because it may be unsafe. "
            "Please rewrite it as a safe, legitimate request."
        )
    if policy["decision"] == "warning":
        return (
            "Please revise this message before sending it to the AI. "
            "Remove requests to bypass rules, reveal hidden instructions, or perform unauthorized actions."
        )
    if llm.get("status") == "ok" and llm.get("content"):
        return str(llm["content"])
    if llm.get("status") == "error":
        return "The AI service is temporarily unavailable. Please try again later."
    if llm.get("status") == "skipped":
        return "The request is allowed, but the backend LLM is not configured yet."
    return "The request is allowed, but the assistant did not return a response."


def _public_source_separation(source_separation: dict[str, Any]) -> dict[str, Any]:
    def compact(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "sourceId": item.get("source_id"),
                "sourceType": item.get("source_type"),
                "trustLevel": item.get("trust_level"),
                "trusted": item.get("trusted"),
                "metadata": item.get("metadata", {}),
            }
            for item in items
        ]

    return {
        "trustedContext": compact(list(source_separation.get("trusted_context", []) or [])),
        "untrustedContent": compact(list(source_separation.get("untrusted_content", []) or [])),
        "sourceRisk": _score(source_separation.get("source_risk")),
    }


def _call_llm_with_output_security(
    *,
    user_message: str,
    project_context: dict[str, Any] | None,
    use_cuda: bool,
    max_regenerations: int,
) -> tuple[dict[str, Any], dict[str, Any] | None, int]:
    llm_result = call_llm(user_message=user_message, project_context=project_context)
    if llm_result.get("status") != "ok" or not str(llm_result.get("content", "")).strip():
        return llm_result, None, int(bool(llm_result.get("called")))

    attempts = 1
    regeneration_count = 0
    output_scan = security_pipeline.scan_output(
        text=str(llm_result.get("content", "")),
        roberta_scanner=roberta_service,
        user_input=user_message,
        use_cuda=use_cuda,
        regeneration_count=regeneration_count,
    )
    while output_scan.get("action") == "REGENERATE" and regeneration_count < max_regenerations:
        regeneration_count += 1
        retry = call_llm(
            user_message=user_message,
            project_context=project_context,
            safety_feedback="OUTPUT_SECURITY_REGENERATION",
        )
        attempts += int(bool(retry.get("called")))
        if retry.get("status") != "ok" or not str(retry.get("content", "")).strip():
            output_scan = {
                **output_scan,
                "decision": "blocked",
                "action": "SAFE_FALLBACK",
                "policyId": "POL-OUTPUT-REGENERATION-FAILED",
                "reasons": [*output_scan.get("reasons", []), "OUTPUT_REGENERATION_FAILED"],
                "finalText": str(
                    (load_runtime_config().get("output_security") or {}).get(
                        "safe_fallback",
                        "The response could not be returned because it did not pass the security checks.",
                    )
                ),
                "regenerationCount": regeneration_count,
            }
            break
        llm_result = retry
        output_scan = security_pipeline.scan_output(
            text=str(llm_result.get("content", "")),
            roberta_scanner=roberta_service,
            user_input=user_message,
            use_cuda=use_cuda,
            regeneration_count=regeneration_count,
        )

    llm_result = {
        **llm_result,
        "content": str(output_scan.get("finalText", "")),
        "outputSecurityAction": output_scan.get("action"),
        "regenerationCount": regeneration_count,
    }
    return llm_result, output_scan, attempts


def run_official_runtime(
    *,
    message: str,
    user_id: str,
    project_id: str | None = None,
    conversation_id: str | None = None,
    project_context: dict[str, Any] | None = None,
    explicit_context: str | None = None,
    document_signal: dict[str, Any] | None = None,
    session_id: str | None = None,
    request_id: str | None = None,
    use_cuda: bool = False,
    user_role: str = "user",
    requested_tools: list[dict[str, Any]] | None = None,
    invoke_llm: bool = True,
    input_type: str = "chat",
) -> dict[str, Any]:
    if message is None or not str(message).strip():
        raise ValueError("Field 'message' must not be empty.")

    started = perf_counter()
    request_id = request_id or f"req_{uuid4().hex[:16]}"
    config = load_runtime_config()
    user_message = str(message).strip()
    normalized_input_type = str(input_type or "chat").strip().lower()
    if normalized_input_type not in {"chat", "document", "web", "email", "tool_output", "rag"}:
        raise ValueError("input_type must be chat, document, web, email, tool_output, or rag.")
    source_type = (
        "external_document"
        if document_signal or normalized_input_type == "document"
        else ("chat" if normalized_input_type == "chat" else normalized_input_type)
    )
    analysis_target = (
        str(explicit_context)
        if explicit_context and normalized_input_type != "chat"
        else user_message
    )
    preprocessing = security_pipeline.preprocess_input(analysis_target)
    source_separation = security_pipeline.separate_sources(
        user_message=user_message,
        user_role=user_role,
        project_context=project_context,
        explicit_context=explicit_context,
        explicit_source_type=normalized_input_type if normalized_input_type != "chat" else "unknown",
    )

    variant_analysis = analyze_security_variants(
        preprocessing=preprocessing,
        user_message=user_message,
        source_type=source_type,
        roberta_scanner=roberta_service,
        use_cuda=use_cuda,
        stage="input",
        rule_detector=detect_rule_signal,
    )
    preprocessing["selected_variant_id"] = variant_analysis.get("selectedVariantId")
    preprocessing["obfuscation_score"] = variant_analysis.get(
        "effectiveObfuscationScore", preprocessing.get("obfuscation_score", 0.0)
    )
    rule_signal = variant_analysis["selectedRuleSignal"]
    roberta_signal = variant_analysis["originalRoBERTaSignal"]
    context_signal = detect_context_signal(
        user_message=user_message,
        project_context=project_context,
        explicit_context=explicit_context,
    )
    context_signal = _merge_document_signal(context_signal, document_signal)
    variant_context = variant_analysis.get("selectedContextSignal", {})
    if _score(variant_context.get("score")) > _score(context_signal.get("score")):
        context_signal = {
            **context_signal,
            "score": _score(variant_context.get("score")),
            "contextRisk": _score(variant_context.get("score")),
            "mismatch": bool(variant_context.get("mismatch")),
            "attackType": "indirect" if source_type != "chat" else "direct",
            "reasonCodes": sorted(set([*(context_signal.get("reasonCodes", []) or []), *(variant_context.get("reasonCodes", []) or [])])),
            "encodedContext": variant_context,
        }

    roberta_score = _score(roberta_signal.get("score")) if roberta_signal.get("available") else 0.0
    benign_variant_context = bool(
        variant_analysis.get("benignReferenceIntent") or variant_analysis.get("benignTransformedPayload")
    ) and not bool(variant_analysis.get("executionIntent"))
    if benign_variant_context:
        roberta_score = min(roberta_score, 0.24)
    matched_codes = {
        str(item.get("code", ""))
        for item in rule_signal.get("matchedRules", []) or []
        if isinstance(item, dict)
    }
    sensitive_target_score = 0.95 if matched_codes.intersection(
        {"PI_DATA_EXFILTRATION", "PI_SYSTEM_PROMPT_EXTRACTION", "PI_CREDENTIAL_REQUEST"}
    ) or variant_analysis.get("sensitiveTarget") else 0.0
    tool_risk = 0.95 if variant_analysis.get("toolActivation") else (0.90 if requested_tools and source_type != "chat" else 0.0)
    fusion = fuse_runtime_scores(
        rule_score=_score(rule_signal.get("score")),
        roberta_score=roberta_score,
        context_score=_score(context_signal.get("score")),
        weights=config.get("weights", {}),
        source_risk=_score(source_separation.get("source_risk")),
        obfuscation_score=_score(preprocessing.get("obfuscation_score")),
        tool_risk=tool_risk,
        sensitive_target_score=sensitive_target_score,
        source_type=source_type,
        selected_variant_score=_score(variant_analysis.get("selectedVariantRiskScore")),
        decode_depth=int(variant_analysis.get("selectedDepth", 0)),
        variant_confidence=_score(variant_analysis.get("selectedConfidence")),
        decoded_malicious=bool(variant_analysis.get("decodedMaliciousContent")),
        execution_intent=bool(variant_analysis.get("executionIntent")),
        benign_reference_intent=benign_variant_context,
        attack_category=variant_analysis.get("attackCategory"),
    )
    policy = apply_policy(
        fusion_score=float(fusion["fusionScore"]),
        rule_signal=rule_signal,
        roberta_signal=roberta_signal,
        context_signal=context_signal,
        config=config,
        variant_analysis=variant_analysis,
    )

    llm_policy = config.get("llm", {})
    should_call_llm = (
        policy["decision"] == "safe"
        and bool(llm_policy.get("callOnSafe", True))
        and bool(invoke_llm)
    )
    if should_call_llm:
        llm_result, output_security, llm_attempts = _call_llm_with_output_security(
            user_message=user_message,
            project_context=project_context,
            use_cuda=use_cuda,
            max_regenerations=int(llm_policy.get("maxRegenerations", 1)),
        )
    else:
        output_security = None
        llm_attempts = 0
        llm_result = {
            "called": False,
            "status": (
                "blocked_by_policy"
                if policy["decision"] == "blocked"
                else ("skipped_by_policy" if invoke_llm else "skipped_for_analysis")
            ),
            "provider": "openai-compatible",
            "model": "",
            "latencyMs": 0.0,
            "tokenUsage": {"input": 0, "output": 0, "total": 0},
            "estimatedCost": 0.0,
            "content": "",
        }

    reason = ", ".join(policy["reasonCodes"])
    detection_types = _detection_types(
        rule_signal=rule_signal,
        roberta_signal=roberta_signal,
        context_signal=context_signal,
        document_signal=document_signal,
        policy=policy,
    )
    if preprocessing.get("detected_encodings"):
        detection_types = sorted({*detection_types, "encoded_input"})
    if preprocessing.get("detected_obfuscations"):
        detection_types = sorted({*detection_types, "obfuscated_input"})
    if variant_analysis.get("decodedMaliciousContent"):
        detection_types = sorted({*detection_types, "encoded_injection"})
    if output_security and output_security.get("decision") != "safe":
        detection_types = sorted({*detection_types, "output_violation"})
    effective_risk_score = float(fusion["fusionScore"])
    if policy["decision"] in {"warning", "blocked"}:
        if any(code.startswith("POLICY_CONTEXT_") for code in policy["reasonCodes"]):
            effective_risk_score = max(effective_risk_score, _score(context_signal.get("score")))
        if "POLICY_RULE_WARN_SIGNAL" in policy["reasonCodes"]:
            effective_risk_score = max(effective_risk_score, _score(rule_signal.get("score")))
        if "POLICY_HARD_BLOCK_RULE" in policy["reasonCodes"]:
            effective_risk_score = max(effective_risk_score, _score(rule_signal.get("score")))
    effective_risk_score = round(effective_risk_score, 6)
    total_latency_ms = round((perf_counter() - started) * 1000, 3)
    output_decision = output_security.get("decision") if output_security else "not_scanned"
    warning = None
    if policy["decision"] == "warning":
        warning = "Please revise the request before it is sent to the AI."
    elif policy["decision"] == "blocked":
        warning = "The request was blocked by the security policy."
    elif output_decision in {"warning", "blocked"}:
        warning = "The AI response was sanitized or replaced by the output security policy."
    return {
        "requestId": request_id,
        "requestMetadata": {
            "requestId": request_id,
            "userId": user_id,
            "projectId": project_id,
            "conversationId": conversation_id,
            "sessionId": session_id,
            "inputType": normalized_input_type,
            "attachmentCount": 1 if document_signal else 0,
            "requestedTools": [
                str(item.get("toolName") or item.get("name") or "unknown")
                for item in requested_tools or []
                if isinstance(item, dict)
            ],
            "createdAt": datetime.now(timezone.utc).isoformat(),
        },
        "decision": policy["decision"],
        "riskScore": effective_risk_score,
        "label": policy["label"],
        "assistantMessage": _assistant_message(policy, effective_risk_score, reason, llm_result),
        "conversationId": conversation_id,
        "projectId": project_id,
        "sessionId": session_id,
        "reasons": policy["reasonCodes"],
        "detectionType": detection_types,
        "details": {
            "ruleScore": rule_signal["score"],
            "robertaScore": roberta_score,
            "contextAwareScore": context_signal["score"],
            "documentScore": None if not document_signal else _score(document_signal.get("score")),
            "fusionScore": fusion["fusionScore"],
            "effectiveRiskScore": effective_risk_score,
            "threshold": policy["warnThreshold"] if policy["decision"] != "blocked" else policy["blockThreshold"],
            "warnThreshold": policy["warnThreshold"],
            "blockThreshold": policy["blockThreshold"],
            "policyAction": policy["action"],
            "policyVersion": policy["policyVersion"],
            "ruleVersion": config["ruleVersion"],
            "modelVersion": roberta_signal.get("modelVersion"),
            "robertaRawScore": roberta_signal.get("rawScore"),
            "robertaScoreUsed": roberta_signal.get("scoreUsed"),
            "robertaIntentCategory": (
                (roberta_signal.get("runtimeBenignIntent") or {}).get("category")
                if isinstance(roberta_signal.get("runtimeBenignIntent"), dict)
                else None
            ),
            "highestRiskSource": fusion["highestRiskSource"],
            "contributions": fusion["contributions"],
            "sourceRisk": _score(source_separation.get("source_risk")),
            "obfuscationScore": _score(preprocessing.get("obfuscation_score")),
            "selectedVariantId": variant_analysis.get("selectedVariantId"),
            "selectedVariantRoBERTaScore": variant_analysis.get("selectedVariantRoBERTaScore"),
            "selectedVariantRiskScore": variant_analysis.get("selectedVariantRiskScore"),
            "decodeDepth": variant_analysis.get("selectedDepth"),
            "transformChain": variant_analysis.get("selectedTransformChain", []),
            "inputPolicyId": policy.get("policyId"),
            "outputPolicyId": None if not output_security else output_security.get("policyId"),
            "thresholdVersion": roberta_signal.get("thresholdVersion"),
            "calibratorVersion": roberta_signal.get("calibratorVersion"),
        },
        "modelScores": {
            "ruleBased": {
                "score": rule_signal["score"],
                "matchedRules": rule_signal["matchedRules"],
                "hardBlock": rule_signal["hardBlock"],
                "highestSeverity": rule_signal.get("highestSeverity"),
            },
            "roberta": roberta_signal,
            "contextAware": context_signal,
            "document": _public_document_signal(document_signal),
            "variantAnalysis": _public_variant_analysis(variant_analysis),
            "finalRisk": fusion["fusionScore"],
        },
        "fusion": fusion,
        "policyResult": policy,
        "llm": llm_result,
        "llmAttempts": llm_attempts,
        "preprocessing": {
            "normalizedText": preprocessing.get("normalized_text"),
            "detectedEncodings": preprocessing.get("detected_encodings", []),
            "obfuscationScore": preprocessing.get("obfuscation_score", 0.0),
            "warnings": preprocessing.get("warnings", []),
            "detectedObfuscations": preprocessing.get("detected_obfuscations", []),
            "variantCount": preprocessing.get("variant_count", 0),
            "maxDecodeDepth": preprocessing.get("max_decode_depth", 0),
            "selectedVariantId": variant_analysis.get("selectedVariantId"),
            "resourceGuard": preprocessing.get("resource_guard", {}),
            "latencyMs": preprocessing.get("preprocessing_latency_ms", 0.0),
        },
        "sourceSeparation": _public_source_separation(source_separation),
        "outputSecurity": output_security,
        "security": {
            "requestId": request_id,
            "inputDecision": policy["action"],
            "outputDecision": output_decision,
            "riskLevel": (
                "critical" if effective_risk_score >= 0.90 else
                "high" if effective_risk_score >= policy["blockThreshold"] else
                "medium" if effective_risk_score >= policy["warnThreshold"] else
                "low"
            ),
            "warning": warning,
        },
        "totalLatencyMs": total_latency_ms,
        "language": rule_signal.get("language"),
    }
