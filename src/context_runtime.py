"""Context-aware signal for the official runtime pipeline."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from src.detector import detect_prompt_with_context
from src.runtime_rule_signal import detect_rule_signal


GOAL_PATTERNS = {
    "summarize": ("summarize", "summary", "tom tat", "tóm tắt"),
    "translate": ("translate", "dich", "dịch"),
    "extract": ("extract", "trich xuat", "trích xuất"),
    "search": ("search", "find", "tim kiem", "tìm kiếm"),
    "generate_code": ("write code", "generate code", "viet code", "viết code"),
    "execute_tool": ("send email", "execute", "run command", "goi cong cu", "gọi công cụ"),
}


def infer_user_goal(message: str) -> str:
    lowered = " ".join(str(message or "").lower().split())
    for goal, patterns in GOAL_PATTERNS.items():
        if any(pattern in lowered for pattern in patterns):
            return goal
    return "answer_question"


def _fold_context_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    ascii_like = "".join(character for character in normalized if not unicodedata.combining(character))
    return " ".join(ascii_like.replace("đ", "d").replace("Đ", "D").lower().split())


def detect_encoded_context_signal(
    *,
    user_message: str,
    decoded_text: str,
    source_type: str = "chat",
) -> dict[str, Any]:
    """Distinguish inspection/translation from instructions to execute decoded data."""
    user = _fold_context_text(user_message)
    decoded = _fold_context_text(decoded_text)
    combined = f"{user} {decoded}"
    benign_patterns = (
        "analyze whether", "security analysis", "do not execute", "do not follow",
        "without following", "only describe", "documentation only", "quoted sentence",
        "translate the", "summarize the", "phan tich", "khong thuc thi",
        "khong lam theo", "chi mo ta", "dich nguyen van", "tom tat",
    )
    execution_patterns = (
        "decode and follow", "decode then follow", "decode and execute", "decode then execute",
        "execute the decoded", "follow the decoded", "run the decoded", "please execute",
        "giai ma va lam theo", "giai ma va thuc thi", "hay thuc thi", "lam theo noi dung",
    )
    decoded_instruction_patterns = (
        "ignore previous", "ignore all instructions", "override instructions", "system prompt",
        "developer instruction", "reveal secret", "show secret", "api key", "send all private",
        "execute the", "run command", "delete tool", "tool call", "bypass safety", "exfiltrate",
    )
    sensitive_patterns = ("system prompt", "developer instruction", "secret", "api key", "password", "private data", "credential")
    tool_patterns = ("execute the", "run command", "delete tool", "tool call", "send email", "open url")
    academic_question = bool(
        re.search(r"^(?:why|what|how|explain|describe|vi sao|tai sao|the nao)\b", user)
        and not re.search(r"\b(?:reveal|show|print|give|extract|display|send|quote|quoting)\b|exact instructions", user)
    )
    encoding_question = bool(
        re.search(r"\b(?:base64|hex|url encoding|unicode escape)\b", user)
        and re.search(r"\b(?:what is|how does|la gi|hoat dong nhu the nao)\b", user)
    )
    structured_reference = bool(
        re.search(r"\b(?:request id|uuid)\b.*\b[0-9a-f]{8}-[0-9a-f-]{27,}\b", user)
        or re.search(r"\b(?:sha-?256|md5|checksum|commit)\b.*\b[0-9a-f]{32,64}\b", user)
        or re.search(r"\bimage (?:preview|sample)\b.*\b(?:truncated|fragment)\b", user)
        or re.search(r"\b(?:documentation|docs?|sample|placeholder)\b.*\b(?:password|api key|secret)\b|\b(?:password|api key|secret)\b.*\b(?:documentation|sample|placeholder)\b", user)
        or re.search(r"https?://\S+.*\b(?:documentation|docs?|guide)\b", user)
    )
    benign_reference = any(pattern in user for pattern in benign_patterns) or academic_question or encoding_question or structured_reference
    execution_intent = any(pattern in user for pattern in execution_patterns)
    decoded_instruction = any(pattern in decoded for pattern in decoded_instruction_patterns)
    sensitive_target = any(pattern in decoded for pattern in sensitive_patterns)
    tool_activation = any(pattern in decoded for pattern in tool_patterns)
    external = source_type not in {"chat", "user_prompt"}

    score = 0.0
    reasons: list[str] = []
    if decoded_instruction:
        score = max(score, 0.55)
        reasons.append("ENC_DECODED_INSTRUCTION")
    if external and decoded_instruction:
        score = max(score, 0.78)
        reasons.append("ENC_UNTRUSTED_SOURCE_INSTRUCTION")
    if execution_intent and decoded_instruction:
        score = max(score, 0.92)
        reasons.append("ENC_DECODE_AND_EXECUTE_INTENT")
    if sensitive_target and decoded_instruction:
        score = max(score, 0.90)
        reasons.append("ENC_SENSITIVE_TARGET")
    if tool_activation and decoded_instruction:
        score = max(score, 0.94)
        reasons.append("ENC_TOOL_ACTIVATION")
    effective_benign_reference = benign_reference and not external
    if effective_benign_reference and not execution_intent:
        score = min(score, 0.24)
        reasons.append("ENC_BENIGN_REFERENCE_CONTEXT")

    category = "none"
    if tool_activation:
        category = "encoded_tool_activation"
    elif any(pattern in decoded for pattern in ("send all private", "exfiltrate", "send the secret", "attacker")):
        category = "encoded_data_exfiltration"
    elif "system prompt" in decoded or "developer instruction" in decoded:
        category = "encoded_system_prompt_extraction"
    elif "jailbreak" in decoded or "bypass safety" in decoded:
        category = "encoded_jailbreak"
    elif sensitive_target:
        category = "encoded_sensitive_extraction"
    elif decoded_instruction:
        category = "encoded_instruction_override"
    return {
        "score": round(score, 6),
        "mismatch": bool(decoded_instruction and not effective_benign_reference),
        "decodedInstruction": decoded_instruction,
        "executionIntent": execution_intent,
        "benignReferenceIntent": effective_benign_reference,
        "educationalOrAnalysisContext": benign_reference,
        "externalContent": external,
        "sensitiveTarget": sensitive_target,
        "toolActivation": tool_activation,
        "category": category,
        "reasonCodes": reasons,
        "goal": infer_user_goal(user_message),
        "quotedContent": bool(re.search(r"['\"]", str(user_message or ""))),
        "instructionSourceMismatch": bool(external and decoded_instruction),
    }


def _score(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def project_context_to_text(project_context: dict[str, Any] | None, explicit_context: str | None = None) -> str:
    if not project_context and not explicit_context:
        return ""
    parts: list[str] = []
    if project_context:
        parts.extend(
            [
                f"Project purpose: {project_context.get('projectDescription', '')}",
                f"Project system instruction: {project_context.get('systemInstruction', '')}",
                f"Project context summary: {project_context.get('contextSummary', '')}",
            ]
        )
        for document in project_context.get("documents", []) or []:
            if not isinstance(document, dict):
                continue
            parts.append(
                f"Untrusted document {document.get('title', 'Untitled')}: {document.get('content', '')}"
            )
    if explicit_context:
        parts.append(f"Explicit user-supplied context: {explicit_context}")
    return "\n\n".join(part for part in parts if part and part.strip())


def detect_context_signal(
    *,
    user_message: str,
    project_context: dict[str, Any] | None = None,
    explicit_context: str | None = None,
) -> dict[str, Any]:
    context_text = project_context_to_text(project_context, explicit_context)
    goal = infer_user_goal(user_message)
    if not context_text.strip():
        return {
            "score": 0.0,
            "mismatch": False,
            "reasonCodes": [],
            "evidence": [],
            "attackType": "none",
            "goal": goal,
            "goalMismatch": 0.0,
            "sourceMismatch": 0.0,
            "privilegeEscalation": 0.0,
            "toolMismatch": 0.0,
            "sensitiveTargetScore": 0.0,
            "contextRisk": 0.0,
        }

    context_result = detect_prompt_with_context(
        user_prompt=user_message,
        context=context_text,
        model_type="hybrid",
    )
    context_rules = detect_rule_signal(context_text, source_type="external_content")
    context_score = max(_score(context_result.get("risk_score")), _score(context_rules.get("score")))
    evidence: list[dict[str, Any]] = []
    for rule in context_result.get("matched_rules", []) or []:
        if isinstance(rule, dict):
            evidence.append(
                {
                    "source": rule.get("source", "context"),
                    "code": rule.get("group", "context_rule"),
                    "severity": rule.get("severity"),
                    "matchedText": "<masked>",
                }
            )
    for rule in context_rules.get("matchedRules", []) or []:
        evidence.append(rule)

    reason_codes: list[str] = []
    attack_type = str(context_result.get("attack_type", "none"))
    if attack_type in {"indirect", "mixed"}:
        reason_codes.append("CTX_INDIRECT_INSTRUCTION")
    if context_rules.get("matchedRules"):
        reason_codes.append("CTX_RULE_MATCH")
    if context_score >= 0.30 and not reason_codes:
        reason_codes.append("CTX_CONTEXT_MISMATCH")

    context_rule_signal = {
        "score": context_rules.get("score", 0.0),
        "matchedRules": context_rules.get("matchedRules", []),
        "hardBlock": context_rules.get("hardBlock", False),
        "highestSeverity": context_rules.get("highestSeverity", "none"),
    }

    matched_codes = {
        str(item.get("code", ""))
        for item in context_rules.get("matchedRules", []) or []
        if isinstance(item, dict)
    }
    tool_signal = any("TOOL" in code or "EXECUTION" in code for code in matched_codes)
    sensitive_signal = any(
        marker in " ".join(matched_codes)
        for marker in ("EXFILTRATION", "SYSTEM_PROMPT", "CREDENTIAL", "SECRET")
    )
    assistant_directed = bool(context_rules.get("matchedRules")) or attack_type in {"indirect", "mixed"}
    goal_mismatch = context_score if assistant_directed and goal in {"summarize", "translate", "extract", "search"} else 0.0
    source_mismatch = min(1.0, context_score + 0.10) if assistant_directed else 0.0
    privilege_escalation = max(0.70, context_score) if tool_signal or sensitive_signal else 0.0
    tool_mismatch = max(0.75, context_score) if tool_signal and goal != "execute_tool" else 0.0
    sensitive_target = max(0.80, context_score) if sensitive_signal else 0.0
    context_risk = max(context_score, goal_mismatch, source_mismatch, privilege_escalation, tool_mismatch, sensitive_target)

    return {
        "score": round(context_risk, 6),
        "mismatch": context_risk >= 0.30,
        "reasonCodes": sorted(set(reason_codes)),
        "evidence": evidence,
        "attackType": attack_type,
        "contextRuleSignal": context_rule_signal,
        "goal": goal,
        "goalMismatch": round(goal_mismatch, 6),
        "sourceMismatch": round(source_mismatch, 6),
        "privilegeEscalation": round(privilege_escalation, 6),
        "toolMismatch": round(tool_mismatch, 6),
        "sensitiveTargetScore": round(sensitive_target, 6),
        "contextRisk": round(context_risk, 6),
    }
