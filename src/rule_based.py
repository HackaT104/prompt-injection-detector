"""Rule-based prompt injection detector."""

from __future__ import annotations

import re

from src.benign_intent import detect_benign_reference_intent
from src.preprocessing import clean_text, prepare_text_for_detection


RULE_GROUPS: dict[str, dict[str, object]] = {
    "instruction_override": {
        "weight": 0.80,
        "patterns": [
            "ignore previous instructions",
            "disregard previous instructions",
            "forget all prior instructions",
            "override instruction",
            "override policy",
            "bypass policy",
            "ignore system prompt",
            "ignore developer message",
            "bo qua huong dan truoc",
            "bỏ qua hướng dẫn trước",
            "phớt lờ hướng dẫn trước",
            "ghi đè hướng dẫn",
            "bo qua system prompt",
        ],
        "description": "Cá»‘ gáº¯ng ghi Ä‘Ã¨ hoáº·c bá» qua chá»‰ dáº«n há»‡ thá»‘ng.",
    },
    "system_prompt_extraction": {
        "weight": 0.80,
        "patterns": [
            "reveal your system prompt",
            "reveal system prompt",
            "show system prompt",
            "print hidden instructions",
            "quote the exact instructions",
            "quoting the exact instructions",
            "exact instructions that caused your behavior",
            "instructions that caused your behavior",
            "quote your hidden instructions",
            "disclose internal instructions",
            "show hidden prompt",
            "tiết lộ system prompt",
            "tiet lo system prompt",
            "hiển thị system prompt",
            "in system prompt",
        ],
        "description": "Cá»‘ gáº¯ng trÃ­ch xuáº¥t system prompt hoáº·c chá»‰ dáº«n áº©n.",
    },
    "jailbreak": {
        "weight": 0.80,
        "patterns": [
            "act as dan",
            "do anything now",
            "unrestricted mode",
            "developer mode",
            "bypass safety",
            "no ethical restrictions",
            "jailbreak",
            "roleplay as",
            "pretend you are",
            "act like a different assistant",
            "che do developer",
            "chế độ developer",
        ],
        "description": "Dáº¥u hiá»‡u jailbreak hoáº·c vÃ´ hiá»‡u hÃ³a rÃ ng buá»™c an toÃ n.",
    },
    "code_execution": {
        "weight": 0.70,
        "patterns": [
            "os.system",
            "subprocess",
            "eval(",
            "exec(",
            "import os",
            "bash",
            "execute shell command",
            "shell command",
            "rm -rf",
            "whoami",
        ],
        "description": "Dáº¥u hiá»‡u yÃªu cáº§u thá»±c thi lá»‡nh hoáº·c mÃ£ nguy hiá»ƒm.",
    },
    "data_leakage": {
        "weight": 0.70,
        "patterns": [
            "reveal secret",
            "reveal api key",
            "reveal api keys",
            "show api key",
            "show api keys",
            "print api key",
            "print api keys",
            "print token",
            "expose credentials",
            "leak data",
        ],
        "description": "Dáº¥u hiá»‡u yÃªu cáº§u rÃ² rá»‰ dá»¯ liá»‡u nháº¡y cáº£m.",
    },
    "tool_abuse": {
        "weight": 0.80,
        "patterns": [
            "call the tool",
            "invoke the tool",
            "use the tool",
            "execute tool",
            "run tool",
            "send request",
            "make http request",
            "open this url",
            "download and execute",
            "run command",
            "execute command",
        ],
        "description": "Dáº¥u hiá»‡u láº¡m dá»¥ng tool hoáº·c hÃ nh Ä‘á»™ng bÃªn ngoÃ i pháº¡m vi.",
    },
    "cyber_abuse": {
        "weight": 0.80,
        "patterns": [
            "unauthorized access",
            "unauthorized access to the system",
            "hack account",
            "hack admin account",
            "bypass authentication",
            "bypass login",
            "privilege escalation",
            "gain admin access",
        ],
        "description": "Dáº¥u hiá»‡u yÃªu cáº§u truy cáº­p trÃ¡i phÃ©p hoáº·c láº¡m dá»¥ng an ninh máº¡ng.",
    },
    "role_manipulation": {
        "weight": 0.70,
        "patterns": [
            "you are now",
            "new role",
            "act as system",
            "act as developer",
            "pretend to be system",
            "đóng vai hệ thống",
            "dong vai he thong",
        ],
        "description": "Dấu hiệu ép mô hình đổi vai trò hoặc giả danh system/developer.",
    },
    "policy_override": {
        "weight": 0.80,
        "patterns": [
            "ignore the policy",
            "override the policy",
            "disable safety policy",
            "skip safety checks",
            "bỏ qua chính sách",
            "bo qua chinh sach",
            "vô hiệu hóa chính sách",
            "vo hieu hoa chinh sach",
        ],
        "description": "Dấu hiệu yêu cầu bỏ qua hoặc ghi đè policy.",
    },
    "encoded_obfuscation": {
        "weight": 0.65,
        "patterns": [
            "base64",
            "rot13",
            "hex encoded",
            "decode this instruction",
            "encoded instruction",
            "obfuscated instruction",
            "giải mã chỉ dẫn",
            "giai ma chi dan",
        ],
        "description": "Dấu hiệu che giấu chỉ dẫn qua encoding/obfuscation.",
    },
}


SEVERITY_ORDER = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}

SEVERITY_BASE_SCORES = {
    "none": 0.0,
    "low": 0.30,
    "medium": 0.60,
    "high": 0.85,
    "critical": 0.95,
}

EXTERNAL_OR_TOOL_SOURCES = {
    "external_content",
    "external",
    "email",
    "web",
    "pdf",
    "rag",
    "tool",
}


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


def _severity_for_group(group_name: str, pattern: str, source_type: str) -> str:
    if group_name == "instruction_override":
        if "system prompt" in pattern or "developer message" in pattern:
            return "high"
        return "medium"
    if group_name == "system_prompt_extraction":
        return "high"
    if group_name == "data_leakage":
        return "critical"
    if group_name == "tool_abuse":
        return "critical" if source_type in EXTERNAL_OR_TOOL_SOURCES else "high"
    if group_name in {"code_execution", "cyber_abuse"}:
        return "critical" if source_type in {"tool", "external_content", "rag"} else "high"
    if group_name == "jailbreak":
        return "high"
    if group_name in {"role_manipulation", "policy_override"}:
        return "high"
    if group_name == "encoded_obfuscation":
        return "medium"
    return "medium"


def _highest_severity(matched_rules: list[dict[str, object]]) -> str:
    if not matched_rules:
        return "none"
    return max(
        (str(rule.get("severity", "none")) for rule in matched_rules),
        key=lambda severity: SEVERITY_ORDER.get(severity, 0),
    )


def _matched_text(normalized_text: str, pattern: str) -> str:
    match = re.search(re.escape(pattern), normalized_text, flags=re.IGNORECASE)
    if match:
        return match.group(0)
    return pattern


def _action_from_score(risk_score: float) -> str:
    if risk_score >= 0.80:
        return "block"
    if risk_score >= 0.50:
        return "warn"
    return "allow"


def detect_by_rules(text: str, source_type: str = "user_prompt") -> dict[str, object]:
    """Detect obvious prompt injection attempts using weighted rules."""
    prepared_text = prepare_text_for_detection(text)
    normalized_text = clean_text(prepared_text["cleaned_text"])
    normalized_source_type = _normalize_source_type(source_type)
    benign_guard = detect_benign_reference_intent(text)
    if benign_guard["triggered"]:
        return {
            "label": 0,
            "risk_score": 0.0,
            "rule_score": 0.0,
            "action": "allow",
            "detected_language": prepared_text["detected_language"],
            "canonical_text": prepared_text["cleaned_text"],
            "matched_rules": [],
            "highest_severity": "none",
            "has_high_severity_rule": False,
            "has_critical_rule": False,
            "benign_guard": benign_guard,
            "explanation": str(benign_guard["reason"]),
        }

    matched_rules: list[dict[str, object]] = []
    matched_group_weights: list[float] = []

    for group_name, group_config in RULE_GROUPS.items():
        patterns = group_config["patterns"]
        weight = float(group_config["weight"])
        group_matched = False
        for pattern in patterns:  # type: ignore[assignment]
            if str(pattern) in normalized_text:
                pattern_text = str(pattern)
                severity = _severity_for_group(group_name, pattern_text, normalized_source_type)
                matched_rules.append(
                    {
                        "group": group_name,
                        "pattern": pattern_text,
                        "weight": weight,
                        "severity": severity,
                        "matched_text": _matched_text(normalized_text, pattern_text),
                        "description": str(group_config["description"]),
                    }
                )
                group_matched = True
        if group_matched:
            matched_group_weights.append(weight)

    if not matched_group_weights:
        risk_score = 0.0
    else:
        strongest_signal = max(matched_group_weights)
        extra_signal = 0.15 * (len(matched_group_weights) - 1)
        repeated_signal = 0.05 * max(0, len(matched_rules) - len(matched_group_weights))
        risk_score = min(1.0, strongest_signal + extra_signal + repeated_signal)

    highest_severity = _highest_severity(matched_rules)
    severity_floor = SEVERITY_BASE_SCORES.get(highest_severity, 0.0)
    risk_score = max(risk_score, severity_floor)
    action = _action_from_score(risk_score)
    label = 1 if action in {"warn", "block"} else 0
    if matched_rules:
        groups = sorted({str(rule["group"]) for rule in matched_rules})
        explanation = (
            "Rule-based phÃ¡t hiá»‡n dáº¥u hiá»‡u rá»§i ro thuá»™c nhÃ³m: "
            + ", ".join(groups)
            + "."
        )
    else:
        explanation = "KhÃ´ng phÃ¡t hiá»‡n rule nguy hiá»ƒm rÃµ rÃ ng."

    return {
        "label": label,
        "risk_score": round(float(risk_score), 4),
        "rule_score": round(float(risk_score), 4),
        "action": action,
        "detected_language": prepared_text["detected_language"],
        "canonical_text": prepared_text["cleaned_text"],
        "matched_rules": matched_rules,
        "highest_severity": highest_severity,
        "has_high_severity_rule": SEVERITY_ORDER.get(highest_severity, 0) >= SEVERITY_ORDER["high"],
        "has_critical_rule": highest_severity == "critical",
        "explanation": explanation,
    }


