"""Rule signal adapter for the official runtime pipeline."""

from __future__ import annotations

import hashlib
from typing import Any

from src.rule_based import detect_by_rules


RULE_CODE_BY_GROUP = {
    "instruction_override": "PI_IGNORE_PREVIOUS",
    "system_prompt_extraction": "PI_REVEAL_SYSTEM_PROMPT",
    "jailbreak": "PI_JAILBREAK",
    "code_execution": "PI_CODE_EXECUTION",
    "data_leakage": "PI_DATA_EXFILTRATION",
    "tool_abuse": "PI_TOOL_ABUSE",
    "cyber_abuse": "PI_CYBER_ABUSE",
    "encoded_obfuscation": "PI_ENCODED_OBFUSCATION",
    "role_manipulation": "PI_ROLE_MANIPULATION",
    "policy_override": "PI_OVERRIDE_POLICY",
}


def _score(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _mask(value: Any) -> str:
    text = str(value or "")
    if not text:
        return "<empty>"
    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"<masked:{digest}:len={len(text)}>"


def _rule_code(rule: dict[str, Any]) -> str:
    base = RULE_CODE_BY_GROUP.get(str(rule.get("group", "")), "PI_RULE_MATCH")
    if rule.get("severity") == "critical" and base in {"PI_TOOL_ABUSE", "PI_CODE_EXECUTION", "PI_CYBER_ABUSE"}:
        return f"{base}_CRITICAL"
    return base


def detect_rule_signal(text: str, source_type: str = "user_prompt") -> dict[str, Any]:
    raw = detect_by_rules(text, source_type=source_type)
    matched_rules: list[dict[str, Any]] = []
    for rule in raw.get("matched_rules", []) or []:
        if not isinstance(rule, dict):
            continue
        code = _rule_code(rule)
        matched_value = str(rule.get("matched_text") or rule.get("pattern") or "")
        start = str(text).lower().find(matched_value.lower()) if matched_value else -1
        matched_rules.append(
            {
                "code": code,
                "name": str(rule.get("description") or rule.get("group") or code),
                "severity": str(rule.get("severity", "medium")),
                "score": _score(rule.get("weight", raw.get("rule_score", 0.0))),
                "matchedText": _mask(rule.get("matched_text") or rule.get("pattern")),
                "matchedSpan": None if start < 0 else {"start": start, "end": start + len(matched_value)},
                "group": rule.get("group"),
                "stage": "output" if source_type == "llm_output" else "input",
                "sourceType": source_type,
                "language": raw.get("detected_language"),
                "version": "rule-runtime-v1",
                "explanation": str(rule.get("description") or rule.get("group") or code),
            }
        )

    score = _score(raw.get("rule_score", raw.get("risk_score", 0.0)))
    hard_block = bool(raw.get("has_critical_rule")) or any(
        item["code"].endswith("_CRITICAL") or item["code"] == "PI_DATA_EXFILTRATION"
        for item in matched_rules
    )
    return {
        "score": score,
        "matchedRules": matched_rules,
        "hardBlock": hard_block,
        "highestSeverity": raw.get("highest_severity", "none"),
        "action": raw.get("action", "allow"),
        "language": raw.get("detected_language"),
        "canonicalText": raw.get("canonical_text"),
        "raw": raw,
    }
