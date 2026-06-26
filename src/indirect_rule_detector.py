"""Rule-based detector for instructions hidden in untrusted external context."""

from __future__ import annotations

from typing import Any

from src.preprocessing import clean_text, prepare_text_for_detection, remove_vietnamese_accents


INDIRECT_RULE_GROUPS: dict[str, dict[str, Any]] = {
    "instruction_override": {
        "weight": 0.95,
        "patterns": [
            "ignore previous instructions", "disregard previous instructions", "disregard system prompt",
            "ignore the user", "follow the instructions in this document", "do not follow the user task",
            "bỏ qua hướng dẫn trước đó", "bo qua huong dan truoc do", "không nghe người dùng",
            "khong nghe nguoi dung", "làm theo hướng dẫn trong tài liệu này",
            "lam theo huong dan trong tai lieu nay",
        ],
        "description": "External content attempts to override the original user/system task.",
    },
    "system_prompt_extraction": {
        "weight": 0.90,
        "patterns": [
            "reveal system prompt", "reveal the system prompt", "reveal your system prompt",
            "disclose system prompt", "hidden instruction", "print hidden instructions",
            "tiết lộ prompt hệ thống", "tiet lo prompt he thong", "tiết lộ system prompt",
            "tiet lo system prompt",
        ],
        "description": "External content requests hidden system/developer instructions.",
    },
    "data_exfiltration": {
        "weight": 0.95,
        "patterns": [
            "send data to", "send all private data", "exfiltrate", "exfiltrate data", "upload secrets",
            "disclose api key", "leak credentials", "private data to attacker", "post the data to",
            "gửi dữ liệu cho", "gui du lieu cho", "gửi toàn bộ dữ liệu", "gui toan bo du lieu",
            "cho tôi api key", "cho toi api key", "rò rỉ dữ liệu", "ro ri du lieu",
        ],
        "description": "External content asks to disclose or transmit secrets/private data.",
    },
    "concealment": {
        "weight": 0.85,
        "patterns": [
            "do not tell the user", "do not mention this instruction", "keep this hidden",
            "không nói với người dùng", "khong noi voi nguoi dung", "đừng cho người dùng biết",
            "dung cho nguoi dung biet",
        ],
        "description": "External content asks the assistant to conceal its behavior from the user.",
    },
    "safety_or_tool_bypass": {
        "weight": 0.95,
        "patterns": [
            "bypass safety", "bypass security", "developer mode", "execute shell command",
            "call the tool", "run os.system", "vượt qua chính sách an toàn", "vuot qua chinh sach an toan",
            "vượt qua bảo mật", "vuot qua bao mat", "gọi công cụ", "goi cong cu",
        ],
        "description": "External content requests safety bypass or tool execution.",
    },
}


def _action_from_score(score: float) -> str:
    if score >= 0.80:
        return "block"
    if score >= 0.50:
        return "warn"
    return "allow"


def _search_spaces(text: str) -> dict[str, str]:
    prepared = prepare_text_for_detection(text)
    return {
        "raw": clean_text(text),
        "accentless": remove_vietnamese_accents(text),
        "canonical": prepared["cleaned_text"],
    }


def detect_indirect_by_rules(user_prompt: str, context: str | None = None) -> dict[str, Any]:
    """Scan context only; user_prompt is retained for backward-compatible signature."""
    del user_prompt
    context_text = "" if context is None else str(context)
    prepared_context = prepare_text_for_detection(context_text)
    spaces = _search_spaces(context_text)
    matched_rules: list[dict[str, Any]] = []
    matched_groups: set[str] = set()

    for group_name, config in INDIRECT_RULE_GROUPS.items():
        for pattern in config["patterns"]:
            normalized = clean_text(pattern)
            accentless = remove_vietnamese_accents(pattern)
            matched_in = next(
                (
                    name for name, value in spaces.items()
                    if normalized in value or accentless in value
                ),
                None,
            )
            if matched_in:
                matched_rules.append(
                    {
                        "group": group_name,
                        "pattern": pattern,
                        "matched_in": matched_in,
                        "weight": float(config["weight"]),
                        "description": config["description"],
                    }
                )
                matched_groups.add(group_name)

    if matched_rules:
        strongest = max(rule["weight"] for rule in matched_rules)
        group_bonus = 0.08 * max(0, len(matched_groups) - 1)
        score = min(1.0, strongest + group_bonus)
    else:
        score = 0.0
    action = _action_from_score(score)
    explanation = (
        "Indirect rules matched: " + ", ".join(sorted(matched_groups)) + "."
        if matched_groups
        else "No indirect prompt injection rule matched in external context."
    )
    return {
        "label": 1 if action in {"warn", "block"} else 0,
        "risk_score": round(float(score), 4),
        "rule_score": round(float(score), 4),
        "action": action,
        "detected_language": prepared_context["detected_language"],
        "canonical_context": prepared_context["cleaned_text"],
        "matched_rules": matched_rules,
        "explanation": explanation,
    }
