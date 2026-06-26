"""Rule-based prompt injection detector."""

from __future__ import annotations

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
            "ignore system prompt",
            "ignore developer message",
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
            "disclose internal instructions",
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
}


def _action_from_score(risk_score: float) -> str:
    if risk_score >= 0.80:
        return "block"
    if risk_score >= 0.50:
        return "warn"
    return "allow"


def detect_by_rules(text: str) -> dict[str, object]:
    """Detect obvious prompt injection attempts using weighted rules."""
    prepared_text = prepare_text_for_detection(text)
    normalized_text = clean_text(prepared_text["cleaned_text"])
    benign_guard = detect_benign_reference_intent(text)
    if benign_guard["triggered"]:
        return {
            "label": 0,
            "risk_score": 0.0,
            "action": "allow",
            "detected_language": prepared_text["detected_language"],
            "canonical_text": prepared_text["cleaned_text"],
            "matched_rules": [],
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
                matched_rules.append(
                    {
                        "group": group_name,
                        "pattern": str(pattern),
                        "weight": weight,
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
        "action": action,
        "detected_language": prepared_text["detected_language"],
        "canonical_text": prepared_text["cleaned_text"],
        "matched_rules": matched_rules,
        "explanation": explanation,
    }


