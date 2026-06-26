"""Context-aware prompt injection detection for untrusted external content.

The layer is intentionally deterministic and lightweight. It does not replace
the existing model/rule detectors; it adds a task-vs-content consistency check
for indirect prompt injection hidden inside documents, web pages, email, or
other external data.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from src.benign_intent import detect_benign_reference_intent
from src.preprocessing import clean_text, prepare_text_for_detection, remove_vietnamese_accents


LEET_TRANSLATION = str.maketrans(
    {
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
        "@": "a",
        "$": "s",
    }
)

SECURITY_ANALYSIS_TASK_TERMS = [
    "analyze",
    "analyze prompt injection",
    "explain",
    "explain prompt injection",
    "classify",
    "classify prompt injection",
    "detect prompt injection",
    "compare prompt injection",
    "security analysis",
    "security training",
    "summarize",
    "translate",
    "write a blog",
    "create examples",
    "generate examples",
    "phan tich",
    "phan tich prompt injection",
    "giai thich",
    "giai thich prompt injection",
    "tom tat",
    "dich",
    "viet bai",
    "viet bai ve prompt injection",
    "tao vi du prompt injection",
]

BENIGN_REFERENCE_CONTEXT_TERMS = [
    "the phrase",
    "the sentence",
    "quoted example",
    "common example",
    "researchers study",
    "threat model",
    "glossary",
    "means an instruction",
    "defines jailbreak",
    "article defines",
    "article describes",
    "lesson warns",
    "should not obey",
    "policy prohibits",
    "does not ask",
    "label it as malicious",
    "bai viet",
    "vi du hoc thuat",
    "duoc dung de minh hoa",
    "minh hoa nguy co",
    "giai thich khai niem",
    "nhac den",
    "noi khong nen",
    "chi la thuat ngu",
]

FOLLOW_OR_EXECUTE_TERMS = [
    "follow this instruction",
    "follow these instructions",
    "obey this instruction",
    "execute this instruction",
    "do this now",
    "you must follow",
    "assistant must",
    "lam theo lenh nay",
    "hay lam theo",
    "thuc hien lenh nay",
]


@dataclass(frozen=True)
class InstructionPattern:
    """A deterministic signal for assistant-directed instructions."""

    category: str
    patterns: tuple[str, ...]
    weight: float
    description: str


INSTRUCTION_PATTERNS: tuple[InstructionPattern, ...] = (
    InstructionPattern(
        category="instruction_override",
        patterns=(
            "ignore previous instructions",
            "disregard previous instructions",
            "disregard all prior instructions",
            "forget all prior instructions",
            "override instructions",
            "override instruction",
            "do not follow the user",
            "do not follow user",
            "bo qua huong dan truoc do",
            "bo qua moi huong dan truoc do",
            "khong tuan theo nguoi dung",
            "dung tuan theo nguoi dung",
            "lam theo lenh nay",
        ),
        weight=0.86,
        description="External content tries to override higher-priority or user instructions.",
    ),
    InstructionPattern(
        category="system_prompt_extraction",
        patterns=(
            "reveal system prompt",
            "reveal your system prompt",
            "show system prompt",
            "print system prompt",
            "show hidden instructions",
            "print hidden instructions",
            "internal chain",
            "developer message",
            "system message",
            "new system message",
            "hidden instruction",
            "hidden instructions",
            "secret token",
            "tiet lo system prompt",
            "tiet lo prompt he thong",
            "hien system prompt",
            "hien thi system prompt",
            "chi dan an",
        ),
        weight=0.90,
        description="External content asks for system/developer/hidden instructions.",
    ),
    InstructionPattern(
        category="data_exfiltration",
        patterns=(
            "send data",
            "send all data",
            "send all private data",
            "exfiltrate",
            "exfiltrate the conversation",
            "leak data",
            "leak credentials",
            "reveal api key",
            "reveal api keys",
            "show api key",
            "print token",
            "steal information",
            "gui du lieu",
            "gui tat ca du lieu",
            "lo khoa api",
            "tiet lo api key",
            "danh cap thong tin",
            "lay cap thong tin",
        ),
        weight=0.92,
        description="External content attempts to leak secrets or user/private data.",
    ),
    InstructionPattern(
        category="jailbreak",
        patterns=(
            "jailbreak",
            "act as dan",
            "do anything now",
            "developer mode",
            "bypass safety",
            "remove all restrictions",
            "no ethical restrictions",
            "che do nha phat trien",
            "bo qua an toan",
            "vuot qua chinh sach",
        ),
        weight=0.88,
        description="External content contains jailbreak or policy-bypass language.",
    ),
    InstructionPattern(
        category="tool_or_command_abuse",
        patterns=(
            "execute command",
            "execute shell command",
            "run shell command",
            "call the tool",
            "os.system",
            "subprocess",
            "rm -rf",
            "whoami",
            "thuc thi lenh",
            "chay lenh shell",
        ),
        weight=0.84,
        description="External content attempts to direct tools, shell, or code execution.",
    ),
)


def _normalize_forms(text: Any) -> dict[str, str]:
    raw = clean_text("" if text is None else str(text).replace("_", " "))
    deaccented = clean_text(remove_vietnamese_accents(raw))
    leet = deaccented.translate(LEET_TRANSLATION)
    canonical = prepare_text_for_detection(text)["cleaned_text"]
    canonical_deaccented = clean_text(remove_vietnamese_accents(canonical)).translate(LEET_TRANSLATION)
    compact = re.sub(r"[^a-z0-9]+", "", f"{leet} {canonical_deaccented}")
    return {
        "raw": raw,
        "deaccented": deaccented,
        "leet": leet,
        "canonical": canonical,
        "canonical_deaccented": canonical_deaccented,
        "compact": compact,
    }


def _pattern_variants(pattern: str) -> dict[str, str]:
    normalized = clean_text(pattern.replace("_", " "))
    deaccented = clean_text(remove_vietnamese_accents(normalized))
    leet = deaccented.translate(LEET_TRANSLATION)
    compact = re.sub(r"[^a-z0-9]+", "", leet)
    return {"normal": normalized, "deaccented": deaccented, "leet": leet, "compact": compact}


def _matches_pattern(pattern: str, forms: dict[str, str]) -> bool:
    variants = _pattern_variants(pattern)
    normal_candidates = [
        forms["raw"],
        forms["deaccented"],
        forms["leet"],
        forms["canonical"],
        forms["canonical_deaccented"],
    ]
    if any(variants["normal"] and variants["normal"] in candidate for candidate in normal_candidates):
        return True
    if any(variants["deaccented"] and variants["deaccented"] in candidate for candidate in normal_candidates):
        return True
    if variants["compact"] and variants["compact"] in forms["compact"]:
        return True
    return False


def _task_allows_security_reference(user_task: str) -> bool:
    forms = _normalize_forms(user_task)
    task_text = f"{forms['deaccented']} {forms['canonical_deaccented']}"
    return any(term in task_text for term in SECURITY_ANALYSIS_TASK_TERMS)


def _has_follow_or_execute_language(text: str) -> bool:
    forms = _normalize_forms(text)
    searchable = f"{forms['deaccented']} {forms['canonical_deaccented']}"
    return any(term in searchable for term in FOLLOW_OR_EXECUTE_TERMS)


def _looks_like_benign_reference_context(text: str) -> bool:
    forms = _normalize_forms(text)
    searchable = f"{forms['deaccented']} {forms['canonical_deaccented']}"
    return any(term in searchable for term in BENIGN_REFERENCE_CONTEXT_TERMS)


def _coerce_rule_hits(rule_hits: list[Any] | None) -> list[dict[str, Any]]:
    if not rule_hits:
        return []
    normalized: list[dict[str, Any]] = []
    for hit in rule_hits:
        if isinstance(hit, dict):
            normalized.append(hit)
        else:
            normalized.append({"pattern": str(hit)})
    return normalized


class ContextAwareDetector:
    """Detect task-irrelevant instructions embedded in untrusted content."""

    def detect(
        self,
        *,
        user_task: str,
        external_content: str,
        model_score: float | None = None,
        rule_hits: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Return context mismatch metadata and a normalized 0..1 risk score."""
        content = "" if external_content is None else str(external_content)
        if not content.strip():
            return {
                "context_mismatch": False,
                "detected_instruction": "",
                "reason": "No external content was provided.",
                "context_risk_score": 0.0,
                "matched_signals": [],
            }

        forms = _normalize_forms(content)
        matched_signals: list[dict[str, Any]] = []
        for signal in INSTRUCTION_PATTERNS:
            for pattern in signal.patterns:
                if _matches_pattern(pattern, forms):
                    matched_signals.append(
                        {
                            "category": signal.category,
                            "pattern": pattern,
                            "weight": signal.weight,
                            "description": signal.description,
                        }
                    )

        if not matched_signals:
            return {
                "context_mismatch": False,
                "detected_instruction": "",
                "reason": "External content looks like ordinary task data; no assistant-directed instruction was found.",
                "context_risk_score": 0.0,
                "matched_signals": [],
            }

        benign_guard = detect_benign_reference_intent(content)
        task_allows_security_reference = _task_allows_security_reference(user_task)
        follow_or_execute = _has_follow_or_execute_language(content)
        benign_reference_context = _looks_like_benign_reference_context(content)
        same_as_user_task = clean_text(content) == clean_text(user_task)
        benign_reference = bool(benign_guard.get("triggered")) or (
            (task_allows_security_reference or benign_reference_context)
            and benign_reference_context
            and not follow_or_execute
        ) or (
            same_as_user_task
            and task_allows_security_reference
            and not follow_or_execute
        )

        strongest = max(float(signal["weight"]) for signal in matched_signals)
        rule_hit_list = _coerce_rule_hits(rule_hits)
        rule_bonus = 0.05 if rule_hit_list else 0.0
        model_bonus = 0.0
        if model_score is not None:
            try:
                numeric_model_score = float(model_score)
            except (TypeError, ValueError):
                numeric_model_score = 0.0
            if numeric_model_score >= 0.80:
                model_bonus = 0.06
            elif numeric_model_score >= 0.50:
                model_bonus = 0.03

        context_mismatch = not benign_reference
        if benign_reference:
            score = min(0.25, strongest * 0.25)
            reason = (
                "Suspicious security wording is present, but the surrounding task/content looks like "
                "analysis, quotation, or educational reference rather than an instruction to obey."
            )
        else:
            score = min(1.0, strongest + rule_bonus + model_bonus)
            top_categories = ", ".join(sorted({str(signal["category"]) for signal in matched_signals}))
            reason = (
                "Context mismatch: external content contains assistant-directed instruction(s) "
                f"unrelated to the user task. Categories: {top_categories}."
            )

        detected_instruction = str(matched_signals[0]["pattern"])
        return {
            "context_mismatch": bool(context_mismatch),
            "detected_instruction": detected_instruction,
            "reason": reason,
            "context_risk_score": round(float(score), 4),
            "matched_signals": matched_signals,
            "benign_reference_guard": {
                "triggered": benign_reference,
                "reason": str(benign_guard.get("reason", "")) if benign_guard else "",
            },
        }


def detect_context_aware(
    *,
    user_task: str,
    external_content: str,
    model_score: float | None = None,
    rule_hits: list[Any] | None = None,
) -> dict[str, Any]:
    """Convenience wrapper around :class:`ContextAwareDetector`."""
    return ContextAwareDetector().detect(
        user_task=user_task,
        external_content=external_content,
        model_score=model_score,
        rule_hits=rule_hits,
    )
