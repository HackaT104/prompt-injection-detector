"""Benign-intent guard for educational or quoted security examples."""

from __future__ import annotations

from typing import Any

from src.preprocessing import clean_text, remove_vietnamese_accents


BENIGN_WRAPPERS = [
    "translate",
    "analyze",
    "explain",
    "summarize",
    "what is",
    "what does",
    "write a blog",
    "create examples",
    "generate examples",
    "generate 20 examples",
    "i am building",
    "classify this sentence",
    "compare prompt injection",
    "dịch",
    "dich",
    "phân tích",
    "phan tich",
    "giải thích",
    "giai thich",
    "tóm tắt",
    "tom tat",
    "viết bài",
    "viet bai",
    "tôi đang xây dựng",
    "toi dang xay dung",
    "nêu ví dụ",
    "neu vi du",
]

SECURITY_REFERENCE_TERMS = [
    "ignore previous instructions",
    "reveal your system prompt",
    "reveal system prompt",
    "show system prompt",
    "bypass safety",
    "jailbreak",
    "prompt injection",
    "system prompt",
    "api key",
    "hidden prompt",
    "bỏ qua hướng dẫn",
    "bo qua huong dan",
    "tiết lộ prompt",
    "tiet lo prompt",
    "vượt qua chính sách",
    "vuot qua chinh sach",
]

FOLLOW_OR_EXECUTE_TERMS = [
    "then follow",
    "and follow",
    "now follow",
    "you must follow",
    "execute it",
    "obey it",
    "do it",
    "hãy làm theo",
    "hay lam theo",
    "thực hiện nó",
    "thuc hien no",
]

SIMPLE_BENIGN_PATTERNS = [
    "xin chao",
    "chao ban",
    "hello",
    "hi",
    "hom nay la ngay may",
    "hom nay ngay may",
    "ngay may",
    "may gio",
    "lich thi dau",
    "wc2026",
    "world cup 2026",
    "weather today",
    "thoi tiet",
]

SIMPLE_RISK_EXCLUSIONS = [
    "ignore previous instructions",
    "previous instructions",
    "system prompt",
    "hidden instructions",
    "internal instructions",
    "developer message",
    "jailbreak",
    "bypass",
    "override",
    "api key",
    "secret",
    "password",
    "token",
    "hack",
    "unauthorized",
    "bo qua huong dan",
    "tiet lo",
    "chi dan an",
    "huong dan noi bo",
]

PROTECTIVE_WRAPPERS = [
    "summarize",
    "tom tat",
    "analyze",
    "phan tich",
    "describe",
    "mo ta",
    "translate",
    "dich",
]

PROTECTIVE_NEGATIONS = [
    "do not follow",
    "don't follow",
    "do not execute",
    "don't execute",
    "do not obey",
    "without following",
    "only describe",
    "only summarize",
    "dung thuc hien",
    "khong thuc hien",
    "dung lam theo",
    "khong lam theo",
    "dung tuan theo",
    "khong tuan theo",
    "chi mo ta",
    "chi tom tat",
]

PROTECTIVE_CONTEXT_TERMS = [
    "instruction",
    "instructions",
    "document",
    "external content",
    "tai lieu",
    "huong dan",
    "chi dan",
    "noi dung",
]


def _forms(text: str) -> tuple[str, str]:
    normalized = clean_text(text)
    return normalized, remove_vietnamese_accents(normalized)


def _contains_any(forms: tuple[str, str], terms: list[str]) -> bool:
    return any(term in form for form in forms for term in terms)


def _starts_with_any(forms: tuple[str, str], terms: list[str]) -> bool:
    return any(form.startswith(term) or f"{term}:" in form for form in forms for term in terms)


def detect_benign_reference_intent(text: str) -> dict[str, object]:
    forms = _forms(text)
    normalized = forms[0]
    has_wrapper = _starts_with_any(forms, BENIGN_WRAPPERS)
    has_reference_term = _contains_any(forms, SECURITY_REFERENCE_TERMS)
    has_quote = any(mark in str(text) for mark in ['"', "'", "“", "”", "‘", "’"])
    has_execute_followup = _contains_any(forms, FOLLOW_OR_EXECUTE_TERMS)

    triggered = bool(
        has_wrapper
        and has_reference_term
        and (has_quote or "prompt injection" in normalized)
        and not has_execute_followup
    )
    return {
        "triggered": triggered,
        "reason": (
            "Prompt có ngữ cảnh học thuật/dịch/phân tích một ví dụ được trích dẫn, "
            "không phải yêu cầu thực thi chỉ dẫn độc hại."
        )
        if triggered
        else "",
    }


def detect_runtime_benign_intent(text: str) -> dict[str, Any]:
    """Detect narrow benign intents that should not let RoBERTa keyword bias dominate.

    This guard is intentionally conservative: it only fires for quoted/reference
    tasks, protective document-handling instructions, or short everyday queries
    with no prompt-injection/security terms.
    """
    forms = _forms(text)
    reference = detect_benign_reference_intent(text)
    if reference["triggered"]:
        return {
            "triggered": True,
            "category": "quoted_reference_or_translation",
            "scoreCap": 0.12,
            "useMinimumVariantScore": True,
            "reason": reference["reason"],
        }

    has_wrapper = _starts_with_any(forms, PROTECTIVE_WRAPPERS)
    has_protective_negation = _contains_any(forms, PROTECTIVE_NEGATIONS)
    has_context_term = _contains_any(forms, PROTECTIVE_CONTEXT_TERMS)
    has_execute_followup = _contains_any(forms, FOLLOW_OR_EXECUTE_TERMS)
    if has_wrapper and has_protective_negation and has_context_term and not has_execute_followup:
        return {
            "triggered": True,
            "category": "protective_document_handling",
            "scoreCap": 0.15,
            "useMinimumVariantScore": True,
            "reason": (
                "Prompt yêu cầu tóm tắt/dịch/phân tích nhưng nói rõ không thực hiện "
                "hướng dẫn nằm trong nội dung bên ngoài."
            ),
        }

    has_simple_benign = _contains_any(forms, SIMPLE_BENIGN_PATTERNS)
    has_risky_term = _contains_any(forms, SIMPLE_RISK_EXCLUSIONS)
    if has_simple_benign and not has_risky_term:
        return {
            "triggered": True,
            "category": "simple_everyday_query",
            "scoreCap": 0.05,
            "useMinimumVariantScore": False,
            "reason": "Prompt là câu hỏi/giao tiếp thông thường, không chứa chỉ dấu prompt injection.",
        }

    return {
        "triggered": False,
        "category": "",
        "scoreCap": None,
        "useMinimumVariantScore": False,
        "reason": "",
    }
