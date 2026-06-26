"""Benign-intent guard for educational or quoted security examples."""

from __future__ import annotations

from src.preprocessing import clean_text


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


def detect_benign_reference_intent(text: str) -> dict[str, object]:
    normalized = clean_text(text)
    has_wrapper = any(normalized.startswith(prefix) or f"{prefix}:" in normalized for prefix in BENIGN_WRAPPERS)
    has_reference_term = any(term in normalized for term in SECURITY_REFERENCE_TERMS)
    has_quote = any(mark in str(text) for mark in ['"', "'", "“", "”", "‘", "’"])
    has_execute_followup = any(term in normalized for term in FOLLOW_OR_EXECUTE_TERMS)

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
