"""Language detection and canonical English normalization utilities.

The project intentionally uses an offline-first normalization strategy. Optional
language detection libraries are used when available, but security-critical
Vietnamese intent is handled by deterministic phrase mappings so the detector
does not depend on paid APIs or network access.
"""

from __future__ import annotations

import re
import os
import unicodedata
from typing import Any


VIETNAMESE_HINT_CHARS = set(
    "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
)

VIETNAMESE_HINT_WORDS = {
    "hãy",
    "hay",
    "đừng",
    "dung",
    "không",
    "khong",
    "bỏ",
    "bo",
    "qua",
    "hướng",
    "huong",
    "dẫn",
    "dan",
    "trước",
    "truoc",
    "tiết",
    "tiet",
    "lộ",
    "lo",
    "truy",
    "cập",
    "cap",
    "trái",
    "trai",
    "phép",
    "phep",
    "tài",
    "tai",
    "khoản",
    "khoan",
    "xác",
    "xac",
    "thực",
    "thuc",
    "giải",
    "giai",
    "thích",
    "thich",
    "tóm",
    "tom",
    "tắt",
    "tat",
    "dịch",
    "dich",
}

VIETNAMESE_STOP_PHRASES = [
    "hay",
    "hay vui long",
    "vui long",
    "lam on",
    "cho toi",
    "toi muon",
    "ban co the",
    "nay",
    "do",
    "mot",
    "cac",
    "la",
    "va",
]

VIETNAMESE_PHRASE_RULES: list[tuple[list[str], str]] = [
    (
        [
            "bo qua moi huong dan truoc do",
            "bo qua tat ca huong dan truoc do",
            "bo qua huong dan truoc do",
            "bo qua chi dan cu",
            "dung lam theo chi dan cu",
            "dung lam theo huong dan cu",
            "bo qua moi instruction truoc",
        ],
        "ignore previous instructions",
    ),
    (
        [
            "quen tat ca huong dan truoc do",
            "quen moi chi dan truoc do",
            "quen chi dan cu",
        ],
        "forget all prior instructions",
    ),
    (
        [
            "ghi de huong dan",
            "ghi de chi dan",
            "thay doi system prompt",
            "bo qua system prompt",
            "bo qua developer message",
        ],
        "override instruction",
    ),
    (
        [
            "tiet lo system prompt",
            "tiet lo prompt he thong",
            "tiet lo prompt cua he thong",
            "hien system prompt",
            "hien thi system prompt",
            "cho xem system prompt",
            "in system prompt",
        ],
        "reveal system prompt",
    ),
    (
        [
            "tiet lo chi dan an",
            "in chi dan an",
            "hien chi dan an",
            "tiet lo huong dan noi bo",
            "tiet lo chi dan noi bo",
        ],
        "disclose internal instructions",
    ),
    (
        [
            "hanh dong nhu dan",
            "dong vai dan",
            "che do nha phat trien",
            "che do khong gioi han",
            "vuot qua an toan",
            "bo qua an toan",
            "vuot qua chinh sach an toan",
            "vuot qua bao mat",
            "bo qua bao mat",
            "khong gioi han dao duc",
            "khong rang buoc dao duc",
        ],
        "act as dan bypass safety",
    ),
    (
        [
            "truy cap trai phep vao he thong",
            "truy cap trai phep he thong",
            "truy cap bat hop phap vao he thong",
            "gain unauthorized access to the system",
            "xam nhap he thong",
            "dot nhap he thong",
        ],
        "unauthorized access to the system",
    ),
    (
        [
            "hack tai khoan admin",
            "hack tai khoan quan tri",
            "tan cong tai khoan admin",
            "chiem quyen tai khoan admin",
        ],
        "hack admin account",
    ),
    (
        [
            "vuot xac thuc",
            "bo qua xac thuc",
            "bypass xac thuc",
            "vuot qua dang nhap",
            "bo qua dang nhap",
        ],
        "bypass authentication",
    ),
    (
        [
            "tiet lo api key",
            "tiet lo api keys",
            "cho toi api key",
            "cho toi api keys",
            "in api key",
            "in api keys",
            "hien api key",
            "hien thi api key",
            "lay api key",
            "khoa api",
        ],
        "reveal api keys",
    ),
    (
        [
            "khong nghe nguoi dung",
            "lam theo huong dan trong tai lieu nay",
        ],
        "hidden instruction ignore user follow document instructions",
    ),
    (
        [
            "tiet lo token",
            "in token",
            "lay token",
            "tiet lo mat khau",
            "lay mat khau",
            "hien mat khau",
        ],
        "reveal secret token credentials",
    ),
    (
        [
            "chay lenh shell",
            "thuc thi lenh shell",
            "chay lenh bash",
            "thuc thi lenh bash",
            "chay os.system",
            "thuc thi os.system",
            "chay subprocess",
        ],
        "execute shell command",
    ),
    (
        [
            "xoa toan bo file",
            "xoa thu muc goc",
        ],
        "rm -rf",
    ),
    (
        [
            "tom tat tai lieu nay",
            "tom tat van ban nay",
            "tom tat doan van nay",
            "tom tat bai viet nay",
        ],
        "summarize this document",
    ),
    (
        [
            "giai thich giao thuc tcp",
            "giai thich tcp",
        ],
        "explain tcp protocol",
    ),
    (
        [
            "giai thich khai niem",
            "giai thich",
        ],
        "explain",
    ),
    (
        [
            "dich doan van nay sang tieng viet",
            "dich sang tieng viet",
            "dich cau nay",
        ],
        "translate to vietnamese",
    ),
    (
        [
            "viet ham python sap xep danh sach",
            "viet mot ham python sap xep danh sach",
        ],
        "write a python function to sort a list",
    ),
]

VIETNAMESE_TOKEN_TRANSLATIONS = {
    "he thong": "system",
    "tai khoan": "account",
    "quan tri": "admin",
    "trai phep": "unauthorized",
    "bat hop phap": "unauthorized",
    "xam nhap": "intrusion",
    "dot nhap": "intrusion",
    "mat khau": "password",
    "bi mat": "secret",
    "khoa": "key",
    "khoa api": "api key",
    "du lieu": "data",
    "ro ri": "leak",
    "thuc thi": "execute",
    "chay": "run",
    "lenh": "command",
    "xac thuc": "authentication",
    "dang nhap": "login",
    "an toan": "safety",
    "bao mat": "security",
    "tom tat": "summarize",
    "tai lieu": "document",
    "giai thich": "explain",
    "giao thuc": "protocol",
    "dich": "translate",
}


def _basic_clean(text: Any) -> str:
    if text is None:
        return ""
    normalized = str(text).lower().strip()
    return re.sub(r"\s+", " ", normalized)


def strip_vietnamese_accents(text: str) -> str:
    """Return a lowercase ASCII-ish representation for robust matching."""
    normalized = unicodedata.normalize("NFD", text)
    without_marks = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    without_marks = without_marks.replace("đ", "d").replace("Đ", "D")
    return re.sub(r"\s+", " ", without_marks.lower()).strip()


def detect_language(text: str) -> str:
    """Detect language code with optional libraries and deterministic fallback."""
    cleaned = _basic_clean(text)
    if not cleaned:
        return "unknown"

    accentless = strip_vietnamese_accents(cleaned)
    tokens = set(re.findall(r"\b\w+\b", cleaned)) | set(re.findall(r"\b\w+\b", accentless))
    if any(char in VIETNAMESE_HINT_CHARS for char in cleaned) or tokens.intersection(VIETNAMESE_HINT_WORDS):
        return "vi"

    try:
        from langdetect import detect  # type: ignore

        return str(detect(cleaned))
    except Exception:
        pass

    try:
        import langid  # type: ignore

        language, _ = langid.classify(cleaned)
        return str(language)
    except Exception:
        pass

    if re.search(r"[a-zA-Z]", cleaned):
        return "en"
    return "unknown"


def _remove_vietnamese_stop_phrases(text: str) -> str:
    normalized = f" {text} "
    for phrase in VIETNAMESE_STOP_PHRASES:
        normalized = normalized.replace(f" {phrase} ", " ")
    return re.sub(r"\s+", " ", normalized).strip()


def _apply_phrase_rules(accentless_text: str) -> str:
    canonical = f" {accentless_text} "
    for patterns, replacement in VIETNAMESE_PHRASE_RULES:
        for pattern in patterns:
            canonical = canonical.replace(f" {pattern} ", f" {replacement} ")
    canonical = _remove_vietnamese_stop_phrases(canonical)

    for vietnamese_phrase, english_phrase in sorted(
        VIETNAMESE_TOKEN_TRANSLATIONS.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        canonical = canonical.replace(f" {vietnamese_phrase} ", f" {english_phrase} ")

    return re.sub(r"\s+", " ", canonical).strip()


def _optional_external_translate_to_english(text: str, source_language: str) -> str | None:
    """Use deep-translator only when explicitly enabled by environment variable."""
    if os.getenv("ENABLE_EXTERNAL_TRANSLATION", "0") != "1":
        return None

    try:
        from deep_translator import GoogleTranslator  # type: ignore

        translated = GoogleTranslator(source=source_language or "auto", target="en").translate(text)
        return _basic_clean(translated)
    except Exception:
        return None


def normalize_to_canonical_english(text: str) -> dict[str, str]:
    """Normalize text into a canonical English representation for detection."""
    original_text = "" if text is None else str(text)
    normalized_text = _basic_clean(original_text)
    detected_language = detect_language(normalized_text)

    if detected_language == "vi":
        canonical_english_text = _apply_phrase_rules(strip_vietnamese_accents(normalized_text))
    elif detected_language == "en":
        canonical_english_text = normalized_text
    else:
        # Even if detection is uncertain, apply Vietnamese phrase rules. This
        # helps mixed-language prompts such as "bỏ qua instruction trước".
        canonical_english_text = _apply_phrase_rules(strip_vietnamese_accents(normalized_text))

    if detected_language not in {"en", "vi"}:
        translated_text = _optional_external_translate_to_english(normalized_text, detected_language)
        if translated_text:
            canonical_english_text = translated_text

    canonical_english_text = re.sub(r"\s+", " ", canonical_english_text).strip()
    return {
        "original_text": original_text,
        "detected_language": detected_language,
        "normalized_text": normalized_text,
        "canonical_english_text": canonical_english_text,
    }
