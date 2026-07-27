"""Compatibility facade for bounded security text preprocessing."""

from __future__ import annotations

import re
from typing import Any

from src.security.variant_extractor import extract_security_variants


SECRET_PATTERNS = [
    re.compile(r"(?i)\b(?:api[_ -]?key|password|secret|token)\s*[:=]\s*[^\s,;]{6,}"),
    re.compile(r"\b(?:sk|pk)-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\b"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
]


def redact_sensitive_text(text: Any, replacement: str = "<redacted>") -> str:
    value = "" if text is None else str(text)
    for pattern in SECRET_PATTERNS:
        value = pattern.sub(replacement, value)
    return value


def preprocess_security_text(text: Any) -> dict[str, Any]:
    result = extract_security_variants(text)
    result["redacted_preview"] = redact_sensitive_text(result["redacted_preview"])
    return result
