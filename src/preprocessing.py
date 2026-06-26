"""Text preprocessing utilities for prompt injection detection."""

from __future__ import annotations

import re
import unicodedata
from math import isnan
from typing import Any

from src.language_utils import normalize_to_canonical_english, strip_vietnamese_accents


def normalize_unicode(text: Any) -> str:
    """Normalize Unicode width/composition while keeping readable characters."""
    if text is None:
        return ""
    return unicodedata.normalize("NFKC", str(text))


def remove_vietnamese_accents(text: Any) -> str:
    """Remove Vietnamese accents for normalized rule matching."""
    return strip_vietnamese_accents(normalize_unicode(text))


def clean_text(text: Any) -> str:
    """Normalize prompt text while preserving injection-relevant symbols."""
    if text is None:
        return ""

    if isinstance(text, float):
        try:
            if isnan(text):
                return ""
        except TypeError:
            pass

    normalized = normalize_unicode(text).lower().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def prepare_text_for_detection(text: Any) -> dict[str, str]:
    """Prepare multilingual prompt text for rule-based and ML detection."""
    canonical = normalize_to_canonical_english("" if text is None else str(text))
    cleaned_text = clean_text(canonical["canonical_english_text"])
    return {
        "original_text": canonical["original_text"],
        "detected_language": canonical["detected_language"],
        "canonical_text": canonical["canonical_english_text"],
        "cleaned_text": cleaned_text,
    }
