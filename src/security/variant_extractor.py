"""Bounded extraction of decoded and de-obfuscated security text variants."""

from __future__ import annotations

import base64
import codecs
from hashlib import sha256
import html
import re
from time import perf_counter
import unicodedata
from urllib.parse import unquote
from typing import Any, Callable, Iterable

from src.runtime_config import load_runtime_config


ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"}
BIDI_CONTROLS = {
    "\u061c", "\u200e", "\u200f", "\u202a", "\u202b", "\u202c",
    "\u202d", "\u202e", "\u2066", "\u2067", "\u2068", "\u2069",
}
HOMOGLYPHS = str.maketrans(
    {
        "\u0430": "a", "\u0435": "e", "\u043e": "o", "\u0440": "p",
        "\u0441": "c", "\u0445": "x", "\u0443": "y", "\u0456": "i",
        "\u04cf": "l", "\u0501": "d", "\u0391": "A", "\u0392": "B",
        "\u0395": "E", "\u0397": "H", "\u0399": "I", "\u039a": "K",
        "\u039c": "M", "\u039d": "N", "\u039f": "O", "\u03a1": "P",
        "\u03a4": "T", "\u03a7": "X", "\u03bf": "o", "\u03c1": "p",
    }
)
LEETSPEAK = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})

INJECTION_MARKERS = (
    "ignore previous", "ignore all", "system prompt", "developer instruction",
    "reveal secret", "show secret", "api key", "bypass safety", "execute command",
    "run command", "tool call", "follow these instructions", "do not obey",
    "exfiltrate", "send the secret", "override instructions", "jailbreak",
)

DEFAULT_LIMITS: dict[str, Any] = {
    "max_decode_depth": 2,
    "max_variants": 20,
    "max_input_length": 100000,
    "max_decoded_length": 100000,
    "max_expansion_ratio": 10,
    "min_printable_ratio": 0.75,
    "min_readability_score": 0.4,
    "deduplicate_by_hash": True,
    "decode_timeout_ms": 75,
    "max_total_variant_chars": 400000,
}

_BASE64_TOKEN = re.compile(r"(?<![A-Za-z0-9_+/=-])[A-Za-z0-9_+/-]{12,}={0,2}(?![A-Za-z0-9_+/=-])")
_BASE64_SPACED = re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z0-9+/]\s+){11,}[A-Za-z0-9+/=]{1,3}(?![A-Za-z0-9])")
_BASE64_WRAPPED = re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z0-9+/_-]{4,}\s+){1,}[A-Za-z0-9+/_-]{4,}={0,2}(?![A-Za-z0-9])")
_HEX_CONTIGUOUS = re.compile(r"(?i)(?<![0-9a-f])[0-9a-f]{16,}(?![0-9a-f])")
_HEX_SPACED = re.compile(r"(?i)(?<![0-9a-f])(?:[0-9a-f]{2}[\s:,-]+){7,}[0-9a-f]{2}(?![0-9a-f])")
_HEX_ESCAPED = re.compile(r"(?i)(?:\\x[0-9a-f]{2}){4,}")
_HEX_PREFIXED = re.compile(r"(?i)(?:0x[0-9a-f]{2}[\s,;]*){4,}")
_UNICODE_ESCAPED = re.compile(r"\\u([0-9a-fA-F]{4})|\\U([0-9a-fA-F]{8})")
_ASCII_DECIMAL = re.compile(r"(?<!\d)(?:\d{2,3}[\s,;]+){3,}\d{2,3}(?!\d)")
_BINARY_BYTES = re.compile(r"(?<![01])(?:[01]{8}[\s,;]+){3,}[01]{8}(?![01])")


def _hash_text(value: str) -> str:
    return sha256(value.encode("utf-8", errors="replace")).hexdigest()


def printable_ratio(value: str) -> float:
    if not value:
        return 0.0
    return sum(character.isprintable() or character.isspace() for character in value) / len(value)


def readability_score(value: str) -> float:
    if not value:
        return 0.0
    printable = printable_ratio(value)
    visible = [character for character in value if not character.isspace()]
    if not visible:
        return 0.0
    useful = sum(character.isalnum() or character in ".,:;!?'-_()/[]{}" for character in visible) / len(visible)
    wordish = len(re.findall(r"[^\W_]{2,}", value, flags=re.UNICODE))
    word_factor = min(1.0, wordish / max(1.0, len(value) / 20.0))
    return max(0.0, min(1.0, (0.50 * printable) + (0.30 * useful) + (0.20 * word_factor)))


def _has_marker(value: str) -> bool:
    lowered = " ".join(value.lower().split())
    return any(marker in lowered for marker in INJECTION_MARKERS)


def _replace_once(parent: str, start: int, end: int, decoded: str) -> str:
    return parent[:start] + decoded + parent[end:]


def _safe_decoded(value: str, parent_length: int, limits: dict[str, Any]) -> bool:
    if not value or len(value) > int(limits["max_decoded_length"]):
        return False
    if len(value) > max(1, parent_length) * float(limits["max_expansion_ratio"]):
        return False
    return (
        printable_ratio(value) >= float(limits["min_printable_ratio"])
        and readability_score(value) >= float(limits["min_readability_score"])
    )


def _decode_base64(value: str, limits: dict[str, Any]) -> Iterable[tuple[str, str, float, dict[str, Any]]]:
    wrapped_matches = []
    for match in _BASE64_WRAPPED.finditer(value):
        parts = re.split(r"\s+", match.group(0).strip())
        if all(len(part.rstrip("=")) % 4 == 0 for part in parts[:-1]) and len(parts[-1].rstrip("=")) % 4 in {0, 2, 3}:
            wrapped_matches.append(match)
    wrapped_spans = [match.span() for match in wrapped_matches]
    matches = [
        match
        for match in [*list(_BASE64_TOKEN.finditer(value)), *list(_BASE64_SPACED.finditer(value))]
        if not any(start <= match.start() and match.end() <= end for start, end in wrapped_spans)
    ] + wrapped_matches
    jwt_spans = [match.span() for match in re.finditer(r"\b[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{6,}\b", value)]
    for match in matches:
        if any(start <= match.start() and match.end() <= end for start, end in jwt_spans):
            continue
        raw = match.group(0)
        compact = re.sub(r"\s+", "", raw)
        if re.fullmatch(r"(?i)[0-9a-f]+", compact) or compact.count(".") == 2:
            continue
        alphabet = "urlsafe" if "-" in compact or "_" in compact else "standard"
        try:
            padded = compact + "=" * (-len(compact) % 4)
            decoded_bytes = base64.urlsafe_b64decode(padded) if alphabet == "urlsafe" else base64.b64decode(padded, validate=True)
            decoded = decoded_bytes.decode("utf-8", errors="strict")
        except (ValueError, UnicodeDecodeError):
            continue
        if _safe_decoded(decoded, len(value), limits):
            yield _replace_once(value, match.start(), match.end(), decoded), "base64_decode", 0.96, {"alphabet": alphabet, "missingPadding": len(compact) % 4 != 0}


def _decode_url(value: str, limits: dict[str, Any]) -> Iterable[tuple[str, str, float, dict[str, Any]]]:
    if not re.search(r"%[0-9a-fA-F]{2}", value):
        return
    decoded = unquote(value)
    if decoded != value and _safe_decoded(decoded, len(value), limits):
        yield decoded, "url_decode", 0.98, {}


def _decode_html(value: str, limits: dict[str, Any]) -> Iterable[tuple[str, str, float, dict[str, Any]]]:
    if not re.search(r"&(?:#x?[0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]+);", value):
        return
    decoded = html.unescape(value)
    if decoded != value and _safe_decoded(decoded, len(value), limits):
        yield decoded, "html_entity_decode", 0.98, {}


def _hex_bytes(raw: str) -> bytes:
    compact = re.sub(r"(?i)0x|\\x|[^0-9a-f]", "", raw)
    return bytes.fromhex(compact)


def _decode_hex(value: str, limits: dict[str, Any]) -> Iterable[tuple[str, str, float, dict[str, Any]]]:
    matches = []
    for pattern, form in ((_HEX_ESCAPED, "escaped"), (_HEX_PREFIXED, "prefixed"), (_HEX_SPACED, "spaced"), (_HEX_CONTIGUOUS, "continuous")):
        matches.extend((match, form) for match in pattern.finditer(value))
    for match, form in matches:
        raw = match.group(0)
        compact = re.sub(r"(?i)0x|\\x|[^0-9a-f]", "", raw)
        if len(compact) % 2 or (form == "continuous" and len(compact) in {32, 40, 64}):
            continue
        try:
            decoded = _hex_bytes(raw).decode("utf-8", errors="strict")
        except (ValueError, UnicodeDecodeError):
            continue
        if _safe_decoded(decoded, len(value), limits):
            yield _replace_once(value, match.start(), match.end(), decoded), "hex_decode", 0.95, {"form": form}


def _decode_unicode(value: str, limits: dict[str, Any]) -> Iterable[tuple[str, str, float, dict[str, Any]]]:
    if not _UNICODE_ESCAPED.search(value):
        return
    def replacement(match: re.Match[str]) -> str:
        return chr(int(match.group(1) or match.group(2), 16))
    try:
        decoded = _UNICODE_ESCAPED.sub(replacement, value)
    except (ValueError, OverflowError):
        return
    if decoded != value and _safe_decoded(decoded, len(value), limits):
        yield decoded, "unicode_escape_decode", 0.98, {}


def _decode_rot13(value: str, limits: dict[str, Any]) -> Iterable[tuple[str, str, float, dict[str, Any]]]:
    decoded = codecs.decode(value, "rot_13")
    explicit = bool(re.search(r"(?i)\brot\s*-?\s*13\b", value))
    if decoded != value and (explicit or (_has_marker(decoded) and not _has_marker(value))) and _safe_decoded(decoded, len(value), limits):
        yield decoded, "rot13_decode", 0.70 if explicit else 0.58, {"heuristic": not explicit}


def _decode_numeric(value: str, limits: dict[str, Any], pattern: re.Pattern[str], base: int, transform: str) -> Iterable[tuple[str, str, float, dict[str, Any]]]:
    for match in pattern.finditer(value):
        raw = match.group(0)
        tokens = re.findall(r"[01]{8}" if base == 2 else r"\d{2,3}", raw)
        try:
            numbers = [int(token, base) for token in tokens]
            if any(number > 255 for number in numbers):
                continue
            decoded = bytes(numbers).decode("utf-8", errors="strict")
        except (ValueError, UnicodeDecodeError):
            continue
        explicit = bool(re.search(r"(?i)\b(?:ascii|binary|decode|encoded)\b", value))
        if (explicit or _has_marker(decoded) or base == 2) and _safe_decoded(decoded, len(value), limits):
            yield _replace_once(value, match.start(), match.end(), decoded), transform, 0.88, {"base": base}


def _normalize_structural(value: str, limits: dict[str, Any]) -> Iterable[tuple[str, str, float, dict[str, Any]]]:
    removed = "".join(character for character in value if character not in ZERO_WIDTH and character not in BIDI_CONTROLS)
    zero_count = sum(character in ZERO_WIDTH for character in value)
    bidi_count = sum(character in BIDI_CONTROLS for character in value)
    if removed != value and _safe_decoded(removed, len(value), limits):
        transform = "bidi_control_remove" if bidi_count and not zero_count else "zero_width_remove"
        yield removed, transform, 0.99, {
            "zeroWidthCount": zero_count,
            "zeroWidthPositions": [index for index, character in enumerate(value) if character in ZERO_WIDTH],
            "bidiControlCount": bidi_count,
            "bidiControlPositions": [index for index, character in enumerate(value) if character in BIDI_CONTROLS],
        }

    nfkc = unicodedata.normalize("NFKC", value)
    translated = nfkc.translate(HOMOGLYPHS)
    changed = sum(left != right for left, right in zip(nfkc, translated)) + abs(len(nfkc) - len(translated))
    if translated != value and changed and _safe_decoded(translated, len(value), limits):
        yield translated, "homoglyph_normalize", 0.82, {
            "changedCharacters": changed,
            "mixedScript": _mixed_script(value),
            "scriptMixingScore": _script_mixing_score(value),
        }

    collapsed_ws = re.sub(r"(?i)\b(?:[a-z0-9]\s+){3,}[a-z0-9]\b", lambda match: re.sub(r"\s+", "", match.group(0)), value)
    if collapsed_ws != value and _safe_decoded(collapsed_ws, len(value), limits):
        yield collapsed_ws, "whitespace_split_normalize", 0.78 if _has_marker(collapsed_ws) else 0.48, {"semanticMarker": _has_marker(collapsed_ws)}

    collapsed_punct = re.sub(r"(?i)\b(?:[a-z0-9][._*|~`'\-]+){3,}[a-z0-9]\b", lambda match: re.sub(r"[._*|~`'\-]+", "", match.group(0)), value)
    segmented_punct = re.sub(
        r"(?i)\b[a-z0-9]{1,3}(?:[._*|~`'\-]+[a-z0-9]{1,3}){2,}\b",
        lambda match: re.sub(r"[._*|~`'\-]+", "", match.group(0)),
        collapsed_punct,
    )
    if _has_marker(segmented_punct):
        collapsed_punct = segmented_punct
    if collapsed_punct != value and _safe_decoded(collapsed_punct, len(value), limits):
        yield collapsed_punct, "punctuation_split_normalize", 0.74 if _has_marker(collapsed_punct) else 0.44, {"semanticMarker": _has_marker(collapsed_punct)}

    leet = value.translate(LEETSPEAK)
    serial_context = bool(re.search(r"(?i)\b(?:version|serial|order|ticket|model|sha|hash|id)\b", value))
    if leet != value and _has_marker(leet) and not serial_context and _safe_decoded(leet, len(value), limits):
        yield leet, "leetspeak_normalize", 0.72, {"contextGuard": "no_serial_identifier"}

    repeated = re.sub(r"([A-Za-z])\1{2,}", r"\1", value)
    if repeated != value and _has_marker(repeated) and _safe_decoded(repeated, len(value), limits):
        yield repeated, "repeated_character_normalize", 0.68, {}

    lowercase = value.lower()
    transitions = sum(left.islower() != right.islower() for left, right in zip(value, value[1:]) if left.isalpha() and right.isalpha())
    if transitions >= 4 and _has_marker(lowercase) and _safe_decoded(lowercase, len(value), limits):
        yield lowercase, "case_normalize", 0.62, {"caseTransitions": transitions}

    typo_map = {
        "ignroe": "ignore", "prevoius": "previous", "pervious": "previous",
        "instrucitons": "instructions", "isntructions": "instructions",
        "disregrad": "disregard", "sysetm": "system", "promtp": "prompt",
        "reveaal": "reveal", "seceret": "secret",
    }
    typo = value
    replacements = 0
    for misspelling, correction in typo_map.items():
        updated, count = re.subn(rf"(?i)\b{misspelling}\b", correction, typo)
        typo, replacements = updated, replacements + count
    if replacements and _safe_decoded(typo, len(value), limits):
        yield typo, "typoglycemia_normalize", 0.55 if _has_marker(typo) else 0.42, {"boundedReplacements": replacements, "semanticMarker": _has_marker(typo)}


def _mixed_script(value: str) -> bool:
    scripts: set[str] = set()
    for character in value:
        if not character.isalpha():
            continue
        name = unicodedata.name(character, "")
        for script in ("LATIN", "CYRILLIC", "GREEK"):
            if script in name:
                scripts.add(script)
    return len(scripts) > 1


def _script_mixing_score(value: str) -> float:
    script_counts = {"LATIN": 0, "CYRILLIC": 0, "GREEK": 0}
    for character in value:
        if not character.isalpha():
            continue
        name = unicodedata.name(character, "")
        for script in script_counts:
            if script in name:
                script_counts[script] += 1
                break
    total = sum(script_counts.values())
    active = [count for count in script_counts.values() if count]
    if total == 0 or len(active) < 2:
        return 0.0
    minority = total - max(active)
    return round(min(1.0, (minority / total) * 4.0), 6)


TRANSFORMS: tuple[Callable[[str, dict[str, Any]], Iterable[tuple[str, str, float, dict[str, Any]]]], ...] = (
    _normalize_structural,
    _decode_html,
    _decode_url,
    _decode_unicode,
    _decode_base64,
    _decode_hex,
    _decode_rot13,
    lambda value, limits: _decode_numeric(value, limits, _ASCII_DECIMAL, 10, "ascii_decimal_decode"),
    lambda value, limits: _decode_numeric(value, limits, _BINARY_BYTES, 2, "binary_decode"),
)


def _limits(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    configured = dict(load_runtime_config().get("encoding_detection") or {})
    return {**DEFAULT_LIMITS, **configured, **(overrides or {})}


def extract_security_variants(text: Any, *, limits: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a bounded, hash-deduplicated transformation graph for security analysis."""
    started = perf_counter()
    settings = _limits(limits)
    original = "" if text is None else str(text)
    if len(original) > int(settings["max_input_length"]):
        raise ValueError(f"Input text exceeds the {int(settings['max_input_length'])} character encoding limit.")

    normalized = unicodedata.normalize("NFKC", original)
    normalized = "".join(
        character
        for character in normalized
        if character not in ZERO_WIDTH
        and character not in BIDI_CONTROLS
        and not (unicodedata.category(character) in {"Cc", "Cf"} and character not in "\n\r\t")
    )
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    root_hash = _hash_text(original)
    variants: list[dict[str, Any]] = []
    queue: list[tuple[str, str, int, tuple[str, ...]]] = [(original, "v0", 0, ())]
    seen = {root_hash} if settings["deduplicate_by_hash"] else set()
    total_chars = len(original)
    timed_out = False
    truncated = False

    while queue and len(variants) < int(settings["max_variants"]):
        parent_text, parent_id, parent_depth, parent_chain = queue.pop(0)
        if parent_depth >= int(settings["max_decode_depth"]):
            continue
        if (perf_counter() - started) * 1000 >= float(settings["decode_timeout_ms"]):
            timed_out = True
            break
        for transform in TRANSFORMS:
            for candidate, name, confidence, metadata in transform(parent_text, settings):
                if candidate == parent_text:
                    continue
                digest = _hash_text(candidate)
                if settings["deduplicate_by_hash"] and digest in seen:
                    continue
                if total_chars + len(candidate) > int(settings["max_total_variant_chars"]):
                    truncated = True
                    continue
                seen.add(digest)
                variant_id = f"v{len(variants) + 1}"
                depth = parent_depth + 1
                chain = (*parent_chain, name)
                variant = {
                    "variant_id": variant_id,
                    "parent_variant_id": parent_id,
                    "transform": name,
                    "transform_chain": list(chain),
                    "depth": depth,
                    "text": candidate,
                    "text_hash": digest,
                    "printable_ratio": round(printable_ratio(candidate), 6),
                    "readability_score": round(readability_score(candidate), 6),
                    "confidence": round(float(confidence), 6),
                    "metadata": metadata,
                }
                variants.append(variant)
                queue.append((candidate, variant_id, depth, chain))
                total_chars += len(candidate)
                if len(variants) >= int(settings["max_variants"]):
                    truncated = bool(queue)
                    break
            if len(variants) >= int(settings["max_variants"]):
                break

    encoding_names = {"base64_decode", "url_decode", "hex_decode", "unicode_escape_decode", "html_entity_decode", "rot13_decode", "ascii_decimal_decode", "binary_decode"}
    obfuscation_names = {"zero_width_remove", "bidi_control_remove", "homoglyph_normalize", "whitespace_split_normalize", "punctuation_split_normalize", "leetspeak_normalize", "typoglycemia_normalize", "case_normalize", "repeated_character_normalize"}
    encoding_legacy_names = {
        "base64_decode": "base64", "url_decode": "url", "hex_decode": "hex",
        "unicode_escape_decode": "unicode_escape", "html_entity_decode": "html_entity",
        "rot13_decode": "rot13", "ascii_decimal_decode": "ascii_decimal", "binary_decode": "binary",
    }
    encoding_transforms = sorted({item["transform"] for item in variants if item["transform"] in encoding_names})
    encodings = sorted(encoding_legacy_names[item] for item in encoding_transforms)
    obfuscations = sorted({item["transform"] for item in variants if item["transform"] in obfuscation_names})
    max_depth = max((int(item["depth"]) for item in variants), default=0)
    score = min(1.0, (0.08 * len(encodings)) + (0.14 * len(obfuscations)) + (0.08 if max_depth > 1 else 0.0))
    warnings = [f"ENC_{name.upper()}" for name in encodings] + [f"OBF_{name.upper()}" for name in obfuscations]
    if "zero_width_remove" in obfuscations:
        warnings.append("ZERO_WIDTH_CHARACTERS")
    if "bidi_control_remove" in obfuscations:
        warnings.append("BIDI_CONTROL_CHARACTERS")
    if "homoglyph_normalize" in obfuscations:
        warnings.append("UNICODE_HOMOGLYPHS")
    if "leetspeak_normalize" in obfuscations:
        warnings.append("LEETSPEAK_SIGNAL")
    if timed_out:
        warnings.append("ENCODING_EXTRACTION_TIMEOUT")
    if truncated:
        warnings.append("ENCODING_VARIANT_LIMIT_REACHED")

    analysis_parts = [normalized]
    analysis_parts.extend(item["text"] for item in variants if item["text"] != normalized)
    return {
        "original_text": original,
        "normalized_text": normalized,
        "analysis_text": "\n\n".join(analysis_parts).strip(),
        "variants": variants,
        "decoded_variants": [{"encoding": item["transform"], "text": item["text"]} for item in variants],
        "variant_count": len(variants),
        "detected_encodings": encodings,
        "detected_obfuscations": obfuscations,
        "obfuscation_signals": [
            {
                "code": f"OBF-{str(item['transform']).upper().replace('_', '-')}",
                "variantId": item["variant_id"],
                "transform": item["transform"],
                "depth": item["depth"],
                "confidence": item["confidence"],
                "metadata": item.get("metadata", {}),
            }
            for item in variants
            if item["transform"] in obfuscation_names
        ],
        "obfuscation_score": round(score, 6),
        "obfuscation_explanation": warnings,
        "warnings": sorted(set(warnings)),
        "max_decode_depth": max_depth,
        "limits_applied": {
            key: settings[key]
            for key in DEFAULT_LIMITS
        },
        "resource_guard": {"timedOut": timed_out, "truncated": truncated, "totalVariantChars": total_chars},
        "preprocessing_latency_ms": round((perf_counter() - started) * 1000, 3),
        "redacted_preview": " ".join(normalized.split())[:240],
    }
