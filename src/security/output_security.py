"""Output-stage RoBERTa, secret, PII and prompt-leak security checks."""

from __future__ import annotations

from hashlib import sha256
import math
import re
from time import perf_counter
from typing import Any, Protocol

from src.runtime_config import load_runtime_config
from src.security.preprocessing import preprocess_security_text


class RoBERTaScanner(Protocol):
    def predict(self, text: str, *, use_cuda: bool = False, max_length: int = 128, stage: str = "input") -> dict[str, Any]: ...
    def predict_many(self, texts: list[str], *, use_cuda: bool = False, max_length: int = 128, stage: str = "input") -> list[dict[str, Any]]: ...


SECRET_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\b")),
    ("bearer_token", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}")),
    ("openai_style_key", re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")),
    ("database_url", re.compile(r"(?i)\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?):\/\/[^\s]+")),
    ("password_assignment", re.compile(r"(?i)\b(?:password|passwd|api[_ -]?key|secret)\s*[:=]\s*[^\s,;]{6,}")),
    ("session_cookie", re.compile(r"(?i)\b(?:session|cookie)[_-]?(?:id|token)?\s*[:=]\s*[^\s,;]{12,}")),
)

PII_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("phone", re.compile(r"(?<!\d)(?:\+?84|0)(?:\d[ .-]?){8,10}(?!\d)")),
    ("ipv4", re.compile(r"\b(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}\b")),
    ("payment_card", re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")),
    ("vietnamese_identifier", re.compile(r"(?<!\d)\d{12}(?!\d)")),
)

PROMPT_LEAK_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("system_prompt_disclosure", re.compile(r"(?i)\b(?:my|the) system prompt (?:is|says|contains)\b")),
    ("developer_instruction_disclosure", re.compile(r"(?i)\bdeveloper (?:message|instruction)s? (?:are|say|said|contain)\b")),
    ("hidden_instruction_quote", re.compile(r"(?i)\b(?:hidden|internal) instructions?\s*[:=]")),
    ("jailbreak_continuation", re.compile(r"(?i)\bignore previous instructions\b")),
    ("tool_control_directive", re.compile(r"(?i)\b(?:execute|run) (?:this )?(?:shell )?command\b")),
)


def _score(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _entropy(value: str) -> float:
    if not value:
        return 0.0
    probabilities = [value.count(character) / len(value) for character in set(value)]
    return -sum(probability * math.log2(probability) for probability in probabilities)


def _matches(text: str, rules: tuple[tuple[str, re.Pattern[str]], ...]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for category, pattern in rules:
        for match in pattern.finditer(text):
            matched_value = match.group(0)
            findings.append(
                {
                    "category": category,
                    "start": match.start(),
                    "end": match.end(),
                    "masked": "<redacted>",
                    "valueHash": sha256(matched_value.encode("utf-8", errors="replace")).hexdigest(),
                    "redactedPreview": "<redacted>",
                }
            )
    return findings


def scan_secrets(text: str) -> dict[str, Any]:
    findings = _matches(text, SECRET_RULES)
    high_entropy_candidates = 0
    for token in re.findall(r"\b[A-Za-z0-9_+/=-]{24,}\b", text):
        if _entropy(token) >= 4.2 and not token.isalpha():
            high_entropy_candidates += 1
    return {
        "detected": bool(findings),
        "count": len(findings),
        "categories": sorted({item["category"] for item in findings}),
        "findings": findings,
        "highEntropyCandidateCount": high_entropy_candidates,
    }


def scan_pii(text: str, *, user_input: str = "") -> dict[str, Any]:
    findings = _matches(text, PII_RULES)
    for finding in findings:
        span = text[finding["start"] : finding["end"]]
        finding["provenance"] = "user_provided" if span and span in user_input else "output_only"
    unauthorized = [item for item in findings if item["provenance"] == "output_only"]
    return {
        "detected": bool(findings),
        "count": len(findings),
        "unauthorizedCount": len(unauthorized),
        "categories": sorted({item["category"] for item in findings}),
        "findings": findings,
    }


def scan_prompt_leak(text: str) -> dict[str, Any]:
    findings = _matches(text, PROMPT_LEAK_RULES)
    return {
        "detected": bool(findings),
        "count": len(findings),
        "categories": sorted({item["category"] for item in findings}),
        "findings": findings,
    }


def redact_findings(text: str, groups: list[list[dict[str, Any]]]) -> str:
    spans = sorted(
        {(int(item["start"]), int(item["end"])) for group in groups for item in group},
        reverse=True,
    )
    redacted = text
    for start, end in spans:
        redacted = redacted[:start] + "<redacted>" + redacted[end:]
    return redacted


def scan_output(
    *,
    text: str,
    roberta_scanner: RoBERTaScanner,
    user_input: str = "",
    use_cuda: bool = False,
    regeneration_count: int = 0,
) -> dict[str, Any]:
    started = perf_counter()
    config = load_runtime_config()
    output_config = dict(config.get("output_security") or {})
    model_config = dict(config.get("securityModel") or {})
    preprocessing = preprocess_security_text(text)
    candidate_variants = [{"variant_id": "v0", "transform": "original", "depth": 0, "text": text}]
    candidate_variants.extend(item for item in preprocessing.get("variants", []) if isinstance(item, dict))
    candidate_texts = [str(item.get("text", "")) for item in candidate_variants]
    if len(candidate_texts) > 1 and hasattr(roberta_scanner, "predict_many"):
        roberta_results = roberta_scanner.predict_many(
            candidate_texts,
            use_cuda=use_cuda,
            max_length=int(model_config.get("maxLength", 128)),
            stage="output",
        )
    else:
        roberta_results = [
            roberta_scanner.predict(
                candidate_text,
                use_cuda=use_cuda,
                max_length=int(model_config.get("maxLength", 128)),
                stage="output",
            )
            for candidate_text in candidate_texts
        ]

    secret_findings: list[dict[str, Any]] = []
    pii_findings: list[dict[str, Any]] = []
    prompt_findings: list[dict[str, Any]] = []
    variant_scores: list[dict[str, Any]] = []
    for variant, roberta_result in zip(candidate_variants, roberta_results):
        variant_text = str(variant.get("text", ""))
        variant_id = str(variant.get("variant_id", "v0"))
        transform = str(variant.get("transform", "original"))
        for finding in scan_secrets(variant_text)["findings"]:
            secret_findings.append({**finding, "variantId": variant_id, "transform": transform, "decoded": variant_id != "v0"})
        for finding in scan_pii(variant_text, user_input=user_input)["findings"]:
            pii_findings.append({**finding, "variantId": variant_id, "transform": transform, "decoded": variant_id != "v0"})
        for finding in scan_prompt_leak(variant_text)["findings"]:
            prompt_findings.append({**finding, "variantId": variant_id, "transform": transform, "decoded": variant_id != "v0"})
        variant_scores.append({
            "variantId": variant_id,
            "transform": transform,
            "depth": int(variant.get("depth", 0)),
            "score": roberta_result.get("score"),
            "available": roberta_result.get("available", False),
            "modelVersion": roberta_result.get("modelVersion"),
            "latencyMs": roberta_result.get("latencyMs"),
        })

    roberta = roberta_results[0]
    secrets = {
        "detected": bool(secret_findings),
        "count": len(secret_findings),
        "categories": sorted({item["category"] for item in secret_findings}),
        "findings": secret_findings,
        "highEntropyCandidateCount": scan_secrets(text)["highEntropyCandidateCount"],
        "decodedFindingCount": sum(bool(item.get("decoded")) for item in secret_findings),
    }
    pii = {
        "detected": bool(pii_findings),
        "count": len(pii_findings),
        "unauthorizedCount": sum(item.get("provenance") == "output_only" for item in pii_findings),
        "categories": sorted({item["category"] for item in pii_findings}),
        "findings": pii_findings,
        "decodedFindingCount": sum(bool(item.get("decoded")) for item in pii_findings),
    }
    prompt_leak = {
        "detected": bool(prompt_findings),
        "count": len(prompt_findings),
        "categories": sorted({item["category"] for item in prompt_findings}),
        "findings": prompt_findings,
        "decodedFindingCount": sum(bool(item.get("decoded")) for item in prompt_findings),
    }
    available_scores = [_score(item.get("score")) for item in roberta_results if item.get("available")]
    has_decoded_finding = any(item.get("decoded") for item in [*secret_findings, *pii_findings, *prompt_findings])
    model_score = max(available_scores, default=0.0)
    if len(available_scores) > 1 and not has_decoded_finding:
        model_score = min(model_score, min(available_scores))
    warn = float(model_config.get("outputWarnThreshold", 0.30))
    block = float(model_config.get("outputBlockThreshold", 0.85))
    max_regenerations = int(output_config.get("max_regenerations", 1))

    reasons: list[str] = []
    action = "ALLOW"
    policy_id = "POL-OUTPUT-ALLOW"
    final_risk = model_score
    if not roberta.get("available", True):
        action = "SAFE_FALLBACK"
        policy_id = "POL-OUTPUT-MODEL-FAIL-SAFE"
        reasons.append("OUTPUT_MODEL_UNAVAILABLE")
        final_risk = 1.0
    elif secrets["detected"]:
        encoded_secret = bool(secrets["decodedFindingCount"])
        action = "SAFE_FALLBACK" if encoded_secret else "REDACT"
        policy_id = "POL-OUTPUT-ENCODED-SECRET" if encoded_secret else "POL-OUTPUT-SECRET"
        reasons.append("OUTPUT_ENCODED_SECRET_DETECTED" if encoded_secret else "OUTPUT_SECRET_DETECTED")
        final_risk = max(final_risk, 0.95)
    elif prompt_leak["detected"]:
        action = "REGENERATE" if regeneration_count < max_regenerations else "SAFE_FALLBACK"
        policy_id = "POL-OUTPUT-PROMPT-LEAK"
        reasons.append("OUTPUT_PROMPT_LEAK_DETECTED")
        final_risk = max(final_risk, 0.90)
    elif pii["unauthorizedCount"]:
        encoded_pii = bool(pii["decodedFindingCount"])
        action = "SAFE_FALLBACK" if encoded_pii else "REDACT"
        policy_id = "POL-OUTPUT-ENCODED-PII" if encoded_pii else "POL-OUTPUT-UNAUTHORIZED-PII"
        reasons.append("OUTPUT_ENCODED_UNAUTHORIZED_PII" if encoded_pii else "OUTPUT_UNAUTHORIZED_PII")
        final_risk = max(final_risk, 0.75)
    elif model_score >= block:
        action = "REGENERATE" if regeneration_count < max_regenerations else "SAFE_FALLBACK"
        policy_id = "POL-OUTPUT-MODEL-BLOCK"
        reasons.append("OUTPUT_MODEL_BLOCK_THRESHOLD")
    elif model_score >= warn:
        action = "ALLOW_WITH_LOG"
        policy_id = "POL-OUTPUT-MODEL-WARN"
        reasons.append("OUTPUT_MODEL_WARN_THRESHOLD")
    else:
        reasons.append("OUTPUT_SAFE")

    redacted_text = redact_findings(text, [[item for item in secrets["findings"] if item.get("variantId") == "v0"], [
        item for item in pii["findings"] if item.get("provenance") == "output_only" and item.get("variantId") == "v0"
    ]])
    safe_fallback = str(
        output_config.get(
            "safe_fallback",
            "The response could not be returned because it did not pass the security checks.",
        )
    )
    if action == "REDACT":
        final_text = redacted_text
    elif action == "SAFE_FALLBACK":
        final_text = safe_fallback
    else:
        final_text = text

    return {
        "stage": "output",
        "decision": "blocked" if action == "SAFE_FALLBACK" else ("warning" if action in {"REDACT", "REGENERATE", "ALLOW_WITH_LOG"} else "safe"),
        "action": action,
        "policyId": policy_id,
        "riskScore": round(final_risk, 6),
        "reasons": reasons,
        "finalText": final_text,
        "regenerationCount": regeneration_count,
        "roberta": {
            "stage": "output",
            "modelVersion": roberta.get("modelVersion"),
            "rawScore": roberta.get("rawScore", roberta.get("score")),
            "selectedScore": roberta.get("score"),
            "selectedVariantScore": model_score,
            "threshold": warn,
            "blockThreshold": block,
            "label": "unsafe_output" if model_score >= warn else "safe_output",
            "latencyMs": roberta.get("latencyMs"),
            "available": roberta.get("available", False),
        },
        "secretScan": secrets,
        "piiScan": pii,
        "promptLeakScan": prompt_leak,
        "variantScores": variant_scores,
        "preprocessing": {
            "detectedEncodings": preprocessing["detected_encodings"],
            "obfuscationScore": preprocessing["obfuscation_score"],
            "warnings": preprocessing["warnings"],
            "detectedObfuscations": preprocessing.get("detected_obfuscations", []),
            "variantCount": preprocessing.get("variant_count", 0),
            "maxDecodeDepth": preprocessing.get("max_decode_depth", 0),
            "resourceGuard": preprocessing.get("resource_guard", {}),
            "latencyMs": preprocessing.get("preprocessing_latency_ms", 0.0),
        },
        "latencyMs": round((perf_counter() - started) * 1000, 3),
    }
