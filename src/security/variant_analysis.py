"""Rule, RoBERTa and context analysis over bounded security variants."""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Callable

from src.context_runtime import detect_encoded_context_signal
from src.runtime_rule_signal import detect_rule_signal


TRANSFORM_RULE_IDS = {
    "base64_decode": "ENC-BASE64-DETECTED",
    "url_decode": "ENC-URL-DETECTED",
    "hex_decode": "ENC-HEX-DETECTED",
    "unicode_escape_decode": "ENC-UNICODE-ESCAPE",
    "html_entity_decode": "ENC-HTML-ENTITY",
    "rot13_decode": "ENC-ROT13",
    "ascii_decimal_decode": "ENC-ASCII-DECIMAL",
    "binary_decode": "ENC-BINARY",
    "zero_width_remove": "OBF-ZERO-WIDTH",
    "homoglyph_normalize": "OBF-HOMOGLYPH",
    "bidi_control_remove": "OBF-BIDI-CONTROL",
    "whitespace_split_normalize": "OBF-WHITESPACE-SPLIT",
    "punctuation_split_normalize": "OBF-PUNCTUATION-SPLIT",
    "leetspeak_normalize": "OBF-LEETSPEAK",
    "typoglycemia_normalize": "OBF-TYPOGLYCEMIA",
    "case_normalize": "OBF-CASE-ALTERNATION",
    "repeated_character_normalize": "OBF-REPEATED-CHARACTER",
}
CATEGORY_RULE_IDS = {
    "encoded_instruction_override": "ENCODED-INSTRUCTION-OVERRIDE",
    "encoded_system_prompt_extraction": "ENCODED-SYSTEM-PROMPT-EXTRACTION",
    "encoded_data_exfiltration": "ENCODED-DATA-EXFILTRATION",
    "encoded_tool_activation": "ENCODED-TOOL-ABUSE",
    "encoded_jailbreak": "ENCODED-JAILBREAK",
    "encoded_sensitive_extraction": "ENCODED-DATA-EXFILTRATION",
}


def _score(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _public_variant(variant: dict[str, Any]) -> dict[str, Any]:
    return {
        "variantId": variant.get("variant_id"),
        "parentVariantId": variant.get("parent_variant_id"),
        "transform": variant.get("transform"),
        "transformChain": variant.get("transform_chain", []),
        "depth": variant.get("depth", 0),
        "textHash": variant.get("text_hash"),
        "printableRatio": variant.get("printable_ratio"),
        "readabilityScore": variant.get("readability_score"),
        "confidence": variant.get("confidence"),
        "metadata": variant.get("metadata", {}),
    }


def analyze_security_variants(
    *,
    preprocessing: dict[str, Any],
    user_message: str,
    source_type: str,
    roberta_scanner: Any,
    use_cuda: bool = False,
    stage: str = "input",
    rule_detector: Callable[..., dict[str, Any]] = detect_rule_signal,
) -> dict[str, Any]:
    """Analyze original, normalized and decoded variants and select highest semantic risk."""
    original = str(preprocessing.get("original_text", ""))
    normalized = str(preprocessing.get("normalized_text", original))
    raw_variants = [item for item in preprocessing.get("variants", []) if isinstance(item, dict)]
    candidates: list[dict[str, Any]] = [{
        "variant_id": "v0",
        "parent_variant_id": None,
        "transform": "original",
        "transform_chain": [],
        "depth": 0,
        "text": original,
        "text_hash": None,
        "printable_ratio": 1.0,
        "readability_score": 1.0,
        "confidence": 1.0,
        "metadata": {},
    }]
    if normalized and normalized != original:
        candidates.append({
            "variant_id": "vn",
            "parent_variant_id": "v0",
            "transform": "nfkc_normalize",
            "transform_chain": ["nfkc_normalize"],
            "depth": 0,
            "text": normalized,
            "text_hash": None,
            "printable_ratio": 1.0,
            "readability_score": 1.0,
            "confidence": 1.0,
            "metadata": {},
        })
    candidates.extend(raw_variants)

    texts = [str(item.get("text", "")) for item in candidates]
    roberta_results = (
        roberta_scanner.predict_many(texts, use_cuda=use_cuda, stage=stage)
        if len(texts) > 1
        else [roberta_scanner.predict(texts[0], use_cuda=use_cuda, stage=stage)]
    )
    analyses: list[dict[str, Any]] = []
    for variant, roberta in zip(candidates, roberta_results):
        text = str(variant.get("text", ""))
        rule = rule_detector(text, source_type="external_content" if source_type != "chat" else "user_prompt")
        context = detect_encoded_context_signal(user_message=user_message, decoded_text=text, source_type=source_type)
        rule_score = _score(rule.get("score"))
        roberta_score = _score(roberta.get("score")) if roberta.get("available", True) else 0.0
        context_score = _score(context.get("score"))
        semantic_score = max(rule_score, roberta_score, context_score)
        if context.get("benignReferenceIntent") and not context.get("executionIntent"):
            effective_score = min(semantic_score, 0.24)
        elif int(variant.get("depth", 0)) > 0 and not context.get("decodedInstruction") and not rule.get("matchedRules"):
            effective_score = min(semantic_score, 0.24)
        else:
            effective_score = semantic_score
        if context.get("decodedInstruction") and int(variant.get("depth", 0)) > 0:
            effective_score = max(effective_score, context_score)
        analyses.append({
            "variant": variant,
            "rule": rule,
            "roberta": roberta,
            "context": context,
            "semanticScore": round(semantic_score, 6),
            "effectiveScore": round(effective_score, 6),
        })

    original_analysis = analyses[0]
    transformed_analyses = [item for item in analyses if int(item["variant"].get("depth", 0)) > 0]
    benign_transformed_payload = bool(
        transformed_analyses
        and not any(item["context"].get("decodedInstruction") or item["rule"].get("matchedRules") for item in transformed_analyses)
    )
    if benign_transformed_payload:
        original_analysis["effectiveScore"] = min(original_analysis["effectiveScore"], 0.24)

    selected = max(
        analyses,
        key=lambda item: (
            item["effectiveScore"],
            int(item["variant"].get("depth", 0)),
            _score(item["variant"].get("confidence")),
        ),
    )
    selected_variant = selected["variant"]
    selected_text = str(selected_variant.get("text", ""))
    selected_hash = str(selected_variant.get("text_hash") or sha256(selected_text.encode("utf-8", errors="replace")).hexdigest())
    decoded_malicious = bool(
        int(selected_variant.get("depth", 0)) > 0
        and selected["context"].get("decodedInstruction")
    )
    benign_reference = bool(selected["context"].get("benignReferenceIntent"))
    execution_intent = bool(selected["context"].get("executionIntent"))
    reason_codes = list(selected["context"].get("reasonCodes", []))
    if decoded_malicious:
        reason_codes.append("ENC_MALICIOUS_DECODED_CONTENT")
    if int(selected_variant.get("depth", 0)) >= 2 and decoded_malicious:
        reason_codes.append("ENC_NESTED_MALICIOUS_CONTENT")

    technique_rules: list[dict[str, Any]] = []
    observed_transforms = [str(item.get("transform")) for item in raw_variants]
    for transform in sorted(set(observed_transforms)):
        code = TRANSFORM_RULE_IDS.get(transform)
        if not code:
            continue
        technique_rules.append({"code": code, "severity": "low", "score": 0.15, "variantId": None})
    if observed_transforms.count("base64_decode") >= 2:
        technique_rules.append({"code": "ENC-BASE64-NESTED", "severity": "medium", "score": 0.35, "variantId": selected_variant.get("variant_id")})
    if observed_transforms.count("url_decode") >= 2:
        technique_rules.append({"code": "ENC-URL-NESTED", "severity": "medium", "score": 0.35, "variantId": selected_variant.get("variant_id")})
    if any(bool((item.get("metadata") or {}).get("mixedScript")) for item in raw_variants):
        technique_rules.append({"code": "OBF-MIXED-SCRIPT", "severity": "medium", "score": 0.40, "variantId": selected_variant.get("variant_id")})
    malicious_rule_id = CATEGORY_RULE_IDS.get(str(selected["context"].get("category"))) if decoded_malicious else None
    if malicious_rule_id:
        technique_rules.append({"code": malicious_rule_id, "severity": "critical" if selected["context"].get("sensitiveTarget") or selected["context"].get("toolActivation") else "high", "score": selected["effectiveScore"], "variantId": selected_variant.get("variant_id")})

    effective_obfuscation_score = _score(preprocessing.get("obfuscation_score"))
    if decoded_malicious:
        effective_obfuscation_score = max(effective_obfuscation_score, 0.75)
    if int(selected_variant.get("depth", 0)) >= 2 and decoded_malicious:
        effective_obfuscation_score = max(effective_obfuscation_score, 0.90)
    if decoded_malicious and any(str(item.get("transform", "")).startswith(("zero_width", "homoglyph", "bidi", "whitespace", "punctuation", "leetspeak", "typoglycemia", "case", "repeated")) for item in raw_variants):
        effective_obfuscation_score = max(effective_obfuscation_score, 0.82)
    if execution_intent:
        effective_obfuscation_score = max(effective_obfuscation_score, 0.90)

    public_analyses = []
    for analysis in analyses:
        public_analyses.append({
            **_public_variant(analysis["variant"]),
            "ruleScore": analysis["rule"].get("score", 0.0),
            "robertaScore": analysis["roberta"].get("score"),
            "contextScore": analysis["context"].get("score", 0.0),
            "effectiveScore": analysis["effectiveScore"],
            "matchedRuleCodes": [item.get("code") for item in analysis["rule"].get("matchedRules", [])],
            "contextCategory": analysis["context"].get("category"),
        })
    return {
        "selectedVariantId": selected_variant.get("variant_id"),
        "selectedTransform": selected_variant.get("transform"),
        "selectedTransformChain": selected_variant.get("transform_chain", []),
        "selectedDepth": int(selected_variant.get("depth", 0)),
        "selectedConfidence": _score(selected_variant.get("confidence", 1.0)),
        "selectedDecodedPreview": f"<redacted:{selected_hash[:12]}:len={len(selected_text)}>",
        "selectedRuleSignal": selected["rule"],
        "selectedRoBERTaSignal": selected["roberta"],
        "selectedContextSignal": selected["context"],
        "originalRoBERTaSignal": original_analysis["roberta"],
        "originalRoBERTaScore": original_analysis["roberta"].get("score"),
        "selectedVariantRoBERTaScore": selected["roberta"].get("score"),
        "selectedVariantRiskScore": selected["effectiveScore"],
        "decodedMaliciousContent": decoded_malicious,
        "executionIntent": execution_intent,
        "benignReferenceIntent": benign_reference,
        "benignTransformedPayload": benign_transformed_payload,
        "sensitiveTarget": bool(selected["context"].get("sensitiveTarget")),
        "toolActivation": bool(selected["context"].get("toolActivation")),
        "attackCategory": selected["context"].get("category"),
        "reasonCodes": sorted(set(reason_codes)),
        "techniqueRules": technique_rules,
        "effectiveObfuscationScore": round(effective_obfuscation_score, 6),
        "obfuscationExplanation": [item["code"] for item in technique_rules],
        "variantCount": len(raw_variants),
        "maxDecodeDepth": preprocessing.get("max_decode_depth", 0),
        "variants": public_analyses,
    }
