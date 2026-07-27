"""Document upload signal for indirect prompt injection runtime checks."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from src.audit_log import sanitize_preview
from src.indirect_pipeline import detect_indirect_content
from src.roberta_runtime import roberta_service
from src.runtime_config import load_runtime_config


SUPPORTED_DOCUMENT_EXTENSIONS = {".txt", ".docx", ".pdf"}
MAX_DOCUMENT_BYTES = 8 * 1024 * 1024

RULE_CODE_BY_GROUP = {
    "instruction_override": "DOC_IGNORE_PREVIOUS",
    "system_prompt_extraction": "DOC_REVEAL_SYSTEM_PROMPT",
    "data_exfiltration": "DOC_DATA_EXFILTRATION",
    "concealment": "DOC_CONCEALMENT",
    "safety_or_tool_bypass": "DOC_TOOL_OR_POLICY_BYPASS",
}


def _score(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _source_type_from_name(file_name: str, source_type: str = "auto") -> str:
    normalized = str(source_type or "auto").strip().lower()
    if normalized != "auto":
        return normalized
    suffix = Path(file_name).suffix.lower()
    if suffix == ".txt":
        return "txt"
    if suffix == ".docx":
        return "docx"
    if suffix == ".pdf":
        return "pdf"
    return "auto"


def validate_document_upload(file_name: str, content: bytes) -> None:
    suffix = Path(file_name or "").suffix.lower()
    if suffix not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise ValueError("Only .txt, .docx, and .pdf documents are supported.")
    if not content:
        raise ValueError("Uploaded document is empty.")
    configured_limit = int(
        (load_runtime_config().get("request_limits") or {}).get("max_file_bytes", MAX_DOCUMENT_BYTES)
    )
    if len(content) > configured_limit:
        raise ValueError(f"Uploaded document exceeds the {configured_limit} byte limit.")
    if suffix == ".pdf" and not content.startswith(b"%PDF"):
        raise ValueError("The uploaded .pdf file does not have a valid PDF signature.")
    if suffix == ".docx" and not content.startswith(b"PK"):
        raise ValueError("The uploaded .docx file does not have a valid ZIP/DOCX signature.")
    if suffix == ".txt" and b"\x00" in content[:4096]:
        raise ValueError("The uploaded .txt file appears to contain binary data.")


def _document_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _roberta_chunk_scorer(text: str, model_name: str, use_cuda: bool) -> dict[str, Any]:
    result = roberta_service.predict(text, use_cuda=use_cuda)
    available = bool(result.get("available"))
    score = result.get("score") if available else None
    return {
        "available": available,
        "model": result.get("modelVersion") or model_name,
        "model_path": "",
        "model_score": None if score is None else _score(score),
        "predicted_label": None if score is None else int(_score(score) >= 0.5),
        "thresholds": result.get("thresholdUsed"),
        "runtime_device": None,
        "latencyMs": result.get("latencyMs"),
        "error": result.get("error"),
    }


def _compact_rule(rule: dict[str, Any]) -> dict[str, Any]:
    group = str(rule.get("group", "document_rule"))
    return {
        "code": RULE_CODE_BY_GROUP.get(group, "DOC_RULE_MATCH"),
        "name": group,
        "score": _score(rule.get("weight", 0.0)),
        "matchedText": "<masked>",
    }


def _top_evidence(chunks: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    ranked = sorted(chunks, key=lambda item: float(item.get("final_score", 0.0) or 0.0), reverse=True)
    evidence: list[dict[str, Any]] = []
    for chunk in ranked[:limit]:
        metadata = chunk.get("metadata", {}) if isinstance(chunk.get("metadata"), dict) else {}
        matched_rules = [_compact_rule(rule) for rule in chunk.get("matched_rules", []) or [] if isinstance(rule, dict)]
        context_signals = [
            {
                "group": signal.get("group"),
                "score": _score(signal.get("weight", 0.0)),
                "matchedText": "<masked>",
            }
            for signal in chunk.get("context_signals", []) or []
            if isinstance(signal, dict)
        ]
        evidence.append(
            {
                "chunkId": metadata.get("chunk_id"),
                "pageNumber": metadata.get("page_number"),
                "score": _score(chunk.get("final_score")),
                "ruleScore": _score(chunk.get("rule_score")),
                "robertaScore": None if chunk.get("model_score") is None else _score(chunk.get("model_score")),
                "contextScore": _score(chunk.get("context_score")),
                "recommendedAction": chunk.get("recommended_action"),
                "preview": sanitize_preview(chunk.get("text"), max_chars=220),
                "matchedRules": matched_rules,
                "contextSignals": context_signals,
                "variantAnalysis": chunk.get("variant_analysis", {}),
            }
        )
    return evidence


def analyze_uploaded_document(
    *,
    user_message: str,
    file_name: str,
    content: bytes,
    source_type: str = "auto",
    use_cuda: bool = False,
) -> dict[str, Any]:
    """Analyze an uploaded document as untrusted external content."""
    validate_document_upload(file_name, content)
    resolved_source_type = _source_type_from_name(file_name, source_type)
    result = detect_indirect_content(
        user_task=user_message,
        content_bytes=content,
        source_type=resolved_source_type,
        source_name=file_name,
        model_name=roberta_service.resolved_model_name,
        safe_context_policy="exclude",
        use_cuda=use_cuda,
    )
    action = str(result.get("recommended_action", "allow"))
    decision = "blocked" if action == "block" else ("warning" if action == "sanitize_or_warn" else "safe")
    chunks = result.get("chunks", []) if isinstance(result.get("chunks"), list) else []
    source_metadata = result.get("source_metadata", {}) if isinstance(result.get("source_metadata"), dict) else {}
    safe_context = result.get("safe_context", {}) if isinstance(result.get("safe_context"), dict) else {}
    evidence = _top_evidence(chunks)
    matched_rules: list[dict[str, Any]] = []
    seen_codes: set[str] = set()
    for item in evidence:
        for rule in item["matchedRules"]:
            code = str(rule.get("code", "DOC_RULE_MATCH"))
            if code not in seen_codes:
                matched_rules.append(rule)
                seen_codes.add(code)

    page_numbers = {
        chunk.get("metadata", {}).get("page_number")
        for chunk in chunks
        if isinstance(chunk.get("metadata"), dict) and chunk.get("metadata", {}).get("page_number") is not None
    }
    return {
        "score": _score(result.get("final_score")),
        "decision": decision,
        "recommendedAction": action,
        "label": {"safe": "SAFE", "warning": "WARNING", "blocked": "BLOCKED"}[decision],
        "ruleScore": _score(result.get("rule_score")),
        "robertaScore": None if result.get("model_score") is None else _score(result.get("model_score")),
        "contextScore": _score(result.get("context_score")),
        "matchedRules": matched_rules,
        "hardBlock": decision == "blocked",
        "reasonCodes": [] if decision == "safe" else ["DOC_INDIRECT_INJECTION"],
        "evidence": evidence,
        "source": {
            "fileName": file_name,
            "sourceType": source_metadata.get("source_type", resolved_source_type),
            "chunkCount": int(source_metadata.get("chunk_count", len(chunks)) or len(chunks)),
            "pageCount": len(page_numbers) if page_numbers else None,
            "sha256": _document_hash(content),
            "sizeBytes": len(content),
        },
        "safeContextText": str(safe_context.get("context_text", "")),
        "safeChunkCount": int(safe_context.get("safe_chunk_count", 0) or 0),
        "unsafeChunkCount": int(safe_context.get("unsafe_chunk_count", 0) or 0),
        "model": result.get("model", roberta_service.resolved_model_name),
        "warnings": result.get("warnings", []),
        "encodingSummary": {
            "affectedChunkCount": sum(bool((chunk.get("variant_analysis") or {}).get("variantCount")) for chunk in chunks),
            "maxDecodeDepth": max((int((chunk.get("variant_analysis") or {}).get("selectedDepth", 0) or 0) for chunk in chunks), default=0),
            "selectedTransforms": sorted({
                str((chunk.get("variant_analysis") or {}).get("selectedTransform"))
                for chunk in chunks
                if (chunk.get("variant_analysis") or {}).get("selectedTransform") not in {None, "original"}
            }),
        },
    }
