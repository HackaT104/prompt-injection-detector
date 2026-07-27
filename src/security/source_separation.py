"""Preserve instruction provenance instead of flattening all request content."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.runtime_config import load_runtime_config


@dataclass(slots=True)
class SourceBlock:
    source_id: str
    source_type: str
    content: str
    trust_level: float
    trusted: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def source_trust(source_type: str) -> float:
    config = load_runtime_config()
    trust = dict(config.get("source_trust") or {})
    normalized = str(source_type or "unknown").strip().lower()
    aliases = {
        "document": "uploaded_document",
        "pdf": "uploaded_document",
        "docx": "uploaded_document",
        "txt": "uploaded_document",
        "external_content": "unknown",
        "user_prompt": "authenticated_user",
    }
    return float(trust.get(aliases.get(normalized, normalized), trust.get("unknown", 0.1)))


def separate_request_sources(
    *,
    user_message: str,
    user_role: str = "user",
    project_context: dict[str, Any] | None = None,
    explicit_context: str | None = None,
    explicit_source_type: str = "unknown",
) -> dict[str, Any]:
    trusted: list[SourceBlock] = []
    untrusted: list[SourceBlock] = []
    role = "admin" if str(user_role).lower() == "admin" else "authenticated_user"
    trusted.append(
        SourceBlock(
            source_id="user:message",
            source_type="user_instruction",
            content=str(user_message),
            trust_level=source_trust(role),
            trusted=True,
            metadata={"role": user_role},
        )
    )

    project = project_context or {}
    system_instruction = str(project.get("systemInstruction", "")).strip()
    if system_instruction:
        trusted.append(
            SourceBlock(
                source_id="project:instruction",
                source_type="project_context",
                content=system_instruction,
                trust_level=source_trust("project_context"),
                trusted=True,
            )
        )
    for index, document in enumerate(project.get("documents", []) or [], start=1):
        if not isinstance(document, dict):
            continue
        untrusted.append(
            SourceBlock(
                source_id=str(document.get("id") or f"project:document:{index}"),
                source_type=str(document.get("type") or "uploaded_document"),
                content=str(document.get("content", "")),
                trust_level=source_trust(str(document.get("type") or "uploaded_document")),
                trusted=False,
                metadata={"title": document.get("title"), "trustLabel": "untrusted"},
            )
        )
    if explicit_context and str(explicit_context).strip():
        resolved_source_type = (
            "uploaded_document" if explicit_source_type == "document" else str(explicit_source_type or "unknown")
        )
        untrusted.append(
            SourceBlock(
                source_id="request:explicit-context",
                source_type=resolved_source_type,
                content=str(explicit_context),
                trust_level=source_trust(resolved_source_type),
                trusted=False,
            )
        )

    minimum_trust = min((block.trust_level for block in untrusted), default=1.0)
    return {
        "user_instruction": str(user_message),
        "trusted_context": [block.to_dict() for block in trusted],
        "untrusted_content": [block.to_dict() for block in untrusted],
        "source_risk": round(1.0 - minimum_trust, 6) if untrusted else 0.0,
    }
