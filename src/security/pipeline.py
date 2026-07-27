"""Reusable Hybrid Sandwich Security orchestration helpers."""

from __future__ import annotations

from typing import Any

from src.runtime_config import load_runtime_config
from src.security.output_security import scan_output
from src.security.preprocessing import preprocess_security_text
from src.security.source_separation import separate_request_sources
from src.security.tool_gateway import authorize_tool


class HybridSandwichSecurityPipeline:
    """Coordinate shared security stages while retaining existing detectors."""

    def preprocess_input(self, text: str) -> dict[str, Any]:
        limits = dict(load_runtime_config().get("request_limits") or {})
        max_chars = int(limits.get("max_text_chars", 50000))
        if not str(text or "").strip():
            raise ValueError("Input text must not be empty.")
        if len(text) > max_chars:
            raise ValueError(f"Input text exceeds the {max_chars} character limit.")
        return preprocess_security_text(text)

    def separate_sources(
        self,
        *,
        user_message: str,
        user_role: str,
        project_context: dict[str, Any] | None,
        explicit_context: str | None,
        explicit_source_type: str = "unknown",
    ) -> dict[str, Any]:
        return separate_request_sources(
            user_message=user_message,
            user_role=user_role,
            project_context=project_context,
            explicit_context=explicit_context,
            explicit_source_type=explicit_source_type,
        )

    def scan_output(
        self,
        *,
        text: str,
        roberta_scanner: Any,
        user_input: str,
        use_cuda: bool,
        regeneration_count: int = 0,
    ) -> dict[str, Any]:
        return scan_output(
            text=text,
            roberta_scanner=roberta_scanner,
            user_input=user_input,
            use_cuda=use_cuda,
            regeneration_count=regeneration_count,
        )

    def authorize_tool(self, **kwargs: Any) -> dict[str, Any]:
        return authorize_tool(**kwargs)


security_pipeline = HybridSandwichSecurityPipeline()
