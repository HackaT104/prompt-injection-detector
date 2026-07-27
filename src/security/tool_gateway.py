"""Policy-only tool authorization gateway for future function calling."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    tool_name: str
    risk_level: str
    allowed_roles: tuple[str, ...]
    requires_confirmation: bool
    allowed_sources: tuple[str, ...]
    required_arguments: tuple[str, ...] = field(default_factory=tuple)


DEFAULT_TOOL_REGISTRY: dict[str, ToolDefinition] = {
    "search": ToolDefinition("search", "low", ("user", "admin"), False, ("user_instruction",), ("query",)),
    "read_document": ToolDefinition(
        "read_document", "low", ("user", "admin"), False, ("user_instruction",), ("document_id",)
    ),
    "send_email": ToolDefinition(
        "send_email", "medium", ("user", "admin"), True, ("user_instruction",), ("to", "subject", "body")
    ),
    "write_data": ToolDefinition(
        "write_data", "high", ("admin",), True, ("user_instruction",), ("target", "value")
    ),
    "delete_data": ToolDefinition(
        "delete_data", "critical", ("admin",), True, ("user_instruction",), ("target",)
    ),
    "execute_command": ToolDefinition(
        "execute_command", "critical", ("admin",), True, ("user_instruction",), ("command",)
    ),
}


def authorize_tool(
    *,
    tool_name: str,
    arguments: dict[str, Any] | None,
    user_role: str,
    instruction_source: str,
    task_relevant: bool,
    confirmed: bool = False,
) -> dict[str, Any]:
    normalized_name = str(tool_name or "").strip().lower()
    definition = DEFAULT_TOOL_REGISTRY.get(normalized_name)
    reasons: list[str] = []
    if definition is None:
        return {
            "authorized": False,
            "decision": "BLOCK",
            "policyId": "TOOL-UNKNOWN-DENY",
            "reasons": ["Unknown tools are denied by default."],
            "tool": normalized_name,
        }

    normalized_role = str(user_role or "user").strip().lower()
    normalized_source = str(instruction_source or "unknown").strip().lower()
    if normalized_source not in definition.allowed_sources:
        reasons.append("External, retrieved, email and tool-output content cannot directly authorize a tool.")
    if normalized_role not in definition.allowed_roles:
        reasons.append(f"Role '{normalized_role}' is not allowed to use this tool.")
    if not task_relevant:
        reasons.append("The proposed tool is not relevant to the user's direct task.")

    supplied = arguments if isinstance(arguments, dict) else {}
    missing = [name for name in definition.required_arguments if supplied.get(name) in {None, ""}]
    if missing:
        reasons.append("Missing required tool arguments: " + ", ".join(missing))
    if reasons:
        return {
            "authorized": False,
            "decision": "BLOCK",
            "policyId": "TOOL-DENY-POLICY",
            "reasons": reasons,
            "tool": asdict(definition),
        }
    if definition.requires_confirmation and not confirmed:
        return {
            "authorized": False,
            "decision": "REQUIRE_CONFIRMATION",
            "policyId": "TOOL-CONFIRMATION-REQUIRED",
            "reasons": ["Explicit user confirmation is required before execution."],
            "tool": asdict(definition),
        }
    return {
        "authorized": True,
        "decision": "ALLOW",
        "policyId": "TOOL-AUTHORIZED",
        "reasons": ["Role, source, relevance and argument checks passed."],
        "tool": asdict(definition),
    }

