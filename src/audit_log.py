"""Structured audit logging for runtime detection and admin monitoring."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from threading import Lock
from typing import Any

from src.security.preprocessing import redact_sensitive_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_LOG_PATH = PROJECT_ROOT / "data" / "audit_log.jsonl"
_LOCK = Lock()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_text(text: str | None) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()


def sanitize_preview(text: str | None, max_chars: int = 180) -> str:
    value = " ".join(str(text or "").split())
    value = redact_sensitive_text(value)
    if len(value) > max_chars:
        return value[:max_chars] + "..."
    return value


def append_audit_log(record: dict[str, Any], path: Path = DEFAULT_AUDIT_LOG_PATH) -> dict[str, Any]:
    payload = {**record, "timestamp": record.get("timestamp") or now_iso()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return payload


def read_audit_logs(path: Path = DEFAULT_AUDIT_LOG_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with _LOCK:
        lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def filter_audit_logs(
    *,
    user_id: str | None = None,
    project_id: str | None = None,
    decision: str | None = None,
    detection_type: str | None = None,
    min_risk: float | None = None,
    max_risk: float | None = None,
    source: str | None = None,
    category: str | None = None,
    risk_level: str | None = None,
    model_version: str | None = None,
    stage: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    rows = read_audit_logs()
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if user_id and row.get("user_id") != user_id:
            continue
        if project_id and row.get("project_id") != project_id:
            continue
        if decision and str(row.get("decision", "")).lower() != decision.lower():
            continue
        if detection_type and detection_type not in row.get("detection_type", []):
            continue
        if source and str(row.get("input_source", "")).lower() != source.lower():
            continue
        if category and category not in row.get("detection_type", []):
            continue
        if risk_level and str(row.get("risk_level", "")).lower() != risk_level.lower():
            continue
        if model_version and str(row.get("model_version", "")) != model_version:
            continue
        if stage and stage.lower() not in {str(item).lower() for item in row.get("stages", []) or []}:
            continue
        timestamp = str(row.get("timestamp", ""))
        if date_from and timestamp < date_from:
            continue
        if date_to and timestamp > date_to:
            continue
        risk = float(row.get("fusion_score", 0.0) or 0.0)
        if min_risk is not None and risk < min_risk:
            continue
        if max_risk is not None and risk > max_risk:
            continue
        filtered.append(row)
    filtered.sort(key=lambda item: item.get("timestamp", ""), reverse=True)
    return filtered[: max(1, min(limit, 1000))]


def audit_summary() -> dict[str, Any]:
    rows = read_audit_logs()
    total = len(rows)
    decisions = {"safe": 0, "warning": 0, "blocked": 0}
    detection_counts: dict[str, int] = {}
    top_rules: dict[str, int] = {}
    risky_projects: dict[str, int] = {}
    total_latency = 0.0
    llm_errors = 0
    llm_total = 0
    total_tokens = 0
    total_cost = 0.0
    latencies: list[float] = []
    output_blocked = 0
    output_warned = 0
    prompt_leaks = 0
    secret_leaks = 0
    tool_abuse = 0
    source_counts: dict[str, int] = {}
    model_versions: dict[str, int] = {}
    encoded_input_count = 0
    decoded_malicious_count = 0
    nested_encoding_count = 0
    encoded_output_count = 0
    for row in rows:
        decision = str(row.get("decision", "")).lower()
        if decision in decisions:
            decisions[decision] += 1
        for item in row.get("detection_type", []) or []:
            detection_counts[str(item)] = detection_counts.get(str(item), 0) + 1
        for rule in row.get("matched_rules", []) or []:
            code = str(rule.get("code", "UNKNOWN")) if isinstance(rule, dict) else str(rule)
            top_rules[code] = top_rules.get(code, 0) + 1
        if decision == "blocked" and row.get("project_id"):
            key = str(row["project_id"])
            risky_projects[key] = risky_projects.get(key, 0) + 1
        latency = float(row.get("total_latency_ms", 0.0) or 0.0)
        total_latency += latency
        latencies.append(latency)
        output_decision = str(row.get("output_decision", "not_scanned")).lower()
        output_blocked += int(output_decision == "blocked")
        output_warned += int(output_decision == "warning")
        prompt_leaks += int(bool(row.get("prompt_leak_detected")))
        secret_leaks += int(bool(row.get("secret_detected")))
        tool_abuse += int("tool_abuse" in (row.get("detection_type", []) or []))
        source = str(row.get("input_source", "chat"))
        source_counts[source] = source_counts.get(source, 0) + 1
        version = str(row.get("model_version", "unknown"))
        model_versions[version] = model_versions.get(version, 0) + 1
        encoded_input_count += int(bool(row.get("detected_encodings") or row.get("detected_obfuscations")))
        decoded_malicious_count += int(bool(row.get("decoded_malicious_content")))
        nested_encoding_count += int(int(row.get("max_decode_depth", 0) or 0) >= 2)
        encoded_output_count += int(
            bool(row.get("output_encoded_secret_count") or row.get("output_encoded_pii_count") or row.get("output_encoded_prompt_leak_count"))
        )
        llm = row.get("llm", {}) if isinstance(row.get("llm"), dict) else {}
        if llm.get("called"):
            llm_total += 1
        if llm.get("status") == "error":
            llm_errors += 1
        usage = llm.get("tokenUsage", {}) if isinstance(llm.get("tokenUsage"), dict) else {}
        total_tokens += int(usage.get("total", 0) or 0)
        total_cost += float(llm.get("estimatedCost", 0.0) or 0.0)
    latencies.sort()
    p95_index = max(0, min(len(latencies) - 1, int(0.95 * len(latencies)))) if latencies else 0
    return {
        "totalRequests": total,
        "safeCount": decisions["safe"],
        "warningCount": decisions["warning"],
        "blockedCount": decisions["blocked"],
        "safeRate": decisions["safe"] / total if total else 0.0,
        "warningRate": decisions["warning"] / total if total else 0.0,
        "blockedRate": decisions["blocked"] / total if total else 0.0,
        "inputBlockRate": decisions["blocked"] / total if total else 0.0,
        "outputBlockRate": output_blocked / total if total else 0.0,
        "outputWarningCount": output_warned,
        "directInjectionCount": detection_counts.get("direct", 0),
        "indirectInjectionCount": detection_counts.get("indirect", 0),
        "contextMismatchCount": detection_counts.get("context_mismatch", 0),
        "averageLatencyMs": total_latency / total if total else 0.0,
        "p95LatencyMs": latencies[p95_index] if latencies else 0.0,
        "promptLeakCount": prompt_leaks,
        "secretLeakCount": secret_leaks,
        "toolAbuseCount": tool_abuse,
        "llmErrorRate": llm_errors / llm_total if llm_total else 0.0,
        "tokenUsage": total_tokens,
        "estimatedLlmCost": round(total_cost, 6),
        "topMatchedRules": sorted(top_rules.items(), key=lambda item: item[1], reverse=True)[:10],
        "topRiskyProjects": sorted(risky_projects.items(), key=lambda item: item[1], reverse=True)[:10],
        "topRiskySources": sorted(source_counts.items(), key=lambda item: item[1], reverse=True)[:10],
        "modelVersions": sorted(model_versions.items(), key=lambda item: item[1], reverse=True),
        "encodedInputCount": encoded_input_count,
        "decodedMaliciousCount": decoded_malicious_count,
        "nestedEncodingCount": nested_encoding_count,
        "encodedOutputStoppedCount": encoded_output_count,
    }
