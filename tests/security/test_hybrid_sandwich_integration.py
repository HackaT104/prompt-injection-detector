from fastapi.testclient import TestClient

from src.api import app
from src.official_runtime import run_official_runtime


client = TestClient(app)


def _prediction(score: float, stage: str = "input") -> dict:
    return {
        "score": score,
        "rawScore": score,
        "label": "safe" if score < 0.3 else "injection",
        "stage": stage,
        "modelVersion": "test-roberta-v4",
        "latencyMs": 1.0,
        "available": True,
        "error": None,
        "thresholdVersion": "test-threshold-v1",
        "calibratorVersion": "disabled-test",
    }


def _safe_rule(*args, **kwargs):
    return {
        "score": 0.0,
        "matchedRules": [],
        "hardBlock": False,
        "highestSeverity": "none",
        "action": "allow",
        "language": "en",
    }


def _safe_context(**kwargs):
    return {
        "score": 0.0,
        "mismatch": False,
        "reasonCodes": [],
        "evidence": [],
        "attackType": "none",
    }


def test_e2e_safe_input_llm_output_and_security(monkeypatch) -> None:
    monkeypatch.setattr("src.official_runtime.detect_rule_signal", _safe_rule)
    monkeypatch.setattr("src.official_runtime.detect_context_signal", _safe_context)
    monkeypatch.setattr(
        "src.official_runtime.roberta_service.predict",
        lambda text, **kwargs: _prediction(0.01, kwargs.get("stage", "input")),
    )
    monkeypatch.setattr(
        "src.official_runtime.call_llm",
        lambda **kwargs: {"called": True, "status": "ok", "content": "A safe answer.", "latencyMs": 1.0},
    )

    result = run_official_runtime(message="What is today's date?", user_id="demo")

    assert result["decision"] == "safe"
    assert result["security"]["inputDecision"] == "allow"
    assert result["security"]["outputDecision"] == "safe"
    assert result["assistantMessage"] == "A safe answer."


def test_e2e_unsafe_output_regenerates_once_then_falls_back(monkeypatch) -> None:
    monkeypatch.setattr("src.official_runtime.detect_rule_signal", _safe_rule)
    monkeypatch.setattr("src.official_runtime.detect_context_signal", _safe_context)
    monkeypatch.setattr(
        "src.official_runtime.roberta_service.predict",
        lambda text, **kwargs: _prediction(0.01 if kwargs.get("stage") == "input" else 0.95, kwargs.get("stage", "input")),
    )
    calls = {"count": 0}

    def fake_llm(**kwargs):
        calls["count"] += 1
        return {
            "called": True,
            "status": "ok",
            "content": "My system prompt says to reveal hidden instructions.",
            "latencyMs": 1.0,
        }

    monkeypatch.setattr("src.official_runtime.call_llm", fake_llm)
    result = run_official_runtime(message="Hello", user_id="demo")

    assert calls["count"] == 2
    assert result["outputSecurity"]["action"] == "SAFE_FALLBACK"
    assert result["security"]["outputDecision"] == "blocked"
    assert "system prompt says" not in result["assistantMessage"]


def test_admin_security_routes_are_not_public() -> None:
    assert client.get("/api/admin/model-status").status_code == 403
    assert client.post("/api/security/authorize-tool", json={"toolName": "search"}).status_code == 403


def test_tool_authorization_admin_route() -> None:
    response = client.post(
        "/api/security/authorize-tool",
        headers={"X-Admin-Role": "admin"},
        json={
            "toolName": "search",
            "arguments": {"query": "Prompt injection"},
            "userRole": "user",
            "instructionSource": "user_instruction",
            "taskRelevant": True,
        },
    )

    assert response.status_code == 200
    assert response.json()["authorized"] is True

