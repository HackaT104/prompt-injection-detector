from src.llm_service import llm_config_status


def test_gemini_key_configures_llm_without_openai_env(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

    status = llm_config_status()

    assert status["configured"] is True
    assert status["provider"] == "gemini"
    assert status["model"] == "gemini-3.5-flash"
    assert status["baseUrlConfigured"] is True
