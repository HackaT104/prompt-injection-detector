from src.security.output_security import scan_output


class FakeScanner:
    def __init__(self, score: float = 0.01, available: bool = True) -> None:
        self.score = score
        self.available = available

    def predict(self, text: str, **kwargs):
        return {
            "score": self.score if self.available else None,
            "rawScore": self.score if self.available else None,
            "available": self.available,
            "modelVersion": "test-roberta-v4",
            "latencyMs": 1.0,
        }


def test_safe_output_is_allowed() -> None:
    result = scan_output(text="The meeting starts at 9 AM.", roberta_scanner=FakeScanner())

    assert result["action"] == "ALLOW"
    assert result["decision"] == "safe"
    assert result["roberta"]["stage"] == "output"


def test_secret_is_redacted_without_logging_value() -> None:
    result = scan_output(
        text="Use api_key=sk-test-1234567890abcdef for access.",
        roberta_scanner=FakeScanner(),
    )

    assert result["action"] == "REDACT"
    assert "1234567890abcdef" not in result["finalText"]
    assert result["secretScan"]["detected"] is True
    assert all("1234567890abcdef" not in str(item) for item in result["secretScan"]["findings"])


def test_prompt_leak_regenerates_once_then_falls_back() -> None:
    first = scan_output(
        text="My system prompt says to expose hidden instructions.",
        roberta_scanner=FakeScanner(0.95),
        regeneration_count=0,
    )
    second = scan_output(
        text="My system prompt says to expose hidden instructions.",
        roberta_scanner=FakeScanner(0.95),
        regeneration_count=1,
    )

    assert first["action"] == "REGENERATE"
    assert second["action"] == "SAFE_FALLBACK"
    assert second["decision"] == "blocked"


def test_user_provided_pii_is_not_treated_as_unauthorized() -> None:
    email = "student@example.edu"
    result = scan_output(
        text=f"Your email is {email}.",
        user_input=f"Please repeat my email {email}",
        roberta_scanner=FakeScanner(),
    )

    assert result["piiScan"]["detected"] is True
    assert result["piiScan"]["unauthorizedCount"] == 0
    assert result["action"] == "ALLOW"


def test_model_failure_uses_safe_fallback() -> None:
    result = scan_output(text="Ordinary answer.", roberta_scanner=FakeScanner(available=False))

    assert result["action"] == "SAFE_FALLBACK"
    assert "OUTPUT_MODEL_UNAVAILABLE" in result["reasons"]

