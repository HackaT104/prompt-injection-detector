"""Adaptive weighted risk fusion for runtime prompt-injection detection.

This module intentionally uses only three runtime signals:

1. rule-based detector score
2. calibrated RoBERTa score
3. calibrated XLM-RoBERTa score

Classical ML models remain useful for research/evaluation baselines, but they
are not part of the runtime hybrid fusion implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SEVERITY_RANK: dict[str, int] = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}


def _clamp_probability(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def _normalize_language(language: str | None) -> str:
    normalized = (language or "unknown").strip().lower()
    if normalized in {"en", "eng", "english"}:
        return "en"
    if normalized in {"vi", "vie", "vietnamese", "vn"}:
        return "vi"
    if normalized in {"mixed", "multi", "multilingual"}:
        return "mixed"
    return "unknown"


def _normalize_source_type(source_type: str | None) -> str:
    normalized = (source_type or "user_prompt").strip().lower()
    aliases = {
        "raw_text": "user_prompt",
        "prompt": "user_prompt",
        "user": "user_prompt",
        "external": "external_content",
        "content": "external_content",
        "retrieval": "rag",
    }
    return aliases.get(normalized, normalized)


@dataclass(slots=True)
class AdaptiveRiskFusion:
    """Fuse model and rule signals with language/source-aware weights."""

    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        defaults = {
            "language_model_weights": {
                "en": {"roberta": 0.75, "xlm_roberta": 0.25},
                "vi": {"roberta": 0.30, "xlm_roberta": 0.70},
                "mixed": {"roberta": 0.50, "xlm_roberta": 0.50},
                "unknown": {"roberta": 0.60, "xlm_roberta": 0.40},
            },
            "source_signal_weights": {
                "user_prompt": {"model": 0.85, "rule": 0.15},
                "external_content": {"model": 0.70, "rule": 0.30},
                "email": {"model": 0.70, "rule": 0.30},
                "web": {"model": 0.70, "rule": 0.30},
                "pdf": {"model": 0.70, "rule": 0.30},
                "rag": {"model": 0.70, "rule": 0.30},
                "tool": {"model": 0.60, "rule": 0.40},
            },
            "high_severity_signal_weights": {"model": 0.55, "rule": 0.45},
        }
        merged = {**defaults, **self.config}
        merged["language_model_weights"] = {
            **defaults["language_model_weights"],
            **self.config.get("language_model_weights", {}),
        }
        merged["source_signal_weights"] = {
            **defaults["source_signal_weights"],
            **self.config.get("source_signal_weights", {}),
        }
        self.config = merged

    def fuse(
        self,
        *,
        text: str | None = None,
        language: str | None = None,
        source_type: str | None = "user_prompt",
        rule_score: float = 0.0,
        rule_matches: list[dict[str, Any]] | None = None,
        roberta_score: float = 0.0,
        xlm_score: float = 0.0,
        scores_are_calibrated: bool = True,
        highest_severity: str | None = None,
        has_high_severity_rule: bool | None = None,
        has_critical_rule: bool | None = None,
    ) -> dict[str, Any]:
        """Return explainable adaptive fusion output."""
        normalized_language = _normalize_language(language)
        normalized_source = _normalize_source_type(source_type)
        rule_matches = rule_matches or []
        reasons: list[str] = []

        severity = (highest_severity or "none").strip().lower()
        severity_rank = SEVERITY_RANK.get(severity, 0)
        if has_high_severity_rule is None:
            has_high_severity_rule = severity_rank >= SEVERITY_RANK["high"]
        if has_critical_rule is None:
            has_critical_rule = severity_rank >= SEVERITY_RANK["critical"]

        model_weight_config = self.config["language_model_weights"].get(
            normalized_language,
            self.config["language_model_weights"]["unknown"],
        )
        w_roberta = float(model_weight_config["roberta"])
        w_xlm = float(model_weight_config["xlm_roberta"])
        model_weight_total = max(w_roberta + w_xlm, 1e-9)
        w_roberta = w_roberta / model_weight_total
        w_xlm = w_xlm / model_weight_total
        reasons.append(
            f"Ngôn ngữ '{normalized_language}' dùng trọng số model RoBERTa={w_roberta:.2f}, XLM-R={w_xlm:.2f}."
        )

        signal_weight_config = self.config["source_signal_weights"].get(
            normalized_source,
            self.config["source_signal_weights"]["user_prompt"],
        )
        w_model = float(signal_weight_config["model"])
        w_rule = float(signal_weight_config["rule"])
        fusion_method = f"adaptive_weighted:{normalized_source}"
        reasons.append(
            f"Nguồn '{normalized_source}' dùng trọng số signal model={w_model:.2f}, rule={w_rule:.2f}."
        )

        if has_high_severity_rule:
            severity_weights = self.config["high_severity_signal_weights"]
            w_model = float(severity_weights["model"])
            w_rule = float(severity_weights["rule"])
            fusion_method = "adaptive_weighted:high_severity_rule"
            reasons.append(
                f"Rule severity '{severity}' kích hoạt tăng trọng số rule: model={w_model:.2f}, rule={w_rule:.2f}."
            )

        signal_weight_total = max(w_model + w_rule, 1e-9)
        w_model = w_model / signal_weight_total
        w_rule = w_rule / signal_weight_total

        if not scores_are_calibrated:
            reasons.append(
                "Cảnh báo: không phải tất cả Transformer score đều là calibrated probability; cần kiểm tra calibrator/checkpoint."
            )

        roberta = _clamp_probability(roberta_score)
        xlm = _clamp_probability(xlm_score)
        rule = _clamp_probability(rule_score)
        model_risk = (w_roberta * roberta) + (w_xlm * xlm)
        final_risk = (w_model * model_risk) + (w_rule * rule)

        return {
            "model_risk": round(float(model_risk), 6),
            "rule_risk": round(float(rule), 6),
            "final_risk": round(float(final_risk), 6),
            "weights": {
                "model": round(float(w_model), 6),
                "rule": round(float(w_rule), 6),
                "roberta": round(float(w_roberta), 6),
                "xlm_roberta": round(float(w_xlm), 6),
            },
            "fusion_method": fusion_method,
            "language": normalized_language,
            "source_type": normalized_source,
            "highest_severity": severity,
            "has_high_severity_rule": bool(has_high_severity_rule),
            "has_critical_rule": bool(has_critical_rule),
            "scores_are_calibrated": bool(scores_are_calibrated),
            "reasons": reasons,
        }
