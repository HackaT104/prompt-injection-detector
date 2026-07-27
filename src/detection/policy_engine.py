"""Decision policy engine for adaptive hybrid prompt-injection detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


EXTERNAL_OR_TOOL_SOURCES = {
    "external_content",
    "external",
    "email",
    "web",
    "pdf",
    "rag",
    "tool",
}

SEVERITY_RANK = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}


def _score(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


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
class DecisionPolicyEngine:
    """Convert fused risk and rule severity into a runtime decision."""

    config: dict[str, Any] = field(default_factory=dict)

    def decide(
        self,
        *,
        final_risk: float,
        model_risk: float,
        rule_score: float,
        roberta_score: float,
        xlm_score: float,
        highest_severity: str | None = "none",
        has_high_severity_rule: bool = False,
        has_critical_rule: bool = False,
        source_type: str | None = "user_prompt",
        language: str | None = None,
        rule_matches: list[dict[str, Any]] | None = None,
        weights: dict[str, Any] | None = None,
        fusion_method: str | None = None,
        scores_are_calibrated: bool = True,
        benign_reference_intent: bool = False,
    ) -> dict[str, Any]:
        """Apply ordered decision rules and return explainable output."""
        final = _score(final_risk)
        model = _score(model_risk)
        rule = _score(rule_score)
        roberta = _score(roberta_score)
        xlm = _score(xlm_score)
        severity = (highest_severity or "none").strip().lower()
        source = _normalize_source_type(source_type)
        rule_matches = rule_matches or []

        reasons: list[str] = []
        decision_policy = "threshold"
        decision = "SAFE"
        risk_level = "safe"

        if has_critical_rule and source in EXTERNAL_OR_TOOL_SOURCES:
            decision = "BLOCK"
            risk_level = "critical"
            decision_policy = "critical_rule_external_or_tool_override"
            reasons.append(
                f"Rule severity critical trong nguồn '{source}' được block để ngăn exfiltration/tool abuse."
            )
        elif has_high_severity_rule and model >= 0.50:
            decision = "BLOCK"
            risk_level = "high"
            decision_policy = "high_rule_plus_model_risk"
            reasons.append(
                f"Rule severity '{severity}' kết hợp model_risk={model:.3f} >= 0.50."
            )
        elif benign_reference_intent:
            if final >= 0.30 or model >= 0.30 or rule >= 0.30:
                decision = "WARN"
                risk_level = "medium"
                decision_policy = "benign_reference_warn_cap"
                reasons.append(
                    "Benign reference guard phát hiện ngữ cảnh học thuật/trích dẫn; không block nếu không có rule high/critical."
                )
            else:
                decision = "SAFE"
                risk_level = "safe"
                decision_policy = "benign_reference_safe"
                reasons.append(
                    "Benign reference guard phát hiện ngữ cảnh học thuật/trích dẫn và risk thấp."
                )
        elif roberta >= 0.90 and xlm >= 0.80:
            decision = "BLOCK"
            risk_level = "high"
            decision_policy = "dual_transformer_high_confidence"
            reasons.append(
                f"Cả RoBERTa ({roberta:.3f}) và XLM-R ({xlm:.3f}) đều rất cao."
            )
        elif rule >= 0.90 and model < 0.30:
            decision = "WARN"
            risk_level = "medium"
            decision_policy = "rule_high_model_low_warn_only"
            reasons.append(
                "Rule score rất cao nhưng model_risk thấp; xem như keyword/reference nghi vấn, cảnh báo thay vì block."
            )
        elif final >= 0.70:
            decision = "BLOCK"
            risk_level = "high"
            decision_policy = "final_risk_block_threshold"
            reasons.append(f"final_risk={final:.3f} >= 0.70.")
        elif final >= 0.30:
            decision = "WARN"
            risk_level = "medium"
            decision_policy = "final_risk_warn_threshold"
            reasons.append(f"final_risk={final:.3f} >= 0.30.")
        else:
            decision = "SAFE"
            risk_level = "safe"
            decision_policy = "final_risk_safe_threshold"
            reasons.append(f"final_risk={final:.3f} < 0.30.")

        if not scores_are_calibrated:
            reasons.append(
                "Cảnh báo vận hành: score Transformer chưa/không được calibration đầy đủ."
            )

        recommendation = self._recommendation(decision, decision_policy)
        return {
            "risk_level": risk_level,
            "decision": decision,
            "decision_policy": decision_policy,
            "reasons": reasons,
            "recommendation": recommendation,
            "policy_inputs": {
                "final_risk": round(final, 6),
                "model_risk": round(model, 6),
                "rule_score": round(rule, 6),
                "roberta_score": round(roberta, 6),
                "xlm_score": round(xlm, 6),
                "highest_severity": severity,
                "has_high_severity_rule": bool(has_high_severity_rule),
                "has_critical_rule": bool(has_critical_rule),
                "source_type": source,
                "language": language,
                "weights": weights or {},
                "fusion_method": fusion_method,
                "matched_rule_count": len(rule_matches),
                "benign_reference_intent": bool(benign_reference_intent),
            },
        }

    @staticmethod
    def _recommendation(decision: str, decision_policy: str) -> str:
        if decision == "BLOCK":
            if "critical_rule" in decision_policy:
                return "Chặn nội dung và yêu cầu kiểm tra thủ công vì có dấu hiệu exfiltration/tool abuse nghiêm trọng."
            return "Chặn hoặc yêu cầu human review trước khi đưa nội dung vào LLM/tool."
        if decision == "WARN":
            return "Cảnh báo người dùng, log lại lý do, và chỉ tiếp tục nếu workflow cho phép rủi ro trung bình."
        return "Cho phép xử lý tiếp; vẫn nên log score để phục vụ monitoring/calibration định kỳ."
