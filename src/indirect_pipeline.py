"""End-to-end indirect prompt injection pipeline for untrusted external content."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

from src.external_content import build_external_chunks, extract_external_content
from src.indirect_context_detector import detect_context_manipulation
from src.indirect_rule_detector import detect_indirect_by_rules
from src.rule_based import detect_by_rules
from src.safe_context_builder import build_safe_context
from src.runtime_config import load_runtime_config
from src.security.preprocessing import preprocess_security_text
from src.roberta_runtime import roberta_service
from src.transformer_utils import (
    is_finetuned_transformer_checkpoint,
    predict_transformer,
    resolve_transformer_model_dir,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "indirect_detection.json"
ModelScorer = Callable[[str, str, bool], dict[str, Any]]


@dataclass(frozen=True)
class IndirectPipelineConfig:
    rule_weight: float = 0.35
    model_weight: float = 0.45
    context_weight: float = 0.20
    allow_threshold: float = 0.50
    block_threshold: float = 0.80
    chunk_max_chars: int = 1200
    chunk_overlap_chars: int = 160
    default_model: str = "roberta"
    safe_context_policy: str = "exclude"

    @property
    def weights(self) -> dict[str, float]:
        return {"rule": self.rule_weight, "model": self.model_weight, "context": self.context_weight}


def load_indirect_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> IndirectPipelineConfig:
    payload: dict[str, Any] = {}
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if config_path is None:
        runtime = load_runtime_config()
        document = dict(runtime.get("document_security") or {})
        chunk_weights = dict(document.get("chunk_weights") or {})
        selected_weights = {
            "rule_score": float(chunk_weights.get("rule", 0.35)),
            "model_score": float(chunk_weights.get("roberta", 0.45)),
            "context_score": float(chunk_weights.get("context", 0.20)),
        }
        payload = {
            "weights": selected_weights,
            "thresholds": {
                "sanitize_or_warn": document.get("quarantine_threshold", 0.50),
                "block": document.get("block_threshold", 0.80),
            },
            "chunking": {
                "max_chars": document.get("chunk_max_chars", 1200),
                "overlap_chars": document.get("chunk_overlap_chars", 160),
            },
            "default_model": (runtime.get("securityModel") or {}).get("modelName", "roberta_v4"),
            "safe_context_policy": document.get("safe_context_policy", "exclude"),
        }
    elif path.exists():
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if overrides:
        payload = {**payload, **overrides}

    weights = payload.get("weights", {})
    thresholds = payload.get("thresholds", {})
    chunking = payload.get("chunking", {})
    values = {
        "rule_weight": float(weights.get("rule_score", payload.get("rule_weight", 0.35))),
        "model_weight": float(weights.get("model_score", payload.get("model_weight", 0.45))),
        "context_weight": float(weights.get("context_score", payload.get("context_weight", 0.20))),
        "allow_threshold": float(thresholds.get("sanitize_or_warn", payload.get("allow_threshold", 0.50))),
        "block_threshold": float(thresholds.get("block", payload.get("block_threshold", 0.80))),
        "chunk_max_chars": int(chunking.get("max_chars", payload.get("chunk_max_chars", 1200))),
        "chunk_overlap_chars": int(chunking.get("overlap_chars", payload.get("chunk_overlap_chars", 160))),
        "default_model": str(payload.get("default_model", "roberta")),
        "safe_context_policy": str(payload.get("safe_context_policy", "exclude")),
    }
    weight_sum = values["rule_weight"] + values["model_weight"] + values["context_weight"]
    if weight_sum <= 0:
        raise ValueError("At least one ensemble weight must be positive.")
    if abs(weight_sum - 1.0) > 1e-9:
        values["rule_weight"] /= weight_sum
        values["model_weight"] /= weight_sum
        values["context_weight"] /= weight_sum
    if not 0 < values["allow_threshold"] < values["block_threshold"] <= 1:
        raise ValueError("Thresholds must satisfy 0 < sanitize_or_warn < block <= 1.")
    return IndirectPipelineConfig(**values)


def action_from_indirect_score(score: float, config: IndirectPipelineConfig) -> str:
    if score >= config.block_threshold:
        return "block"
    if score >= config.allow_threshold:
        return "sanitize_or_warn"
    return "allow"


def risk_level_from_score(score: float, config: IndirectPipelineConfig) -> str:
    if score >= config.block_threshold:
        return "high"
    if score >= config.allow_threshold:
        return "medium"
    return "low"


def _default_model_scorer(text: str, model_name: str, use_cuda: bool) -> dict[str, Any]:
    model_dir = resolve_transformer_model_dir(model_name)
    if not is_finetuned_transformer_checkpoint(model_dir):
        return {
            "available": False,
            "model": model_name,
            "model_path": str(model_dir),
            "model_score": None,
            "predicted_label": None,
            "error": "Model checkpoint not found or not fine-tuned.",
        }
    try:
        result = predict_transformer(
            text=text,
            model_path=model_dir,
            model_name=model_name,
            max_length=128,
            use_cuda=use_cuda,
        )
    except Exception as exc:
        return {
            "available": False,
            "model": model_name,
            "model_path": str(model_dir),
            "model_score": None,
            "predicted_label": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "available": True,
        "model": model_dir.name,
        "model_path": str(model_dir),
        "model_score": float(result["risk_score"]),
        "predicted_label": int(result["evaluation_label"]),
        "thresholds": result.get("thresholds", {}),
        "runtime_device": result.get("runtime_device"),
        "error": None,
    }


def _ensemble_score(
    rule_score: float,
    model_score: float | None,
    context_score: float,
    config: IndirectPipelineConfig,
) -> tuple[float, bool, dict[str, float]]:
    weights = config.weights
    if model_score is not None:
        score = weights["rule"] * rule_score + weights["model"] * model_score + weights["context"] * context_score
        return min(1.0, max(0.0, score)), False, weights

    available_weight = weights["rule"] + weights["context"]
    score = (weights["rule"] * rule_score + weights["context"] * context_score) / available_weight
    effective_weights = {
        "rule": weights["rule"] / available_weight,
        "model": 0.0,
        "context": weights["context"] / available_weight,
    }
    return min(1.0, max(0.0, score)), True, effective_weights


def detect_indirect_content(
    *,
    user_task: str,
    external_content: str | None = None,
    content_bytes: bytes | None = None,
    source_type: str = "raw_text",
    source_name: str = "inline-content",
    model_name: str | None = None,
    safe_context_policy: str | None = None,
    config_overrides: dict[str, Any] | None = None,
    use_cuda: bool = True,
    model_scorer: ModelScorer | None = None,
) -> dict[str, Any]:
    """Run extract -> clean -> chunk -> rule -> ML -> context -> ensemble -> policy."""
    if user_task is None or not str(user_task).strip():
        raise ValueError("user_task must not be empty.")
    config = load_indirect_config(overrides=config_overrides)
    selected_model = model_name or config.default_model
    metadata, segments = extract_external_content(
        source_type=source_type,
        source_name=source_name,
        raw_text=external_content,
        content_bytes=content_bytes,
    )
    chunks = build_external_chunks(
        metadata,
        segments,
        max_chars=config.chunk_max_chars,
        overlap_chars=config.chunk_overlap_chars,
    )
    if not chunks:
        raise ValueError("No chunks were produced from external content.")

    scorer = model_scorer or _default_model_scorer
    analyzed_chunks: list[dict[str, Any]] = []
    warnings: list[str] = []
    for chunk in chunks:
        preprocessing = preprocess_security_text(chunk.text)
        variants = [{
            "variant_id": "v0", "parent_variant_id": None, "transform": "original",
            "transform_chain": [], "depth": 0, "text": chunk.text,
            "confidence": 1.0, "text_hash": None,
        }]
        variants.extend(item for item in preprocessing.get("variants", []) if isinstance(item, dict))
        if model_scorer is None and selected_model == roberta_service.resolved_model_name and len(variants) > 1:
            batch_results = roberta_service.predict_many(
                [str(item.get("text", "")) for item in variants],
                use_cuda=use_cuda,
                stage="input",
            )
            ml_results = [
                {
                    "available": result.get("available", False),
                    "model": result.get("modelName"),
                    "model_path": result.get("modelPath"),
                    "model_score": result.get("score"),
                    "predicted_label": 1 if float(result.get("score") or 0.0) >= float((result.get("thresholdUsed") or {}).get("evaluation", 0.5)) else 0,
                    "thresholds": result.get("thresholdUsed", {}),
                    "runtime_device": "runtime_service",
                    "error": result.get("error"),
                }
                for result in batch_results
            ]
        else:
            ml_results = [scorer(str(item.get("text", "")), selected_model, use_cuda) for item in variants]

        variant_analyses: list[dict[str, Any]] = []
        for variant, variant_ml in zip(variants, ml_results):
            variant_text = str(variant.get("text", ""))
            variant_rule = detect_indirect_by_rules(str(user_task), variant_text)
            variant_context = detect_context_manipulation(str(user_task), variant_text)
            variant_model_score = variant_ml.get("model_score")
            numeric_variant_model = None if variant_model_score is None else float(variant_model_score)
            variant_final, variant_degraded, variant_weights = _ensemble_score(
                float(variant_rule["risk_score"]),
                numeric_variant_model,
                float(variant_context["context_score"]),
                config,
            )
            has_semantic_signal = bool(variant_rule.get("matched_rules")) or float(variant_context["context_score"]) >= config.allow_threshold
            if int(variant.get("depth", 0)) > 0 and not has_semantic_signal:
                variant_final = min(variant_final, 0.24)
            variant_analyses.append({
                "variant": variant,
                "rule": variant_rule,
                "ml": variant_ml,
                "context": variant_context,
                "model_score": numeric_variant_model,
                "final_score": variant_final,
                "degraded": variant_degraded,
                "effective_weights": variant_weights,
                "has_semantic_signal": has_semantic_signal,
            })

        transformed = [item for item in variant_analyses if int(item["variant"].get("depth", 0)) > 0]
        if transformed and not any(item["has_semantic_signal"] for item in transformed):
            variant_analyses[0]["final_score"] = min(variant_analyses[0]["final_score"], 0.24)
        selected_analysis = max(
            variant_analyses,
            key=lambda item: (item["final_score"], int(item["variant"].get("depth", 0))),
        )
        rule_result = selected_analysis["rule"]
        ml_result = selected_analysis["ml"]
        context_result = selected_analysis["context"]
        numeric_model_score = selected_analysis["model_score"]
        final_score = float(selected_analysis["final_score"])
        degraded = bool(selected_analysis["degraded"])
        effective_weights = selected_analysis["effective_weights"]
        action = action_from_indirect_score(final_score, config)
        if degraded:
            warning = f"ML detector unavailable for {chunk.chunk_id}; ensemble weights were renormalized."
            if warning not in warnings:
                warnings.append(warning)
        analyzed_chunks.append(
            {
                "metadata": {
                    "source_type": chunk.source_type,
                    "source_name": chunk.source_name,
                    "trust_level": "untrusted",
                    "chunk_id": chunk.chunk_id,
                    "page_number": chunk.page_number,
                },
                "text": chunk.text,
                "cleaned_text": chunk.cleaned_text,
                "rule_score": float(rule_result["risk_score"]),
                "model_score": numeric_model_score,
                "context_score": float(context_result["context_score"]),
                "final_score": round(final_score, 4),
                "predicted_label": ml_result.get("predicted_label"),
                "matched_rules": rule_result.get("matched_rules", []),
                "context_signals": context_result.get("matched_signals", []),
                "recommended_action": action,
                "ensemble_degraded": degraded,
                "effective_weights": effective_weights,
                "ml_result": ml_result,
                "variant_analysis": {
                    "selectedVariantId": selected_analysis["variant"].get("variant_id"),
                    "selectedTransform": selected_analysis["variant"].get("transform"),
                    "selectedTransformChain": selected_analysis["variant"].get("transform_chain", []),
                    "selectedDepth": selected_analysis["variant"].get("depth", 0),
                    "variantCount": preprocessing.get("variant_count", 0),
                    "detectedEncodings": preprocessing.get("detected_encodings", []),
                    "detectedObfuscations": preprocessing.get("detected_obfuscations", []),
                    "obfuscationScore": preprocessing.get("obfuscation_score", 0.0),
                    "variants": [
                        {
                            "variantId": item["variant"].get("variant_id"),
                            "parentVariantId": item["variant"].get("parent_variant_id"),
                            "transform": item["variant"].get("transform"),
                            "transformChain": item["variant"].get("transform_chain", []),
                            "depth": item["variant"].get("depth", 0),
                            "textHash": item["variant"].get("text_hash"),
                            "confidence": item["variant"].get("confidence", 1.0),
                            "ruleScore": item["rule"].get("risk_score", 0.0),
                            "modelScore": item["model_score"],
                            "contextScore": item["context"].get("context_score", 0.0),
                            "finalScore": round(float(item["final_score"]), 4),
                        }
                        for item in variant_analyses
                    ],
                },
            }
        )

    rule_score = max(float(chunk["rule_score"]) for chunk in analyzed_chunks)
    model_values = [float(chunk["model_score"]) for chunk in analyzed_chunks if chunk["model_score"] is not None]
    model_score = max(model_values) if model_values else None
    context_score = max(float(chunk["context_score"]) for chunk in analyzed_chunks)
    final_score, degraded, effective_weights = _ensemble_score(rule_score, model_score, context_score, config)
    recommended_action = action_from_indirect_score(final_score, config)
    direct_rule_result = detect_by_rules(str(user_task))
    direct_signal = float(direct_rule_result.get("risk_score", 0.0)) >= config.allow_threshold
    external_signal = recommended_action != "allow"
    attack_type = "indirect" if external_signal else ("direct" if direct_signal else "unknown")

    matched_rules: list[dict[str, Any]] = []
    for chunk in analyzed_chunks:
        for rule in chunk["matched_rules"]:
            matched_rules.append({"chunk_id": chunk["metadata"]["chunk_id"], **rule})

    safe_context = build_safe_context(
        analyzed_chunks,
        policy=safe_context_policy or config.safe_context_policy,
    )
    if recommended_action == "block":
        explanation = "High-risk assistant-directed instructions were detected in untrusted external content; block this content from the LLM/RAG context."
    elif recommended_action == "sanitize_or_warn":
        explanation = "Suspicious instructions were detected in untrusted external content; sanitize, exclude, or quote affected chunks before use."
    else:
        explanation = "No indirect prompt injection signal reached the warning threshold; content remains explicitly untrusted data."
    if degraded:
        explanation += " The ML signal was unavailable, so the ensemble used renormalized rule/context weights."

    return {
        "is_injection": bool(recommended_action != "allow" or direct_signal),
        "attack_type": attack_type,
        "risk_level": risk_level_from_score(final_score, config),
        "final_score": round(final_score, 4),
        "rule_score": round(rule_score, 4),
        "model_score": None if model_score is None else round(model_score, 4),
        "context_score": round(context_score, 4),
        "matched_rules": matched_rules,
        "source_metadata": {
            "source_type": metadata.source_type,
            "source_name": metadata.source_name,
            "trust_level": "untrusted",
            "chunk_count": len(analyzed_chunks),
        },
        "recommended_action": recommended_action,
        "explanation": explanation,
        "ensemble": {
            "configured_weights": config.weights,
            "effective_weights": effective_weights,
            "formula": (
                f"final_score = {config.rule_weight:.2f} * rule_score + "
                f"{config.model_weight:.2f} * model_score + {config.context_weight:.2f} * context_score"
            ),
            "degraded": degraded,
            "thresholds": {
                "allow": f"score < {config.allow_threshold:.2f}",
                "sanitize_or_warn": f"{config.allow_threshold:.2f} <= score < {config.block_threshold:.2f}",
                "block": f"score >= {config.block_threshold:.2f}",
            },
        },
        "model": selected_model,
        "chunks": analyzed_chunks,
        "safe_context": safe_context,
        "warnings": warnings,
        "direct_rule_result": direct_rule_result,
    }
