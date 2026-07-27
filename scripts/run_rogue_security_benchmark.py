"""Evaluate the runtime on rogue-security/prompt-injections-benchmark.

The dataset is an external holdout. This script never trains, calibrates, tunes
thresholds, changes production checkpoints, or calls the downstream LLM.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.official_runtime as official_runtime
from src.runtime_config import load_runtime_config
from src.runtime_rule_signal import detect_rule_signal
from src.transformer_utils import predict_transformer, resolve_transformer_model_dir


DATASET_NAME = "rogue-security/prompt-injections-benchmark"
DATASET_URL = "https://huggingface.co/datasets/rogue-security/prompt-injections-benchmark"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "rogue_security"
PINT_RESULTS_PATH = PROJECT_ROOT / "reports" / "pint_compatible_results.csv"
MAX_PREVIEW_CHARS = 220
INFERENCE_MAX_LENGTH = 128
MODE_LABELS = {
    "roberta_only": "RoBERTa-only",
    "rule_roberta": "Rule + RoBERTa",
    "full_runtime": "Full runtime",
}
RESULT_FIELDNAMES = [
    "dataset_name",
    "split",
    "sample_id",
    "text_preview",
    "expected_label",
    "expected_label_name",
    "mode",
    "mode_label",
    "predicted_label",
    "predicted_label_name",
    "correct",
    "raw_roberta_score",
    "calibrated_score",
    "intent_adjusted_score",
    "score_used_by_policy",
    "rule_score",
    "matched_rule_ids",
    "highest_rule_severity",
    "context_score",
    "fusion_score",
    "final_decision",
    "decision_source",
    "triggered_policy",
    "threshold",
    "threshold_source",
    "intent_category",
    "intent_guard_applied",
    "intent_guard_reason",
    "token_count",
    "original_token_count",
    "was_truncated",
    "internal_duplicate",
    "leakage_duplicate",
    "latency_ms",
    "model_version",
    "checkpoint_path",
    "error",
]


@dataclass(frozen=True)
class BenchmarkSample:
    split: str
    index: int
    sample_id: str
    text: str
    expected_label: int
    expected_label_name: str
    token_count: int = 0
    was_truncated: bool = False
    text_sha256: str = ""
    normalized_sha256: str = ""
    internal_duplicate: bool = False
    leakage_duplicate: bool = False


@dataclass
class ModeResult:
    predicted_label: int
    raw_roberta_score: float | None
    calibrated_score: float | None
    intent_adjusted_score: float | None
    score_used_by_policy: str
    rule_score: float | None
    matched_rule_ids: list[str]
    highest_rule_severity: str
    context_score: float | None
    fusion_score: float | None
    final_decision: str
    decision_source: str
    triggered_policy: str
    threshold: float | None
    threshold_source: str
    intent_category: str
    intent_guard_applied: bool
    intent_guard_reason: str
    latency_ms: float
    model_version: str
    checkpoint_path: str
    error: str = ""


def _score(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text.strip().lower())
    return text


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()


def safe_preview(text: Any, max_chars: int = MAX_PREVIEW_CHARS) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    value = value.replace("\r", " ").replace("\n", " ")
    if len(value) > max_chars:
        value = value[: max_chars - 3].rstrip() + "..."
    return value


def normalize_label(value: Any) -> tuple[int, str]:
    if isinstance(value, str):
        label = value.strip().lower()
        if label == "benign":
            return 0, "benign"
        if label == "jailbreak":
            return 1, "jailbreak"
    raise ValueError(f"Unsupported label value: {value!r}. Expected 'benign' or 'jailbreak'.")


def normalize_hf_record(record: dict[str, Any], *, split: str, index: int) -> BenchmarkSample:
    if not isinstance(record, dict):
        raise ValueError(f"Record {split}:{index} must be an object.")
    text = record.get("text")
    if text is None or not str(text).strip():
        raise ValueError(f"Record {split}:{index} has null/empty text.")
    expected_label, label_name = normalize_label(record.get("label"))
    raw_text = str(text)
    normalized = normalize_text(raw_text)
    return BenchmarkSample(
        split=str(split),
        index=index,
        sample_id=f"{split}:{index}",
        text=raw_text,
        expected_label=expected_label,
        expected_label_name=label_name,
        text_sha256=hash_text(raw_text),
        normalized_sha256=hash_text(normalized),
    )


def _tokenizer_for_profile() -> Any | None:
    try:
        from transformers import AutoTokenizer

        model_dir = resolve_transformer_model_dir("roberta")
        return AutoTokenizer.from_pretrained(model_dir)
    except Exception:
        return None


def annotate_token_counts(samples: list[BenchmarkSample], *, max_length: int = INFERENCE_MAX_LENGTH) -> list[BenchmarkSample]:
    tokenizer = _tokenizer_for_profile()
    annotated: list[BenchmarkSample] = []
    for sample in samples:
        if tokenizer is None:
            token_count = max(1, len(normalize_text(sample.text).split()))
        else:
            token_count = len(tokenizer.encode(sample.text, add_special_tokens=True, truncation=False))
        annotated.append(
            BenchmarkSample(
                **{
                    **sample.__dict__,
                    "token_count": int(token_count),
                    "was_truncated": int(token_count) > int(max_length),
                }
            )
        )
    return annotated


def load_rogue_security_dataset(max_samples: int | None = None) -> tuple[list[BenchmarkSample], dict[str, Any]]:
    from datasets import load_dataset

    dataset_dict = load_dataset(DATASET_NAME)
    splits = list(dataset_dict.keys())
    samples: list[BenchmarkSample] = []
    schema: dict[str, Any] = {"splits": {}, "features": {}}
    for split in splits:
        dataset = dataset_dict[split]
        schema["splits"][split] = len(dataset)
        schema["features"][split] = list(dataset.column_names)
        for index, record in enumerate(dataset):
            samples.append(normalize_hf_record(dict(record), split=split, index=index))
            if max_samples is not None and len(samples) >= max_samples:
                break
        if max_samples is not None and len(samples) >= max_samples:
            break

    normalized_counts = Counter(sample.normalized_sha256 for sample in samples)
    exact_counts = Counter(sample.text_sha256 for sample in samples)
    marked: list[BenchmarkSample] = []
    for sample in samples:
        marked.append(
            BenchmarkSample(
                **{
                    **sample.__dict__,
                    "internal_duplicate": exact_counts[sample.text_sha256] > 1
                    or normalized_counts[sample.normalized_sha256] > 1,
                }
            )
        )
    return annotate_token_counts(marked), schema


def _maybe_tqdm(iterable: Iterable[Any], *, total: int | None, desc: str) -> Iterable[Any]:
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        return iterable


def device_to_use_cuda(device: str) -> bool:
    normalized = str(device or "auto").lower()
    if normalized == "cpu":
        return False
    if normalized == "cuda":
        return True
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _threshold_from_transformer(result: dict[str, Any]) -> tuple[float, str]:
    threshold_used = result.get("threshold_used")
    if isinstance(threshold_used, dict):
        return _score(threshold_used.get("warn", threshold_used.get("evaluation", 0.5))), str(
            result.get("threshold_source") or "unknown"
        )
    thresholds = result.get("thresholds")
    if isinstance(thresholds, dict):
        return _score(thresholds.get("runtime_warn_threshold", thresholds.get("evaluation_threshold", 0.5))), str(
            result.get("threshold_source") or "unknown"
        )
    return 0.5, str(result.get("threshold_source") or "default")


def evaluate_roberta_only(text: str, *, use_cuda: bool) -> ModeResult:
    started = time.perf_counter()
    model_dir = resolve_transformer_model_dir("roberta")
    result = predict_transformer(
        text=text,
        model_path=model_dir,
        model_name="roberta",
        max_length=INFERENCE_MAX_LENGTH,
        use_cuda=use_cuda,
        use_intent_guard=False,
        use_runtime_calibration=False,
    )
    latency_ms = round((time.perf_counter() - started) * 1000, 3)
    raw_score = _score(result.get("raw_score", result.get("raw_risk_score", result.get("risk_score"))))
    threshold, threshold_source = _threshold_from_transformer(result)
    predicted = int(raw_score >= threshold)
    return ModeResult(
        predicted_label=predicted,
        raw_roberta_score=raw_score,
        calibrated_score=None,
        intent_adjusted_score=None,
        score_used_by_policy="raw_softmax_probability",
        rule_score=None,
        matched_rule_ids=[],
        highest_rule_severity="none",
        context_score=None,
        fusion_score=None,
        final_decision="warning" if predicted else "safe",
        decision_source="roberta_raw" if predicted else "none",
        triggered_policy="roberta_threshold" if predicted else "none",
        threshold=threshold,
        threshold_source=threshold_source,
        intent_category="",
        intent_guard_applied=False,
        intent_guard_reason="",
        latency_ms=latency_ms,
        model_version=str(result.get("model") or "roberta"),
        checkpoint_path=str(model_dir),
    )


def evaluate_rule_roberta(text: str, *, use_cuda: bool) -> ModeResult:
    started = time.perf_counter()
    rule_signal = detect_rule_signal(text, source_type="user_prompt")
    roberta = evaluate_roberta_only(text, use_cuda=use_cuda)
    matched_rules = [
        str(item.get("code") or item.get("group") or "RULE_MATCH")
        for item in (rule_signal.get("matchedRules") or [])
        if isinstance(item, dict)
    ]
    rule_action = str(rule_signal.get("action", "allow")).lower()
    highest_severity = str(rule_signal.get("highestSeverity") or "none")
    rule_positive = bool(rule_signal.get("hardBlock")) or rule_action == "block"
    roberta_positive = bool(roberta.predicted_label)
    predicted = int(rule_positive or roberta_positive)
    if rule_positive and roberta_positive:
        decision_source = "rule_roberta"
    elif rule_positive:
        decision_source = "rule"
    elif roberta_positive:
        decision_source = "roberta_raw"
    else:
        decision_source = "none"
    triggered = []
    if rule_positive:
        triggered.append("rule_block_signal")
    if roberta_positive:
        triggered.append("roberta_threshold")
    return ModeResult(
        predicted_label=predicted,
        raw_roberta_score=roberta.raw_roberta_score,
        calibrated_score=None,
        intent_adjusted_score=None,
        score_used_by_policy="rule_block_or_roberta_raw",
        rule_score=_score(rule_signal.get("score")),
        matched_rule_ids=matched_rules,
        highest_rule_severity=highest_severity,
        context_score=None,
        fusion_score=None,
        final_decision="blocked" if rule_signal.get("hardBlock") else ("warning" if predicted else "safe"),
        decision_source=decision_source,
        triggered_policy=";".join(triggered) if triggered else "none",
        threshold=roberta.threshold,
        threshold_source=roberta.threshold_source,
        intent_category="",
        intent_guard_applied=False,
        intent_guard_reason="",
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
        model_version=roberta.model_version,
        checkpoint_path=roberta.checkpoint_path,
    )


def _benchmark_llm_skip(*_: Any, **__: Any) -> dict[str, Any]:
    return {
        "called": False,
        "status": "skipped",
        "provider": "benchmark-no-llm",
        "model": "",
        "latencyMs": 0.0,
        "tokenUsage": {"input": 0, "output": 0, "total": 0},
        "estimatedCost": 0.0,
        "content": "",
    }


def evaluate_full_runtime(text: str, *, use_cuda: bool) -> ModeResult:
    started = time.perf_counter()
    original_call_llm = official_runtime.call_llm
    official_runtime.call_llm = _benchmark_llm_skip
    try:
        result = official_runtime.run_official_runtime(
            message=text,
            user_id="rogue-security-benchmark",
            use_cuda=use_cuda,
            request_id="rogue_security_benchmark",
        )
    finally:
        official_runtime.call_llm = original_call_llm

    details = result.get("details", {}) if isinstance(result.get("details"), dict) else {}
    model_scores = result.get("modelScores", {}) if isinstance(result.get("modelScores"), dict) else {}
    roberta = model_scores.get("roberta", {}) if isinstance(model_scores.get("roberta"), dict) else {}
    rule_based = model_scores.get("ruleBased", {}) if isinstance(model_scores.get("ruleBased"), dict) else {}
    context = model_scores.get("contextAware", {}) if isinstance(model_scores.get("contextAware"), dict) else {}
    policy = result.get("policyResult", {}) if isinstance(result.get("policyResult"), dict) else {}
    decision = str(result.get("decision", "safe")).lower()
    matched_rules = [
        str(item.get("code") or item.get("group") or "RULE_MATCH")
        for item in (rule_based.get("matchedRules") or [])
        if isinstance(item, dict)
    ]
    runtime_intent = roberta.get("runtimeBenignIntent") if isinstance(roberta.get("runtimeBenignIntent"), dict) else {}
    raw_score = roberta.get("rawScore", details.get("robertaRawScore"))
    intent_adjusted = roberta.get("intentAdjustedScore")
    calibrated = roberta.get("calibratedScore")
    return ModeResult(
        predicted_label=int(decision in {"warning", "blocked"}),
        raw_roberta_score=None if raw_score is None else _score(raw_score),
        calibrated_score=None if calibrated is None else _score(calibrated),
        intent_adjusted_score=None if intent_adjusted is None else _score(intent_adjusted),
        score_used_by_policy=str(roberta.get("scoreUsed") or details.get("robertaScoreUsed") or ""),
        rule_score=None if rule_based.get("score") is None else _score(rule_based.get("score")),
        matched_rule_ids=matched_rules,
        highest_rule_severity=str(rule_based.get("highestSeverity") or "none"),
        context_score=None if details.get("contextAwareScore") is None else _score(details.get("contextAwareScore")),
        fusion_score=None if details.get("fusionScore") is None else _score(details.get("fusionScore")),
        final_decision=decision,
        decision_source=str(details.get("highestRiskSource") or "policy"),
        triggered_policy=";".join(policy.get("reasonCodes", []) or result.get("reasons", []) or []),
        threshold=None if details.get("threshold") is None else _score(details.get("threshold")),
        threshold_source=str((roberta.get("thresholdUsed") or {}) if isinstance(roberta.get("thresholdUsed"), dict) else ""),
        intent_category=str(runtime_intent.get("category") or details.get("robertaIntentCategory") or ""),
        intent_guard_applied=bool(runtime_intent.get("triggered")),
        intent_guard_reason=str(runtime_intent.get("reason") or runtime_intent.get("category") or ""),
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
        model_version=str(roberta.get("modelVersion") or details.get("modelVersion") or ""),
        checkpoint_path=str(resolve_transformer_model_dir("roberta")),
    )


def safe_evaluate(mode: str, text: str, *, use_cuda: bool) -> ModeResult:
    try:
        if mode == "roberta_only":
            return evaluate_roberta_only(text, use_cuda=use_cuda)
        if mode == "rule_roberta":
            return evaluate_rule_roberta(text, use_cuda=use_cuda)
        if mode == "full_runtime":
            return evaluate_full_runtime(text, use_cuda=use_cuda)
        raise ValueError(f"Unsupported mode: {mode}")
    except Exception as exc:
        return ModeResult(
            predicted_label=0,
            raw_roberta_score=None,
            calibrated_score=None,
            intent_adjusted_score=None,
            score_used_by_policy="error",
            rule_score=None,
            matched_rule_ids=[],
            highest_rule_severity="none",
            context_score=None,
            fusion_score=None,
            final_decision="error",
            decision_source=f"error:{exc.__class__.__name__}",
            triggered_policy="error",
            threshold=None,
            threshold_source="",
            intent_category="",
            intent_guard_applied=False,
            intent_guard_reason="",
            latency_ms=0.0,
            model_version="",
            checkpoint_path="",
            error=str(exc),
        )


def row_from_result(sample: BenchmarkSample, mode: str, result: ModeResult) -> dict[str, Any]:
    return {
        "dataset_name": DATASET_NAME,
        "split": sample.split,
        "sample_id": sample.sample_id,
        "text_preview": safe_preview(sample.text),
        "expected_label": sample.expected_label,
        "expected_label_name": sample.expected_label_name,
        "mode": mode,
        "mode_label": MODE_LABELS.get(mode, mode),
        "predicted_label": result.predicted_label,
        "predicted_label_name": "jailbreak" if int(result.predicted_label) == 1 else "benign",
        "correct": int(result.predicted_label == sample.expected_label),
        "raw_roberta_score": "" if result.raw_roberta_score is None else round(float(result.raw_roberta_score), 8),
        "calibrated_score": "" if result.calibrated_score is None else round(float(result.calibrated_score), 8),
        "intent_adjusted_score": ""
        if result.intent_adjusted_score is None
        else round(float(result.intent_adjusted_score), 8),
        "score_used_by_policy": result.score_used_by_policy,
        "rule_score": "" if result.rule_score is None else round(float(result.rule_score), 8),
        "matched_rule_ids": ";".join(result.matched_rule_ids),
        "highest_rule_severity": result.highest_rule_severity,
        "context_score": "" if result.context_score is None else round(float(result.context_score), 8),
        "fusion_score": "" if result.fusion_score is None else round(float(result.fusion_score), 8),
        "final_decision": result.final_decision,
        "decision_source": result.decision_source,
        "triggered_policy": result.triggered_policy,
        "threshold": "" if result.threshold is None else round(float(result.threshold), 8),
        "threshold_source": result.threshold_source,
        "intent_category": result.intent_category,
        "intent_guard_applied": int(bool(result.intent_guard_applied)),
        "intent_guard_reason": result.intent_guard_reason,
        "token_count": sample.token_count,
        "original_token_count": sample.token_count,
        "was_truncated": int(bool(sample.was_truncated)),
        "latency_ms": result.latency_ms,
        "model_version": result.model_version,
        "checkpoint_path": result.checkpoint_path,
        "error": result.error,
    }


def load_existing_keys(results_path: Path) -> set[tuple[str, str]]:
    if not results_path.exists():
        return set()
    with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return {(row.get("mode", ""), row.get("sample_id", "")) for row in reader}


def append_result_rows(results_path: Path, rows: list[dict[str, Any]]) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    exists = results_path.exists() and results_path.stat().st_size > 0
    with results_path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDNAMES, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()


def read_results_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _float_or_none(value: Any) -> float | None:
    try:
        if value == "" or value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_binary_metrics(rows: list[dict[str, Any]], *, score_field: str = "raw_roberta_score") -> dict[str, Any]:
    total = len(rows)
    y_true = [_int(row["expected_label"]) for row in rows]
    y_pred = [_int(row["predicted_label"]) for row in rows]
    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)
    tn = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 0)
    fp = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
    fn = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    mcc_denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / mcc_denom if mcc_denom else 0.0
    balanced = (recall + specificity) / 2 if total else 0.0
    auc = None
    pr_auc = None
    scores = [_float_or_none(row.get(score_field)) for row in rows]
    if total and len(set(y_true)) == 2 and all(score is not None for score in scores):
        score_values = [float(score) for score in scores if score is not None]
        auc = float(roc_auc_score(y_true, score_values))
        pr_auc = float(average_precision_score(y_true, score_values))
    return {
        "total": total,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "negative_predictive_value": npv,
        "matthews_correlation_coefficient": mcc,
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * (percent / 100)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[int(index)])
    return float(ordered[lower] * (upper - index) + ordered[upper] * (index - lower))


def format_percent(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_Không có dữ liệu._"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def length_bin(token_count: int) -> str:
    if token_count <= 32:
        return "0-32"
    if token_count <= 64:
        return "33-64"
    if token_count <= 128:
        return "65-128"
    if token_count <= 256:
        return "129-256"
    if token_count <= 512:
        return "257-512"
    return ">512"


def classify_false_positive(row: dict[str, Any]) -> str:
    text = str(row.get("text_preview", "")).lower()
    if any(term in text for term in ["roleplay", "pretend", "act as"]):
        return "Harmless roleplay"
    if any(term in text for term in ["story", "fiction", "novel", "character"]):
        return "Fiction/story"
    if any(term in text for term in ["cybersecurity", "security", "vulnerability", "ctf"]):
        return "Cybersecurity discussion"
    if any(term in text for term in ["quote", "'", '"']):
        return "Quotation"
    if any(term in text for term in ["translate", "dịch"]):
        return "Translation"
    if any(term in text for term in ["research", "paper", "study", "system prompt"]):
        return "Research context"
    if any(term in text for term in ["policy", "safety"]):
        return "Safety policy discussion"
    if _int(row.get("token_count")) > 128:
        return "Long benign instruction"
    if any(term in text for term in ["ignore", "instruction", "developer", "system"]):
        return "Sensitive keyword without attack intent"
    return "Other"


def classify_false_negative(row: dict[str, Any]) -> str:
    text = str(row.get("text_preview", "")).lower()
    if any(term in text for term in ["ignore previous", "disregard", "forget"]):
        return "Direct override"
    if any(term in text for term in ["system prompt", "hidden instruction", "developer message"]):
        return "System prompt extraction"
    if any(term in text for term in ["you are now", "new role", "act as"]):
        return "Role manipulation"
    if any(term in text for term in ["dan", "developer mode", "jailbreak"]):
        return "Jailbreak persona"
    if any(term in text for term in ["bypass", "policy", "safety"]):
        return "Policy bypass"
    if any(term in text for term in ["base64", "rot13", "decode", "encoded"]):
        return "Encoding/obfuscation"
    if any(term in text for term in ["first", "then", "step"]):
        return "Multi-step attack"
    if any(term in text for term in ["explain why", "quote"]):
        return "Indirectly phrased extraction"
    if _int(row.get("token_count")) > 128:
        return "Long-context attack"
    return "Other"


def profile_samples(samples: list[BenchmarkSample], schema: dict[str, Any], *, output_path: Path) -> None:
    label_counts = Counter(sample.expected_label_name for sample in samples)
    char_lengths = [len(sample.text) for sample in samples]
    token_lengths = [sample.token_count for sample in samples]
    empty_count = sum(1 for sample in samples if not sample.text.strip())
    exact_duplicates = sum(count - 1 for count in Counter(sample.text_sha256 for sample in samples).values() if count > 1)
    normalized_duplicates = sum(
        count - 1 for count in Counter(sample.normalized_sha256 for sample in samples).values() if count > 1
    )
    label_examples: dict[str, list[str]] = defaultdict(list)
    for sample in samples:
        if len(label_examples[sample.expected_label_name]) < 3:
            label_examples[sample.expected_label_name].append(safe_preview(sample.text, max_chars=120))

    language_counts = Counter()
    try:
        from src.language_utils import detect_language

        for sample in samples[: min(300, len(samples))]:
            language_counts[str(detect_language(sample.text))] += 1
    except Exception:
        language_counts["not_detected"] = min(300, len(samples))

    lines = [
        "# Rogue Security dataset profile",
        "",
        f"- Dataset: `{DATASET_NAME}`",
        f"- URL: {DATASET_URL}",
        f"- Actual splits: `{schema.get('splits', {})}`",
        f"- Actual columns: `{schema.get('features', {})}`",
        f"- Total samples loaded: `{len(samples)}`",
        f"- Label distribution: `{dict(label_counts)}`",
        f"- Benign ratio: `{(label_counts.get('benign', 0) / len(samples)) if samples else 0:.4f}`",
        f"- Jailbreak ratio: `{(label_counts.get('jailbreak', 0) / len(samples)) if samples else 0:.4f}`",
        f"- Empty/null text count: `{empty_count}`",
        f"- Internal exact duplicate count: `{exact_duplicates}`",
        f"- Internal normalized duplicate count: `{normalized_duplicates}`",
        f"- Main language estimate on first {min(300, len(samples))} samples: `{dict(language_counts)}`",
        "",
        "## Character length",
        "",
        markdown_table(
            ["min", "mean", "median", "p95", "max"],
            [[min(char_lengths), round(statistics.mean(char_lengths), 2), round(statistics.median(char_lengths), 2), round(percentile(char_lengths, 95), 2), max(char_lengths)]]
            if char_lengths
            else [],
        ),
        "",
        "## Token length",
        "",
        markdown_table(
            ["min", "mean", "median", "p95", "max", f"truncated > {INFERENCE_MAX_LENGTH}"],
            [
                [
                    min(token_lengths),
                    round(statistics.mean(token_lengths), 2),
                    round(statistics.median(token_lengths), 2),
                    round(percentile(token_lengths, 95), 2),
                    max(token_lengths),
                    sum(1 for sample in samples if sample.was_truncated),
                ]
            ]
            if token_lengths
            else [],
        ),
        "",
        "## Representative previews by label",
        "",
    ]
    for label, previews in sorted(label_examples.items()):
        lines.append(f"### {label}")
        lines.extend(f"- {preview}" for preview in previews)
        lines.append("")
    lines.extend(
        [
            "## Dataset limitations",
            "",
            "- Đây là external holdout direct jailbreak/benign, không đánh giá upload PDF/DOCX/TXT.",
            "- Không đánh giá document extraction, OCR, chunk location, hoặc user-task/document mismatch.",
            "- Benchmark không được dùng để train, calibrate, sửa rule, sửa intent guard, hoặc tối ưu threshold production.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


TEXT_KEYS = {
    "text",
    "prompt",
    "input",
    "content",
    "instruction",
    "query",
    "message",
    "ml_text",
    "model_text",
    "canonical_text",
}


def infer_source_split(path: Path) -> str:
    lowered = str(path).lower()
    if "train" in lowered:
        return "train"
    if "valid" in lowered or "validation" in lowered or "\\dev" in lowered or "/dev" in lowered:
        return "validation"
    if "test" in lowered:
        return "test"
    if "fixture" in lowered or "\\tests" in lowered or "/tests" in lowered:
        return "fixture"
    if "benchmark" in lowered or "holdout" in lowered:
        return "benchmark"
    if "report" in lowered:
        return "report"
    return "unknown"


def should_scan_file(path: Path) -> bool:
    lowered = str(path).lower()
    excluded_parts = [
        "\\.git\\",
        "\\.venv\\",
        "\\models\\",
        "\\outputs\\",
        "\\__pycache__\\",
        "\\data\\external_benchmark\\pint\\raw\\",
        "\\reports\\rogue_security\\",
        "\\reports\\rogue_security_smoke\\",
        "\\reports\\direct_external_evaluation\\",
        "\\reports\\bipia_evaluation\\",
        "\\reports\\final_vi_report\\",
        "\\reports\\latest_completed_work_vi\\",
        "\\reports\\latest_model_training_results_vi\\",
        "\\reports\\project_audit\\",
    ]
    if any(part in lowered for part in excluded_parts):
        return False
    if path.name in {"rogue_security_results.csv", "false_positives.csv", "false_negatives.csv"}:
        return False
    if path.name in {
        "rogue_security_dataset_profile.md",
        "rogue_security_data_leakage_check.md",
        "BAO_CAO_ROGUE_SECURITY_BENCHMARK.md",
        "BAO_CAO_ROGUE_SECURITY_BENCHMARK_DAY_DU.md",
        "TOM_TAT_TRINH_BAY_GIANG_VIEN.md",
    }:
        return False
    if "\\reports\\" in lowered and path.suffix.lower() != ".md":
        return False
    if path.suffix.lower() not in {".csv", ".json", ".jsonl", ".yaml", ".yml", ".md", ".txt", ".py", ".ipynb"}:
        return False
    try:
        max_size = 2 * 1024 * 1024 if "\\reports\\" in lowered else 50 * 1024 * 1024
        return path.stat().st_size <= max_size
    except OSError:
        return False


def iter_structured_texts(path: Path) -> Iterable[tuple[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                return
            fields = [field for field in reader.fieldnames if str(field).lower() in TEXT_KEYS]
            for row in reader:
                for field in fields:
                    value = row.get(field)
                    if value and str(value).strip():
                        yield field, str(value)
    elif suffix == ".jsonl":
        with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    for key, value in item.items():
                        if str(key).lower() in TEXT_KEYS and value and str(value).strip():
                            yield str(key), str(value)
    elif suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig", errors="ignore"))
        except Exception:
            return
        yield from _walk_text_payload(payload)
    elif suffix in {".yaml", ".yml"}:
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8-sig", errors="ignore"))
        except Exception:
            return
        yield from _walk_text_payload(payload)


def _walk_text_payload(payload: Any) -> Iterable[tuple[str, str]]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key).lower() in TEXT_KEYS and value and str(value).strip():
                yield str(key), str(value)
            else:
                yield from _walk_text_payload(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from _walk_text_payload(item)


def simhash(text: str) -> int:
    tokens = re.findall(r"[a-z0-9_]+", normalize_text(text))
    if len(tokens) < 3:
        grams = tokens
    else:
        grams = [" ".join(tokens[index : index + 3]) for index in range(len(tokens) - 2)]
    vector = [0] * 64
    for gram in grams:
        digest = int(hashlib.blake2b(gram.encode("utf-8"), digest_size=8).hexdigest(), 16)
        for bit in range(64):
            vector[bit] += 1 if digest & (1 << bit) else -1
    result = 0
    for bit, value in enumerate(vector):
        if value > 0:
            result |= 1 << bit
    return result


def hamming_distance(left: int, right: int) -> int:
    return int((left ^ right).bit_count())


def leakage_check(
    samples: list[BenchmarkSample],
    *,
    output_path: Path,
    duplicates_path: Path,
) -> tuple[list[BenchmarkSample], dict[str, Any]]:
    raw_hashes = {sample.text_sha256: sample for sample in samples}
    normalized_hashes = {sample.normalized_sha256: sample for sample in samples}
    exact_matches: list[dict[str, Any]] = []
    normalized_matches: list[dict[str, Any]] = []
    name_hits: list[str] = []

    near_duplicate_enabled = str(os.getenv("ROGUE_NEAR_DUPLICATE", "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    near_candidate_limit = int(os.getenv("ROGUE_NEAR_DUPLICATE_LIMIT", "2000") or 2000)
    near_time_budget_seconds = float(os.getenv("ROGUE_NEAR_DUPLICATE_SECONDS", "45") or 45)
    near_started = time.perf_counter()
    near_candidates_checked = 0
    benchmark_simhashes = {sample.sample_id: simhash(sample.text) for sample in samples}
    buckets: dict[tuple[int, int], list[tuple[str, int]]] = defaultdict(list)
    for sample_id, value in benchmark_simhashes.items():
        for band in range(4):
            buckets[(band, (value >> (band * 16)) & 0xFFFF)].append((sample_id, value))
    near_matches: list[dict[str, Any]] = []

    scan_roots = [PROJECT_ROOT / name for name in ["datasets", "data", "tests", "reports", "scripts", "src"]]
    for root in scan_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or not should_scan_file(path):
                continue
            text_for_name_search = ""
            if path.suffix.lower() in {".md", ".txt", ".py", ".ipynb"}:
                try:
                    text_for_name_search = path.read_text(encoding="utf-8-sig", errors="ignore")
                except OSError:
                    text_for_name_search = ""
                if "rogue-security" in text_for_name_search or "prompt-injections-benchmark" in text_for_name_search:
                    name_hits.append(str(path.relative_to(PROJECT_ROOT)))
            for field, value in iter_structured_texts(path):
                if not value.strip():
                    continue
                raw_hash = hash_text(value)
                normalized_hash = hash_text(normalize_text(value))
                source_split = infer_source_split(path)
                if raw_hash in raw_hashes:
                    exact_matches.append(
                        {
                            "sample_id": raw_hashes[raw_hash].sample_id,
                            "source_file": str(path.relative_to(PROJECT_ROOT)),
                            "source_split": source_split,
                            "field": field,
                        }
                    )
                if normalized_hash in normalized_hashes:
                    normalized_matches.append(
                        {
                            "sample_id": normalized_hashes[normalized_hash].sample_id,
                            "source_file": str(path.relative_to(PROJECT_ROOT)),
                            "source_split": source_split,
                            "field": field,
                        }
                    )
                if (
                    near_duplicate_enabled
                    and near_candidates_checked < near_candidate_limit
                    and (time.perf_counter() - near_started) <= near_time_budget_seconds
                ):
                    near_candidates_checked += 1
                    candidate_simhash = simhash(value)
                    checked: set[str] = set()
                    for band in range(4):
                        key = (band, (candidate_simhash >> (band * 16)) & 0xFFFF)
                        for sample_id, sample_hash in buckets.get(key, []):
                            if sample_id in checked:
                                continue
                            checked.add(sample_id)
                            if hamming_distance(candidate_simhash, sample_hash) <= 4:
                                near_matches.append(
                                    {
                                        "sample_id": sample_id,
                                        "source_file": str(path.relative_to(PROJECT_ROOT)),
                                        "source_split": source_split,
                                        "field": field,
                                        "method": "simhash_hamming<=4",
                                    }
                                )

    duplicate_sample_ids = {item["sample_id"] for item in exact_matches + normalized_matches + near_matches}
    updated_samples: list[BenchmarkSample] = []
    for sample in samples:
        updated_samples.append(
            BenchmarkSample(
                **{
                    **sample.__dict__,
                    "leakage_duplicate": sample.sample_id in duplicate_sample_ids,
                }
            )
        )

    def count_by_split(matches: list[dict[str, Any]], split: str) -> int:
        return len({item["sample_id"] for item in matches if item.get("source_split") == split})

    summary = {
        "total_samples": len(samples),
        "exact_duplicate_count": len({item["sample_id"] for item in exact_matches}),
        "normalized_duplicate_count": len({item["sample_id"] for item in normalized_matches}),
        "near_duplicate_count": len({item["sample_id"] for item in near_matches}),
        "exact_duplicate_train": count_by_split(exact_matches, "train"),
        "exact_duplicate_validation": count_by_split(exact_matches, "validation"),
        "exact_duplicate_test": count_by_split(exact_matches, "test"),
        "name_hits": sorted(set(name_hits)),
        "independent_enough": len(duplicate_sample_ids) == 0,
        "duplicate_sample_ids": sorted(duplicate_sample_ids),
        "near_duplicate_enabled": near_duplicate_enabled,
        "near_candidates_checked": near_candidates_checked,
        "near_candidate_limit": near_candidate_limit,
        "near_time_budget_seconds": near_time_budget_seconds,
    }
    duplicate_rows = [
        {**item, "match_type": "exact"} for item in exact_matches
    ] + [
        {**item, "match_type": "normalized"} for item in normalized_matches
    ] + [
        {**item, "match_type": "near_duplicate"} for item in near_matches
    ]
    write_csv(
        duplicates_path,
        duplicate_rows,
        fieldnames=["sample_id", "source_file", "source_split", "field", "method", "match_type"],
    )
    sample_rows = duplicate_rows[:50]
    lines = [
        "# Rogue Security data leakage check",
        "",
        f"- Dataset: `{DATASET_NAME}`",
        f"- Total benchmark samples: `{len(samples)}`",
        f"- Dataset/repository name hits in local repo: `{len(set(name_hits))}`",
        f"- Exact duplicate samples: `{summary['exact_duplicate_count']}`",
        f"- Exact duplicate with train: `{summary['exact_duplicate_train']}`",
        f"- Exact duplicate with validation: `{summary['exact_duplicate_validation']}`",
        f"- Exact duplicate with test: `{summary['exact_duplicate_test']}`",
        f"- Normalized duplicate samples: `{summary['normalized_duplicate_count']}`",
        f"- Near-duplicate samples by SimHash best-effort: `{summary['near_duplicate_count']}`",
        f"- Near-duplicate enabled: `{near_duplicate_enabled}`",
        f"- Near-duplicate candidates checked: `{near_candidates_checked}` / `{near_candidate_limit}` within `{near_time_budget_seconds}` seconds",
        f"- Conclusion: `{'independent enough for first external holdout evaluation' if summary['independent_enough'] else 'duplicates found; report full and deduplicated metrics'}`",
        "",
        "## Dataset/repository name hits",
        "",
        "\n".join(f"- `{item}`" for item in sorted(set(name_hits))[:50]) or "_Không tìm thấy._",
        "",
        "## Duplicate samples",
        "",
        markdown_table(
            ["sample_id", "source_file", "source_split", "field", "method"],
            [
                [
                    item.get("sample_id", ""),
                    item.get("source_file", ""),
                    item.get("source_split", ""),
                    item.get("field", ""),
                    item.get("method", item.get("match_type", "exact_or_normalized")),
                ]
                for item in sample_rows
            ],
        ),
        "",
        "Ghi chú: near-duplicate là kiểm tra SimHash best-effort, không dùng để xóa dữ liệu âm thầm.",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return updated_samples, summary


def group_by_mode(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("mode"))].append(row)
    return grouped


def write_metrics_outputs(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, dict[str, Any]]:
    metrics_rows: list[dict[str, Any]] = []
    metrics_by_mode: dict[str, dict[str, Any]] = {}
    for mode, mode_rows in sorted(group_by_mode(rows).items()):
        score_field = "fusion_score" if mode == "full_runtime" else "raw_roberta_score"
        metrics = compute_binary_metrics(mode_rows, score_field=score_field)
        metrics_by_mode[mode] = metrics
        metrics_rows.append({"mode": mode, "mode_label": MODE_LABELS.get(mode, mode), **metrics})

        dedup_rows = [row for row in mode_rows if str(row.get("leakage_duplicate", "0")) not in {"1", "true", "True"}]
        if len(dedup_rows) != len(mode_rows):
            metrics_rows.append(
                {
                    "mode": f"{mode}_deduplicated",
                    "mode_label": f"{MODE_LABELS.get(mode, mode)} deduplicated",
                    **compute_binary_metrics(dedup_rows, score_field=score_field),
                }
            )
    write_csv(output_dir / "metrics_summary.csv", metrics_rows)
    return metrics_by_mode


def write_threshold_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    roberta_rows = [row for row in rows if row.get("mode") == "roberta_only" and row.get("error") in {"", None}]
    if not roberta_rows:
        write_csv(output_dir / "threshold_sweep.csv", [])
        return
    current_thresholds = [_float_or_none(row.get("threshold")) for row in roberta_rows]
    current_threshold = next((value for value in current_thresholds if value is not None), 0.5)
    thresholds = sorted({round(index / 100, 2) for index in range(101)} | {round(float(current_threshold), 8)})
    sweep_rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        eval_rows = []
        for row in roberta_rows:
            score = _float_or_none(row.get("raw_roberta_score")) or 0.0
            eval_rows.append({**row, "predicted_label": int(score >= threshold)})
        metrics = compute_binary_metrics(eval_rows, score_field="raw_roberta_score")
        sweep_rows.append(
            {
                "threshold": threshold,
                **metrics,
                "fpr": metrics["false_positive_rate"],
                "fnr": metrics["false_negative_rate"],
            }
        )

    best_f1 = max(sweep_rows, key=lambda row: (row["f1"], row["balanced_accuracy"], -row["threshold"]))["threshold"]
    best_balanced = max(sweep_rows, key=lambda row: (row["balanced_accuracy"], row["f1"], -row["threshold"]))["threshold"]
    recall_95_candidates = [row for row in sweep_rows if row["recall"] >= 0.95]
    fpr_5_candidates = [row for row in sweep_rows if row["false_positive_rate"] <= 0.05]
    recall_95 = max(recall_95_candidates, key=lambda row: row["threshold"])["threshold"] if recall_95_candidates else None
    fpr_5 = min(fpr_5_candidates, key=lambda row: row["false_negative_rate"])["threshold"] if fpr_5_candidates else None
    for row in sweep_rows:
        row["is_current_production_threshold"] = int(float(row["threshold"]) == float(current_threshold))
        row["is_best_f1"] = int(float(row["threshold"]) == float(best_f1))
        row["is_best_balanced_accuracy"] = int(float(row["threshold"]) == float(best_balanced))
        row["is_recall_95_threshold"] = int(recall_95 is not None and float(row["threshold"]) == float(recall_95))
        row["is_fpr_5_threshold"] = int(fpr_5 is not None and float(row["threshold"]) == float(fpr_5))
    write_csv(output_dir / "threshold_sweep.csv", sweep_rows)

    y_true = [_int(row["expected_label"]) for row in roberta_rows]
    y_score = [float(_float_or_none(row.get("raw_roberta_score")) or 0.0) for row in roberta_rows]
    if len(set(y_true)) == 2:
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
        write_csv(
            output_dir / "roc_curve.csv",
            [
                {"fpr": float(left), "tpr": float(right), "threshold": float(threshold)}
                for left, right, threshold in zip(fpr, tpr, roc_thresholds)
            ],
        )
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        pr_rows = []
        for index, (prec, rec) in enumerate(zip(precision, recall)):
            threshold = "" if index >= len(pr_thresholds) else float(pr_thresholds[index])
            pr_rows.append({"precision": float(prec), "recall": float(rec), "threshold": threshold})
        write_csv(output_dir / "pr_curve.csv", pr_rows)

    distribution_rows = []
    for label_name, label_value in [("benign", 0), ("jailbreak", 1)]:
        scores = [
            float(_float_or_none(row.get("raw_roberta_score")) or 0.0)
            for row in roberta_rows
            if _int(row.get("expected_label")) == label_value
        ]
        distribution_rows.append(
            {
                "label": label_name,
                "count": len(scores),
                "p5": percentile(scores, 5),
                "p25": percentile(scores, 25),
                "p50": percentile(scores, 50),
                "p75": percentile(scores, 75),
                "p95": percentile(scores, 95),
                "high_score_count_at_current_threshold": sum(1 for score in scores if score >= current_threshold)
                if label_value == 0
                else "",
                "low_score_count_below_current_threshold": sum(1 for score in scores if score < current_threshold)
                if label_value == 1
                else "",
            }
        )
    write_csv(output_dir / "score_distribution.csv", distribution_rows)


def write_error_analysis(rows: list[dict[str, Any]], output_dir: Path) -> None:
    fp_rows = []
    fn_rows = []
    for row in rows:
        if row.get("error"):
            continue
        if _int(row.get("expected_label")) == 0 and _int(row.get("predicted_label")) == 1:
            fp_rows.append({**row, "false_positive_group": classify_false_positive(row)})
        if _int(row.get("expected_label")) == 1 and _int(row.get("predicted_label")) == 0:
            fn_rows.append({**row, "false_negative_group": classify_false_negative(row)})
    write_csv(output_dir / "false_positives.csv", fp_rows)
    write_csv(output_dir / "false_negatives.csv", fn_rows)
    fp_summary: list[dict[str, Any]] = []
    for category, grouped in sorted(_group_by_key(fp_rows, "false_positive_group").items()):
        scores = [float(_float_or_none(row.get("raw_roberta_score")) or 0.0) for row in grouped]
        rules = Counter(
            rule
            for row in grouped
            for rule in str(row.get("matched_rule_ids", "")).split(";")
            if rule
        )
        fp_summary.append(
            {
                "category": category,
                "false_positive_count": len(grouped),
                "percentage": len(grouped) / len(fp_rows) if fp_rows else 0.0,
                "mean_score": statistics.mean(scores) if scores else 0.0,
                "median_score": statistics.median(scores) if scores else 0.0,
                "top_triggered_rules": ";".join(rule for rule, _ in rules.most_common(5)),
            }
        )
    write_csv(output_dir / "false_positive_categories.csv", fp_summary)

    fn_summary: list[dict[str, Any]] = []
    for category, grouped in sorted(_group_by_key(fn_rows, "false_negative_group").items()):
        scores = [float(_float_or_none(row.get("raw_roberta_score")) or 0.0) for row in grouped]
        fn_summary.append(
            {
                "category": category,
                "false_negative_count": len(grouped),
                "percentage": len(grouped) / len(fn_rows) if fn_rows else 0.0,
                "mean_score": statistics.mean(scores) if scores else 0.0,
                "median_score": statistics.median(scores) if scores else 0.0,
                "truncated_count": sum(1 for row in grouped if _int(row.get("was_truncated")) == 1),
                "near_threshold_count": sum(
                    1
                    for row in grouped
                    if abs((_float_or_none(row.get("raw_roberta_score")) or 0.0) - (_float_or_none(row.get("threshold")) or 0.0)) <= 0.05
                ),
            }
        )
    write_csv(output_dir / "false_negative_categories.csv", fn_summary)


def _group_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "Other")].append(row)
    return grouped


def write_intent_guard_audit(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    full_rows = [row for row in rows if row.get("mode") == "full_runtime"]
    audit_rows = []
    for row in full_rows:
        raw = _float_or_none(row.get("raw_roberta_score"))
        adjusted = _float_or_none(row.get("intent_adjusted_score"))
        threshold = _float_or_none(row.get("threshold")) or 0.5
        raw_pred = int((raw or 0.0) >= threshold)
        after_pred = _int(row.get("predicted_label"))
        expected = _int(row.get("expected_label"))
        audit_rows.append(
            {
                "sample_id": row.get("sample_id"),
                "expected_label": row.get("expected_label"),
                "raw_score": "" if raw is None else raw,
                "adjusted_score": "" if adjusted is None else adjusted,
                "score_reduction": "" if raw is None or adjusted is None else max(0.0, raw - adjusted),
                "intent_category": row.get("intent_category"),
                "intent_guard_applied": row.get("intent_guard_applied"),
                "intent_guard_reason": row.get("intent_guard_reason"),
                "predicted_before_guard": raw_pred,
                "predicted_after_guard": after_pred,
                "correct_before_guard": int(raw_pred == expected),
                "correct_after_guard": int(after_pred == expected),
                "final_decision": row.get("final_decision"),
                "decision_source": row.get("decision_source"),
                "triggered_policy": row.get("triggered_policy"),
            }
        )
    write_csv(output_dir / "intent_guard_audit.csv", audit_rows)

    before_rows = [{**row, "predicted_label": item["predicted_before_guard"]} for row, item in zip(full_rows, audit_rows)]
    after_rows = full_rows
    summary = {
        "benign_corrected_by_guard": sum(
            1
            for before, after in zip(before_rows, after_rows)
            if _int(before["expected_label"]) == 0
            and _int(before["predicted_label"]) == 1
            and _int(after["predicted_label"]) == 0
        ),
        "jailbreak_lowered_wrongly_by_guard": sum(
            1
            for before, after in zip(before_rows, after_rows)
            if _int(before["expected_label"]) == 1
            and _int(before["predicted_label"]) == 1
            and _int(after["predicted_label"]) == 0
        ),
        "before": compute_binary_metrics(before_rows) if before_rows else {},
        "after": compute_binary_metrics(after_rows) if after_rows else {},
    }
    write_csv(
        output_dir / "intent_guard_summary.csv",
        [
            {
                "benign_corrected_by_guard": summary["benign_corrected_by_guard"],
                "jailbreak_lowered_wrongly_by_guard": summary["jailbreak_lowered_wrongly_by_guard"],
                "recall_before_guard": summary["before"].get("recall"),
                "recall_after_guard": summary["after"].get("recall"),
                "fpr_before_guard": summary["before"].get("false_positive_rate"),
                "fpr_after_guard": summary["after"].get("false_positive_rate"),
                "bypass_due_to_guard": summary["jailbreak_lowered_wrongly_by_guard"],
            }
        ],
    )
    reduction_rows = []
    for category, grouped in _group_by_key(audit_rows, "intent_category").items():
        reductions = [
            float(row.get("score_reduction"))
            for row in grouped
            if row.get("score_reduction") not in {"", None}
        ]
        reduction_rows.append(
            {
                "intent_category": category,
                "count": len(grouped),
                "guard_applied_count": sum(1 for row in grouped if _int(row.get("intent_guard_applied")) == 1),
                "mean_score_reduction": statistics.mean(reductions) if reductions else 0.0,
                "jailbreak_lowered_wrongly": sum(
                    1
                    for row in grouped
                    if _int(row.get("expected_label")) == 1
                    and _int(row.get("predicted_before_guard")) == 1
                    and _int(row.get("predicted_after_guard")) == 0
                ),
            }
        )
    write_csv(output_dir / "intent_guard_reduction_by_category.csv", reduction_rows)
    return summary


def write_rule_contribution(rows: list[dict[str, Any]], output_dir: Path) -> None:
    by_sample_mode = {(row.get("sample_id"), row.get("mode")): row for row in rows}
    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "trigger_count": 0,
            "true_positive_count": 0,
            "false_positive_count": 0,
            "corrected_false_negatives": 0,
            "introduced_false_positives": 0,
            "rule_scores": [],
            "severities": [],
        }
    )
    for row in rows:
        if row.get("mode") != "rule_roberta":
            continue
        rule_ids = [item for item in str(row.get("matched_rule_ids", "")).split(";") if item]
        if not rule_ids:
            continue
        roberta = by_sample_mode.get((row.get("sample_id"), "roberta_only"))
        for rule_id in rule_ids:
            stats[rule_id]["trigger_count"] += 1
            if _float_or_none(row.get("rule_score")) is not None:
                stats[rule_id]["rule_scores"].append(float(_float_or_none(row.get("rule_score")) or 0.0))
            stats[rule_id]["severities"].append(str(row.get("highest_rule_severity") or "none"))
            if _int(row.get("expected_label")) == 1 and _int(row.get("predicted_label")) == 1:
                stats[rule_id]["true_positive_count"] += 1
            if _int(row.get("expected_label")) == 0 and _int(row.get("predicted_label")) == 1:
                stats[rule_id]["false_positive_count"] += 1
            if roberta and _int(roberta.get("expected_label")) == 1 and _int(roberta.get("predicted_label")) == 0 and _int(row.get("predicted_label")) == 1:
                stats[rule_id]["corrected_false_negatives"] += 1
            if roberta and _int(roberta.get("expected_label")) == 0 and _int(roberta.get("predicted_label")) == 0 and _int(row.get("predicted_label")) == 1:
                stats[rule_id]["introduced_false_positives"] += 1

    output_rows = []
    for rule_id, values in sorted(stats.items()):
        trigger_count = values["trigger_count"]
        precision = (
            values["true_positive_count"] / (values["true_positive_count"] + values["false_positive_count"])
            if (values["true_positive_count"] + values["false_positive_count"])
            else 0.0
        )
        severity_order = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
        highest_severity = max(values["severities"] or ["none"], key=lambda item: severity_order.get(item, 0))
        output_rows.append(
            {
                "rule_id": rule_id,
                "trigger_count": trigger_count,
                "true_positive_count": values["true_positive_count"],
                "false_positive_count": values["false_positive_count"],
                "precision_when_triggered": precision,
                "corrected_false_negatives": values["corrected_false_negatives"],
                "introduced_false_positives": values["introduced_false_positives"],
                "mean_rule_score": statistics.mean(values["rule_scores"]) if values["rule_scores"] else 0.0,
                "highest_severity": highest_severity,
            }
        )
    write_csv(output_dir / "rule_contribution.csv", output_rows)


def write_policy_decision_analysis(rows: list[dict[str, Any]], output_dir: Path) -> None:
    full_rows = [row for row in rows if row.get("mode") == "full_runtime"]
    output_rows: list[dict[str, Any]] = []
    total = len(full_rows)
    for decision in ["safe", "warning", "blocked", "error"]:
        grouped = [row for row in full_rows if str(row.get("final_decision", "")).lower() == decision]
        if not grouped:
            continue
        benign_count = sum(1 for row in grouped if _int(row.get("expected_label")) == 0)
        jailbreak_count = sum(1 for row in grouped if _int(row.get("expected_label")) == 1)
        output_rows.append(
            {
                "analysis_type": "decision_count",
                "key": decision.upper(),
                "count": len(grouped),
                "percentage": len(grouped) / total if total else 0.0,
                "benign_count": benign_count,
                "jailbreak_count": jailbreak_count,
                "benign_rate_within_decision": benign_count / len(grouped) if grouped else 0.0,
                "jailbreak_rate_within_decision": jailbreak_count / len(grouped) if grouped else 0.0,
            }
        )
    jailbreak_rows = [row for row in full_rows if _int(row.get("expected_label")) == 1]
    benign_rows = [row for row in full_rows if _int(row.get("expected_label")) == 0]
    for label_name, grouped in [("jailbreak", jailbreak_rows), ("benign", benign_rows)]:
        for decision in ["warning", "blocked"]:
            count = sum(1 for row in grouped if str(row.get("final_decision", "")).lower() == decision)
            output_rows.append(
                {
                    "analysis_type": f"{label_name}_decision_rate",
                    "key": decision.upper(),
                    "count": count,
                    "percentage": count / len(grouped) if grouped else 0.0,
                    "benign_count": "",
                    "jailbreak_count": "",
                    "benign_rate_within_decision": "",
                    "jailbreak_rate_within_decision": "",
                }
            )
    for source, grouped in sorted(_group_by_key(full_rows, "decision_source").items()):
        output_rows.append(
            {
                "analysis_type": "decision_source_distribution",
                "key": source,
                "count": len(grouped),
                "percentage": len(grouped) / total if total else 0.0,
                "benign_count": sum(1 for row in grouped if _int(row.get("expected_label")) == 0),
                "jailbreak_count": sum(1 for row in grouped if _int(row.get("expected_label")) == 1),
                "benign_rate_within_decision": "",
                "jailbreak_rate_within_decision": "",
            }
        )
    policy_counter = Counter(
        policy
        for row in full_rows
        for policy in str(row.get("triggered_policy", "")).split(";")
        if policy and policy != "none"
    )
    for policy, count in policy_counter.most_common():
        output_rows.append(
            {
                "analysis_type": "triggered_policy_distribution",
                "key": policy,
                "count": count,
                "percentage": count / total if total else 0.0,
                "benign_count": "",
                "jailbreak_count": "",
                "benign_rate_within_decision": "",
                "jailbreak_rate_within_decision": "",
            }
        )
    write_csv(output_dir / "policy_decision_analysis.csv", output_rows)


def estimate_attack_position(text_preview: str) -> str:
    text = str(text_preview or "").lower()
    positions = []
    for keyword in [
        "ignore",
        "disregard",
        "system prompt",
        "developer mode",
        "bypass",
        "jailbreak",
        "reveal",
        "hidden instruction",
    ]:
        index = text.find(keyword)
        if index >= 0:
            positions.append(index / max(1, len(text)))
    if not positions:
        return "unknown"
    position = min(positions)
    if position < 0.33:
        return "early"
    if position < 0.66:
        return "middle"
    return "late"


def write_truncation_cases(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_rows = []
    for row in rows:
        if _int(row.get("was_truncated")) != 1:
            continue
        output_rows.append(
            {
                "sample_id": row.get("sample_id"),
                "mode": row.get("mode"),
                "expected_label": row.get("expected_label"),
                "predicted_label": row.get("predicted_label"),
                "original_token_count": row.get("original_token_count") or row.get("token_count"),
                "used_token_count": min(_int(row.get("token_count")), INFERENCE_MAX_LENGTH),
                "was_truncated": row.get("was_truncated"),
                "attack_position_estimate": estimate_attack_position(str(row.get("text_preview", ""))),
                "correct": row.get("correct"),
                "raw_roberta_score": row.get("raw_roberta_score"),
                "threshold": row.get("threshold"),
            }
        )
    write_csv(output_dir / "truncation_cases.csv", output_rows)


def _device_description(use_cuda: bool) -> str:
    if not use_cuda:
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.get_device_name(0)}"
    except Exception:
        pass
    return "cuda"


def write_length_and_latency(
    rows: list[dict[str, Any]],
    output_dir: Path,
    warmup_latencies: dict[str, float],
    *,
    use_cuda: bool,
    batch_size: int,
) -> None:
    length_rows = []
    for mode, mode_rows in sorted(group_by_mode(rows).items()):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in mode_rows:
            groups[length_bin(_int(row.get("token_count")))].append(row)
        for bin_name, grouped in sorted(groups.items()):
            length_rows.append(
                {
                    "mode": mode,
                    "length_bin": bin_name,
                    **compute_binary_metrics(grouped),
                    "truncation_rate": sum(1 for row in grouped if _int(row.get("was_truncated")) == 1) / len(grouped)
                    if grouped
                    else 0.0,
                }
            )
    write_csv(output_dir / "metrics_by_length.csv", length_rows)

    latency_rows = []
    for mode, mode_rows in sorted(group_by_mode(rows).items()):
        values = [float(_float_or_none(row.get("latency_ms")) or 0.0) for row in mode_rows if not row.get("error")]
        if not values:
            latency_rows.append({"mode": mode, "error": "no successful rows"})
            continue
        total_seconds = sum(values) / 1000.0
        latency_rows.append(
            {
                "mode": mode,
                "mean_latency_ms": statistics.mean(values),
                "median_latency_ms": statistics.median(values),
                "p90_latency_ms": percentile(values, 90),
                "p95_latency_ms": percentile(values, 95),
                "p99_latency_ms": percentile(values, 99),
                "min_latency_ms": min(values),
                "max_latency_ms": max(values),
                "throughput_samples_per_second": len(values) / total_seconds if total_seconds else 0.0,
                "total_runtime_seconds": total_seconds,
                "cold_start_latency_ms": warmup_latencies.get(mode, ""),
                "warm_inference_latency_ms": statistics.median(values),
                "device": _device_description(use_cuda),
                "batch_size": batch_size,
                "workers": 1,
                "machine": "local Windows",
                "model_cache_state": "process singleton/lru cache after warm-up",
            }
        )
    write_csv(output_dir / "latency_summary.csv", latency_rows)


def load_previous_comparison_rows(rogue_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparison_rows = []
    if PINT_RESULTS_PATH.exists():
        pint_rows = read_results_csv(PINT_RESULTS_PATH)
        for (benchmark, mode), grouped in _group_for_comparison(pint_rows).items():
            metrics = _metrics_for_pint_group(grouped)
            comparison_rows.append({"Benchmark": benchmark, "Mode": mode, "N": metrics["total"], **metrics})
    for mode, grouped in group_by_mode(rogue_rows).items():
        metrics = compute_binary_metrics(grouped, score_field="fusion_score" if mode == "full_runtime" else "raw_roberta_score")
        comparison_rows.append(
            {
                "Benchmark": "Rogue Security external benchmark",
                "Mode": MODE_LABELS.get(mode, mode),
                "N": metrics["total"],
                **metrics,
            }
        )
    return comparison_rows


def _group_for_comparison(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    output: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("benchmark_type") == "indirect_document":
            benchmark = "Indirect document benchmark"
            mode = str(row.get("mode_label") or row.get("mode"))
        elif row.get("dataset_name") == "pint_example_smoke":
            benchmark = "PINT example smoke"
            mode = str(row.get("mode_label") or row.get("mode"))
        elif row.get("dataset_name") == "pint_compatible_internal_holdout":
            benchmark = "Internal regression set"
            mode = str(row.get("mode_label") or row.get("mode"))
        else:
            continue
        output[(benchmark, mode)].append(row)
    return output


def _metrics_for_pint_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    normalized = [
        {
            "expected_label": int(str(row.get("label", row.get("expected_label"))).lower() in {"true", "1"}),
            "predicted_label": int(str(row.get("prediction")).lower() in {"true", "1"}),
            "raw_roberta_score": row.get("score", ""),
        }
        for row in rows
    ]
    return compute_binary_metrics(normalized, score_field="raw_roberta_score")


def write_final_report(
    *,
    samples: list[BenchmarkSample],
    rows: list[dict[str, Any]],
    output_dir: Path,
    leakage_summary: dict[str, Any],
    warmup_latencies: dict[str, float],
    access_error: str | None = None,
) -> None:
    report_path = output_dir / "BAO_CAO_ROGUE_SECURITY_BENCHMARK.md"
    full_report_path = output_dir / "BAO_CAO_ROGUE_SECURITY_BENCHMARK_DAY_DU.md"
    teacher_summary_path = output_dir / "TOM_TAT_TRINH_BAY_GIANG_VIEN.md"
    if access_error:
        access_text = "\n".join(
                [
                    "# Báo cáo Rogue Security external benchmark",
                    "",
                    "## Trạng thái",
                    "",
                    "Chưa thể chạy benchmark vì dataset Hugging Face đang bị gated và môi trường này chưa có token đã được cấp quyền.",
                    "",
                    f"- Dataset: `{DATASET_NAME}`",
                    f"- URL: {DATASET_URL}",
                    f"- Lỗi: `{access_error}`",
                    "",
                    "Cần đăng nhập Hugging Face và accept điều kiện dataset trước khi chạy lại script.",
                ]
        )
        report_path.write_text(access_text, encoding="utf-8")
        full_report_path.write_text(access_text, encoding="utf-8")
        teacher_summary_path.write_text(access_text, encoding="utf-8")
        return

    metrics_by_mode = {mode: compute_binary_metrics(grouped, score_field="fusion_score" if mode == "full_runtime" else "raw_roberta_score") for mode, grouped in group_by_mode(rows).items()}
    metric_table_rows = [
        [
            MODE_LABELS.get(mode, mode),
            metrics["total"],
            format_percent(metrics["balanced_accuracy"]),
            format_percent(metrics["accuracy"]),
            format_percent(metrics["precision"]),
            format_percent(metrics["recall"]),
            format_percent(metrics["f1"]),
            format_percent(metrics["false_positive_rate"]),
            format_percent(metrics["false_negative_rate"]),
            f"TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}",
        ]
        for mode, metrics in sorted(metrics_by_mode.items())
    ]
    fp_rows = read_results_csv(output_dir / "false_positives.csv")[:10]
    fn_rows = read_results_csv(output_dir / "false_negatives.csv")[:10]
    intent_summary_rows = read_results_csv(output_dir / "intent_guard_summary.csv")
    rule_rows = read_results_csv(output_dir / "rule_contribution.csv")[:15]
    latency_rows = read_results_csv(output_dir / "latency_summary.csv")
    comparison = load_previous_comparison_rows(rows)
    comparison_table = [
        [
            item["Benchmark"],
            item["Mode"],
            item["N"],
            format_percent(item["balanced_accuracy"]),
            format_percent(item["precision"]),
            format_percent(item["recall"]),
            format_percent(item["f1"]),
            format_percent(item["false_positive_rate"]),
            format_percent(item["false_negative_rate"]),
        ]
        for item in comparison
    ]
    label_counts = Counter(sample.expected_label_name for sample in samples)
    threshold_source = next((row.get("threshold_source") for row in rows if row.get("mode") == "roberta_only"), "")
    threshold = next((row.get("threshold") for row in rows if row.get("mode") == "roberta_only"), "")
    config = load_runtime_config()
    lines = [
        "# Báo cáo Rogue Security external benchmark đầy đủ",
        "",
        "## Tóm tắt điều hành",
        "",
        f"Benchmark xử lý `{len(samples)}` mẫu từ split `test` của Rogue Security và so sánh ba mode: RoBERTa-only, Rule + RoBERTa, Full runtime. Positive class là `jailbreak`; negative class là `benign`.",
        "",
        "Kết quả cần đọc theo hai tầng: RoBERTa-only đo năng lực checkpoint, còn Full runtime đo năng lực toàn hệ thống hybrid. Nếu Full runtime cao hơn RoBERTa-only, kết luận đúng là hệ thống hybrid hiệu quả hơn, không tự động suy ra RoBERTa đã hiểu tốt.",
        "",
        "## 1. Mục tiêu benchmark",
        "",
        "Đánh giá khả năng tổng quát hóa của RoBERTa-only, Rule + RoBERTa và Full runtime trên external holdout direct jailbreak/benign.",
        "",
        "## 2. Nguồn dataset",
        "",
        f"- Dataset: `{DATASET_NAME}`",
        f"- URL: {DATASET_URL}",
        "- Dataset card hiện mô tả 5.000 prompt với nhãn `benign` và `jailbreak`.",
        "",
        "## 3. License và điều kiện sử dụng",
        "",
        "- License trên dataset card: `cc-by-nc-4.0`.",
        "- Hugging Face yêu cầu accept điều kiện truy cập/chia sẻ contact info trước khi tải file dataset.",
        "",
        "## 4. Cấu trúc dataset",
        "",
        f"- Số mẫu đã load: `{len(samples)}`",
        f"- Label distribution: `{dict(label_counts)}`",
        "- Cột bắt buộc đã xác minh: `text`, `label`.",
        "",
        "## 5. Kiểm tra data leakage",
        "",
        f"- Exact duplicate: `{leakage_summary.get('exact_duplicate_count')}`",
        f"- Normalized duplicate: `{leakage_summary.get('normalized_duplicate_count')}`",
        f"- Near-duplicate best-effort: `{leakage_summary.get('near_duplicate_count')}`",
        f"- Kết luận: `{'đủ độc lập cho external holdout đầu tiên' if leakage_summary.get('independent_enough') else 'có duplicate, cần đọc cả full và deduplicated metrics'}`",
        "",
        "## 6. Label mapping",
        "",
        "- `benign` -> 0, negative.",
        "- `jailbreak` -> 1, positive.",
        "- Không dùng index label của Hugging Face làm index model.",
        "",
        "## 7. Cấu hình model",
        "",
        f"- Runtime model: `{config.get('runtimeModel')}`",
        f"- Model version config: `{config.get('modelVersion')}`",
        "- Downstream LLM bị skip trong benchmark.",
        "",
        "## 8. Checkpoint và threshold",
        "",
        f"- RoBERTa-only threshold: `{threshold}`",
        f"- Threshold source: `{threshold_source}`",
        "- Threshold sweep chỉ để phân tích, không tự động áp dụng vào production.",
        "",
        "## 9. Ba mode đánh giá",
        "",
        "- RoBERTa-only: rule/context/fusion/policy/intent guard/calibration OFF, dùng raw probability.",
        "- Rule + RoBERTa: rule block signal hoặc raw RoBERTa vượt threshold thì positive.",
        "- Full runtime: pipeline production hiện tại, gồm policy/fusion/context nếu đang có, nhưng không gọi LLM.",
        "",
        "## 10. Metric tổng quan",
        "",
        markdown_table(
            ["Mode", "N", "Balanced Accuracy", "Accuracy", "Precision", "Recall", "F1", "FPR", "FNR", "Confusion matrix"],
            metric_table_rows,
        ),
        "",
        "### Công thức metric",
        "",
        "- Accuracy = `(TP + TN) / N`.",
        "- Balanced Accuracy = `(Recall + Specificity) / 2`.",
        "- Precision = `TP / (TP + FP)`.",
        "- Recall = `TP / (TP + FN)`.",
        "- Specificity = `TN / (TN + FP)`.",
        "- F1 = `2 * Precision * Recall / (Precision + Recall)`.",
        "- FPR = `FP / (FP + TN)`.",
        "- FNR = `FN / (FN + TP)`.",
        "- MCC đo tương quan nhị phân giữa prediction và label, hữu ích khi dữ liệu lệch lớp.",
        "- ROC-AUC và PR-AUC dùng score liên tục, không chỉ nhãn sau threshold.",
        "",
        "## 11. Confusion matrix",
        "",
        "Confusion matrix được ghi trong bảng metric tổng quan và `metrics_summary.csv`.",
        "",
        "## 12. So sánh ba mode",
        "",
        "So sánh chính cần đọc theo recall/FPR, không chỉ accuracy. Rule contribution và intent guard audit nằm ở các phần dưới.",
        "",
        "## 13. Phân tích false positive",
        "",
        markdown_table(
            ["mode", "sample_id", "score", "decision_source", "group", "preview"],
            [
                [
                    row.get("mode"),
                    row.get("sample_id"),
                    row.get("raw_roberta_score"),
                    row.get("decision_source"),
                    row.get("false_positive_group"),
                    row.get("text_preview"),
                ]
                for row in fp_rows
            ],
        ),
        "",
        "## 14. Phân tích false negative",
        "",
        markdown_table(
            ["mode", "sample_id", "score", "decision_source", "group", "preview"],
            [
                [
                    row.get("mode"),
                    row.get("sample_id"),
                    row.get("raw_roberta_score"),
                    row.get("decision_source"),
                    row.get("false_negative_group"),
                    row.get("text_preview"),
                ]
                for row in fn_rows
            ],
        ),
        "",
        "## 15. Phân tích intent guard",
        "",
        markdown_table(list(intent_summary_rows[0].keys()), [list(intent_summary_rows[0].values())] if intent_summary_rows else []),
        "",
        "## 16. Phân tích đóng góp của rule",
        "",
        markdown_table(
            ["rule_id", "trigger_count", "true_positive_count", "false_positive_count", "precision_when_triggered", "corrected_false_negatives", "introduced_false_positives"],
            [
                [
                    row.get("rule_id"),
                    row.get("trigger_count"),
                    row.get("true_positive_count"),
                    row.get("false_positive_count"),
                    row.get("precision_when_triggered"),
                    row.get("corrected_false_negatives"),
                    row.get("introduced_false_positives"),
                ]
                for row in rule_rows
            ],
        ),
        "",
        "## 17. Phân tích theo độ dài",
        "",
        "Chi tiết nằm trong `metrics_by_length.csv`; bảng này có N, accuracy, balanced accuracy, precision, recall, F1, FPR, FNR và truncation rate theo từng nhóm token length.",
        "",
        "## 18. Phân tích truncation",
        "",
        f"- Số mẫu vượt `{INFERENCE_MAX_LENGTH}` token trước truncation: `{sum(1 for sample in samples if sample.was_truncated)}`.",
        "",
        "## 19. Phân tích threshold",
        "",
        "- `threshold_sweep.csv` đánh dấu current threshold, best F1, best balanced accuracy, recall >= 95% và FPR <= 5% nếu tồn tại.",
        "- Không tự động áp dụng threshold tối ưu từ benchmark vì sẽ overfit external holdout.",
        "",
        "## 20. Phân tích latency",
        "",
        markdown_table(
            list(latency_rows[0].keys()) if latency_rows else [],
            [list(row.values()) for row in latency_rows],
        ),
        "",
        "## 21. So sánh với benchmark trước",
        "",
        markdown_table(
            ["Benchmark", "Mode", "N", "Balanced Accuracy", "Precision", "Recall", "F1", "FPR", "FNR"],
            comparison_table,
        ),
        "",
        "Internal set trước chỉ có 12 mẫu; Rogue Security có 5.000 mẫu khi chạy full nên đáng tin hơn để đánh giá tổng quát hóa direct jailbreak/benign. Dataset này không thay thế indirect document benchmark.",
        "",
        "## 22. Hạn chế",
        "",
        "- Đây không phải indirect benchmark.",
        "- Không đánh giá upload PDF/DOCX/TXT, extraction, chunk location, OCR, hoặc mismatch giữa user task và document.",
        "- Nếu dataset có duplicate với train/test local, phải đọc cả full metrics và deduplicated metrics.",
        "",
        "## 23. Kết luận",
        "",
        (
            f"Đã xử lý đủ `{len(samples)}` mẫu cho mỗi mode. Data leakage hiện là exact="
            f"`{leakage_summary.get('exact_duplicate_count')}`, normalized=`{leakage_summary.get('normalized_duplicate_count')}`; "
            "vì vậy không gọi kết quả này là external holdout hoàn toàn độc lập nếu duplicate đáng kể."
        ),
        (
            f"RoBERTa-only đạt balanced accuracy `{format_percent(metrics_by_mode.get('roberta_only', {}).get('balanced_accuracy'))}`, "
            f"recall `{format_percent(metrics_by_mode.get('roberta_only', {}).get('recall'))}`, "
            f"FPR `{format_percent(metrics_by_mode.get('roberta_only', {}).get('false_positive_rate'))}`. "
            "Đây là năng lực checkpoint thô, không có rule/guard che lỗi."
        ),
        (
            f"Rule + RoBERTa đạt balanced accuracy `{format_percent(metrics_by_mode.get('rule_roberta', {}).get('balanced_accuracy'))}`. "
            "Nếu chỉ nhỉnh hơn RoBERTa-only, rule đang đóng góp nhỏ và cần đọc `rule_contribution.csv` để tránh match quá rộng."
        ),
        (
            f"Full runtime đạt precision `{format_percent(metrics_by_mode.get('full_runtime', {}).get('precision'))}` nhưng recall "
            f"`{format_percent(metrics_by_mode.get('full_runtime', {}).get('recall'))}`. Nếu recall thấp hơn RoBERTa-only, "
            "kết luận đúng là policy/intent guard giảm false positive nhưng có nguy cơ bypass attack."
        ),
        "",
        "## 24. Khuyến nghị retrain/calibration",
        "",
        "Không retrain hoặc calibration từ benchmark này trong cùng lần chạy.",
        "Nếu RoBERTa-only chỉ đạt mức trung bình hoặc bỏ sót nhiều long-context attack, nên lên kế hoạch retrain/evaluate lại bằng tập tách biệt, không dùng Rogue Security benchmark này làm train set.",
        "Nếu score distribution cho thấy nhiều benign/jailbreak bị overconfident sai, có thể cần calibration mới trên validation set độc lập; không fit calibrator trên benchmark này.",
        "",
        "## 25. Đường dẫn file kết quả",
        "",
        f"- Dataset profile: `{output_dir / 'dataset_profile.md'}`",
        f"- Data leakage check: `{output_dir / 'data_leakage_check.md'}`",
        f"- Raw per-sample results: `{output_dir / 'rogue_security_results.csv'}`",
        f"- Metrics summary: `{output_dir / 'metrics_summary.csv'}`",
        f"- False positives: `{output_dir / 'false_positives.csv'}`",
        f"- False negatives: `{output_dir / 'false_negatives.csv'}`",
        f"- Rule contribution: `{output_dir / 'rule_contribution.csv'}`",
        f"- Intent guard audit: `{output_dir / 'intent_guard_audit.csv'}`",
        f"- Policy decision analysis: `{output_dir / 'policy_decision_analysis.csv'}`",
        f"- Threshold sweep: `{output_dir / 'threshold_sweep.csv'}`",
        f"- ROC curve: `{output_dir / 'roc_curve.csv'}`",
        f"- PR curve: `{output_dir / 'pr_curve.csv'}`",
        f"- Metrics by length: `{output_dir / 'metrics_by_length.csv'}`",
        f"- Truncation cases: `{output_dir / 'truncation_cases.csv'}`",
        f"- Latency summary: `{output_dir / 'latency_summary.csv'}`",
        f"- Bản tóm tắt giảng viên: `{teacher_summary_path}`",
    ]
    report_text = "\n".join(lines)
    report_path.write_text(report_text, encoding="utf-8")
    full_report_path.write_text(report_text, encoding="utf-8")
    teacher_lines = [
        "# Tóm tắt trình bày giảng viên - Rogue Security Benchmark",
        "",
        "## Dataset là gì?",
        "",
        f"Rogue Security `prompt-injections-benchmark` là external benchmark gồm `{len(samples)}` mẫu direct prompt với hai nhãn `benign` và `jailbreak`. Dataset này không đánh giá indirect document injection.",
        "",
        "## Vì sao chọn benchmark này?",
        "",
        "Dataset có quy mô lớn hơn nhiều so với regression set nội bộ, nên phù hợp hơn để đánh giá generalization cho direct jailbreak detection.",
        "",
        "## Ba mode đánh giá",
        "",
        "- RoBERTa-only: đo checkpoint thô, tắt rule/intent guard/calibration/context/fusion/policy.",
        "- Rule + RoBERTa: đo rule giúp sửa FN hoặc tạo thêm FP như thế nào.",
        "- Full runtime: đo toàn hệ thống production, không gọi downstream LLM.",
        "",
        "## Kết quả chính",
        "",
        markdown_table(
            ["Mode", "N", "Balanced Accuracy", "Accuracy", "Precision", "Recall", "F1", "FPR", "FNR"],
            [
                [
                    MODE_LABELS.get(mode, mode),
                    metrics["total"],
                    format_percent(metrics["balanced_accuracy"]),
                    format_percent(metrics["accuracy"]),
                    format_percent(metrics["precision"]),
                    format_percent(metrics["recall"]),
                    format_percent(metrics["f1"]),
                    format_percent(metrics["false_positive_rate"]),
                    format_percent(metrics["false_negative_rate"]),
                ]
                for mode, metrics in sorted(metrics_by_mode.items())
            ],
        ),
        "",
        "## Diễn giải trung thực",
        "",
        "RoBERTa-only là năng lực model thật. Rule + RoBERTa và Full runtime là năng lực hệ thống hybrid. Nếu Full runtime tốt hơn, cần ghi rõ đóng góp đến từ rule/policy/intent guard thay vì quy toàn bộ cho RoBERTa.",
        "",
        "## False positive / false negative",
        "",
        f"- False positives chi tiết: `{output_dir / 'false_positives.csv'}`",
        f"- False negatives chi tiết: `{output_dir / 'false_negatives.csv'}`",
        "",
        "## Có cần retrain không?",
        "",
        "Chỉ quyết định sau khi đọc FP/FN. Nếu RoBERTa-only thấp rõ rệt trong khi Full runtime cao nhờ rule/guard, nên cân nhắc retrain hoặc bổ sung dữ liệu hard negative thay vì chỉ tăng threshold.",
        "",
        "## Có cần calibration mới không?",
        "",
        "Không áp dụng calibration từ benchmark này trong cùng lần chạy. Nếu score distribution cho thấy raw probability lệch mạnh, có thể lập kế hoạch calibration riêng trên validation set tách biệt.",
        "",
        "## Kết luận",
        "",
        "Benchmark này đáng tin hơn internal regression set nhỏ cho direct jailbreak detection, nhưng không thay thế benchmark indirect document injection.",
    ]
    teacher_summary_path.write_text("\n".join(teacher_lines), encoding="utf-8")


def write_access_error_reports(error: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "rogue_security_results.csv", [], fieldnames=RESULT_FIELDNAMES)
    write_csv(
        output_dir / "metrics_summary.csv",
        [],
        fieldnames=[
            "mode",
            "mode_label",
            "total",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "specificity",
            "false_positive_rate",
            "false_negative_rate",
            "negative_predictive_value",
            "matthews_correlation_coefficient",
            "roc_auc",
            "pr_auc",
            "tp",
            "tn",
            "fp",
            "fn",
        ],
    )
    write_csv(output_dir / "threshold_sweep.csv", [], fieldnames=["threshold", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "specificity", "fpr", "fnr", "tp", "tn", "fp", "fn"])
    write_csv(output_dir / "false_positives.csv", [], fieldnames=[*RESULT_FIELDNAMES, "false_positive_group"])
    write_csv(output_dir / "false_negatives.csv", [], fieldnames=[*RESULT_FIELDNAMES, "false_negative_group"])
    write_csv(output_dir / "false_positive_categories.csv", [], fieldnames=["category", "false_positive_count", "percentage", "mean_score", "median_score", "top_triggered_rules"])
    write_csv(output_dir / "false_negative_categories.csv", [], fieldnames=["category", "false_negative_count", "percentage", "mean_score", "median_score", "truncated_count", "near_threshold_count"])
    write_csv(output_dir / "intent_guard_audit.csv", [], fieldnames=["sample_id", "expected_label", "raw_score", "adjusted_score", "score_reduction", "intent_category", "intent_guard_applied", "intent_guard_reason", "predicted_before_guard", "predicted_after_guard", "correct_before_guard", "correct_after_guard", "final_decision", "decision_source", "triggered_policy"])
    write_csv(output_dir / "intent_guard_summary.csv", [], fieldnames=["benign_corrected_by_guard", "jailbreak_lowered_wrongly_by_guard", "recall_before_guard", "recall_after_guard", "fpr_before_guard", "fpr_after_guard"])
    write_csv(output_dir / "intent_guard_reduction_by_category.csv", [], fieldnames=["intent_category", "count", "guard_applied_count", "mean_score_reduction", "jailbreak_lowered_wrongly"])
    write_csv(output_dir / "rule_contribution.csv", [], fieldnames=["rule_id", "trigger_count", "true_positive_count", "false_positive_count", "precision_when_triggered", "corrected_false_negatives", "introduced_false_positives", "mean_rule_score", "highest_severity"])
    write_csv(output_dir / "policy_decision_analysis.csv", [], fieldnames=["analysis_type", "key", "count", "percentage", "benign_count", "jailbreak_count", "benign_rate_within_decision", "jailbreak_rate_within_decision"])
    write_csv(output_dir / "metrics_by_length.csv", [], fieldnames=["mode", "length_bin", "total", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "specificity", "false_positive_rate", "false_negative_rate", "tp", "tn", "fp", "fn", "truncation_rate"])
    write_csv(output_dir / "latency_summary.csv", [], fieldnames=["mode", "mean_latency_ms", "median_latency_ms", "p90_latency_ms", "p95_latency_ms", "p99_latency_ms", "min_latency_ms", "max_latency_ms", "throughput_samples_per_second", "total_runtime_seconds", "cold_start_latency_ms", "warm_inference_latency_ms", "device", "batch_size", "workers", "machine", "model_cache_state"])
    write_csv(output_dir / "truncation_cases.csv", [], fieldnames=["sample_id", "mode", "expected_label", "predicted_label", "original_token_count", "used_token_count", "was_truncated", "attack_position_estimate", "correct", "raw_roberta_score", "threshold"])
    profile = [
        "# Rogue Security dataset profile",
        "",
        f"- Dataset: `{DATASET_NAME}`",
        f"- URL: {DATASET_URL}",
        "- Total samples: chưa tải được do dataset gated.",
        "- Label distribution: chưa xác minh local được.",
        "- License trên dataset card: `cc-by-nc-4.0`.",
        "- Access condition: cần accept điều kiện Hugging Face và đăng nhập token.",
        "",
        f"Load error: `{error}`",
    ]
    (output_dir / "dataset_profile.md").write_text("\n".join(profile), encoding="utf-8")
    leakage = [
        "# Rogue Security data leakage check",
        "",
        "Chưa thể chạy exact/normalized/near-duplicate check vì dataset chưa tải được.",
        "",
        f"- Dataset: `{DATASET_NAME}`",
        f"- Load error: `{error}`",
        "",
        "Sau khi đăng nhập Hugging Face và accept điều kiện dataset, chạy lại script để tạo báo cáo leakage đầy đủ.",
    ]
    (output_dir / "data_leakage_check.md").write_text("\n".join(leakage), encoding="utf-8")
    write_csv(output_dir / "data_leakage_duplicates.csv", [], fieldnames=["sample_id", "source_file", "source_split", "field", "method", "match_type"])
    write_final_report(
        samples=[],
        rows=[],
        output_dir=output_dir,
        leakage_summary={},
        warmup_latencies={},
        access_error=error,
    )


def run_warmup(samples: list[BenchmarkSample], modes: list[str], *, use_cuda: bool) -> dict[str, float]:
    if not samples:
        return {}
    text = samples[0].text
    warmup_latencies: dict[str, float] = {}
    for mode in modes:
        started = time.perf_counter()
        safe_evaluate(mode, text, use_cuda=use_cuda)
        warmup_latencies[mode] = round((time.perf_counter() - started) * 1000, 3)
    return warmup_latencies


def run_benchmark(
    *,
    samples: list[BenchmarkSample],
    modes: list[str],
    output_dir: Path,
    batch_size: int,
    use_cuda: bool,
    resume: bool,
    warmup: bool,
) -> dict[str, float]:
    results_path = output_dir / "rogue_security_results.csv"
    existing = load_existing_keys(results_path) if resume else set()
    warmup_latencies = run_warmup(samples, modes, use_cuda=use_cuda) if warmup else {}
    total = len(samples) * len(modes)
    processed = 0
    for mode in modes:
        batched = [samples[index : index + batch_size] for index in range(0, len(samples), batch_size)]
        for batch in _maybe_tqdm(batched, total=len(batched), desc=mode):
            rows_to_write = []
            for sample in batch:
                key = (mode, sample.sample_id)
                if key in existing:
                    continue
                result = safe_evaluate(mode, sample.text, use_cuda=use_cuda)
                row = row_from_result(sample, mode, result)
                row["internal_duplicate"] = int(sample.internal_duplicate)
                row["leakage_duplicate"] = int(sample.leakage_duplicate)
                rows_to_write.append(row)
                existing.add(key)
                processed += 1
            if rows_to_write:
                append_result_rows(results_path, rows_to_write)
        if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
            print(f"Finished mode {mode}. Processed new rows: {processed}/{total}")
    return warmup_latencies


def generate_analysis_outputs(
    output_dir: Path,
    warmup_latencies: dict[str, float],
    *,
    use_cuda: bool,
    batch_size: int,
) -> list[dict[str, Any]]:
    rows = read_results_csv(output_dir / "rogue_security_results.csv")
    write_metrics_outputs(rows, output_dir)
    write_threshold_outputs(rows, output_dir)
    write_error_analysis(rows, output_dir)
    write_intent_guard_audit(rows, output_dir)
    write_rule_contribution(rows, output_dir)
    write_policy_decision_analysis(rows, output_dir)
    write_truncation_cases(rows, output_dir)
    write_length_and_latency(rows, output_dir, warmup_latencies, use_cuda=use_cuda, batch_size=batch_size)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", choices=sorted(MODE_LABELS), default=list(MODE_LABELS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    return parser.parse_args()


def archive_existing_outputs(output_dir: Path, filenames: list[str]) -> Path | None:
    existing = [output_dir / filename for filename in filenames if (output_dir / filename).exists()]
    if not existing:
        return None
    archive_dir = output_dir / "archive" / time.strftime("%Y%m%d_%H%M%S")
    archive_dir.mkdir(parents=True, exist_ok=True)
    for path in existing:
        path.replace(archive_dir / path.name)
    return archive_dir


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.force:
        archived = archive_existing_outputs(
            output_dir,
            [
            "rogue_security_results.csv",
            "metrics_summary.csv",
            "threshold_sweep.csv",
            "false_positives.csv",
            "false_negatives.csv",
            "intent_guard_audit.csv",
            "intent_guard_summary.csv",
            "rule_contribution.csv",
            "metrics_by_length.csv",
            "latency_summary.csv",
            "roc_curve.csv",
            "pr_curve.csv",
            "score_distribution.csv",
            "dataset_profile.md",
            "data_leakage_check.md",
            "data_leakage_duplicates.csv",
            "policy_decision_analysis.csv",
            "truncation_cases.csv",
            "false_positive_categories.csv",
            "false_negative_categories.csv",
            "intent_guard_reduction_by_category.csv",
            "BAO_CAO_ROGUE_SECURITY_BENCHMARK.md",
            "BAO_CAO_ROGUE_SECURITY_BENCHMARK_DAY_DU.md",
            "TOM_TAT_TRINH_BAY_GIANG_VIEN.md",
            ],
        )
        if archived:
            print(f"Archived previous outputs to: {archived}")

    try:
        samples, schema = load_rogue_security_dataset(max_samples=args.max_samples)
    except Exception as exc:
        error = f"{exc.__class__.__name__}: {exc}"
        write_access_error_reports(error, output_dir)
        print(f"Dataset access failed: {error}", file=sys.stderr)
        print(f"Wrote access-error reports under {output_dir}", file=sys.stderr)
        return 2

    if args.shuffle:
        random.Random(args.seed).shuffle(samples)
    print(f"Loaded {len(samples)} samples. Writing dataset profile...", flush=True)
    profile_samples(samples, schema, output_path=output_dir / "dataset_profile.md")
    print("Running data leakage check...", flush=True)
    samples, leakage_summary = leakage_check(
        samples,
        output_path=output_dir / "data_leakage_check.md",
        duplicates_path=output_dir / "data_leakage_duplicates.csv",
    )
    print(
        "Leakage check done: "
        f"exact={leakage_summary.get('exact_duplicate_count')}, "
        f"normalized={leakage_summary.get('normalized_duplicate_count')}, "
        f"near={leakage_summary.get('near_duplicate_count')}",
        flush=True,
    )
    use_cuda = device_to_use_cuda(args.device)
    print(f"Starting benchmark modes={args.modes}, use_cuda={use_cuda}, batch_size={max(1, int(args.batch_size))}", flush=True)
    warmup_latencies = run_benchmark(
        samples=samples,
        modes=args.modes,
        output_dir=output_dir,
        batch_size=max(1, int(args.batch_size)),
        use_cuda=use_cuda,
        resume=not args.no_resume,
        warmup=not args.no_warmup,
    )
    rows = generate_analysis_outputs(output_dir, warmup_latencies, use_cuda=use_cuda, batch_size=max(1, int(args.batch_size)))
    write_final_report(
        samples=samples,
        rows=rows,
        output_dir=output_dir,
        leakage_summary=leakage_summary,
        warmup_latencies=warmup_latencies,
    )
    print(f"Wrote results: {output_dir / 'rogue_security_results.csv'}")
    print(f"Wrote report: {output_dir / 'BAO_CAO_ROGUE_SECURITY_BENCHMARK.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
