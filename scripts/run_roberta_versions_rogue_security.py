"""Compare every RoBERTa checkpoint on rogue-security/prompt-injections-benchmark.

This is a diagnostic evaluation script only. It does not train, tune,
calibrate, or modify production checkpoints. Each checkpoint is evaluated in
RoBERTa-only mode: rule detector off, intent guard off, runtime calibration off.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_rogue_security_benchmark import (  # noqa: E402
    DATASET_NAME,
    DATASET_URL,
    INFERENCE_MAX_LENGTH,
    BenchmarkSample,
    archive_existing_outputs,
    compute_binary_metrics,
    device_to_use_cuda,
    format_percent,
    leakage_check,
    load_rogue_security_dataset,
    markdown_table,
    percentile,
    profile_samples,
    safe_preview,
    write_csv,
)
from src.transformer_utils import (  # noqa: E402
    clear_transformer_runtime_cache,
    is_finetuned_transformer_checkpoint,
    predict_transformer,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "roberta_versions_rogue_security"
TRANSFORMER_DIR = PROJECT_ROOT / "models" / "transformers"
RESULTS_FILENAME = "roberta_versions_results.csv"
SUMMARY_FILENAME = "roberta_versions_summary.csv"
REPORT_FILENAME = "BAO_CAO_SO_SANH_ROBERTA_VERSIONS.md"

RESULT_FIELDNAMES = [
    "dataset_name",
    "dataset_url",
    "split",
    "sample_id",
    "text_preview",
    "expected_label",
    "expected_label_name",
    "model_version",
    "model_dir",
    "base_model",
    "train_dataset",
    "trained_at",
    "train_rows",
    "validation_rows",
    "test_rows",
    "label_mapping",
    "current_eval_threshold",
    "runtime_warn_threshold",
    "runtime_block_threshold",
    "threshold_source",
    "predicted_label",
    "predicted_label_name",
    "argmax_label",
    "argmax_label_name",
    "correct",
    "raw_roberta_score",
    "softmax_safe",
    "softmax_injection",
    "logit_safe",
    "logit_injection",
    "score_used",
    "runtime_device",
    "calibration_enabled",
    "intent_guard_enabled",
    "token_count",
    "was_truncated",
    "latency_ms",
    "error",
]


ARCHIVE_FILENAMES = [
    RESULTS_FILENAME,
    SUMMARY_FILENAME,
    "score_distribution.csv",
    "latency_summary.csv",
    "threshold_sweep.csv",
    "false_positives.csv",
    "false_negatives.csv",
    "dataset_profile.md",
    "data_leakage_check.md",
    "data_leakage_duplicates.csv",
    REPORT_FILENAME,
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        return payload if isinstance(payload, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _first_present(payload: dict[str, Any], keys: list[str], default: Any = "") -> Any:
    for key in keys:
        value = payload.get(key)
        if value not in {None, ""}:
            return value
    return default


def _float_or_none(value: Any) -> float | None:
    try:
        if value in {None, ""}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_blank(value: Any) -> int | str:
    try:
        if value in {None, ""}:
            return ""
        return int(value)
    except (TypeError, ValueError):
        return ""


def _metric_value(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    if value is None:
        return 0.0
    return float(value)


def _format_float(value: Any, digits: int = 6) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def discover_roberta_checkpoints(model_names: list[str] | None = None) -> list[Path]:
    if model_names:
        candidates = [TRANSFORMER_DIR / name for name in model_names]
    else:
        candidates = sorted(
            [
                item
                for item in TRANSFORMER_DIR.iterdir()
                if item.is_dir() and (item.name == "roberta" or item.name.startswith("roberta_"))
            ],
            key=lambda item: (
                0 if item.name == "roberta" else 1,
                item.name,
            ),
        )
    return [path for path in candidates if is_finetuned_transformer_checkpoint(path)]


def checkpoint_metadata(model_dir: Path) -> dict[str, Any]:
    training_metadata = _read_json(model_dir / "training_metadata.json")
    metrics_json = _read_json(model_dir / "metrics.json")
    config = _read_json(model_dir / "config.json")
    threshold_payload = (
        training_metadata.get("thresholds")
        or metrics_json.get("thresholds")
        or training_metadata.get("metrics", {}).get("thresholds")
        or {}
    )
    if not threshold_payload and "evaluation_threshold" in training_metadata.get("metrics", {}):
        metric_values = training_metadata.get("metrics", {})
        threshold_payload = {
            "evaluation_threshold": metric_values.get("evaluation_threshold"),
            "runtime_warn_threshold": metric_values.get("runtime_warn_threshold"),
            "runtime_block_threshold": metric_values.get("runtime_block_threshold"),
        }
    return {
        "model_version": model_dir.name,
        "model_dir": str(model_dir),
        "base_model": _first_present(training_metadata, ["base_model"], config.get("model_type", "")),
        "train_dataset": _first_present(
            training_metadata,
            ["dataset_name", "dataset", "train_dataset", "dataset_path", "dataset_source"],
            "",
        ),
        "trained_at": _first_present(training_metadata, ["trained_at", "created_at", "created"], ""),
        "train_rows": _int_or_blank(_first_present(training_metadata, ["train_rows", "train_size"], "")),
        "validation_rows": _int_or_blank(
            _first_present(training_metadata, ["validation_rows", "validation_size", "val_rows"], "")
        ),
        "test_rows": _int_or_blank(_first_present(training_metadata, ["test_rows", "test_size"], "")),
        "label_mapping": json.dumps(
            training_metadata.get("label_mapping") or config.get("id2label") or {},
            ensure_ascii=False,
            sort_keys=True,
        ),
        "metadata_eval_threshold": _float_or_none(threshold_payload.get("evaluation_threshold")),
        "metadata_warn_threshold": _float_or_none(threshold_payload.get("runtime_warn_threshold")),
        "metadata_block_threshold": _float_or_none(threshold_payload.get("runtime_block_threshold")),
    }


def label_name(label: Any) -> str:
    try:
        return "jailbreak" if int(label) == 1 else "benign"
    except (TypeError, ValueError):
        return "unknown"


def append_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def read_existing_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def evaluate_one(
    sample: BenchmarkSample,
    *,
    model_dir: Path,
    metadata: dict[str, Any],
    use_cuda: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = predict_transformer(
            text=sample.text,
            model_path=model_dir,
            model_name=model_dir.name,
            max_length=INFERENCE_MAX_LENGTH,
            use_cuda=use_cuda,
            use_intent_guard=False,
            use_runtime_calibration=False,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        raw_score = float(result.get("raw_risk_score", result.get("risk_score", 0.0)))
        threshold_used = result.get("threshold_used") or {}
        eval_threshold = float(threshold_used.get("evaluation", 0.5))
        predicted_label = 1 if raw_score >= eval_threshold else 0
        raw_probabilities = result.get("raw_probabilities") or result.get("probabilities") or {}
        logits = result.get("logits") or []
        return {
            "dataset_name": DATASET_NAME,
            "dataset_url": DATASET_URL,
            "split": sample.split,
            "sample_id": sample.sample_id,
            "text_preview": safe_preview(sample.text),
            "expected_label": sample.expected_label,
            "expected_label_name": sample.expected_label_name,
            **metadata,
            "current_eval_threshold": eval_threshold,
            "runtime_warn_threshold": threshold_used.get("warn", ""),
            "runtime_block_threshold": threshold_used.get("block", ""),
            "threshold_source": result.get("threshold_source", ""),
            "predicted_label": predicted_label,
            "predicted_label_name": label_name(predicted_label),
            "argmax_label": result.get("raw_predicted_label", result.get("predicted_label", "")),
            "argmax_label_name": label_name(result.get("raw_predicted_label", result.get("predicted_label", ""))),
            "correct": int(int(predicted_label) == int(sample.expected_label)),
            "raw_roberta_score": raw_score,
            "softmax_safe": raw_probabilities.get("safe", ""),
            "softmax_injection": raw_probabilities.get("injection", ""),
            "logit_safe": logits[0] if len(logits) >= 1 else "",
            "logit_injection": logits[1] if len(logits) >= 2 else "",
            "score_used": "raw_softmax_probability",
            "runtime_device": result.get("runtime_device", ""),
            "calibration_enabled": False,
            "intent_guard_enabled": False,
            "token_count": sample.token_count,
            "was_truncated": int(bool(sample.was_truncated)),
            "latency_ms": latency_ms,
            "error": "",
        }
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return {
            "dataset_name": DATASET_NAME,
            "dataset_url": DATASET_URL,
            "split": sample.split,
            "sample_id": sample.sample_id,
            "text_preview": safe_preview(sample.text),
            "expected_label": sample.expected_label,
            "expected_label_name": sample.expected_label_name,
            **metadata,
            "current_eval_threshold": metadata.get("metadata_eval_threshold", ""),
            "runtime_warn_threshold": metadata.get("metadata_warn_threshold", ""),
            "runtime_block_threshold": metadata.get("metadata_block_threshold", ""),
            "threshold_source": "",
            "predicted_label": "",
            "predicted_label_name": "",
            "argmax_label": "",
            "argmax_label_name": "",
            "correct": "",
            "raw_roberta_score": "",
            "softmax_safe": "",
            "softmax_injection": "",
            "logit_safe": "",
            "logit_injection": "",
            "score_used": "raw_softmax_probability",
            "runtime_device": "",
            "calibration_enabled": False,
            "intent_guard_enabled": False,
            "token_count": sample.token_count,
            "was_truncated": int(bool(sample.was_truncated)),
            "latency_ms": latency_ms,
            "error": f"{exc.__class__.__name__}: {exc}",
        }


def run_model(
    *,
    samples: list[BenchmarkSample],
    model_dir: Path,
    output_path: Path,
    use_cuda: bool,
    resume: bool,
    progress_every: int,
) -> list[dict[str, Any]]:
    existing_rows = read_existing_results(output_path) if resume else []
    done = {
        (row.get("model_version"), row.get("sample_id"))
        for row in existing_rows
        if row.get("model_version") == model_dir.name and row.get("error", "") == ""
    }
    model_rows = [row for row in existing_rows if row.get("model_version") == model_dir.name]
    metadata = checkpoint_metadata(model_dir)
    pending = [sample for sample in samples if (model_dir.name, sample.sample_id) not in done]
    print(
        f"[{model_dir.name}] samples={len(samples)}, done={len(done)}, pending={len(pending)}, "
        f"checkpoint={model_dir}",
        flush=True,
    )
    if not pending:
        return model_rows

    batch: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, sample in enumerate(pending, start=1):
        row = evaluate_one(sample, model_dir=model_dir, metadata=metadata, use_cuda=use_cuda)
        batch.append(row)
        model_rows.append(row)
        if len(batch) >= 25:
            append_rows(output_path, batch, RESULT_FIELDNAMES)
            batch = []
        if index == 1 or index % progress_every == 0 or index == len(pending):
            elapsed = max(time.perf_counter() - started, 0.001)
            rate = index / elapsed
            remaining = (len(pending) - index) / rate if rate > 0 else 0.0
            print(
                f"[{model_dir.name}] {index}/{len(pending)} "
                f"({rate:.1f} sample/s, eta {remaining/60:.1f} min)",
                flush=True,
            )
    if batch:
        append_rows(output_path, batch, RESULT_FIELDNAMES)
    clear_transformer_runtime_cache()
    return model_rows


def valid_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if not row.get("error")]


def summarize_metrics(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    metrics_by_model: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in valid_rows(rows):
        grouped[str(row.get("model_version", ""))].append(row)
    for model_version, model_rows in sorted(grouped.items()):
        metrics = compute_binary_metrics(model_rows, score_field="raw_roberta_score")
        metrics_by_model[model_version] = metrics
        thresholds = sorted({_float_or_none(row.get("current_eval_threshold")) for row in model_rows})
        threshold = next((value for value in thresholds if value is not None), None)
        sources = sorted({str(row.get("threshold_source", "")) for row in model_rows if row.get("threshold_source")})
        metadata = checkpoint_metadata(TRANSFORMER_DIR / model_version)
        latencies = [float(row.get("latency_ms") or 0.0) for row in model_rows]
        summary_rows.append(
            {
                "model_version": model_version,
                "checkpoint_path": str(TRANSFORMER_DIR / model_version),
                "base_model": metadata.get("base_model", ""),
                "train_dataset": metadata.get("train_dataset", ""),
                "trained_at": metadata.get("trained_at", ""),
                "train_rows": metadata.get("train_rows", ""),
                "validation_rows": metadata.get("validation_rows", ""),
                "test_rows": metadata.get("test_rows", ""),
                "threshold": threshold if threshold is not None else "",
                "threshold_source": "; ".join(sources),
                "total": metrics["total"],
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "specificity": metrics["specificity"],
                "false_positive_rate": metrics["false_positive_rate"],
                "false_negative_rate": metrics["false_negative_rate"],
                "negative_predictive_value": metrics["negative_predictive_value"],
                "matthews_correlation_coefficient": metrics["matthews_correlation_coefficient"],
                "roc_auc": "" if metrics.get("roc_auc") is None else metrics["roc_auc"],
                "pr_auc": "" if metrics.get("pr_auc") is None else metrics["pr_auc"],
                "tp": metrics["tp"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "avg_latency_ms": statistics.mean(latencies) if latencies else 0.0,
                "median_latency_ms": statistics.median(latencies) if latencies else 0.0,
                "p95_latency_ms": percentile(latencies, 95) if latencies else 0.0,
            }
        )
    return summary_rows, metrics_by_model


def write_score_distribution(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in valid_rows(rows):
        score = _float_or_none(row.get("raw_roberta_score"))
        if score is None:
            continue
        grouped[(str(row.get("model_version", "")), str(row.get("expected_label_name", "")))].append(score)
    for (model_version, label), scores in sorted(grouped.items()):
        output_rows.append(
            {
                "model_version": model_version,
                "expected_label_name": label,
                "count": len(scores),
                "min": min(scores),
                "p05": percentile(scores, 5),
                "p25": percentile(scores, 25),
                "mean": statistics.mean(scores),
                "median": statistics.median(scores),
                "p75": percentile(scores, 75),
                "p95": percentile(scores, 95),
                "p99": percentile(scores, 99),
                "max": max(scores),
            }
        )
    write_csv(output_dir / "score_distribution.csv", output_rows)


def write_latency_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in valid_rows(rows):
        grouped[str(row.get("model_version", ""))].append(float(row.get("latency_ms") or 0.0))
    for model_version, latencies in sorted(grouped.items()):
        output_rows.append(
            {
                "model_version": model_version,
                "count": len(latencies),
                "total_seconds": sum(latencies) / 1000.0,
                "avg_latency_ms": statistics.mean(latencies),
                "median_latency_ms": statistics.median(latencies),
                "p95_latency_ms": percentile(latencies, 95),
                "min_latency_ms": min(latencies),
                "max_latency_ms": max(latencies),
            }
        )
    write_csv(output_dir / "latency_summary.csv", output_rows)


def metrics_at_threshold(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    adjusted: list[dict[str, Any]] = []
    for row in valid_rows(rows):
        score = _float_or_none(row.get("raw_roberta_score"))
        if score is None:
            continue
        copied = dict(row)
        copied["predicted_label"] = 1 if score >= threshold else 0
        adjusted.append(copied)
    return compute_binary_metrics(adjusted, score_field="raw_roberta_score")


def write_threshold_sweep(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    base_thresholds = [
        0.0,
        0.001,
        0.005,
        0.01,
        0.02,
        0.05,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.95,
        0.99,
        0.999,
        0.9999,
        0.99999,
        1.0,
    ]
    for row in valid_rows(rows):
        grouped[str(row.get("model_version", ""))].append(row)
    for model_version, model_rows in sorted(grouped.items()):
        current_thresholds = {
            value
            for value in (_float_or_none(row.get("current_eval_threshold")) for row in model_rows)
            if value is not None
        }
        thresholds = sorted(set(base_thresholds) | current_thresholds)
        best_balanced = {"threshold": None, "metrics": None}
        best_f1 = {"threshold": None, "metrics": None}
        for threshold in thresholds:
            metrics = metrics_at_threshold(model_rows, threshold)
            if (
                best_balanced["metrics"] is None
                or metrics["balanced_accuracy"] > best_balanced["metrics"]["balanced_accuracy"]
            ):
                best_balanced = {"threshold": threshold, "metrics": metrics}
            if best_f1["metrics"] is None or metrics["f1"] > best_f1["metrics"]["f1"]:
                best_f1 = {"threshold": threshold, "metrics": metrics}
            output_rows.append(
                {
                    "model_version": model_version,
                    "threshold": threshold,
                    "accuracy": metrics["accuracy"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "specificity": metrics["specificity"],
                    "false_positive_rate": metrics["false_positive_rate"],
                    "false_negative_rate": metrics["false_negative_rate"],
                    "tp": metrics["tp"],
                    "tn": metrics["tn"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"],
                    "is_current_threshold": int(threshold in current_thresholds),
                    "is_best_balanced_on_this_dataset": 0,
                    "is_best_f1_on_this_dataset": 0,
                }
            )
        for output_row in output_rows:
            if output_row["model_version"] != model_version:
                continue
            if output_row["threshold"] == best_balanced["threshold"]:
                output_row["is_best_balanced_on_this_dataset"] = 1
            if output_row["threshold"] == best_f1["threshold"]:
                output_row["is_best_f1_on_this_dataset"] = 1
    write_csv(output_dir / "threshold_sweep.csv", output_rows)


def write_error_cases(rows: list[dict[str, Any]], output_dir: Path) -> None:
    false_positives: list[dict[str, Any]] = []
    false_negatives: list[dict[str, Any]] = []
    for row in valid_rows(rows):
        expected = str(row.get("expected_label"))
        predicted = str(row.get("predicted_label"))
        item = {
            "model_version": row.get("model_version"),
            "sample_id": row.get("sample_id"),
            "expected_label_name": row.get("expected_label_name"),
            "predicted_label_name": row.get("predicted_label_name"),
            "raw_roberta_score": row.get("raw_roberta_score"),
            "threshold": row.get("current_eval_threshold"),
            "text_preview": row.get("text_preview"),
        }
        if expected == "0" and predicted == "1":
            false_positives.append(item)
        elif expected == "1" and predicted == "0":
            false_negatives.append(item)
    false_positives.sort(key=lambda row: float(row.get("raw_roberta_score") or 0.0), reverse=True)
    false_negatives.sort(key=lambda row: float(row.get("raw_roberta_score") or 0.0))
    write_csv(output_dir / "false_positives.csv", false_positives)
    write_csv(output_dir / "false_negatives.csv", false_negatives)


def write_final_report(
    *,
    samples: list[BenchmarkSample],
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
    leakage_summary: dict[str, Any] | None,
    elapsed_seconds: float,
    use_cuda: bool,
    max_samples: int | None,
) -> None:
    label_counts = Counter(sample.expected_label_name for sample in samples)
    generated_at = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    model_rows = [
        [
            row["model_version"],
            row.get("base_model", ""),
            row.get("train_dataset", ""),
            row.get("trained_at", ""),
            row.get("threshold", ""),
            row.get("threshold_source", ""),
            row.get("train_rows", ""),
            row.get("validation_rows", ""),
            row.get("test_rows", ""),
        ]
        for row in summary_rows
    ]
    metric_rows = [
        [
            row["model_version"],
            format_percent(row["accuracy"]),
            format_percent(row["balanced_accuracy"]),
            format_percent(row["precision"]),
            format_percent(row["recall"]),
            format_percent(row["f1"]),
            format_percent(row["specificity"]),
            format_percent(row["false_positive_rate"]),
            format_percent(row["false_negative_rate"]),
            format_percent(row["roc_auc"]) if row.get("roc_auc") != "" else "n/a",
            format_percent(row["pr_auc"]) if row.get("pr_auc") != "" else "n/a",
            row["tp"],
            row["tn"],
            row["fp"],
            row["fn"],
            _format_float(row["avg_latency_ms"], 2),
        ]
        for row in summary_rows
    ]
    best_balanced = max(summary_rows, key=lambda row: _metric_value(row, "balanced_accuracy"), default=None)
    best_f1 = max(summary_rows, key=lambda row: _metric_value(row, "f1"), default=None)
    best_auc = max(summary_rows, key=lambda row: _metric_value(row, "roc_auc"), default=None)
    lowest_fpr = min(summary_rows, key=lambda row: _metric_value(row, "false_positive_rate"), default=None)

    leakage_lines = []
    if leakage_summary:
        leakage_lines = [
            f"- Exact duplicate unique samples: {leakage_summary.get('exact_duplicate_count', 'n/a')}",
            f"- Normalized duplicate unique samples: {leakage_summary.get('normalized_duplicate_count', 'n/a')}",
            f"- Near duplicate unique samples: {leakage_summary.get('near_duplicate_count', 'n/a')}",
            "- Ghi chú: nếu có trùng lặp với dữ liệu trong workspace, không xem đây là holdout độc lập tuyệt đối.",
        ]
    else:
        leakage_lines = ["- Không chạy leakage check trong lần này."]

    lines = [
        "# Báo cáo so sánh các version RoBERTa",
        "",
        f"- Thời gian chạy: {generated_at}",
        f"- Dataset: `{DATASET_NAME}` ({DATASET_URL})",
        f"- Split: `test`",
        f"- Số mẫu: {len(samples)}"
        + (f" (giới hạn smoke test: {max_samples})" if max_samples else ""),
        f"- Phân bố label: benign={label_counts.get('benign', 0)}, jailbreak={label_counts.get('jailbreak', 0)}",
        f"- Thiết bị: {'CUDA nếu khả dụng' if use_cuda else 'CPU'}",
        f"- Thời gian lần chạy script gần nhất: {elapsed_seconds / 60:.2f} phút",
        "",
        "## Nguyên tắc đánh giá",
        "",
        "- Chỉ chạy RoBERTa-only cho từng checkpoint.",
        "- Rule-based detector: OFF.",
        "- Intent guard: OFF.",
        "- Runtime calibration: OFF.",
        "- Policy/fusion/context runtime: OFF.",
        "- Prediction được tính bằng raw injection score so với `evaluation_threshold` hiện tại của checkpoint.",
        "- Threshold sweep trong file CSV chỉ để chẩn đoán trên dataset này, không dùng để chỉnh production.",
        "- Các checkpoint `xlm_roberta_*` dùng backbone `xlm-roberta-base`; nên đọc như nhóm XLM-RoBERTa riêng khi so với `roberta-base`.",
        "",
        "## Leakage check",
        "",
        *leakage_lines,
        "",
        "## Checkpoint được đánh giá",
        "",
        markdown_table(
            [
                "Model",
                "Base",
                "Training dataset",
                "Trained at",
                "Eval threshold",
                "Threshold source",
                "Train",
                "Val",
                "Test",
            ],
            model_rows,
        ),
        "",
        "## Kết quả chính",
        "",
        markdown_table(
            [
                "Model",
                "Accuracy",
                "Balanced Acc",
                "Precision",
                "Recall",
                "F1",
                "Specificity",
                "FPR",
                "FNR",
                "ROC-AUC",
                "PR-AUC",
                "TP",
                "TN",
                "FP",
                "FN",
                "Avg ms",
            ],
            metric_rows,
        ),
        "",
        "## Nhận xét nhanh",
        "",
    ]
    if best_balanced:
        lines.append(
            f"- Balanced accuracy cao nhất: `{best_balanced['model_version']}` "
            f"({format_percent(best_balanced['balanced_accuracy'])})."
        )
    if best_f1:
        lines.append(f"- F1 cao nhất: `{best_f1['model_version']}` ({format_percent(best_f1['f1'])}).")
    if best_auc and best_auc.get("roc_auc") != "":
        lines.append(f"- ROC-AUC cao nhất: `{best_auc['model_version']}` ({format_percent(best_auc['roc_auc'])}).")
    if lowest_fpr:
        lines.append(
            f"- FPR thấp nhất tại threshold hiện tại: `{lowest_fpr['model_version']}` "
            f"({format_percent(lowest_fpr['false_positive_rate'])})."
        )
    lines.extend(
        [
            "",
            "## File xuất ra",
            "",
            f"- `{output_dir / RESULTS_FILENAME}`: kết quả từng sample.",
            f"- `{output_dir / SUMMARY_FILENAME}`: bảng metric tổng hợp từng model.",
            f"- `{output_dir / 'score_distribution.csv'}`: phân bố raw score theo label.",
            f"- `{output_dir / 'threshold_sweep.csv'}`: sweep threshold chẩn đoán.",
            f"- `{output_dir / 'false_positives.csv'}` và `{output_dir / 'false_negatives.csv'}`: mẫu sai để phân tích lỗi.",
            f"- `{output_dir / 'latency_summary.csv'}`: latency từng checkpoint.",
            "",
            "## Kết luận sử dụng",
            "",
            "Báo cáo này dùng để so sánh các checkpoint RoBERTa đang có trong dự án trên cùng một bộ dữ liệu. "
            "Không dùng kết quả này để train, calibrate hoặc chọn threshold production nếu chưa có quy trình holdout sạch.",
            "",
        ]
    )
    (output_dir / REPORT_FILENAME).write_text("\n".join(lines), encoding="utf-8")


def generate_outputs(
    *,
    samples: list[BenchmarkSample],
    output_dir: Path,
    leakage_summary: dict[str, Any] | None,
    elapsed_seconds: float,
    use_cuda: bool,
    max_samples: int | None,
) -> list[dict[str, Any]]:
    rows = read_existing_results(output_dir / RESULTS_FILENAME)
    summary_rows, _metrics_by_model = summarize_metrics(rows)
    write_csv(output_dir / SUMMARY_FILENAME, summary_rows)
    write_score_distribution(rows, output_dir)
    write_latency_summary(rows, output_dir)
    write_threshold_sweep(rows, output_dir)
    write_error_cases(rows, output_dir)
    write_final_report(
        samples=samples,
        output_dir=output_dir,
        summary_rows=summary_rows,
        leakage_summary=leakage_summary,
        elapsed_seconds=elapsed_seconds,
        use_cuda=use_cuda,
        max_samples=max_samples,
    )
    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare all classic RoBERTa checkpoints on rogue-security/prompt-injections-benchmark."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", nargs="*", help="Specific RoBERTa checkpoint directory names to evaluate.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for a smoke test.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--force", action="store_true", help="Archive old outputs and start fresh.")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip already completed sample/model rows.")
    parser.add_argument("--no-leakage-check", action="store_true", help="Skip workspace leakage duplicate scan.")
    parser.add_argument("--progress-every", type=int, default=250)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.force:
        archived = archive_existing_outputs(output_dir, ARCHIVE_FILENAMES)
        if archived:
            print(f"Archived previous outputs to: {archived}", flush=True)

    checkpoints = discover_roberta_checkpoints(args.models)
    if not checkpoints:
        print(f"No valid RoBERTa checkpoints found under {TRANSFORMER_DIR}.", file=sys.stderr)
        return 2
    print("Checkpoints:", ", ".join(path.name for path in checkpoints), flush=True)

    samples, schema = load_rogue_security_dataset(max_samples=args.max_samples)
    if args.shuffle:
        random.Random(args.seed).shuffle(samples)
    print(f"Loaded {len(samples)} samples from {DATASET_NAME}.", flush=True)
    profile_samples(samples, schema, output_path=output_dir / "dataset_profile.md")

    leakage_summary: dict[str, Any] | None = None
    if not args.no_leakage_check:
        print("Running leakage check...", flush=True)
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
    started = time.perf_counter()
    output_path = output_dir / RESULTS_FILENAME
    if args.force and output_path.exists():
        output_path.unlink()
    for model_dir in checkpoints:
        run_model(
            samples=samples,
            model_dir=model_dir,
            output_path=output_path,
            use_cuda=use_cuda,
            resume=not args.no_resume,
            progress_every=max(1, int(args.progress_every)),
        )
    elapsed = time.perf_counter() - started
    summary_rows = generate_outputs(
        samples=samples,
        output_dir=output_dir,
        leakage_summary=leakage_summary,
        elapsed_seconds=elapsed,
        use_cuda=use_cuda,
        max_samples=args.max_samples,
    )
    print(f"Wrote sample results: {output_dir / RESULTS_FILENAME}", flush=True)
    print(f"Wrote summary: {output_dir / SUMMARY_FILENAME}", flush=True)
    print(f"Wrote report: {output_dir / REPORT_FILENAME}", flush=True)
    print("Metric summary:", flush=True)
    for row in summary_rows:
        print(
            f"- {row['model_version']}: bal_acc={format_percent(row['balanced_accuracy'])}, "
            f"f1={format_percent(row['f1'])}, recall={format_percent(row['recall'])}, "
            f"fpr={format_percent(row['false_positive_rate'])}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
