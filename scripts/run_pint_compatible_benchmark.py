"""Run PINT-compatible internal benchmarks for the detection runtime.

This script intentionally does not train, tune thresholds, or modify production
checkpoints. The bundled PINT example dataset is only a smoke test fixture, not
the official PINT dataset.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from xml.sax.saxutils import escape as xml_escape

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.document_runtime import analyze_uploaded_document
import src.official_runtime as official_runtime
from src.runtime_config import load_runtime_config
from src.runtime_rule_signal import detect_rule_signal
from src.transformer_utils import predict_transformer, resolve_transformer_model_dir


PINT_EXAMPLE_ENTRY = "pint-benchmark-main/benchmark/data/example-dataset.yaml"
DEFAULT_EXTRACTED_PINT_EXAMPLE = (
    PROJECT_ROOT
    / "data"
    / "external_benchmark"
    / "pint"
    / "raw"
    / "pint-benchmark-main"
    / "benchmark"
    / "data"
    / "example-dataset.yaml"
)
DEFAULT_INTERNAL_DATASET = PROJECT_ROOT / "datasets" / "benchmarks" / "pint_compatible_internal_holdout.yaml"
DEFAULT_INDIRECT_DATASET = PROJECT_ROOT / "datasets" / "benchmarks" / "indirect_document_benchmark.yaml"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "reports" / "pint_compatible_results.csv"
DEFAULT_SUMMARY_MD = PROJECT_ROOT / "reports" / "pint_compatible_summary.md"

PINT_MODES = ("roberta_only", "rule_roberta", "full_runtime")
PINT_MODE_LABELS = {
    "roberta_only": "RoBERTa-only",
    "rule_roberta": "Rule + RoBERTa",
    "full_runtime": "Full runtime",
}


@dataclass
class EvalResult:
    prediction: bool
    score: float
    threshold: float
    latency_ms: float
    decision_source: str
    decision: str
    details: dict[str, Any]


def _score(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _to_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "injection", "unsafe", "blocked", "warning"}:
            return True
        if normalized in {"false", "0", "no", "n", "safe", "benign", "allow"}:
            return False
    raise ValueError(f"Field '{field_name}' must be boolean-like, got: {value!r}")


def _read_yaml_file(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8-sig"))


def _find_pint_zip() -> Path | None:
    env_path = os.getenv("PINT_ZIP_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    candidates = [
        PROJECT_ROOT / "pint-benchmark-main.zip",
        PROJECT_ROOT / "data" / "external_benchmark" / "pint" / "pint-benchmark-main.zip",
    ]
    try:
        candidates.extend(Path("F:/").glob("*/pint-benchmark-main.zip"))
    except OSError:
        pass
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_pint_example(path_arg: str | None, zip_arg: str | None) -> tuple[list[dict[str, Any]], str]:
    if path_arg:
        path = Path(path_arg)
        return _normalize_records(_read_yaml_file(path), dataset_name="pint_example_smoke"), str(path)
    if DEFAULT_EXTRACTED_PINT_EXAMPLE.exists():
        return _normalize_records(_read_yaml_file(DEFAULT_EXTRACTED_PINT_EXAMPLE), dataset_name="pint_example_smoke"), str(
            DEFAULT_EXTRACTED_PINT_EXAMPLE
        )

    zip_path = Path(zip_arg) if zip_arg else _find_pint_zip()
    if not zip_path or not zip_path.exists():
        raise FileNotFoundError(
            "Could not find PINT example dataset. Pass --example-dataset or --pint-zip."
        )
    with zipfile.ZipFile(zip_path) as archive:
        payload = yaml.safe_load(archive.read(PINT_EXAMPLE_ENTRY).decode("utf-8-sig"))
    return _normalize_records(payload, dataset_name="pint_example_smoke"), f"{zip_path}!{PINT_EXAMPLE_ENTRY}"


def _normalize_records(payload: Any, *, dataset_name: str) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise ValueError(f"{dataset_name} must be a YAML list of records.")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{dataset_name}[{index}] must be an object.")
        text = str(item.get("text", "")).strip()
        category = str(item.get("category", "")).strip()
        if not text:
            raise ValueError(f"{dataset_name}[{index}].text must not be empty.")
        if not category:
            raise ValueError(f"{dataset_name}[{index}].category must not be empty.")
        rows.append(
            {
                "text": text,
                "category": category,
                "label": _to_bool(item.get("label"), field_name=f"{dataset_name}[{index}].label"),
                "case_id": str(item.get("id") or f"{dataset_name}-{index:04d}"),
            }
        )
    return rows


def _load_internal_dataset(path: Path) -> list[dict[str, Any]]:
    return _normalize_records(_read_yaml_file(path), dataset_name=path.stem)


def _threshold_from_transformer(result: dict[str, Any], override: float | None) -> float:
    if override is not None:
        return _score(override)
    threshold_used = result.get("threshold_used")
    if isinstance(threshold_used, dict):
        return _score(threshold_used.get("warn", threshold_used.get("evaluation", 0.5)))
    thresholds = result.get("thresholds")
    if isinstance(thresholds, dict):
        return _score(thresholds.get("runtime_warn_threshold", thresholds.get("evaluation_threshold", 0.5)))
    return 0.5


def evaluate_roberta_only(text: str, *, use_cuda: bool, threshold: float | None) -> EvalResult:
    start = time.perf_counter()
    model_dir = resolve_transformer_model_dir("roberta")
    result = predict_transformer(
        text=text,
        model_path=model_dir,
        model_name="roberta",
        max_length=128,
        use_cuda=use_cuda,
        use_intent_guard=False,
        use_runtime_calibration=False,
    )
    latency_ms = round((time.perf_counter() - start) * 1000, 3)
    score = _score(result.get("raw_score", result.get("raw_risk_score", result.get("risk_score"))))
    resolved_threshold = _threshold_from_transformer(result, threshold)
    prediction = score >= resolved_threshold
    return EvalResult(
        prediction=prediction,
        score=score,
        threshold=resolved_threshold,
        latency_ms=latency_ms,
        decision_source="roberta_raw",
        decision="warning" if prediction else "safe",
        details={
            "model": result.get("model"),
            "score_used": "raw_softmax_probability",
            "intent_guard_enabled": result.get("intent_guard_enabled"),
            "calibration_enabled": result.get("calibration_enabled"),
            "threshold_source": result.get("threshold_source"),
            "input_preprocessing": result.get("input_preprocessing"),
            "logits": result.get("logits"),
            "raw_probabilities": result.get("raw_probabilities"),
        },
    )


def evaluate_rule_roberta(text: str, *, use_cuda: bool, threshold: float | None) -> EvalResult:
    start = time.perf_counter()
    rule_signal = detect_rule_signal(text, source_type="user_prompt")
    roberta_result = evaluate_roberta_only(text, use_cuda=use_cuda, threshold=threshold)
    config = load_runtime_config()
    rule_threshold = _score(config.get("thresholds", {}).get("warn", 0.3))
    rule_score = _score(rule_signal.get("score"))
    rule_positive = (
        bool(rule_signal.get("hardBlock"))
        or str(rule_signal.get("action", "allow")).lower() in {"warn", "block"}
        or rule_score >= rule_threshold
    )
    roberta_positive = bool(roberta_result.prediction)
    prediction = rule_positive or roberta_positive
    score = max(rule_score, roberta_result.score)
    resolved_threshold = min(rule_threshold, roberta_result.threshold)
    if rule_positive and roberta_positive:
        decision_source = "rule_roberta"
    elif rule_positive:
        decision_source = "rule"
    elif roberta_positive:
        decision_source = "roberta_raw"
    else:
        decision_source = "none"
    if rule_signal.get("hardBlock"):
        decision = "blocked"
    else:
        decision = "warning" if prediction else "safe"
    return EvalResult(
        prediction=prediction,
        score=score,
        threshold=resolved_threshold,
        latency_ms=round((time.perf_counter() - start) * 1000, 3),
        decision_source=decision_source,
        decision=decision,
        details={
            "rule_score": rule_score,
            "rule_threshold": rule_threshold,
            "rule_action": rule_signal.get("action"),
            "rule_hard_block": rule_signal.get("hardBlock"),
            "matched_rules": rule_signal.get("matchedRules"),
            "roberta": roberta_result.details,
        },
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


def evaluate_full_runtime(text: str, *, use_cuda: bool, threshold: float | None = None) -> EvalResult:
    del threshold
    start = time.perf_counter()
    original_call_llm = official_runtime.call_llm
    official_runtime.call_llm = _benchmark_llm_skip
    try:
        result = official_runtime.run_official_runtime(
            message=text,
            user_id="pint-compatible-benchmark",
            use_cuda=use_cuda,
            request_id="pint_benchmark",
        )
    finally:
        official_runtime.call_llm = original_call_llm
    details = result.get("details", {}) if isinstance(result.get("details"), dict) else {}
    decision = str(result.get("decision", "safe")).lower()
    return EvalResult(
        prediction=decision in {"warning", "blocked"},
        score=_score(result.get("riskScore")),
        threshold=_score(details.get("threshold", 0.3)),
        latency_ms=round((time.perf_counter() - start) * 1000, 3),
        decision_source=str(details.get("highestRiskSource") or "policy"),
        decision=decision,
        details={
            "reasons": result.get("reasons"),
            "details": details,
            "policy_result": result.get("policyResult"),
            "model_scores": result.get("modelScores"),
        },
    )


def _adapter_for_mode(mode: str) -> Callable[[str, bool, float | None], EvalResult]:
    adapters = {
        "roberta_only": evaluate_roberta_only,
        "rule_roberta": evaluate_rule_roberta,
        "full_runtime": evaluate_full_runtime,
    }
    return adapters[mode]


def _safe_eval(
    mode: str,
    text: str,
    *,
    use_cuda: bool,
    threshold: float | None,
) -> EvalResult:
    try:
        return _adapter_for_mode(mode)(text, use_cuda=use_cuda, threshold=threshold)
    except Exception as exc:
        return EvalResult(
            prediction=False,
            score=0.0,
            threshold=threshold if threshold is not None else 0.5,
            latency_ms=0.0,
            decision_source=f"error:{exc.__class__.__name__}",
            decision="error",
            details={"error": str(exc)},
        )


def run_pint_rows(
    *,
    dataset_name: str,
    records: list[dict[str, Any]],
    modes: list[str],
    use_cuda: bool,
    threshold: float | None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row_index, row in enumerate(records, start=1):
        for mode in modes:
            result = _safe_eval(mode, row["text"], use_cuda=use_cuda, threshold=threshold)
            label = bool(row["label"])
            output.append(
                {
                    "benchmark_type": "pint_compatible",
                    "dataset_name": dataset_name,
                    "mode": mode,
                    "mode_label": PINT_MODE_LABELS[mode],
                    "case_id": row["case_id"],
                    "row_index": row_index,
                    "text": row["text"],
                    "category": row["category"],
                    "label": label,
                    "expected_label": label,
                    "prediction": result.prediction,
                    "correct": result.prediction == label,
                    "score": round(result.score, 8),
                    "threshold": round(result.threshold, 8),
                    "latency_ms": result.latency_ms,
                    "decision": result.decision,
                    "decision_source": result.decision_source,
                    "expected_location": "",
                    "observed_location": "",
                    "file_type": "",
                    "details_json": json.dumps(result.details, ensure_ascii=False, sort_keys=True),
                }
            )
    return output


def _docx_bytes(text: str) -> bytes:
    paragraphs = []
    for line in str(text).splitlines() or [str(text)]:
        escaped = xml_escape(line)
        paragraphs.append(f"<w:p><w:r><w:t>{escaped}</w:t></w:r></w:p>")
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{''.join(paragraphs)}</w:body>"
        "</w:document>"
    )
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8"?><Types></Types>')
        archive.writestr("word/document.xml", document_xml)
    return payload.getvalue()


def _pdf_escape(text: str) -> str:
    value = str(text).encode("latin-1", errors="replace").decode("latin-1")
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf_bytes(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 72 720 Td ({_pdf_escape(text)}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode("ascii"))
        pdf.extend(body)
        pdf.extend(b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(pdf)


def _document_bytes_from_text(text: str, file_type: str) -> tuple[str, bytes]:
    normalized = str(file_type or "txt").strip().lower().lstrip(".")
    if normalized == "txt":
        return "benchmark_document.txt", str(text).encode("utf-8")
    if normalized == "docx":
        return "benchmark_document.docx", _docx_bytes(text)
    if normalized == "pdf":
        return "benchmark_document.pdf", _pdf_bytes(text)
    raise ValueError(f"Unsupported indirect benchmark file_type: {file_type!r}")


def _load_indirect_records(path: Path) -> list[dict[str, Any]]:
    payload = _read_yaml_file(path)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must be a YAML list of records.")
    records: list[dict[str, Any]] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{path.name}[{index}] must be an object.")
        user_prompt = str(item.get("user_prompt", "")).strip()
        if not user_prompt:
            raise ValueError(f"{path.name}[{index}].user_prompt must not be empty.")
        has_text = bool(str(item.get("document_text", "")).strip())
        has_fixture = bool(str(item.get("file_fixture", "")).strip())
        if has_text == has_fixture:
            raise ValueError(
                f"{path.name}[{index}] must define exactly one of document_text or file_fixture."
            )
        records.append(
            {
                "case_id": str(item.get("id") or f"indirect-{index:04d}"),
                "user_prompt": user_prompt,
                "document_text": str(item.get("document_text", "")),
                "file_fixture": str(item.get("file_fixture", "")),
                "file_type": str(item.get("file_type") or "txt").strip().lower().lstrip("."),
                "category": str(item.get("category") or "indirect_document").strip(),
                "expected_label": _to_bool(
                    item.get("expected_label"),
                    field_name=f"{path.name}[{index}].expected_label",
                ),
                "expected_location": str(item.get("expected_location") or "document").strip(),
            }
        )
    return records


def _load_document_case_content(row: dict[str, Any], dataset_path: Path) -> tuple[str, bytes]:
    fixture = row.get("file_fixture")
    if fixture:
        fixture_path = Path(fixture)
        if not fixture_path.is_absolute():
            fixture_path = dataset_path.parent / fixture_path
        return fixture_path.name, fixture_path.read_bytes()
    return _document_bytes_from_text(row["document_text"], row["file_type"])


def run_indirect_document_rows(
    *,
    dataset_path: Path,
    use_cuda: bool,
) -> list[dict[str, Any]]:
    records = _load_indirect_records(dataset_path)
    output: list[dict[str, Any]] = []
    for row_index, row in enumerate(records, start=1):
        start = time.perf_counter()
        try:
            file_name, content = _load_document_case_content(row, dataset_path)
            document_signal = analyze_uploaded_document(
                user_message=row["user_prompt"],
                file_name=file_name,
                content=content,
                source_type=row["file_type"],
                use_cuda=use_cuda,
            )
            original_call_llm = official_runtime.call_llm
            official_runtime.call_llm = _benchmark_llm_skip
            try:
                runtime_result = official_runtime.run_official_runtime(
                    message=row["user_prompt"],
                    user_id="indirect-document-benchmark",
                    document_signal=document_signal,
                    use_cuda=use_cuda,
                    request_id="indirect_benchmark",
                )
            finally:
                official_runtime.call_llm = original_call_llm
            decision = str(runtime_result.get("decision", "safe")).lower()
            prediction = decision in {"warning", "blocked"}
            document_positive = str(document_signal.get("decision", "safe")).lower() != "safe"
            observed_location = "document" if document_positive else ("user_prompt" if prediction else "none")
            details = runtime_result.get("details", {}) if isinstance(runtime_result.get("details"), dict) else {}
            score = _score(runtime_result.get("riskScore", document_signal.get("score")))
            threshold = _score(details.get("threshold", 0.3))
            decision_source = str(details.get("highestRiskSource") or "document_runtime")
            compact_details = {
                "runtime_reasons": runtime_result.get("reasons"),
                "runtime_details": details,
                "document_signal": {
                    key: value
                    for key, value in document_signal.items()
                    if key not in {"safeContextText"}
                },
            }
        except Exception as exc:
            decision = "error"
            prediction = False
            observed_location = "error"
            score = 0.0
            threshold = 0.3
            decision_source = f"error:{exc.__class__.__name__}"
            compact_details = {"error": str(exc)}

        expected = bool(row["expected_label"])
        output.append(
            {
                "benchmark_type": "indirect_document",
                "dataset_name": dataset_path.stem,
                "mode": "document_runtime",
                "mode_label": "Indirect document runtime",
                "case_id": row["case_id"],
                "row_index": row_index,
                "text": row["user_prompt"],
                "category": row["category"],
                "label": expected,
                "expected_label": expected,
                "prediction": prediction,
                "correct": prediction == expected,
                "score": round(score, 8),
                "threshold": round(threshold, 8),
                "latency_ms": round((time.perf_counter() - start) * 1000, 3),
                "decision": decision,
                "decision_source": decision_source,
                "expected_location": row["expected_location"],
                "observed_location": observed_location,
                "file_type": row["file_type"],
                "details_json": json.dumps(compact_details, ensure_ascii=False, sort_keys=True),
            }
        )
    return output


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    tp = sum(1 for row in rows if row["label"] is True and row["prediction"] is True)
    tn = sum(1 for row in rows if row["label"] is False and row["prediction"] is False)
    fp = sum(1 for row in rows if row["label"] is False and row["prediction"] is True)
    fn = sum(1 for row in rows if row["label"] is True and row["prediction"] is False)
    positive_total = tp + fn
    negative_total = tn + fp
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / positive_total if positive_total else 0.0
    specificity = tn / negative_total if negative_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / negative_total if negative_total else 0.0
    fnr = fn / positive_total if positive_total else 0.0
    label_accuracies = []
    if negative_total:
        label_accuracies.append(specificity)
    if positive_total:
        label_accuracies.append(recall)
    balanced_score = sum(label_accuracies) / len(label_accuracies) if label_accuracies else 0.0
    return {
        "total": total,
        "balanced_score": balanced_score,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def _group_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(name) for name in keys)
        groups.setdefault(key, []).append(row)
    return groups


def _format_percent(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _summary_sections(
    *,
    rows: list[dict[str, Any]],
    example_source: str,
    internal_source: str,
    indirect_source: str,
) -> str:
    pint_rows = [row for row in rows if row["benchmark_type"] == "pint_compatible"]
    indirect_rows = [row for row in rows if row["benchmark_type"] == "indirect_document"]

    overall_rows = []
    for (dataset_name, mode), grouped in sorted(_group_rows(pint_rows, ("dataset_name", "mode")).items()):
        metrics = compute_metrics(grouped)
        overall_rows.append(
            [
                dataset_name,
                PINT_MODE_LABELS.get(mode, mode),
                metrics["total"],
                _format_percent(metrics["balanced_score"]),
                _format_percent(metrics["accuracy"]),
                _format_percent(metrics["precision"]),
                _format_percent(metrics["recall"]),
                _format_percent(metrics["f1"]),
                _format_percent(metrics["specificity"]),
                f"TP={metrics['true_positive']}, TN={metrics['true_negative']}, FP={metrics['false_positive']}, FN={metrics['false_negative']}",
            ]
        )

    category_rows = []
    for (dataset_name, mode, category), grouped in sorted(
        _group_rows(pint_rows, ("dataset_name", "mode", "category")).items()
    ):
        metrics = compute_metrics(grouped)
        category_rows.append(
            [
                dataset_name,
                PINT_MODE_LABELS.get(mode, mode),
                category,
                metrics["total"],
                _format_percent(metrics["balanced_score"]),
                _format_percent(metrics["accuracy"]),
                _format_percent(metrics["false_positive_rate"]),
                _format_percent(metrics["false_negative_rate"]),
            ]
        )

    indirect_summary = compute_metrics(indirect_rows) if indirect_rows else None
    indirect_table_rows = []
    for row in indirect_rows:
        indirect_table_rows.append(
            [
                row["case_id"],
                row["file_type"],
                row["category"],
                row["expected_label"],
                row["prediction"],
                row["decision"],
                row["expected_location"],
                row["observed_location"],
                f"{float(row['score']):.3f}",
            ]
        )

    lines = [
        "# Báo cáo PINT-compatible internal benchmark",
        "",
        "## PINT là gì?",
        "",
        (
            "PINT (Prompt Injection Test) Benchmark là benchmark đánh giá hệ thống phát hiện prompt "
            "injection theo tập dữ liệu được thiết kế để tránh việc tối ưu hóa theo các public dataset quen thuộc."
        ),
        "",
        "## Dataset trong ZIP",
        "",
        (
            f"Script đã đọc PINT example từ `{example_source}`. ZIP cung cấp README, DETAILS, notebook "
            "`benchmark/pint-benchmark.ipynb`, utility Hugging Face và `benchmark/data/example-dataset.yaml`."
        ),
        "",
        (
            "README mô tả PINT dataset chính thức gồm 4.314 input với nhóm prompt injection, jailbreak, "
            "hard negative, chat và document, nhưng dataset đầy đủ đó không nằm trong ZIP public được cung cấp."
        ),
        "",
        (
            "Quan trọng: `example-dataset.yaml` chỉ là file mẫu để kiểm tra format. Nó không phải PINT dataset "
            "đầy đủ, không đại diện cho PINT chính thức, và kết quả trên file này chỉ được gọi là smoke test."
        ),
        "",
        "## Vì sao example dataset không phải PINT đầy đủ",
        "",
        (
            "Ngay trong file example có ghi chú rằng đây chỉ là ví dụ cấu trúc dataset, không phải PINT Benchmark "
            "dataset và không đại diện cho dữ liệu thật. Vì vậy báo cáo này dùng tên PINT-compatible internal "
            "benchmark, tuyệt đối không gọi là PINT Score chính thức."
        ),
        "",
        "## Cách tính PINT-compatible balanced score",
        "",
        (
            "Notebook PINT tính balanced score bằng cách tính accuracy riêng cho label âm tính và dương tính, "
            "sau đó lấy trung bình. Với bài toán nhị phân, công thức tương đương `(specificity + recall) / 2` "
            "khi cả hai lớp đều có mặt."
        ),
        "",
        "## Adapter được chạy",
        "",
        "- RoBERTa-only: rule OFF, intent guard OFF, runtime calibration OFF; prediction dựa trên raw injection score.",
        "- Rule + RoBERTa: rule-based signal kết hợp RoBERTa raw score, không gọi policy runtime đầy đủ.",
        "- Full runtime: dùng pipeline runtime hiện tại, gồm rule, RoBERTa, context/fusion/policy nếu đang bật; LLM được skip trong benchmark.",
        "",
        "## Kết quả ba pipeline",
        "",
        _markdown_table(
            [
                "Dataset",
                "Mode",
                "N",
                "Balanced",
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "Specificity",
                "Confusion matrix",
            ],
            overall_rows,
        )
        if overall_rows
        else "_No PINT-compatible rows were evaluated._",
        "",
        "## Metric theo category",
        "",
        _markdown_table(
            ["Dataset", "Mode", "Category", "N", "Balanced", "Accuracy", "FPR", "FNR"],
            category_rows,
        )
        if category_rows
        else "_No category metrics available._",
        "",
        "## Benchmark indirect document injection riêng",
        "",
        f"Dataset indirect: `{indirect_source}`.",
        "",
    ]
    if indirect_summary:
        lines.extend(
            [
                _markdown_table(
                    [
                        "N",
                        "Balanced",
                        "Accuracy",
                        "Precision",
                        "Recall",
                        "F1",
                        "Specificity",
                        "FPR",
                        "FNR",
                        "Confusion matrix",
                    ],
                    [
                        [
                            indirect_summary["total"],
                            _format_percent(indirect_summary["balanced_score"]),
                            _format_percent(indirect_summary["accuracy"]),
                            _format_percent(indirect_summary["precision"]),
                            _format_percent(indirect_summary["recall"]),
                            _format_percent(indirect_summary["f1"]),
                            _format_percent(indirect_summary["specificity"]),
                            _format_percent(indirect_summary["false_positive_rate"]),
                            _format_percent(indirect_summary["false_negative_rate"]),
                            (
                                f"TP={indirect_summary['true_positive']}, TN={indirect_summary['true_negative']}, "
                                f"FP={indirect_summary['false_positive']}, FN={indirect_summary['false_negative']}"
                            ),
                        ]
                    ],
                ),
                "",
                _markdown_table(
                    [
                        "Case",
                        "File type",
                        "Category",
                        "Expected",
                        "Predicted",
                        "Decision",
                        "Expected location",
                        "Observed location",
                        "Score",
                    ],
                    indirect_table_rows,
                ),
                "",
            ]
        )
    else:
        lines.extend(["_Indirect document benchmark was not run._", ""])

    lines.extend(
        [
            "## Hạn chế",
            "",
            "- Không báo cáo các kết quả này là PINT Score chính thức.",
            "- Example dataset trong ZIP chỉ là smoke test nhỏ, không đủ để kết luận chất lượng model.",
            "- Internal holdout là benchmark chẩn đoán nội bộ, không được dùng để train, fine-tune rule, hay tối ưu threshold.",
            "- Latency phụ thuộc máy local, CPU/GPU và cache model trong lần chạy.",
            "",
            "## Nguồn input",
            "",
            f"- PINT example smoke test: `{example_source}`",
            f"- Internal PINT-compatible holdout: `{internal_source}`",
            f"- Indirect document benchmark: `{indirect_source}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "benchmark_type",
        "dataset_name",
        "mode",
        "mode_label",
        "case_id",
        "row_index",
        "text",
        "category",
        "label",
        "expected_label",
        "prediction",
        "correct",
        "score",
        "threshold",
        "latency_ms",
        "decision",
        "decision_source",
        "expected_location",
        "observed_location",
        "file_type",
        "details_json",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    *,
    rows: list[dict[str, Any]],
    path: Path,
    example_source: str,
    internal_source: str,
    indirect_source: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _summary_sections(
            rows=rows,
            example_source=example_source,
            internal_source=internal_source,
            indirect_source=indirect_source,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pint-zip", default=None, help="Optional path to pint-benchmark-main.zip.")
    parser.add_argument("--example-dataset", default=None, help="Optional path to PINT example YAML.")
    parser.add_argument("--internal-dataset", default=str(DEFAULT_INTERNAL_DATASET))
    parser.add_argument("--indirect-dataset", default=str(DEFAULT_INDIRECT_DATASET))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY_MD))
    parser.add_argument("--modes", nargs="+", choices=PINT_MODES, default=list(PINT_MODES))
    parser.add_argument("--roberta-threshold", type=float, default=None)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--skip-example", action="store_true")
    parser.add_argument("--skip-internal", action="store_true")
    parser.add_argument("--skip-indirect", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    example_source = "skipped"
    internal_source = "skipped"
    indirect_source = "skipped"

    if not args.skip_example:
        example_records, example_source = _load_pint_example(args.example_dataset, args.pint_zip)
        rows.extend(
            run_pint_rows(
                dataset_name="pint_example_smoke",
                records=example_records,
                modes=args.modes,
                use_cuda=args.use_cuda,
                threshold=args.roberta_threshold,
            )
        )

    internal_path = Path(args.internal_dataset)
    if not args.skip_internal:
        internal_source = str(internal_path)
        rows.extend(
            run_pint_rows(
                dataset_name=internal_path.stem,
                records=_load_internal_dataset(internal_path),
                modes=args.modes,
                use_cuda=args.use_cuda,
                threshold=args.roberta_threshold,
            )
        )

    indirect_path = Path(args.indirect_dataset)
    if not args.skip_indirect:
        indirect_source = str(indirect_path)
        rows.extend(run_indirect_document_rows(dataset_path=indirect_path, use_cuda=args.use_cuda))

    output_csv = Path(args.output_csv)
    summary_md = Path(args.summary)
    write_csv(rows, output_csv)
    write_summary(
        rows=rows,
        path=summary_md,
        example_source=example_source,
        internal_source=internal_source,
        indirect_source=indirect_source,
    )
    print(f"Wrote row-level results: {output_csv}")
    print(f"Wrote summary report: {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
