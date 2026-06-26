"""Run batch dataset evaluation from the command line or a tkinter file picker.

Examples:
    python scripts/run_dataset_evaluation.py --file "datasets/examples/batch_test_sample.csv"
    python scripts/run_dataset_evaluation.py --run-name "neuralchemy_test_500"
    python scripts/run_dataset_evaluation.py
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.batch_evaluation import (  # noqa: E402
    DEFAULT_MAX_BATCH_ITEMS,
    SUPPORTED_BATCH_MODELS,
    evaluate_batch_items,
    parse_dataset_content,
)


def configure_console_encoding() -> None:
    """Avoid Windows charmap crashes when dataset text contains Vietnamese Unicode."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


configure_console_encoding()
warnings.filterwarnings(
    "ignore",
    message=r"`sklearn\.utils\.parallel\.delayed` should be used with `sklearn\.utils\.parallel\.Parallel`.*",
    category=UserWarning,
)
REPORT_DIR = PROJECT_ROOT / "reports" / "batch_evaluation"
DEFAULT_MODELS = list(SUPPORTED_BATCH_MODELS)
METRIC_FIELDS = [
    "model",
    "status",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "tp",
    "fp",
    "tn",
    "fn",
    "avg_latency_ms",
    "available_prompts",
    "unavailable_prompts",
]
INDEX_FIELDS = [
    "run_id",
    "timestamp",
    "dataset_path",
    "dataset_name",
    "total_rows",
    "has_ground_truth",
    "model_count",
    "best_model_by_f1",
    "best_f1",
    "output_folder",
]
MISTAKE_FIELDS = [
    "model",
    "id",
    "text",
    "ground_truth_label",
    "predicted_label",
    "risk_score",
    "confidence",
    "action",
    "latency_ms",
    "category",
    "source",
]
DISAGREEMENT_FIELDS = [
    "id",
    "text",
    "ground_truth_label",
    "category",
    "source",
    "model_labels",
    "model_risk_scores",
    "model_actions",
]
MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic",
    "linear_svm": "SVM",
    "random_forest": "RF",
    "distilbert": "DistilBERT",
    "roberta": "RoBERTa",
    "hybrid": "Hybrid",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Prompt Injection Detector batch evaluation for CSV/JSON/JSONL/TXT datasets."
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Dataset file path. If omitted, a tkinter file picker is opened.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=(
            "Models to run. Supported: logistic_regression linear_svm random_forest "
            "distilbert roberta hybrid. Default: all."
        ),
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=DEFAULT_MAX_BATCH_ITEMS,
        help=f"Maximum dataset rows to process. Default: {DEFAULT_MAX_BATCH_ITEMS}.",
    )
    parser.add_argument(
        "--hybrid-traditional",
        default="all",
        help="Hybrid traditional model: all, logistic_regression, linear_svm, random_forest.",
    )
    parser.add_argument(
        "--hybrid-transformer",
        default="distilbert",
        help="Hybrid transformer model: distilbert or roberta.",
    )
    parser.add_argument(
        "--hybrid-strategy",
        default="maximum_risk",
        help="Hybrid strategy: maximum_risk, majority_vote, weighted_voting.",
    )
    parser.add_argument(
        "--disable-rule-based",
        action="store_true",
        help="Disable rule-based detector inside hybrid mode.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPORT_DIR.relative_to(PROJECT_ROOT)),
        help="Base output directory. A unique runs/<timestamp>_<name>/ folder is created inside it.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name used in the run folder name.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=20,
        help="Number of prediction preview rows to include in Markdown reports for small datasets.",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Compatibility option. CSV reports are always exported.",
    )
    return parser.parse_args()


def choose_file_with_tkinter() -> Path:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("tkinter is not available. Pass --file explicitly.") from exc

    root = tk.Tk()
    root.withdraw()
    selected = filedialog.askopenfilename(
        title="Select dataset file",
        filetypes=[
            ("Supported datasets", "*.csv *.json *.jsonl *.txt"),
            ("CSV", "*.csv"),
            ("JSON", "*.json"),
            ("JSONL", "*.jsonl"),
            ("Text", "*.txt"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    if not selected:
        raise RuntimeError("No dataset file selected.")
    return Path(selected)


def resolve_dataset_path(path_value: str | None) -> Path:
    if path_value:
        path = Path(path_value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
    else:
        path = choose_file_with_tkinter()

    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    if path.suffix.lower() not in {".csv", ".json", ".jsonl", ".txt"}:
        raise ValueError("Unsupported dataset file. Use .csv, .json, .jsonl or .txt.")
    return path


def resolve_output_base(path_value: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _detect_text_encoding(path: Path) -> str:
    sample = path.read_bytes()[:1024 * 1024]
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            sample.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "latin1"


def _open_text_with_fallback(path: Path):
    return path.open("r", encoding=_detect_text_encoding(path), errors="replace", newline="")


def read_dataset(path: Path, max_items: int | None = None) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv" and max_items and max_items > 0:
        rows: list[dict[str, Any]] = []
        with _open_text_with_fallback(path) as file:
            reader = csv.DictReader(file)
            for row in reader:
                row["_dataset_format"] = "csv"
                rows.append(row)
                if len(rows) >= max_items:
                    break
        return rows

    with _open_text_with_fallback(path) as file:
        content = file.read()
    return parse_dataset_content(str(path), content)


def format_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:80] or "dataset"


def create_run_folder(base_output_dir: Path, dataset_path: Path, run_name: str | None) -> tuple[str, str, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = slugify(run_name or dataset_path.stem)
    run_id = f"{timestamp}_{suffix}"
    runs_dir = base_output_dir / "runs"
    run_folder = runs_dir / run_id
    counter = 2
    while run_folder.exists():
        run_id = f"{timestamp}_{suffix}_{counter:02d}"
        run_folder = runs_dir / run_id
        counter += 1
    run_folder.mkdir(parents=True, exist_ok=False)
    return timestamp, run_id, run_folder


def label_distribution(results: list[dict[str, Any]]) -> dict[str, int]:
    distribution = {"0": 0, "1": 0, "missing": 0}
    for row in results:
        label = row.get("ground_truth_label")
        if label is None:
            distribution["missing"] += 1
        elif int(label) == 1:
            distribution["1"] += 1
        else:
            distribution["0"] += 1
    return distribution


def normalize_metric_rows(summary: dict[str, Any], selected_models: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metrics_by_model = summary.get("models", {}) or {}
    for model_name in selected_models:
        metrics = metrics_by_model.get(model_name, {}) or {}
        available = int(metrics.get("available_prompts") or 0)
        status = "available" if available > 0 and not metrics.get("message") else "unavailable"
        rows.append(
            {
                "model": model_name,
                "status": status,
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "tp": metrics.get("tp"),
                "fp": metrics.get("fp"),
                "tn": metrics.get("tn"),
                "fn": metrics.get("fn"),
                "avg_latency_ms": metrics.get("avg_latency_ms"),
                "available_prompts": metrics.get("available_prompts", 0),
                "unavailable_prompts": metrics.get("unavailable_prompts", 0),
                "message": metrics.get("message"),
            }
        )
    return rows


def best_model_by_f1(metric_rows: list[dict[str, Any]]) -> tuple[str, float | None]:
    candidates = [row for row in metric_rows if row.get("f1") is not None and row.get("status") == "available"]
    if not candidates:
        return "", None
    best = max(candidates, key=lambda row: float(row.get("f1") or 0.0))
    return str(best["model"]), float(best["f1"])


def metrics_summary_csv(metric_rows: list[dict[str, Any]]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=METRIC_FIELDS)
    writer.writeheader()
    for row in metric_rows:
        writer.writerow({field: row.get(field) for field in METRIC_FIELDS})
    return output.getvalue()


def rows_to_csv(rows: list[dict[str, Any]], fields: list[str]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field) for field in fields})
    return output.getvalue()


def collect_mistake_rows(results: list[dict[str, Any]], mistake_type: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        truth = result.get("ground_truth_label")
        if truth is None:
            continue
        for model_name, prediction in result.get("predictions", {}).items():
            if not prediction.get("available") or prediction.get("predicted_label") is None:
                continue
            predicted = int(prediction["predicted_label"])
            is_false_positive = int(truth) == 0 and predicted == 1
            is_false_negative = int(truth) == 1 and predicted == 0
            if (mistake_type == "fp" and is_false_positive) or (mistake_type == "fn" and is_false_negative):
                rows.append(
                    {
                        "model": model_name,
                        "id": result.get("id"),
                        "text": result.get("text"),
                        "ground_truth_label": truth,
                        "predicted_label": predicted,
                        "risk_score": prediction.get("risk_score"),
                        "confidence": prediction.get("confidence"),
                        "action": prediction.get("action"),
                        "latency_ms": prediction.get("latency_ms"),
                        "category": result.get("category"),
                        "source": result.get("source"),
                    }
                )
    return rows


def collect_disagreement_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        labels: dict[str, Any] = {}
        risks: dict[str, Any] = {}
        actions: dict[str, Any] = {}
        for model_name, prediction in result.get("predictions", {}).items():
            if prediction.get("available") and prediction.get("predicted_label") is not None:
                labels[model_name] = prediction.get("predicted_label")
                risks[model_name] = prediction.get("risk_score")
                actions[model_name] = prediction.get("action")
        if len(set(labels.values())) > 1:
            rows.append(
                {
                    "id": result.get("id"),
                    "text": result.get("text"),
                    "ground_truth_label": result.get("ground_truth_label"),
                    "category": result.get("category"),
                    "source": result.get("source"),
                    "model_labels": json.dumps(labels, ensure_ascii=False),
                    "model_risk_scores": json.dumps(risks, ensure_ascii=False),
                    "model_actions": json.dumps(actions, ensure_ascii=False),
                }
            )
    return rows


def truncate_text(text: Any, max_len: int = 80) -> str:
    value = "" if text is None else str(text)
    value = re.sub(r"\s+", " ", value.replace("\r", " ").replace("\n", " ")).strip()
    if max_len and len(value) > max_len:
        return value[: max_len - 3].rstrip() + "..."
    return value


def markdown_escape(value: Any, limit: int | None = None) -> str:
    text = truncate_text(value, limit) if limit else ("" if value is None else str(value))
    return text.replace("|", "\\|").replace("\n", " ")


def markdown_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return markdown_escape(value)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    normalized_rows: list[list[str]] = []
    for row in rows:
        if len(row) != len(headers):
            raise ValueError(
                f"Markdown table row has {len(row)} columns but header has {len(headers)} columns."
            )
        normalized_rows.append([markdown_cell(cell) for cell in row])

    lines = [
        "| " + " | ".join(markdown_escape(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in normalized_rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def markdown_metrics_table(metric_rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Model | Status | Accuracy | Precision | Recall | F1 | TP | FP | TN | FN | Avg Latency |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        avg_latency = "" if row.get("avg_latency_ms") is None else f"{float(row['avg_latency_ms']):.2f} ms"
        lines.append(
            "| "
            + " | ".join(
                [
                    markdown_escape(row.get("model")),
                    markdown_escape(row.get("status")),
                    markdown_escape(format_number(row.get("accuracy"))),
                    markdown_escape(format_number(row.get("precision"))),
                    markdown_escape(format_number(row.get("recall"))),
                    markdown_escape(format_number(row.get("f1"))),
                    markdown_escape(row.get("tp")),
                    markdown_escape(row.get("fp")),
                    markdown_escape(row.get("tn")),
                    markdown_escape(row.get("fn")),
                    markdown_escape(avg_latency),
                ]
            )
            + " |"
        )
    return lines


def markdown_confusion_matrices(metric_rows: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for row in metric_rows:
        if row.get("status") != "available":
            continue
        lines.extend(
            [
                f"### {row['model']}",
                "",
                "| Actual / Predicted | SAFE 0 | INJECTION 1 |",
                "|---|---:|---:|",
                f"| SAFE 0 | {row.get('tn', '')} | {row.get('fp', '')} |",
                f"| INJECTION 1 | {row.get('fn', '')} | {row.get('tp', '')} |",
                "",
            ]
        )
    return lines or ["No available model metrics.", ""]


def format_risk(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def format_prediction_cell(prediction: dict[str, Any] | None) -> str:
    if not prediction or not prediction.get("available"):
        return "not_ready"
    label = prediction.get("predicted_label")
    risk = prediction.get("risk_score")
    action = prediction.get("action") or ""
    risk_text = "null" if risk is None else format_risk(risk)
    return f"{label} / {risk_text} / {action}"


def markdown_mistake_section(
    title: str,
    csv_filename: str,
    rows: list[dict[str, Any]],
    case_prefix: str,
    max_cases: int = 20,
) -> list[str]:
    lines = [f"## {title}", "", f"Full list: `{csv_filename}`", ""]
    if not rows:
        lines.append("No cases found.")
        return lines

    shown_rows = rows[:max_cases]
    table_rows: list[list[Any]] = []
    for index, row in enumerate(shown_rows, start=1):
        table_rows.append(
            [
                index,
                row.get("model"),
                row.get("id"),
                row.get("ground_truth_label"),
                row.get("predicted_label"),
                format_risk(row.get("risk_score")),
                row.get("action"),
                truncate_text(row.get("text"), 80),
            ]
        )
    lines.extend(
        markdown_table(
            ["#", "Model", "ID", "Truth", "Pred", "Risk", "Action", "Prompt Preview"],
            table_rows,
        )
    )

    if len(rows) > max_cases:
        lines.extend(["", f"Showing first {max_cases} of {len(rows)} cases. See `{csv_filename}` for all cases."])

    lines.append("")
    for index, row in enumerate(shown_rows, start=1):
        prompt = "" if row.get("text") is None else str(row.get("text")).replace("```", "'''")
        lines.extend(
            [
                f"### {case_prefix} Case {index}",
                "",
                "```text",
                f"ID: {row.get('id')}",
                f"Model: {row.get('model')}",
                f"Ground truth: {row.get('ground_truth_label')}",
                f"Predicted: {row.get('predicted_label')}",
                f"Risk: {format_risk(row.get('risk_score'))}",
                f"Action: {row.get('action')}",
                "Prompt:",
                prompt,
                "```",
                "",
            ]
        )
    return lines


def markdown_prediction_preview(payload: dict[str, Any], preview_rows: int) -> list[str]:
    results = payload.get("results", []) or []
    selected_models = payload.get("metadata", {}).get("selected_models", []) or []
    total_rows = len(results)
    lines = ["## Prediction Preview", ""]

    if total_rows > 50:
        lines.extend(
            [
                "Full predictions are available in `predictions_full.csv`.",
                f"Total rows: {total_rows}",
            ]
        )
        return lines

    if not results:
        lines.append("No predictions found.")
        return lines

    shown_count = min(max(preview_rows, 0), total_rows)
    headers = ["ID", "Truth"] + [MODEL_DISPLAY_NAMES.get(model, model) for model in selected_models]
    table_rows: list[list[Any]] = []
    for result in results[:shown_count]:
        predictions = result.get("predictions", {}) or {}
        table_rows.append(
            [
                result.get("id"),
                result.get("ground_truth_label"),
                *[format_prediction_cell(predictions.get(model)) for model in selected_models],
            ]
        )
    lines.extend(markdown_table(headers, table_rows))
    if shown_count < total_rows:
        lines.extend(["", f"Showing first {shown_count} of {total_rows} rows. Full predictions are in `predictions_full.csv`."])
    return lines


def build_markdown_report(
    payload: dict[str, Any],
    dataset_path: Path,
    metric_rows: list[dict[str, Any]],
    false_positive_rows: list[dict[str, Any]],
    false_negative_rows: list[dict[str, Any]],
    disagreement_rows: list[dict[str, Any]],
    preview_rows: int = 20,
) -> str:
    summary = payload.get("summary", {})
    validation = payload.get("validation", {})
    selected_models = payload.get("metadata", {}).get("selected_models", [])
    distribution = label_distribution(payload.get("results", []))
    lines = [
        "# Batch Dataset Evaluation Report",
        "",
        "## Dataset",
        "",
        f"- Dataset path: `{dataset_path}`",
        f"- Total prompts: `{summary.get('total_prompts', 0)}`",
        f"- Ground-truth labels: `{summary.get('has_ground_truth')}`",
        f"- Text column detected: `{validation.get('text_column_detected')}`",
        f"- Label column detected: `{validation.get('label_column_detected')}`",
        f"- Label distribution: SAFE/0=`{distribution['0']}`, INJECTION/1=`{distribution['1']}`, missing=`{distribution['missing']}`",
        "",
        "## Selected Models",
        "",
        *[f"- `{model_name}`" for model_name in selected_models],
        "",
        "## Model Metrics",
        "",
    ]
    if summary.get("has_ground_truth"):
        lines.extend(markdown_metrics_table(metric_rows))
        lines.extend(["", "## Confusion Matrix Per Model", ""])
        lines.extend(markdown_confusion_matrices(metric_rows))
        lines.extend(markdown_mistake_section("Top False Positives", "false_positives.csv", false_positive_rows, "FP"))
        lines.extend([""])
        lines.extend(markdown_mistake_section("Top False Negatives", "false_negatives.csv", false_negative_rows, "FN"))
    else:
        lines.append("Ground-truth label not found. Metrics are not available.")

    lines.extend(
        [
            "",
            "## Model Disagreements",
            "",
            f"Full list: `model_disagreements.csv` ({len(disagreement_rows)} cases).",
            "",
        ]
    )
    lines.extend(markdown_prediction_preview(payload, preview_rows))
    lines.extend(["", "Full per-prompt predictions are exported to `predictions_full.csv` and `predictions_full.json`."])
    return "\n".join(lines) + "\n"


def load_index_rows(index_path: Path) -> list[dict[str, str]]:
    if not index_path.exists():
        return []
    with index_path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_index(base_output_dir: Path, row: dict[str, Any]) -> None:
    base_output_dir.mkdir(parents=True, exist_ok=True)
    index_path = base_output_dir / "index.csv"
    existing_rows = load_index_rows(index_path)
    normalized_row = {field: str(row.get(field, "")) for field in INDEX_FIELDS}
    with index_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        for existing in existing_rows:
            writer.writerow({field: existing.get(field, "") for field in INDEX_FIELDS})
        writer.writerow(normalized_row)


def write_readme(base_output_dir: Path, recent_limit: int = 30) -> None:
    index_rows = load_index_rows(base_output_dir / "index.csv")
    index_rows = sorted(index_rows, key=lambda row: row.get("timestamp", ""), reverse=True)
    lines = [
        "# Batch Evaluation Runs",
        "",
        "| Time | Dataset | Rows | Best Model | Best F1 | Folder |",
        "| ---- | ------- | ---: | ---------- | ------: | ------ |",
    ]
    for row in index_rows[:recent_limit]:
        lines.append(
            f"| {markdown_escape(row.get('timestamp'))} | "
            f"{markdown_escape(row.get('dataset_name'))} | "
            f"{markdown_escape(row.get('total_rows'))} | "
            f"{markdown_escape(row.get('best_model_by_f1'))} | "
            f"{markdown_escape(row.get('best_f1'))} | "
            f"{markdown_escape(row.get('output_folder'))} |"
        )
    (base_output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_reports(
    payload: dict[str, Any],
    base_output_dir: Path,
    dataset_path: Path,
    run_name: str | None,
    preview_rows: int = 20,
) -> tuple[dict[str, Path], dict[str, Any]]:
    timestamp, run_id, run_folder = create_run_folder(base_output_dir, dataset_path, run_name)
    exports = payload.get("exports", {})
    summary = payload.get("summary", {})
    selected_models = payload.get("metadata", {}).get("selected_models", [])
    metric_rows = normalize_metric_rows(summary, selected_models) if summary.get("has_ground_truth") else []
    false_positive_rows = collect_mistake_rows(payload.get("results", []), "fp")
    false_negative_rows = collect_mistake_rows(payload.get("results", []), "fn")
    disagreement_rows = collect_disagreement_rows(payload.get("results", []))
    best_model, best_f1 = best_model_by_f1(metric_rows)
    distribution = label_distribution(payload.get("results", []))

    run_metadata = {
        "run_id": run_id,
        "timestamp": timestamp,
        "dataset_path": str(dataset_path),
        "dataset_name": dataset_path.name,
        "total_rows": summary.get("total_prompts", 0),
        "has_ground_truth": bool(summary.get("has_ground_truth")),
        "label_distribution": distribution,
        "selected_models": selected_models,
        "model_count": len(selected_models),
        "best_model_by_f1": best_model,
        "best_f1": best_f1,
        "output_folder": display_path(run_folder),
        "validation": payload.get("validation", {}),
    }
    metrics_payload = {
        "run_metadata": run_metadata,
        "metrics": metric_rows,
        "note": None if summary.get("has_ground_truth") else "Ground-truth label not found. Metrics are not available.",
    }

    paths = {
        "metrics_summary_csv": run_folder / "metrics_summary.csv",
        "metrics_summary_json": run_folder / "metrics_summary.json",
        "predictions_full_csv": run_folder / "predictions_full.csv",
        "predictions_full_json": run_folder / "predictions_full.json",
        "false_positives_csv": run_folder / "false_positives.csv",
        "false_negatives_csv": run_folder / "false_negatives.csv",
        "model_disagreements_csv": run_folder / "model_disagreements.csv",
        "markdown_report": run_folder / "batch_evaluation_report.md",
        "run_metadata_json": run_folder / "run_metadata.json",
    }
    paths["metrics_summary_csv"].write_text(metrics_summary_csv(metric_rows), encoding="utf-8")
    paths["metrics_summary_json"].write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["predictions_full_csv"].write_text(exports.get("csv_content", ""), encoding="utf-8")
    paths["predictions_full_json"].write_text(
        exports.get("json_content", json.dumps(payload, ensure_ascii=False, indent=2)),
        encoding="utf-8",
    )
    paths["false_positives_csv"].write_text(rows_to_csv(false_positive_rows, MISTAKE_FIELDS), encoding="utf-8")
    paths["false_negatives_csv"].write_text(rows_to_csv(false_negative_rows, MISTAKE_FIELDS), encoding="utf-8")
    paths["model_disagreements_csv"].write_text(rows_to_csv(disagreement_rows, DISAGREEMENT_FIELDS), encoding="utf-8")
    paths["markdown_report"].write_text(
        build_markdown_report(
            payload,
            dataset_path,
            metric_rows,
            false_positive_rows,
            false_negative_rows,
            disagreement_rows,
            preview_rows=preview_rows,
        ),
        encoding="utf-8",
    )
    paths["run_metadata_json"].write_text(json.dumps(run_metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    index_row = {
        "run_id": run_id,
        "timestamp": timestamp,
        "dataset_path": str(dataset_path),
        "dataset_name": dataset_path.name,
        "total_rows": summary.get("total_prompts", 0),
        "has_ground_truth": bool(summary.get("has_ground_truth")),
        "model_count": len(selected_models),
        "best_model_by_f1": best_model,
        "best_f1": "" if best_f1 is None else f"{best_f1:.6f}",
        "output_folder": display_path(run_folder),
    }
    write_index(base_output_dir, index_row)
    write_readme(base_output_dir)
    return paths, run_metadata


def print_completion(run_folder: Path) -> None:
    print("Evaluation completed.")
    print("")
    print("Run folder:")
    print(display_path(run_folder) + "/")
    print("")
    print("Main report:")
    print("batch_evaluation_report.md")
    print("")
    print("Metrics:")
    print("metrics_summary.csv")
    print("")
    print("Predictions:")
    print("predictions_full.csv")


def main() -> int:
    args = parse_args()
    try:
        dataset_path = resolve_dataset_path(args.file)
        items = read_dataset(dataset_path, max_items=args.max_items)
        hybrid_config = {
            "traditional_model": args.hybrid_traditional,
            "transformer_model": args.hybrid_transformer,
            "use_rule_based": not args.disable_rule_based,
            "decision_strategy": args.hybrid_strategy,
        }
        payload = evaluate_batch_items(
            items=items,
            models=args.models,
            hybrid_config=hybrid_config,
            dataset_name=dataset_path.name,
            max_items=args.max_items,
        )

        base_output_dir = resolve_output_base(args.output_dir)
        report_paths, _run_metadata = write_reports(
            payload,
            base_output_dir,
            dataset_path,
            args.run_name,
            preview_rows=args.preview_rows,
        )
        print_completion(report_paths["markdown_report"].parent)
        return 0 if payload.get("status") == "completed" else 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
