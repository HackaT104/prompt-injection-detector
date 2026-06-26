"""Calibrate runtime thresholds for traditional TF-IDF and Transformer models.

This script does not train or overwrite model checkpoints. It only loads existing
models, computes positive-class risk scores on a labeled calibration dataset, scans
thresholds from 0.01 to 0.99, and writes threshold/report artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.batch_evaluation import parse_dataset_content, validate_batch_items  # noqa: E402
from src.detector import MODEL_FILES, load_model_artifacts, load_thresholds  # noqa: E402
from src.preprocessing import prepare_text_for_detection  # noqa: E402
from src.thresholding import choose_threshold, metrics_at_threshold  # noqa: E402
from src.transformer_utils import (  # noqa: E402
    _load_transformer_artifacts_cached,
    import_optional,
    is_finetuned_transformer_checkpoint,
    resolve_transformer_model_dir,
    safe_model_dir_name,
    softmax_positive_scores,
)


DEFAULT_DATASET = PROJECT_ROOT / "datasets" / "test" / "Prompt_INJECTION_And_Benign_DATASET.jsonl"
TRADITIONAL_MODELS = ["logistic_regression", "linear_svm", "random_forest"]
TRANSFORMER_MODELS = ["distilbert_v3", "roberta_v3", "xlm_roberta_v3"]
ALL_MODELS = [*TRADITIONAL_MODELS, *TRANSFORMER_MODELS]
MODEL_THRESHOLDS_PATH = PROJECT_ROOT / "models" / "model_thresholds.json"
TRANSFORMER_THRESHOLDS_PATH = PROJECT_ROOT / "models" / "transformer_thresholds.json"
COMBINED_THRESHOLDS_PATH = PROJECT_ROOT / "models" / "thresholds.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
SEARCH_REPORT_PATH = REPORTS_DIR / "threshold_search_report.csv"
PR_REPORT_PATH = REPORTS_DIR / "threshold_precision_recall_by_threshold.csv"
SUMMARY_REPORT_PATH = REPORTS_DIR / "threshold_summary.md"
SUMMARY_JSON_PATH = REPORTS_DIR / "threshold_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate thresholds for all prompt injection models.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="CSV/JSON/JSONL/TXT calibration dataset.")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS, help="Models to calibrate.")
    parser.add_argument("--metric", choices=["f1", "f2", "constraint"], default="f1")
    parser.add_argument("--min-recall", type=float, default=0.90)
    parser.add_argument("--min-precision", type=float, default=0.80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--use-cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-items", type=int, default=100000)
    return parser.parse_args()


def _detect_text_encoding(path: Path) -> str:
    sample = path.read_bytes()[:1024 * 1024]
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            sample.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "latin1"


def read_dataset(path: Path) -> list[dict[str, Any]]:
    content = path.read_text(encoding=_detect_text_encoding(path), errors="replace")
    return parse_dataset_content(path.name, content)


def load_labeled_dataset(path: Path, max_items: int) -> tuple[list[str], list[int], dict[str, Any]]:
    items = read_dataset(path)
    validation = validate_batch_items(items, max_items=max_items, dataset_name=path.name)
    if not validation["valid"]:
        raise ValueError("Dataset validation failed: " + "; ".join(validation["errors"]))
    if not validation["has_ground_truth"]:
        raise ValueError("Calibration dataset must contain ground-truth labels.")

    rows = validation["items"]
    texts = [str(row["text"]) for row in rows]
    labels = [int(row["ground_truth_label"]) for row in rows]
    dataset_report = {
        "path": str(path),
        "rows": len(rows),
        "total_rows": validation["total_rows"],
        "text_column_detected": validation["text_column_detected"],
        "label_column_detected": validation["label_column_detected"],
        "label_mapping": validation["label_mapping"],
        "label_distribution": {
            "safe_0": int(sum(1 for label in labels if label == 0)),
            "injection_1": int(sum(1 for label in labels if label == 1)),
        },
    }
    return texts, labels, dataset_report


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def traditional_scores(model_name: str, texts: list[str]) -> tuple[np.ndarray, list[str]]:
    model, vectorizer = load_model_artifacts(model_name)
    prepared_texts = [prepare_text_for_detection(text)["cleaned_text"] for text in texts]
    vectorized = vectorizer.transform(prepared_texts)
    warnings: list[str] = []

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vectorized)
        classes = list(getattr(model, "classes_", [0, 1]))
        positive_index = classes.index(1) if 1 in classes else 1
        return probabilities[:, positive_index].astype(float), warnings

    if hasattr(model, "decision_function"):
        raw_scores = model.decision_function(vectorized)
        warnings.append("Model does not expose predict_proba; decision_function was normalized with sigmoid.")
        return np.asarray([_sigmoid(float(score)) for score in raw_scores], dtype=float), warnings

    warnings.append("Model does not expose predict_proba or decision_function; predict() labels were used as scores.")
    return model.predict(vectorized).astype(float), warnings


def _transformer_scores_impl(
    model_name: str,
    texts: list[str],
    batch_size: int,
    max_length: int,
    use_cuda: bool,
) -> tuple[np.ndarray, Path]:
    model_dir = resolve_transformer_model_dir(model_name)
    if not is_finetuned_transformer_checkpoint(model_dir):
        raise FileNotFoundError(f"Transformer checkpoint not ready: {model_dir}")

    torch, tokenizer, model, device = _load_transformer_artifacts_cached(str(model_dir.resolve()), use_cuda)
    scores: list[float] = []
    prepared_texts = [prepare_text_for_detection(text)["cleaned_text"] for text in texts]
    for start in range(0, len(prepared_texts), batch_size):
        batch = prepared_texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
        scores.extend(float(score) for score in softmax_positive_scores(outputs.logits.detach().cpu().numpy()))

    _load_transformer_artifacts_cached.cache_clear()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return np.asarray(scores, dtype=float), model_dir


def transformer_scores(
    model_name: str,
    texts: list[str],
    batch_size: int,
    max_length: int,
    use_cuda: bool,
) -> tuple[np.ndarray, list[str], Path]:
    warnings: list[str] = []
    try:
        scores, model_dir = _transformer_scores_impl(model_name, texts, batch_size, max_length, use_cuda)
        return scores, warnings, model_dir
    except RuntimeError as exc:
        is_oom = "out of memory" in str(exc).lower()
        if not use_cuda or not is_oom:
            raise
        retry_batch = max(1, batch_size // 2)
        while retry_batch >= 1:
            try:
                _load_transformer_artifacts_cached.cache_clear()
                torch = import_optional("torch")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                warnings.append(
                    f"CUDA out of memory with batch_size={batch_size}; retried with batch_size={retry_batch}."
                )
                scores, model_dir = _transformer_scores_impl(model_name, texts, retry_batch, max_length, True)
                return scores, warnings, model_dir
            except RuntimeError as retry_exc:
                if "out of memory" not in str(retry_exc).lower():
                    raise
                retry_batch //= 2
        try:
            _load_transformer_artifacts_cached.cache_clear()
            torch = import_optional("torch")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        warnings.append("CUDA out of memory during calibration; fell back to CPU for this model.")
        scores, model_dir = _transformer_scores_impl(model_name, texts, batch_size, max_length, False)
        return scores, warnings, model_dir


def is_transformer_candidate(model_name: str) -> bool:
    if model_name in TRANSFORMER_MODELS:
        return True
    try:
        return is_finetuned_transformer_checkpoint(resolve_transformer_model_dir(model_name))
    except Exception:
        return False


def summary_row(model_name: str, analysis: dict[str, Any], warnings: list[str], model_path: str | None = None) -> dict[str, Any]:
    selected = analysis["selected_metrics"]
    return {
        "model": model_name,
        "best_metric": analysis["best_metric"],
        "evaluation_threshold": analysis["evaluation_threshold"],
        "warn_threshold": analysis["runtime_warn_threshold"],
        "block_threshold": analysis["runtime_block_threshold"],
        "precision": selected["precision"],
        "recall": selected["recall"],
        "f1": selected["f1"],
        "f2": selected["f2"],
        "tn": selected["true_negative"],
        "fp": selected["false_positive"],
        "fn": selected["false_negative"],
        "tp": selected["true_positive"],
        "selection_reason": analysis["selection_reason"],
        "runtime_reason": analysis["runtime_reason"],
        "warnings": warnings,
        "model_path": model_path,
    }


def threshold_payload(row: dict[str, Any], dataset_report: dict[str, Any]) -> dict[str, Any]:
    return {
        "evaluation_threshold": float(row["evaluation_threshold"]),
        "runtime_warn_threshold": float(row["warn_threshold"]),
        "runtime_block_threshold": float(row["block_threshold"]),
        "best_metric": row["best_metric"],
        "precision": float(row["precision"]),
        "recall": float(row["recall"]),
        "f1": float(row["f1"]),
        "f2": float(row["f2"]),
        "tn": int(row["tn"]),
        "fp": int(row["fp"]),
        "fn": int(row["fn"]),
        "tp": int(row["tp"]),
        "dataset": dataset_report["path"],
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
        "model_path": row.get("model_path"),
        "warnings": row.get("warnings", []),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_json_if_exists(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(default)
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return dict(default)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return "" if value is None else str(value)


def write_markdown_summary(path: Path, summary: dict[str, Any]) -> None:
    rows = summary["models"]
    lines = [
        "# Threshold Calibration Summary",
        "",
        "Report này được tạo bằng cách load model đã train/fine-tune, lấy risk_score của class attack = 1,",
        "quét threshold từ 0.01 đến 0.99 với step 0.01 và chọn threshold theo metric cấu hình.",
        "",
        "## Dataset",
        "",
        f"- Path: `{summary['dataset']['path']}`",
        f"- Rows used: `{summary['dataset']['rows']}` / total `{summary['dataset']['total_rows']}`",
        f"- Text column: `{summary['dataset']['text_column_detected']}`",
        f"- Label column: `{summary['dataset']['label_column_detected']}`",
        f"- Label distribution: SAFE/0=`{summary['dataset']['label_distribution']['safe_0']}`, INJECTION/1=`{summary['dataset']['label_distribution']['injection_1']}`",
        "",
        "## Threshold Table",
        "",
        "| model | best_metric | evaluation_threshold | warn_threshold | block_threshold | precision | recall | f1 | f2 | tn | fp | fn | tp |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model"]),
                    str(row["best_metric"]),
                    fmt(float(row["evaluation_threshold"])),
                    fmt(float(row["warn_threshold"])),
                    fmt(float(row["block_threshold"])),
                    fmt(float(row["precision"])),
                    fmt(float(row["recall"])),
                    fmt(float(row["f1"])),
                    fmt(float(row["f2"])),
                    str(row["tn"]),
                    str(row["fp"]),
                    str(row["fn"]),
                    str(row["tp"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Runtime Rule",
            "",
            "- `risk_score < warn_threshold` -> allow",
            "- `warn_threshold <= risk_score < block_threshold` -> warn",
            "- `risk_score >= block_threshold` -> block",
            "",
            "## Notes",
            "",
            "- `evaluation_threshold` dùng cho report/evaluate.",
            "- `warn_threshold` và `block_threshold` dùng cho API/runtime.",
            "- `warn_threshold` luôn nhỏ hơn `block_threshold`.",
            "- File chi tiết theo từng threshold: `reports/threshold_search_report.csv`.",
        ]
    )
    warnings = [f"- {row['model']}: {'; '.join(row['warnings'])}" for row in rows if row.get("warnings")]
    if warnings:
        lines.extend(["", "## Warnings", "", *warnings])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    dataset_path = dataset_path.resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    torch = import_optional("torch")
    if args.use_cuda and not torch.cuda.is_available():
        print("[threshold] CUDA is not available; Transformer calibration will use CPU.")

    texts, y_true, dataset_report = load_labeled_dataset(dataset_path, args.max_items)
    print(f"[threshold] Loaded {len(texts)} labeled rows from {dataset_path}")

    traditional_payload: dict[str, Any] = load_json_if_exists(MODEL_THRESHOLDS_PATH, {})
    transformer_payload: dict[str, Any] = load_json_if_exists(TRANSFORMER_THRESHOLDS_PATH, {"models": {}})
    transformer_payload.setdefault("models", {})
    summary_rows: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []

    selected_models = list(dict.fromkeys(args.models))
    for model_name in selected_models:
        print(f"[threshold] Calibrating {model_name}...")
        model_warnings: list[str] = []
        model_path: str | None = None
        try:
            if model_name in TRADITIONAL_MODELS:
                scores, model_warnings = traditional_scores(model_name, texts)
                model_path = str(MODEL_FILES[model_name]["model"])
            elif is_transformer_candidate(model_name):
                scores, model_warnings, model_dir = transformer_scores(
                    model_name,
                    texts,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    use_cuda=args.use_cuda,
                )
                model_path = str(model_dir)
            else:
                print(f"[threshold] Skip unsupported model: {model_name}")
                continue
        except Exception as exc:
            print(f"[threshold] {model_name} unavailable: {exc}")
            continue

        analysis = choose_threshold(
            y_true,
            scores,
            min_recall=args.min_recall if args.metric == "constraint" else None,
            min_precision=args.min_precision if args.metric == "constraint" else None,
            optimization_metric=args.metric,
        )
        row = summary_row(model_name, analysis, model_warnings, model_path=model_path)
        summary_rows.append(row)

        for candidate in analysis["candidate_metrics"]:
            search_rows.append(
                {
                    "model": model_name,
                    "threshold": candidate["threshold"],
                    "accuracy": candidate["accuracy"],
                    "precision": candidate["precision"],
                    "recall": candidate["recall"],
                    "f1": candidate["f1"],
                    "f2": candidate["f2"],
                    "tn": candidate["true_negative"],
                    "fp": candidate["false_positive"],
                    "fn": candidate["false_negative"],
                    "tp": candidate["true_positive"],
                    "is_selected": float(candidate["threshold"]) == float(analysis["evaluation_threshold"]),
                }
            )

        payload = threshold_payload(row, dataset_report)
        if model_name in TRADITIONAL_MODELS:
            traditional_payload[model_name] = payload
        else:
            model_dir_name = Path(str(model_path)).name if model_path else safe_model_dir_name(model_name)
            transformer_payload["models"][model_name] = payload
            transformer_payload["models"][model_dir_name] = payload

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "models").mkdir(parents=True, exist_ok=True)

    MODEL_THRESHOLDS_PATH.write_text(json.dumps(traditional_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    TRANSFORMER_THRESHOLDS_PATH.write_text(json.dumps(transformer_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    combined_payload = {
        **traditional_payload,
        "models": {
            **transformer_payload.get("models", {}),
        },
        "traditional_models": traditional_payload,
        "transformer_models": transformer_payload.get("models", {}),
    }
    COMBINED_THRESHOLDS_PATH.write_text(json.dumps(combined_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    write_csv(
        SEARCH_REPORT_PATH,
        search_rows,
        ["model", "threshold", "accuracy", "precision", "recall", "f1", "f2", "tn", "fp", "fn", "tp", "is_selected"],
    )
    write_csv(
        PR_REPORT_PATH,
        search_rows,
        ["model", "threshold", "precision", "recall", "f1", "f2", "fp", "fn", "is_selected"],
    )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_report,
        "metric": args.metric,
        "models": summary_rows,
        "outputs": {
            "traditional_thresholds": str(MODEL_THRESHOLDS_PATH),
            "transformer_thresholds": str(TRANSFORMER_THRESHOLDS_PATH),
            "combined_thresholds": str(COMBINED_THRESHOLDS_PATH),
            "threshold_search_report": str(SEARCH_REPORT_PATH),
            "precision_recall_report": str(PR_REPORT_PATH),
            "threshold_summary": str(SUMMARY_REPORT_PATH),
        },
    }
    SUMMARY_JSON_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_summary(SUMMARY_REPORT_PATH, summary)

    load_thresholds.cache_clear()
    print(json.dumps(summary["outputs"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
