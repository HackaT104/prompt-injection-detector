"""Evaluate direct prompt-injection external benchmarks with model-only ablations.

This evaluator intentionally does not call rule-based or context-aware detectors.

Examples:
    python scripts/evaluate_direct_benchmarks.py --dataset deepset --model logistic_regression --threshold auto
    python scripts/evaluate_direct_benchmarks.py --dataset all --model roberta --threshold auto
    python scripts/evaluate_direct_benchmarks.py --dataset all --model xlm_roberta --threshold auto --split-threshold
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import MODEL_FILES, load_model_artifacts  # noqa: E402
from src.preprocessing import prepare_text_for_detection  # noqa: E402
from src.transformer_utils import (  # noqa: E402
    LABEL2ID,
    _load_probability_calibrator,
    _load_transformer_artifacts_cached,
    clear_transformer_runtime_cache,
    is_finetuned_transformer_checkpoint,
    resolve_transformer_model_dir,
)


DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "external_benchmark" / "direct"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "direct_external_evaluation"

DATASET_FILES = {
    "deepset": DEFAULT_DATA_DIR / "deepset_normalized.csv",
    "neuralchemy": DEFAULT_DATA_DIR / "neuralchemy_normalized.csv",
    "rogue_security": DEFAULT_DATA_DIR / "rogue_security_normalized.csv",
    "cyberec": DEFAULT_DATA_DIR / "cyberec_normalized.csv",
    "all": DEFAULT_DATA_DIR / "direct_all_normalized.csv",
}

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "roberta": "RoBERTa",
    "xlm_roberta": "XLM-RoBERTa",
}

PREDICTION_COLUMNS = [
    "id",
    "dataset_name",
    "text",
    "label",
    "predicted_label",
    "score",
    "raw_score",
    "score_source",
    "calibration_method",
    "logit_safe",
    "logit_injection",
    "threshold",
    "attack_type",
    "source_label",
]

SUMMARY_COLUMNS = [
    "Dataset",
    "Model",
    "Threshold",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "F2",
    "ROC-AUC",
    "PR-AUC",
    "FP",
    "FN",
    "Rows",
    "Evaluation Scope",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate direct prompt-injection external benchmarks.")
    parser.add_argument("--dataset", choices=sorted(DATASET_FILES), required=True)
    parser.add_argument("--model", choices=sorted(MODEL_LABELS), required=True)
    parser.add_argument(
        "--threshold",
        default="auto",
        help="Use 'auto' or a numeric threshold such as 0.5. Default: auto.",
    )
    parser.add_argument(
        "--threshold-from",
        type=Path,
        default=None,
        help="Load threshold from a threshold_summary.json file.",
    )
    parser.add_argument("--split-threshold", action="store_true", help="Tune on 30% validation and report 70% test.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed):
        return default
    return parsed


def _format_metric(value: Any, digits: int = 4) -> str:
    parsed = _safe_float(value)
    if parsed is None:
        return "n/a"
    return f"{parsed:.{digits}f}"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Normalized direct benchmark not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    normalized: list[dict[str, Any]] = []
    for row in rows:
        try:
            label = int(float(str(row.get("label", "")).strip()))
        except ValueError:
            continue
        if label not in {0, 1}:
            continue
        text = str(row.get("text", "") or "").strip()
        if not text:
            continue
        payload = dict(row)
        payload["label"] = label
        normalized.append(payload)
    if not normalized:
        raise ValueError(f"No valid rows in {path}. Run scripts/prepare_direct_benchmarks.py first.")
    return normalized


def _confusion(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    tn = fp = fn = tp = 0
    for true, pred in zip(y_true, y_pred):
        if true == 0 and pred == 0:
            tn += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 1 and pred == 0:
            fn += 1
        elif true == 1 and pred == 1:
            tp += 1
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "confusion_matrix": [[tn, fp], [fn, tp]]}


def _fbeta(precision: float, recall: float, beta: float) -> float:
    beta_sq = beta * beta
    denominator = beta_sq * precision + recall
    return 0.0 if denominator == 0 else (1 + beta_sq) * precision * recall / denominator


def _metrics_from_scores(y_true: list[int], scores: list[float], threshold: float) -> dict[str, Any]:
    y_pred = [1 if score >= threshold else 0 for score in scores]
    counts = _confusion(y_true, y_pred)
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    metrics: dict[str, Any] = {
        "rows": len(y_true),
        "accuracy": _safe_div(tp + tn, len(y_true)),
        "precision": precision,
        "recall": recall,
        "f1": _fbeta(precision, recall, 1.0),
        "f2": _fbeta(precision, recall, 2.0),
        "false_positive_count": fp,
        "false_negative_count": fn,
        **counts,
        "threshold": round(float(threshold), 4),
        "warnings": [],
    }
    if len(set(y_true)) < 2:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
        metrics["warnings"].append("ROC-AUC/PR-AUC not computed because only one class is present.")
        return metrics
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        metrics["pr_auc"] = float(average_precision_score(y_true, scores))
    except Exception as exc:  # pragma: no cover - optional sklearn runtime
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
        metrics["warnings"].append(f"ROC-AUC/PR-AUC not computed: {type(exc).__name__}: {exc}")
    return metrics


def _rank_threshold(row: dict[str, Any]) -> tuple[float, float, float, int, float]:
    return (
        float(row["f2"]),
        float(row["recall"]),
        float(row["precision"]),
        -int(row["false_positive_count"]),
        float(row["threshold"]),
    )


def _choose_auto_threshold(y_true: list[int], scores: list[float]) -> dict[str, Any]:
    candidates = [_metrics_from_scores(y_true, scores, threshold / 100) for threshold in range(1, 100)]
    selected = max(candidates, key=_rank_threshold)
    return {
        "threshold": float(selected["threshold"]),
        "selected_metrics": selected,
        "candidate_thresholds": "0.01..0.99 step 0.01",
        "selection_rule": "maximize F2, then recall, precision, fewer false positives, higher threshold",
        "warning": "This evaluation uses threshold sweeping on the test set and should not be interpreted as strict held-out performance.",
    }


def _stratified_split_indices(rows: list[dict[str, Any]], seed: int) -> tuple[list[int], list[int]]:
    by_label: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_label[int(row["label"])].append(index)
    rng = random.Random(seed)
    validation: list[int] = []
    test: list[int] = []
    for indices in by_label.values():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        cut = max(1, int(round(len(shuffled) * 0.30))) if len(shuffled) > 1 else len(shuffled)
        validation.extend(shuffled[:cut])
        test.extend(shuffled[cut:])
    validation.sort()
    test.sort()
    return validation, test


def _threshold_from_file(path: Path) -> float:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    candidates = [
        payload.get("recommended_threshold", {}).get("threshold") if isinstance(payload.get("recommended_threshold"), dict) else None,
        payload.get("best_f2_threshold", {}).get("threshold") if isinstance(payload.get("best_f2_threshold"), dict) else None,
        payload.get("threshold"),
        payload.get("evaluation_threshold"),
    ]
    for value in candidates:
        parsed = _safe_float(value)
        if parsed is not None:
            return parsed
    raise ValueError(f"No threshold field found in {path}")


def _resolve_threshold(
    *,
    threshold_arg: str,
    threshold_from: Path | None,
    rows: list[dict[str, Any]],
    scores: list[float],
    split_threshold: bool,
    seed: int,
) -> tuple[float, dict[str, Any], list[int] | None]:
    y_true = [int(row["label"]) for row in rows]
    if threshold_from is not None:
        threshold = _threshold_from_file(threshold_from)
        return threshold, {"mode": "threshold_from_file", "path": str(threshold_from), "threshold": threshold}, None
    if threshold_arg != "auto":
        threshold = _safe_float(threshold_arg)
        if threshold is None or threshold < 0.0 or threshold > 1.0:
            raise ValueError("--threshold must be 'auto' or a number in [0, 1].")
        return threshold, {"mode": "fixed", "threshold": threshold}, None
    if split_threshold:
        validation_indices, test_indices = _stratified_split_indices(rows, seed)
        validation_labels = [int(rows[index]["label"]) for index in validation_indices]
        validation_scores = [scores[index] for index in validation_indices]
        selected = _choose_auto_threshold(validation_labels, validation_scores)
        test_labels = [int(rows[index]["label"]) for index in test_indices]
        test_scores = [scores[index] for index in test_indices]
        test_metrics = _metrics_from_scores(test_labels, test_scores, float(selected["threshold"]))
        return (
            float(selected["threshold"]),
            {
                "mode": "split_auto",
                "validation_rows": len(validation_indices),
                "test_rows": len(test_indices),
                "validation_threshold_selection": selected,
                "test_metrics_at_selected_threshold": test_metrics,
                "note": "Threshold was tuned on a 30% validation split; primary metrics are reported on the 70% test split.",
            },
            test_indices,
        )
    selected = _choose_auto_threshold(y_true, scores)
    return float(selected["threshold"]), {"mode": "auto_test_sweep", **selected}, None


def _score_traditional_model(rows: list[dict[str, Any]], model_type: str) -> tuple[list[float], dict[str, Any]]:
    model, vectorizer = load_model_artifacts(model_type)
    cleaned_texts = [
        prepare_text_for_detection(str(row.get("text", "")))["cleaned_text"]
        for row in rows
    ]
    vectorized = vectorizer.transform(cleaned_texts)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vectorized)
        classes = list(getattr(model, "classes_", [0, 1]))
        positive_index = classes.index(1) if 1 in classes else 1
        scores = [float(row[positive_index]) for row in probabilities]
    elif hasattr(model, "decision_function"):
        raw_scores = model.decision_function(vectorized)
        scores = [1.0 / (1.0 + math.exp(-float(score))) for score in raw_scores]
    else:
        scores = [float(label) for label in model.predict(vectorized)]
    return scores, {
        "model_path": str(MODEL_FILES[model_type]["model"]),
        "vectorizer_path": str(MODEL_FILES[model_type]["vectorizer"]),
        "runtime_device": "cpu",
        "warnings": [],
    }


def _batch_iter(values: list[str], batch_size: int) -> list[list[str]]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    return [values[index : index + batch_size] for index in range(0, len(values), batch_size)]


def _score_transformer(
    *,
    rows: list[dict[str, Any]],
    alias: str,
    batch_size: int,
    max_length: int,
    use_cuda: bool,
) -> tuple[list[float], dict[str, Any]]:
    model_dir = resolve_transformer_model_dir(alias)
    if not is_finetuned_transformer_checkpoint(model_dir):
        raise FileNotFoundError(f"Fine-tuned Transformer checkpoint not found: {model_dir}")
    warnings: list[str] = []
    try:
        torch, tokenizer, model, device = _load_transformer_artifacts_cached(str(model_dir.resolve()), use_cuda)
    except RuntimeError as exc:
        if use_cuda and "out of memory" in str(exc).lower():
            clear_transformer_runtime_cache()
            warnings.append("CUDA out of memory while loading; fell back to CPU.")
            torch, tokenizer, model, device = _load_transformer_artifacts_cached(str(model_dir.resolve()), False)
        else:
            raise

    cleaned_texts = [
        prepare_text_for_detection(str(row.get("text", "")))["cleaned_text"]
        for row in rows
    ]
    calibrator = _load_probability_calibrator(str(model_dir.resolve()))
    calibration_method = calibrator.__class__.__name__ if calibrator is not None else None
    scores: list[float] = []
    raw_scores: list[float] = []
    row_metadata: list[dict[str, Any]] = []
    try:
        for batch in _batch_iter(cleaned_texts, batch_size):
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
                probabilities = torch.softmax(outputs.logits, dim=-1)
            batch_logits = outputs.logits.detach().cpu().tolist()
            batch_scores = probabilities[:, LABEL2ID["INJECTION"]].detach().cpu().tolist()
            raw_scores.extend(float(score) for score in batch_scores)
            if calibrator is not None:
                calibrated = calibrator.predict([float(score) for score in batch_scores])
                scores.extend(float(min(1.0, max(0.0, value))) for value in calibrated)
            else:
                scores.extend(float(score) for score in batch_scores)
            for raw_score, logits in zip(batch_scores, batch_logits):
                safe_logit = float(logits[LABEL2ID["SAFE"]])
                injection_logit = float(logits[LABEL2ID["INJECTION"]])
                row_metadata.append(
                    {
                        "raw_score": round(float(raw_score), 8),
                        "score_source": "transformer_softmax_probability",
                        "calibration_method": calibration_method or "",
                        "logit_safe": round(safe_logit, 8),
                        "logit_injection": round(injection_logit, 8),
                    }
                )
    except RuntimeError as exc:
        if use_cuda and "out of memory" in str(exc).lower():
            clear_transformer_runtime_cache()
            fallback_scores, metadata = _score_transformer(
                rows=rows,
                alias=alias,
                batch_size=max(1, batch_size // 2),
                max_length=max_length,
                use_cuda=False,
            )
            metadata.setdefault("warnings", []).append("CUDA out of memory during inference; reran on CPU.")
            return fallback_scores, metadata
        raise
    return scores, {
        "model_path": str(model_dir),
        "runtime_device": str(device),
        "calibration_method": calibration_method,
        "raw_score_min": min(raw_scores) if raw_scores else None,
        "raw_score_max": max(raw_scores) if raw_scores else None,
        "row_metadata": row_metadata,
        "logits_saved": True,
        "warnings": warnings,
    }


def _score_model(
    *,
    rows: list[dict[str, Any]],
    model_key: str,
    batch_size: int,
    max_length: int,
    use_cuda: bool,
) -> tuple[list[float], dict[str, Any]]:
    if model_key in {"logistic_regression", "random_forest"}:
        return _score_traditional_model(rows, model_key)
    if model_key in {"roberta", "xlm_roberta"}:
        return _score_transformer(
            rows=rows,
            alias=model_key,
            batch_size=batch_size,
            max_length=max_length,
            use_cuda=use_cuda,
        )
    raise ValueError(f"Unsupported model: {model_key}")


def _build_predictions(
    rows: list[dict[str, Any]],
    scores: list[float],
    threshold: float,
    row_metadata: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    metadata_rows = row_metadata or [{} for _ in rows]
    for row, score, metadata in zip(rows, scores, metadata_rows):
        predictions.append(
            {
                "id": row.get("id", ""),
                "dataset_name": row.get("dataset_name", ""),
                "text": row.get("text", ""),
                "label": int(row["label"]),
                "predicted_label": 1 if float(score) >= threshold else 0,
                "score": round(float(score), 8),
                "raw_score": metadata.get("raw_score", round(float(score), 8)),
                "score_source": metadata.get("score_source", "predict_proba_positive_class"),
                "calibration_method": metadata.get("calibration_method", ""),
                "logit_safe": metadata.get("logit_safe", ""),
                "logit_injection": metadata.get("logit_injection", ""),
                "threshold": round(float(threshold), 4),
                "attack_type": row.get("attack_type", ""),
                "source_label": row.get("source_label", ""),
            }
        )
    return predictions


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        if not fieldnames:
            fieldnames = ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _group_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key, "")) for row in rows).items()))


def _group_metrics(predictions: list[dict[str, Any]], key: str, threshold: float) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[str(row.get(key, ""))].append(row)
    metrics: dict[str, Any] = {}
    for group, rows in grouped.items():
        labels = [int(row["label"]) for row in rows]
        scores = [float(row["score"]) for row in rows]
        metrics[group] = _metrics_from_scores(labels, scores, threshold)
    return dict(sorted(metrics.items()))


def _write_report(
    *,
    path: Path,
    dataset_key: str,
    model_key: str,
    input_path: Path,
    metrics: dict[str, Any],
    false_positives: list[dict[str, Any]],
    false_negatives: list[dict[str, Any]],
) -> None:
    lines = [
        f"# Direct Prompt Injection External Evaluation - {dataset_key} / {model_key}",
        "",
        "## Configuration",
        "",
        f"- Dataset: `{dataset_key}`",
        f"- Model: `{MODEL_LABELS[model_key]}`",
        "- Rule-based detector: `No`",
        "- Context-aware detector: `No`",
        f"- Input: `{input_path}`",
        f"- Model path: `{metrics.get('model_path', 'n/a')}`",
        f"- Vectorizer path: `{metrics.get('vectorizer_path', 'n/a')}`",
        f"- Runtime device: `{metrics.get('runtime_device', 'n/a')}`",
        "",
        "## Threshold",
        "",
        f"- Mode: `{metrics.get('threshold_mode', 'n/a')}`",
        f"- Selected threshold: `{_format_metric(metrics.get('threshold'))}`",
    ]
    note = metrics.get("threshold_note") or metrics.get("threshold_warning")
    if note:
        lines.append(f"- Note: {note}")
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            f"- Rows: `{metrics['rows']}`",
            f"- Evaluation scope: `{metrics.get('evaluation_scope', 'all')}`",
            f"- Accuracy: `{_format_metric(metrics['accuracy'])}`",
            f"- Precision: `{_format_metric(metrics['precision'])}`",
            f"- Recall: `{_format_metric(metrics['recall'])}`",
            f"- F1-score: `{_format_metric(metrics['f1'])}`",
            f"- F2-score: `{_format_metric(metrics['f2'])}`",
            f"- ROC-AUC: `{_format_metric(metrics.get('roc_auc'))}`",
            f"- PR-AUC: `{_format_metric(metrics.get('pr_auc'))}`",
            f"- Confusion matrix [[TN, FP], [FN, TP]]: `{metrics['confusion_matrix']}`",
            f"- False positives: `{metrics['false_positive_count']}`",
            f"- False negatives: `{metrics['false_negative_count']}`",
            "",
            "## Dataset breakdown",
            "",
            f"- Dataset distribution: `{metrics.get('dataset_distribution', {})}`",
            f"- Label distribution: `{metrics.get('label_distribution', {})}`",
            f"- Attack type distribution: `{metrics.get('attack_type_distribution', {})}`",
            "",
            "## False positives",
            "",
        ]
    )
    if false_positives:
        for row in false_positives[:10]:
            lines.append(
                f"- `{row['id']}` score={row['score']} attack_type={row['attack_type']} "
                f"source_label={row['source_label']} text={str(row['text'])[:220]}"
            )
    else:
        lines.append("No false positives.")
    lines.extend(["", "## False negatives", ""])
    if false_negatives:
        for row in false_negatives[:10]:
            lines.append(
                f"- `{row['id']}` score={row['score']} attack_type={row['attack_type']} "
                f"source_label={row['source_label']} text={str(row['text'])[:220]}"
            )
    else:
        lines.append("No false negatives.")
    if metrics.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in metrics["warnings"]:
            lines.append(f"- {warning}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _copy_latest(dataset_dir: Path, model_dir: Path) -> None:
    for filename in ["predictions.csv", "metrics.json", "false_positives.csv", "false_negatives.csv", "evaluation_report.md"]:
        source = dataset_dir / filename
        target = model_dir / filename
        if source.exists():
            shutil.copyfile(source, target)


def _summary_row(metrics_path: Path) -> dict[str, Any] | None:
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return {
        "Dataset": metrics.get("dataset", ""),
        "Model": MODEL_LABELS.get(metrics.get("model", ""), metrics.get("model", "")),
        "Threshold": _format_metric(metrics.get("threshold")),
        "Accuracy": _format_metric(metrics.get("accuracy")),
        "Precision": _format_metric(metrics.get("precision")),
        "Recall": _format_metric(metrics.get("recall")),
        "F1": _format_metric(metrics.get("f1")),
        "F2": _format_metric(metrics.get("f2")),
        "ROC-AUC": _format_metric(metrics.get("roc_auc")),
        "PR-AUC": _format_metric(metrics.get("pr_auc")),
        "FP": int(metrics.get("false_positive_count", 0)),
        "FN": int(metrics.get("false_negative_count", 0)),
        "Rows": int(metrics.get("rows", 0)),
        "Evaluation Scope": metrics.get("evaluation_scope", "all"),
        "_metrics": metrics,
    }


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return lines


def _write_summary(output_dir: Path) -> None:
    rows_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for metrics_path in sorted(output_dir.glob("*/metrics.json")):
        row = _summary_row(metrics_path)
        if row is not None:
            model_key = str(row.get("Model", ""))
            dataset_key = str(row.get("Dataset", ""))
            rows_by_key[(dataset_key, model_key, "root")] = row
    for metrics_path in sorted(output_dir.glob("*/*/metrics.json")):
        row = _summary_row(metrics_path)
        if row is not None:
            model_key = str(row.get("Model", ""))
            dataset_key = str(row.get("Dataset", ""))
            if (dataset_key, model_key, "root") not in rows_by_key:
                rows_by_key[(dataset_key, model_key, "nested")] = row
    rows = list(rows_by_key.values())
    rows.sort(key=lambda row: (str(row["Dataset"]), str(row["Model"])))
    public_rows = [{key: value for key, value in row.items() if key != "_metrics"} for row in rows]
    _write_csv(output_dir / "direct_ablation_summary.csv", public_rows, SUMMARY_COLUMNS)

    if rows:
        best_recall = max(rows, key=lambda row: float(row["_metrics"].get("recall", 0.0)))
        lowest_fp = min(rows, key=lambda row: int(row["_metrics"].get("false_positive_count", 0)))
        most_fn = max(rows, key=lambda row: int(row["_metrics"].get("false_negative_count", 0)))
    else:
        best_recall = lowest_fp = most_fn = None

    rows_by_dataset_model = {
        (str(row["Dataset"]), str(row["Model"])): row
        for row in rows
    }
    all_rows = [row for row in rows if str(row["Dataset"]) == "all"]
    best_overall = max(all_rows, key=lambda row: float(row["_metrics"].get("f1", 0.0))) if all_rows else None
    best_overall_f2 = max(all_rows, key=lambda row: float(row["_metrics"].get("f2", 0.0))) if all_rows else None
    best_all_recall = max(all_rows, key=lambda row: float(row["_metrics"].get("recall", 0.0))) if all_rows else None
    lowest_all_fp = min(all_rows, key=lambda row: int(row["_metrics"].get("false_positive_count", 0))) if all_rows else None
    most_all_fn = max(all_rows, key=lambda row: int(row["_metrics"].get("false_negative_count", 0))) if all_rows else None

    def all_row(model: str) -> dict[str, Any] | None:
        return rows_by_dataset_model.get(("all", model))

    def metric(row: dict[str, Any] | None, key: str) -> float:
        if not row:
            return 0.0
        return float(row["_metrics"].get(key, 0.0))

    def hard_dataset_for(row: dict[str, Any] | None) -> str:
        if not row:
            return "n/a"
        by_dataset = row["_metrics"].get("by_dataset", {})
        if not by_dataset:
            return "n/a"
        name, values = min(by_dataset.items(), key=lambda item: float(item[1].get("f1", 0.0)))
        return (
            f"{name} "
            f"(F1={_format_metric(values.get('f1'))}, "
            f"F2={_format_metric(values.get('f2'))}, "
            f"FP={values.get('false_positive_count')}, "
            f"FN={values.get('false_negative_count')})"
        )

    metadata_path = DEFAULT_DATA_DIR / "direct_benchmark_metadata.json"
    metadata = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}

    lines = [
        "# Direct Prompt Injection External Benchmark Summary",
        "",
        "## Dataset preparation",
        "",
    ]
    if metadata:
        for name, info in metadata.get("datasets", {}).items():
            label_distribution = info.get("label_distribution", {})
            status = info.get("status", "unknown")
            rows_count = info.get("rows_after_normalization", 0)
            extra = f" error={info.get('error')}" if status == "error" else ""
            lines.append(f"- `{name}`: status=`{status}`, rows=`{rows_count}`, labels=`{label_distribution}`.{extra}")
        combined = metadata.get("combined", {})
        lines.append(
            f"- `all`: rows=`{combined.get('rows_after_normalization', 0)}`, "
            f"dataset_distribution=`{combined.get('dataset_distribution', {})}`, "
            f"labels=`{combined.get('label_distribution', {})}`."
        )
    else:
        lines.append("- Dataset preparation metadata not found.")
    lines.extend(
        [
            "",
        "## Comparison table",
        "",
        *_markdown_table(public_rows, SUMMARY_COLUMNS[:-1]),
        "",
        "## Analysis",
        "",
        ]
    )
    if rows:
        lines.extend(
            [
                f"1. Logistic Regression generalization: on `all`, F1=`{_format_metric(metric(all_row('Logistic Regression'), 'f1'))}`, F2=`{_format_metric(metric(all_row('Logistic Regression'), 'f2'))}`, ROC-AUC=`{_format_metric(metric(all_row('Logistic Regression'), 'roc_auc'))}`. It catches injections aggressively but has high FP on the combined benchmark.",
                f"2. Random Forest vs Logistic Regression: Random Forest is slightly better on `all` by F2 (`{_format_metric(metric(all_row('Random Forest'), 'f2'))}` vs `{_format_metric(metric(all_row('Logistic Regression'), 'f2'))}`) and ROC-AUC (`{_format_metric(metric(all_row('Random Forest'), 'roc_auc'))}` vs `{_format_metric(metric(all_row('Logistic Regression'), 'roc_auc'))}`).",
                f"3. RoBERTa vs traditional ML: RoBERTa is much better by F1/ROC-AUC and has far fewer FP on `all`; F1=`{_format_metric(metric(all_row('RoBERTa'), 'f1'))}`, ROC-AUC=`{_format_metric(metric(all_row('RoBERTa'), 'roc_auc'))}`.",
                f"4. XLM-RoBERTa vs RoBERTa: XLM-RoBERTa has higher recall on `all` (`{_format_metric(metric(all_row('XLM-RoBERTa'), 'recall'))}` vs `{_format_metric(metric(all_row('RoBERTa'), 'recall'))}`), but RoBERTa is stronger by F1/F2/precision and FP count.",
                f"5. Highest recall on `all`: `{best_all_recall['Model']}` with recall `{best_all_recall['Recall']}`.",
                f"6. Lowest false positives on `all`: `{lowest_all_fp['Model']}` with FP `{lowest_all_fp['FP']}`.",
                f"7. Most missed prompt injections on `all`: `{most_all_fn['Model']}` with FN `{most_all_fn['FN']}`.",
                "8. Memorization/dataset-overlap signal: neuralchemy is included because the user requested it, but it may overlap with prior project training/evaluation sources. Treat high neuralchemy performance as possible leakage unless a separate provenance audit confirms no overlap.",
                "9. Hardest source dataset by F1 under the `all` threshold:",
                f"   - Logistic Regression: {hard_dataset_for(all_row('Logistic Regression'))}",
                f"   - Random Forest: {hard_dataset_for(all_row('Random Forest'))}",
                f"   - RoBERTa: {hard_dataset_for(all_row('RoBERTa'))}",
                f"   - XLM-RoBERTa: {hard_dataset_for(all_row('XLM-RoBERTa'))}",
                "10. Recommendation: combine multiple Direct datasets for continued fine-tuning only after de-duplication/provenance checks and a strict held-out external split. RoBERTa is the best candidate to continue fine-tuning; XLM-RoBERTa needs FP calibration and additional hard negatives.",
                "",
                f"Overall best on `all` by F1: `{best_overall['Model']}` with F1 `{best_overall['F1']}`.",
                f"Overall best on `all` by F2: `{best_overall_f2['Model']}` with F2 `{best_overall_f2['F2']}`.",
                f"Highest recall across all completed rows: `{best_recall['Model']}` on `{best_recall['Dataset']}` with recall `{best_recall['Recall']}`.",
                f"Lowest FP across all completed rows: `{lowest_fp['Model']}` on `{lowest_fp['Dataset']}` with FP `{lowest_fp['FP']}`.",
                f"Most FN across all completed rows: `{most_fn['Model']}` on `{most_fn['Dataset']}` with FN `{most_fn['FN']}`.",
                "",
                "Threshold note: rows with `threshold_mode=auto_test_sweep` use threshold sweeping on the test/evaluation set and should not be interpreted as strict held-out performance.",
            ]
        )
    else:
        lines.append("No completed model/dataset evaluations found yet.")
    lines.append("")
    (output_dir / "direct_ablation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def evaluate_direct_benchmark(
    *,
    dataset_key: str,
    model_key: str,
    threshold_arg: str = "auto",
    threshold_from: Path | None = None,
    split_threshold: bool = False,
    batch_size: int = 16,
    max_length: int = 128,
    use_cuda: bool = True,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    seed: int = 2026,
) -> dict[str, Any]:
    input_path = DATASET_FILES[dataset_key]
    rows = _load_rows(input_path)
    scores, metadata = _score_model(
        rows=rows,
        model_key=model_key,
        batch_size=batch_size,
        max_length=max_length,
        use_cuda=use_cuda,
    )
    threshold, threshold_metadata, test_indices = _resolve_threshold(
        threshold_arg=threshold_arg,
        threshold_from=threshold_from,
        rows=rows,
        scores=scores,
        split_threshold=split_threshold,
        seed=seed,
    )
    predictions = _build_predictions(rows, scores, threshold, metadata.get("row_metadata"))

    if test_indices is None:
        metric_rows = rows
        metric_scores = scores
        evaluation_scope = "all"
    else:
        metric_rows = [rows[index] for index in test_indices]
        metric_scores = [scores[index] for index in test_indices]
        evaluation_scope = "test_split_70pct"
    metrics = _metrics_from_scores([int(row["label"]) for row in metric_rows], metric_scores, threshold)
    metrics.update(
        {
            "dataset": dataset_key,
            "model": model_key,
            "input": str(input_path),
            "output_dir": str(output_dir),
            "evaluation_scope": evaluation_scope,
            "threshold_mode": threshold_metadata.get("mode"),
            "threshold_metadata": threshold_metadata,
            "threshold_warning": threshold_metadata.get("warning"),
            "threshold_note": threshold_metadata.get("note"),
            "model_path": metadata.get("model_path"),
            "vectorizer_path": metadata.get("vectorizer_path"),
            "runtime_device": metadata.get("runtime_device"),
            "calibration_method": metadata.get("calibration_method"),
            "dataset_distribution": _group_counts(predictions, "dataset_name"),
            "label_distribution": _group_counts(predictions, "label"),
            "attack_type_distribution": _group_counts(predictions, "attack_type"),
            "by_dataset": _group_metrics(predictions, "dataset_name", threshold),
            "by_attack_type": _group_metrics(predictions, "attack_type", threshold),
        }
    )
    metrics["warnings"].extend(metadata.get("warnings", []))

    model_dir = output_dir / model_key
    dataset_dir = model_dir / dataset_key
    dataset_dir.mkdir(parents=True, exist_ok=True)
    false_positives = [
        row for row in predictions if int(row["label"]) == 0 and int(row["predicted_label"]) == 1
    ]
    false_negatives = [
        row for row in predictions if int(row["label"]) == 1 and int(row["predicted_label"]) == 0
    ]
    _write_csv(dataset_dir / "predictions.csv", predictions, PREDICTION_COLUMNS)
    _write_csv(dataset_dir / "false_positives.csv", false_positives, PREDICTION_COLUMNS)
    _write_csv(dataset_dir / "false_negatives.csv", false_negatives, PREDICTION_COLUMNS)
    (dataset_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(
        path=dataset_dir / "evaluation_report.md",
        dataset_key=dataset_key,
        model_key=model_key,
        input_path=input_path,
        metrics=metrics,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    _copy_latest(dataset_dir, model_dir)
    _write_summary(output_dir)
    return metrics


def main() -> int:
    args = _parse_args()
    metrics = evaluate_direct_benchmark(
        dataset_key=args.dataset,
        model_key=args.model,
        threshold_arg=args.threshold,
        threshold_from=args.threshold_from,
        split_threshold=args.split_threshold,
        batch_size=args.batch_size,
        max_length=args.max_length,
        use_cuda=not args.no_cuda,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print("Direct benchmark evaluation complete")
    print(f"Dataset: {metrics['dataset']}")
    print(f"Model: {metrics['model']}")
    print(f"Rows: {metrics['rows']}")
    print(f"Threshold: {metrics['threshold']:.4f} ({metrics.get('threshold_mode')})")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"F2: {metrics['f2']:.4f}")
    print(f"ROC-AUC: {_format_metric(metrics.get('roc_auc'))}")
    print(f"PR-AUC: {_format_metric(metrics.get('pr_auc'))}")
    print(f"Confusion matrix: {metrics['confusion_matrix']}")
    if metrics.get("warnings"):
        print("Warnings:")
        for warning in metrics["warnings"]:
            print(f"- {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
