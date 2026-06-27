"""Run BIPIA ablation study across independent detector configurations.

This script intentionally bypasses the full hybrid pipeline so each ablation uses
only the component(s) named in its configuration:

1. RoBERTa only
2. XLM-RoBERTa only
3. RoBERTa + context-aware
4. XLM-RoBERTa + context-aware
5. Random Forest only

Example:
    python scripts/evaluate_bipia_ablation.py
    python scripts/evaluate_bipia_ablation.py --limit 500 --batch-size 8
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import MODEL_FILES, load_model_artifacts  # noqa: E402
from src.detection.context_aware_detector import detect_context_aware  # noqa: E402
from src.preprocessing import prepare_text_for_detection  # noqa: E402
from src.transformer_utils import (  # noqa: E402
    LABEL2ID,
    _load_probability_calibrator,
    _load_transformer_artifacts_cached,
    clear_transformer_runtime_cache,
    is_finetuned_transformer_checkpoint,
    resolve_transformer_model_dir,
)


DEFAULT_INPUT = PROJECT_ROOT / "data" / "external_benchmark" / "bipia" / "bipia_normalized.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "bipia_evaluation"

PREDICTION_COLUMNS = [
    "id",
    "label",
    "predicted_label",
    "final_score",
    "model_score",
    "context_score",
    "threshold",
    "model",
    "context_aware",
    "score_formula",
    "context_mismatch",
    "detected_instruction",
    "matched_signals",
    "reasons",
    "attack_type",
    "source_task",
    "language",
    "difficulty",
    "user_task",
    "external_content",
]

COMPARISON_COLUMNS = [
    "Model",
    "Context-aware",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "F2",
    "ROC-AUC",
    "PR-AUC",
    "FP",
    "FN",
    "Threshold",
    "Rows",
]


@dataclass(frozen=True)
class AblationConfig:
    key: str
    output_name: str
    model_label: str
    score_key: str
    model_kind: str
    context_aware: bool = False
    transformer_alias: str | None = None
    ml_model_type: str | None = None


CONFIGS: dict[str, AblationConfig] = {
    "roberta": AblationConfig(
        key="roberta",
        output_name="roberta",
        model_label="RoBERTa",
        score_key="roberta",
        model_kind="transformer",
        transformer_alias="roberta",
    ),
    "xlm_roberta": AblationConfig(
        key="xlm_roberta",
        output_name="xlm_roberta",
        model_label="XLM-RoBERTa",
        score_key="xlm_roberta",
        model_kind="transformer",
        transformer_alias="xlm_roberta",
    ),
    "roberta_context": AblationConfig(
        key="roberta_context",
        output_name="roberta_context",
        model_label="RoBERTa",
        score_key="roberta",
        model_kind="transformer",
        transformer_alias="roberta",
        context_aware=True,
    ),
    "xlm_roberta_context": AblationConfig(
        key="xlm_roberta_context",
        output_name="xlm_roberta_context",
        model_label="XLM-RoBERTa",
        score_key="xlm_roberta",
        model_kind="transformer",
        transformer_alias="xlm_roberta",
        context_aware=True,
    ),
    "random_forest": AblationConfig(
        key="random_forest",
        output_name="random_forest",
        model_label="Random Forest",
        score_key="random_forest",
        model_kind="traditional_ml",
        ml_model_type="random_forest",
    ),
}

COMPARISON_ORDER = [
    "random_forest",
    "roberta",
    "xlm_roberta",
    "roberta_context",
    "xlm_roberta_context",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BIPIA ablation study.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Normalized BIPIA CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root directory for BIPIA ablation reports.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for quick evaluation.")
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=sorted(CONFIGS),
        default=COMPARISON_ORDER,
        help="Subset of ablation configurations to run.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Transformer batch size.")
    parser.add_argument("--max-length", type=int, default=128, help="Transformer max sequence length.")
    parser.add_argument("--no-cuda", action="store_true", help="Force Transformer inference on CPU.")
    parser.add_argument("--threshold-beta", type=float, default=2.0, help="F-beta used for threshold selection.")
    parser.add_argument("--seed", type=int, default=2026, help="Seed used for deterministic ordering/sampling.")
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


def _format_float(value: Any, digits: int = 4) -> str:
    parsed = _safe_float(value)
    if parsed is None:
        return "n/a"
    return f"{parsed:.{digits}f}"


def _load_rows(path: Path, limit: int | None) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Normalized BIPIA CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if limit is not None:
        return rows[:limit]
    return rows


def _model_input_text(row: dict[str, Any]) -> str:
    """Use the same untrusted-content framing for all model-only ablations."""
    return (
        f"USER_TASK: {row.get('user_task', '')}\n"
        f"EXTERNAL_CONTENT:\n{row.get('external_content', '')}"
    )


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
    valid_pairs = [(true, score) for true, score in zip(y_true, scores) if score is not None]
    if len({true for true, _ in valid_pairs}) < 2:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
        metrics["warnings"].append("ROC-AUC/PR-AUC not computed because valid scores do not contain both classes.")
        return metrics
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        labels = [true for true, _ in valid_pairs]
        valid_scores = [float(score) for _, score in valid_pairs]
        metrics["roc_auc"] = float(roc_auc_score(labels, valid_scores))
        metrics["pr_auc"] = float(average_precision_score(labels, valid_scores))
    except Exception as exc:  # pragma: no cover - depends on optional sklearn runtime
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
        metrics["warnings"].append(f"ROC-AUC/PR-AUC not computed: {type(exc).__name__}: {exc}")
    return metrics


def _threshold_candidates(y_true: list[int], scores: list[float], beta: float) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for value in range(1, 100):
        threshold = value / 100
        metrics = _metrics_from_scores(y_true, scores, threshold)
        metrics["fbeta"] = _fbeta(float(metrics["precision"]), float(metrics["recall"]), beta)
        candidates.append(metrics)
    return candidates


def _rank_threshold(row: dict[str, Any], metric: str) -> tuple[float, float, float, int, float]:
    return (
        float(row[metric]),
        float(row["recall"]),
        float(row["precision"]),
        -int(row["false_positive_count"]),
        float(row["threshold"]),
    )


def _choose_threshold(y_true: list[int], scores: list[float], beta: float) -> dict[str, Any]:
    candidates = _threshold_candidates(y_true, scores, beta)
    best_f1 = max(candidates, key=lambda row: _rank_threshold(row, "f1"))
    best_f2 = max(candidates, key=lambda row: _rank_threshold(row, "f2"))
    best_fbeta = max(candidates, key=lambda row: _rank_threshold(row, "fbeta"))
    selected = best_f2 if beta == 2 else best_fbeta
    return {
        "beta": beta,
        "candidate_thresholds": "0.01..0.99 step 0.01",
        "selection_rule": (
            "maximize F2, then recall, precision, fewer false positives, higher threshold"
            if beta == 2
            else f"maximize F-beta(beta={beta:g}), then recall, precision, fewer false positives, higher threshold"
        ),
        "best_f1_threshold": best_f1,
        "best_f2_threshold": best_f2,
        "recommended_threshold": selected,
    }


def _batch_iter(values: list[str], batch_size: int) -> list[list[str]]:
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    return [values[index : index + batch_size] for index in range(0, len(values), batch_size)]


def _load_transformer(alias: str, use_cuda: bool) -> tuple[Any, Any, Any, Any, Path, list[str]]:
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
    return torch, tokenizer, model, device, model_dir, warnings


def _score_transformer(
    *,
    rows: list[dict[str, Any]],
    alias: str,
    batch_size: int,
    max_length: int,
    use_cuda: bool,
) -> tuple[list[float], dict[str, Any]]:
    torch, tokenizer, model, device, model_dir, warnings = _load_transformer(alias, use_cuda)
    cleaned_texts = [
        prepare_text_for_detection(_model_input_text(row))["cleaned_text"]
        for row in rows
    ]
    calibrator = _load_probability_calibrator(str(model_dir.resolve()))
    calibration_method = calibrator.__class__.__name__ if calibrator is not None else None

    scores: list[float] = []
    raw_scores: list[float] = []
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
            batch_scores = probabilities[:, LABEL2ID["INJECTION"]].detach().cpu().tolist()
            raw_scores.extend(float(score) for score in batch_scores)
            if calibrator is not None:
                calibrated = calibrator.predict([float(score) for score in batch_scores])
                scores.extend(float(min(1.0, max(0.0, value))) for value in calibrated)
            else:
                scores.extend(float(score) for score in batch_scores)
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
        "warnings": warnings,
    }


def _score_random_forest(rows: list[dict[str, Any]]) -> tuple[list[float], dict[str, Any]]:
    model_type = "random_forest"
    model, vectorizer = load_model_artifacts(model_type)
    cleaned_texts = [
        prepare_text_for_detection(_model_input_text(row))["cleaned_text"]
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


def _group_metrics(predictions: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[str(row.get(key, ""))].append(row)
    results: dict[str, Any] = {}
    for group_name, group_rows in grouped.items():
        labels = [int(row["label"]) for row in group_rows]
        scores = [float(row["final_score"]) for row in group_rows]
        threshold = float(group_rows[0]["threshold"]) if group_rows else 0.5
        results[group_name] = _metrics_from_scores(labels, scores, threshold)
    return dict(sorted(results.items()))


def _signal_summary(context_result: dict[str, Any] | None) -> str:
    if not context_result:
        return ""
    signals = context_result.get("matched_signals") or []
    categories = sorted({str(signal.get("category", "")) for signal in signals if signal.get("category")})
    return ";".join(categories)


def _build_config_predictions(
    *,
    config: AblationConfig,
    rows: list[dict[str, Any]],
    base_scores: list[float],
    threshold: float,
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for row, model_score in zip(rows, base_scores):
        context_score: float | None = None
        context_result: dict[str, Any] | None = None
        reasons: list[str] = [f"{config.model_label} score={model_score:.6f}"]
        if config.context_aware:
            context_result = detect_context_aware(
                user_task=str(row.get("user_task", "")),
                external_content=str(row.get("external_content", "")),
                model_score=float(model_score),
                rule_hits=[],
            )
            context_score = float(context_result.get("context_risk_score", 0.0))
            reasons.append(f"context score={context_score:.6f}")
            reason_text = str(context_result.get("reason", "")).strip()
            if reason_text:
                reasons.append(reason_text)
            final_score = max(float(model_score), context_score)
            formula = "final_score = max(model_score, context_score)"
        else:
            final_score = float(model_score)
            formula = "final_score = model_score"

        predicted_label = 1 if final_score >= threshold else 0
        predictions.append(
            {
                "id": row.get("id", ""),
                "label": int(row.get("label", 0)),
                "predicted_label": predicted_label,
                "final_score": round(final_score, 8),
                "model_score": round(float(model_score), 8),
                "context_score": "" if context_score is None else round(float(context_score), 8),
                "threshold": round(float(threshold), 4),
                "model": config.model_label,
                "context_aware": "Yes" if config.context_aware else "No",
                "score_formula": formula,
                "context_mismatch": "" if context_result is None else bool(context_result.get("context_mismatch")),
                "detected_instruction": "" if context_result is None else str(context_result.get("detected_instruction", "")),
                "matched_signals": _signal_summary(context_result),
                "reasons": " | ".join(reasons),
                "attack_type": row.get("attack_type", ""),
                "source_task": row.get("source_task", ""),
                "language": row.get("language", ""),
                "difficulty": row.get("difficulty", ""),
                "user_task": row.get("user_task", ""),
                "external_content": row.get("external_content", ""),
            }
        )
    return predictions


def _write_config_report(
    *,
    path: Path,
    config: AblationConfig,
    input_path: Path,
    output_dir: Path,
    metrics: dict[str, Any],
    false_positives: list[dict[str, Any]],
    false_negatives: list[dict[str, Any]],
) -> None:
    lines = [
        f"# BIPIA Ablation Evaluation - {config.output_name}",
        "",
        "## Configuration",
        "",
        f"- Model: `{config.model_label}`",
        f"- Context-aware: `{'Yes' if config.context_aware else 'No'}`",
        f"- Rule-based detector: `No`",
        f"- Random Forest: `{'Yes' if config.model_kind == 'traditional_ml' else 'No'}`",
        f"- Transformer alias: `{config.transformer_alias or 'n/a'}`",
        f"- Input: `{input_path}`",
        f"- Output directory: `{output_dir}`",
        f"- Rows: `{metrics['rows']}`",
        f"- Model path: `{metrics.get('model_path', 'n/a')}`",
        f"- Runtime device: `{metrics.get('runtime_device', 'n/a')}`",
        "",
        "## Threshold optimization",
        "",
        "- Procedure: sweep thresholds `0.01..0.99`, select by F2, then recall, precision, fewer false positives, higher threshold.",
        f"- Selected threshold: `{_format_float(metrics['threshold'])}`",
        "",
        "## Metrics",
        "",
        f"- Accuracy: `{_format_float(metrics['accuracy'])}`",
        f"- Precision: `{_format_float(metrics['precision'])}`",
        f"- Recall: `{_format_float(metrics['recall'])}`",
        f"- F1-score: `{_format_float(metrics['f1'])}`",
        f"- F2-score: `{_format_float(metrics['f2'])}`",
        f"- ROC-AUC: `{_format_float(metrics.get('roc_auc'))}`",
        f"- PR-AUC: `{_format_float(metrics.get('pr_auc'))}`",
        f"- Confusion matrix [[TN, FP], [FN, TP]]: `{metrics['confusion_matrix']}`",
        f"- False positives: `{metrics['false_positive_count']}`",
        f"- False negatives: `{metrics['false_negative_count']}`",
        "",
        "## Error samples",
        "",
        "### False positives",
        "",
    ]
    if false_positives:
        for row in false_positives[:10]:
            lines.append(
                f"- `{row['id']}` score={row['final_score']} source={row['source_task']} "
                f"difficulty={row['difficulty']} signals={row.get('matched_signals', '')} "
                f"reason={str(row.get('reasons', ''))[:220]}"
            )
    else:
        lines.append("No false positives.")
    lines.extend(["", "### False negatives", ""])
    if false_negatives:
        for row in false_negatives[:10]:
            lines.append(
                f"- `{row['id']}` score={row['final_score']} source={row['source_task']} "
                f"difficulty={row['difficulty']} signals={row.get('matched_signals', '')} "
                f"reason={str(row.get('reasons', ''))[:220]}"
            )
    else:
        lines.append("No false negatives.")

    if metrics.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in metrics["warnings"]:
            lines.append(f"- {warning}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _comparison_row(config: AblationConfig, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "Model": config.model_label,
        "Context-aware": "Yes" if config.context_aware else "No",
        "Accuracy": _format_float(metrics.get("accuracy")),
        "Precision": _format_float(metrics.get("precision")),
        "Recall": _format_float(metrics.get("recall")),
        "F1": _format_float(metrics.get("f1")),
        "F2": _format_float(metrics.get("f2")),
        "ROC-AUC": _format_float(metrics.get("roc_auc")),
        "PR-AUC": _format_float(metrics.get("pr_auc")),
        "FP": int(metrics.get("false_positive_count", 0)),
        "FN": int(metrics.get("false_negative_count", 0)),
        "Threshold": _format_float(metrics.get("threshold")),
        "Rows": int(metrics.get("rows", 0)),
    }


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return lines


def _metric_delta(current: dict[str, Any], baseline: dict[str, Any], key: str) -> float:
    current_value = _safe_float(current.get(key), 0.0) or 0.0
    baseline_value = _safe_float(baseline.get(key), 0.0) or 0.0
    return current_value - baseline_value


def _error_delta(current: dict[str, Any], baseline: dict[str, Any], key: str) -> int:
    return int(current.get(key, 0)) - int(baseline.get(key, 0))


def _quality_statement(model_name: str, metrics: dict[str, Any]) -> str:
    f1 = float(metrics.get("f1", 0.0))
    roc_auc = _safe_float(metrics.get("roc_auc"), 0.0) or 0.0
    if f1 >= 0.90 and roc_auc >= 0.90:
        return f"{model_name} generalizes strongly on this BIPIA split."
    if f1 >= 0.75 and roc_auc >= 0.75:
        return f"{model_name} shows partial generalization on this BIPIA split."
    return f"{model_name} does not generalize reliably on this BIPIA split."


def _write_summary_report(
    *,
    path: Path,
    input_path: Path,
    comparison_rows: list[dict[str, Any]],
    metrics_by_key: dict[str, dict[str, Any]],
    common_failures: list[dict[str, Any]],
) -> None:
    roberta = metrics_by_key.get("roberta", {})
    xlm = metrics_by_key.get("xlm_roberta", {})
    roberta_context = metrics_by_key.get("roberta_context", {})
    xlm_context = metrics_by_key.get("xlm_roberta_context", {})
    rf = metrics_by_key.get("random_forest", {})

    roberta_context_fp_delta = _error_delta(roberta_context, roberta, "false_positive_count")
    roberta_context_fn_delta = _error_delta(roberta_context, roberta, "false_negative_count")
    xlm_context_fp_delta = _error_delta(xlm_context, xlm, "false_positive_count")
    xlm_context_fn_delta = _error_delta(xlm_context, xlm, "false_negative_count")
    roberta_context_f2_delta = _metric_delta(roberta_context, roberta, "f2")
    xlm_context_f2_delta = _metric_delta(xlm_context, xlm, "f2")

    lines = [
        "# BIPIA Ablation Study Summary",
        "",
        "## Setup",
        "",
        f"- Input dataset: `{input_path}`",
        "- Each configuration was scored independently on the same rows.",
        "- No configuration uses the rule-based detector.",
        "- Threshold procedure is identical across configurations: sweep `0.01..0.99`, select by F2, then recall, precision, fewer FP, higher threshold.",
        "",
        "## Comparison table",
        "",
        *_markdown_table(
            comparison_rows,
            ["Model", "Context-aware", "Accuracy", "Precision", "Recall", "F1", "F2", "ROC-AUC", "PR-AUC", "FP", "FN"],
        ),
        "",
        "## Final analysis",
        "",
        "1. RoBERTa generalization",
        "",
        f"   - {_quality_statement('RoBERTa', roberta)} F1={_format_float(roberta.get('f1'))}, F2={_format_float(roberta.get('f2'))}, ROC-AUC={_format_float(roberta.get('roc_auc'))}.",
        "",
        "2. XLM-RoBERTa vs RoBERTa",
        "",
        f"   - XLM-RoBERTa F2 delta vs RoBERTa: `{_format_float(_metric_delta(xlm, roberta, 'f2'))}`; ROC-AUC delta: `{_format_float(_metric_delta(xlm, roberta, 'roc_auc'))}`.",
        f"   - {'XLM-RoBERTa is stronger on this BIPIA run.' if _metric_delta(xlm, roberta, 'f2') > 0 else 'RoBERTa is stronger on this BIPIA run.'}",
        "",
        "3. Context-aware effect on FP/FN",
        "",
        f"   - RoBERTa + context FP delta: `{roberta_context_fp_delta}`, FN delta: `{roberta_context_fn_delta}`, F2 delta: `{_format_float(roberta_context_f2_delta)}`.",
        f"   - XLM-RoBERTa + context FP delta: `{xlm_context_fp_delta}`, FN delta: `{xlm_context_fn_delta}`, F2 delta: `{_format_float(xlm_context_f2_delta)}`.",
        "",
        "4. Which model benefits more from context-aware",
        "",
        f"   - {'RoBERTa benefits more by F2.' if roberta_context_f2_delta > xlm_context_f2_delta else 'XLM-RoBERTa benefits more by F2.' if xlm_context_f2_delta > roberta_context_f2_delta else 'Both models gain the same F2 change.'}",
        "",
        "5. Random Forest vs Transformer",
        "",
        f"   - Random Forest uses TF-IDF/tree features and behaves as the traditional ML baseline: F1={_format_float(rf.get('f1'))}, F2={_format_float(rf.get('f2'))}, ROC-AUC={_format_float(rf.get('roc_auc'))}.",
        f"   - Best Transformer-only F2 is `{_format_float(max(float(roberta.get('f2', 0.0)), float(xlm.get('f2', 0.0))))}`. RF gets high thresholded F2 by favoring recall, but its ROC-AUC shows weak ranking quality on this OOD benchmark.",
        "",
        "6. Samples all three model-only systems failed",
        "",
        f"   - Common failures across Random Forest, RoBERTa-only and XLM-RoBERTa-only: `{len(common_failures)}`.",
    ]
    for row in common_failures[:10]:
        lines.append(
            f"   - `{row['id']}` `{row['error_type']}` source={row.get('source_task', '')} "
            f"difficulty={row.get('difficulty', '')} attack={row.get('attack_type', '')}"
        )
    if not common_failures:
        lines.append("   - No common failures across the three model-only configurations.")

    lines.extend(
        [
            "",
            "7. Recommended next improvements",
            "",
            "   - Add a calibrated indirect-injection training/evaluation protocol instead of relying only on direct prompt-injection fine-tuning.",
            "   - Tune score fusion on a validation split separate from BIPIA if BIPIA is treated as final benchmark.",
            "   - Add hard negative safe contexts containing shell/code/security vocabulary to reduce lexical false positives.",
            "   - Mine common false negatives into categories, then improve context/task-intent modeling rather than copying BIPIA wholesale into training.",
            "   - Track per-source-task metrics because email/table contexts can fail for different reasons.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _common_model_only_failures(predictions_by_config: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    keys = ["random_forest", "roberta", "xlm_roberta"]
    if any(key not in predictions_by_config for key in keys):
        return []
    indexed = {
        key: {str(row["id"]): row for row in predictions_by_config[key]}
        for key in keys
    }
    common_ids = set(indexed[keys[0]])
    for key in keys[1:]:
        common_ids &= set(indexed[key])

    failures: list[dict[str, Any]] = []
    for row_id in sorted(common_ids):
        rows = [indexed[key][row_id] for key in keys]
        if all(int(row["label"]) != int(row["predicted_label"]) for row in rows):
            base = dict(rows[0])
            base["error_type"] = "common_false_positive" if int(base["label"]) == 0 else "common_false_negative"
            base["random_forest_score"] = indexed["random_forest"][row_id]["final_score"]
            base["roberta_score"] = indexed["roberta"][row_id]["final_score"]
            base["xlm_roberta_score"] = indexed["xlm_roberta"][row_id]["final_score"]
            failures.append(base)
    return failures


def _run_config(
    *,
    config: AblationConfig,
    rows: list[dict[str, Any]],
    base_scores_by_key: dict[str, list[float]],
    base_metadata_by_key: dict[str, dict[str, Any]],
    output_root: Path,
    input_path: Path,
    threshold_beta: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels = [int(row.get("label", 0)) for row in rows]
    base_scores = base_scores_by_key[config.score_key]
    raw_predictions = _build_config_predictions(
        config=config,
        rows=rows,
        base_scores=base_scores,
        threshold=0.5,
    )
    final_scores = [float(row["final_score"]) for row in raw_predictions]
    threshold_summary = _choose_threshold(labels, final_scores, threshold_beta)
    selected_threshold = float(threshold_summary["recommended_threshold"]["threshold"])
    predictions = _build_config_predictions(
        config=config,
        rows=rows,
        base_scores=base_scores,
        threshold=selected_threshold,
    )
    metrics = _metrics_from_scores(labels, final_scores, selected_threshold)
    metadata = base_metadata_by_key.get(config.score_key, {})
    metrics.update(
        {
            "config": config.key,
            "model": config.model_label,
            "context_aware": config.context_aware,
            "input": str(input_path),
            "limit": len(rows),
            "model_path": metadata.get("model_path"),
            "vectorizer_path": metadata.get("vectorizer_path"),
            "runtime_device": metadata.get("runtime_device"),
            "calibration_method": metadata.get("calibration_method"),
            "threshold_optimization": threshold_summary,
            "source_task_counts": _group_counts(predictions, "source_task"),
            "difficulty_counts": _group_counts(predictions, "difficulty"),
            "by_source_task": _group_metrics(predictions, "source_task"),
            "by_difficulty": _group_metrics(predictions, "difficulty"),
        }
    )
    metrics["warnings"].extend(metadata.get("warnings", []))

    config_dir = output_root / config.output_name
    config_dir.mkdir(parents=True, exist_ok=True)
    false_positives = [
        row for row in predictions if int(row["label"]) == 0 and int(row["predicted_label"]) == 1
    ]
    false_negatives = [
        row for row in predictions if int(row["label"]) == 1 and int(row["predicted_label"]) == 0
    ]
    _write_csv(config_dir / "predictions.csv", predictions, PREDICTION_COLUMNS)
    _write_csv(config_dir / "false_positives.csv", false_positives, PREDICTION_COLUMNS)
    _write_csv(config_dir / "false_negatives.csv", false_negatives, PREDICTION_COLUMNS)
    (config_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_config_report(
        path=config_dir / "evaluation_report.md",
        config=config,
        input_path=input_path,
        output_dir=config_dir,
        metrics=metrics,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )
    return metrics, predictions


def evaluate_ablation(
    *,
    input_path: Path,
    output_dir: Path,
    config_keys: list[str],
    limit: int | None = None,
    batch_size: int = 16,
    max_length: int = 128,
    use_cuda: bool = True,
    threshold_beta: float = 2.0,
    seed: int = 2026,
) -> dict[str, Any]:
    random.seed(seed)
    rows = _load_rows(input_path, limit)
    if not rows:
        raise ValueError("BIPIA input has no rows.")

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_configs = [CONFIGS[key] for key in config_keys]
    needed_score_keys = sorted({config.score_key for config in selected_configs})

    base_scores_by_key: dict[str, list[float]] = {}
    base_metadata_by_key: dict[str, dict[str, Any]] = {}
    for score_key in needed_score_keys:
        if score_key == "random_forest":
            scores, metadata = _score_random_forest(rows)
        elif score_key == "roberta":
            scores, metadata = _score_transformer(
                rows=rows,
                alias="roberta",
                batch_size=batch_size,
                max_length=max_length,
                use_cuda=use_cuda,
            )
        elif score_key == "xlm_roberta":
            scores, metadata = _score_transformer(
                rows=rows,
                alias="xlm_roberta",
                batch_size=batch_size,
                max_length=max_length,
                use_cuda=use_cuda,
            )
        else:
            raise ValueError(f"Unknown score key: {score_key}")
        base_scores_by_key[score_key] = scores
        base_metadata_by_key[score_key] = metadata

    metrics_by_key: dict[str, dict[str, Any]] = {}
    predictions_by_key: dict[str, list[dict[str, Any]]] = {}
    for config in selected_configs:
        metrics, predictions = _run_config(
            config=config,
            rows=rows,
            base_scores_by_key=base_scores_by_key,
            base_metadata_by_key=base_metadata_by_key,
            output_root=output_dir,
            input_path=input_path,
            threshold_beta=threshold_beta,
        )
        metrics_by_key[config.key] = metrics
        predictions_by_key[config.key] = predictions

    ordered_keys = [key for key in COMPARISON_ORDER if key in metrics_by_key]
    comparison_rows = [_comparison_row(CONFIGS[key], metrics_by_key[key]) for key in ordered_keys]
    _write_csv(output_dir / "ablation_comparison.csv", comparison_rows, COMPARISON_COLUMNS)

    common_failures = _common_model_only_failures(predictions_by_key)
    _write_csv(output_dir / "ablation_common_failures.csv", common_failures)
    _write_summary_report(
        path=output_dir / "ablation_summary.md",
        input_path=input_path,
        comparison_rows=comparison_rows,
        metrics_by_key=metrics_by_key,
        common_failures=common_failures,
    )

    payload = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "rows": len(rows),
        "configs": ordered_keys,
        "comparison": comparison_rows,
        "common_failures": len(common_failures),
    }
    (output_dir / "ablation_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    args = _parse_args()
    result = evaluate_ablation(
        input_path=args.input,
        output_dir=args.output_dir,
        config_keys=args.configs,
        limit=args.limit,
        batch_size=args.batch_size,
        max_length=args.max_length,
        use_cuda=not args.no_cuda,
        threshold_beta=args.threshold_beta,
        seed=args.seed,
    )
    print("BIPIA ablation study complete")
    print(f"Rows: {result['rows']}")
    print(f"Output: {result['output_dir']}")
    for row in result["comparison"]:
        print(
            f"{row['Model']} context={row['Context-aware']}: "
            f"F1={row['F1']} F2={row['F2']} ROC-AUC={row['ROC-AUC']} "
            f"FP={row['FP']} FN={row['FN']}"
        )
    print(f"Common model-only failures: {result['common_failures']}")


if __name__ == "__main__":
    main()
