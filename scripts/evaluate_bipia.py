"""Evaluate the context-aware detector on normalized BIPIA samples.

Default quick deterministic evaluation:
    python scripts/evaluate_bipia.py --limit 500

Optional model signals:
    python scripts/evaluate_bipia.py --limit 500 --use-ml
    python scripts/evaluate_bipia.py --limit 100 --use-transformer
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.pipeline import run_hybrid_detection  # noqa: E402


DEFAULT_INPUT = PROJECT_ROOT / "data" / "external_benchmark" / "bipia" / "bipia_normalized.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "bipia_evaluation"

PREDICTION_COLUMNS = [
    "id",
    "label",
    "predicted_label",
    "risk_level",
    "final_score",
    "rule_score",
    "ml_score",
    "transformer_score",
    "context_score",
    "threshold_warn",
    "threshold_block",
    "user_task",
    "external_content",
    "reasons",
    "attack_type",
    "source_task",
    "difficulty",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Prompt Injection Detection on BIPIA.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Normalized BIPIA CSV.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Report output directory.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for quick evaluation.")
    parser.add_argument("--use-ml", action="store_true", help="Enable traditional ML score.")
    parser.add_argument("--use-transformer", action="store_true", help="Enable Transformer score; slower.")
    parser.add_argument("--transformer-model", default="roberta", help="Transformer model alias.")
    parser.add_argument("--ml-model-type", default="logistic_regression", help="Traditional ML model type.")
    parser.add_argument("--split-threshold", action="store_true", help="Tune threshold on 30% validation and report 70% test metrics.")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _load_rows(path: Path, limit: int | None) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Normalized BIPIA CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if limit is not None:
        return rows[:limit]
    return rows


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


def _metrics_from_labels(y_true: list[int], y_pred: list[int], scores: list[float | None]) -> dict[str, Any]:
    warnings: list[str] = []
    counts = _confusion(y_true, y_pred)
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _fbeta(precision, recall, 1.0)
    f2 = _fbeta(precision, recall, 2.0)
    metrics: dict[str, Any] = {
        "rows": len(y_true),
        "accuracy": _safe_div(tp + tn, len(y_true)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "false_positive_count": fp,
        "false_negative_count": fn,
        **counts,
        "warnings": warnings,
    }
    valid_pairs = [(true, score) for true, score in zip(y_true, scores) if score is not None]
    if len({true for true, _ in valid_pairs}) < 2:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
        warnings.append("ROC-AUC/PR-AUC not computed because valid scores do not contain both classes.")
        return metrics
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        labels = [true for true, _ in valid_pairs]
        valid_scores = [float(score) for _, score in valid_pairs]
        metrics["roc_auc"] = float(roc_auc_score(labels, valid_scores))
        metrics["pr_auc"] = float(average_precision_score(labels, valid_scores))
    except Exception as exc:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None
        warnings.append(f"ROC-AUC/PR-AUC not computed: {type(exc).__name__}: {exc}")
    return metrics


def _metrics_at_threshold(predictions: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    y_true = [int(row["label"]) for row in predictions]
    scores = [_safe_float(row["final_score"]) for row in predictions]
    y_pred = [1 if (score is not None and score >= threshold) else 0 for score in scores]
    metrics = _metrics_from_labels(y_true, y_pred, scores)
    metrics["threshold"] = threshold
    return metrics


def _choose_best_f2_threshold(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [_metrics_at_threshold(predictions, threshold / 100) for threshold in range(1, 100)]
    return max(
        candidates,
        key=lambda row: (
            float(row["f2"]),
            float(row["recall"]),
            float(row["precision"]),
            -int(row["false_positive_count"]),
            float(row["threshold"]),
        ),
    )


def _stratified_split(predictions: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_label[int(row["label"])].append(row)
    validation: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for rows in by_label.values():
        shuffled = rows[:]
        rng.shuffle(shuffled)
        cut = max(1, int(round(len(shuffled) * 0.30))) if len(shuffled) > 1 else len(shuffled)
        validation.extend(shuffled[:cut])
        test.extend(shuffled[cut:])
    return validation, test


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
    for group_name, rows in grouped.items():
        y_true = [int(row["label"]) for row in rows]
        y_pred = [int(row["predicted_label"]) for row in rows]
        scores = [_safe_float(row["final_score"]) for row in rows]
        results[group_name] = _metrics_from_labels(y_true, y_pred, scores)
    return dict(sorted(results.items()))


def _strongest_layer(row: dict[str, Any]) -> str:
    layer_scores = {
        "rule": _safe_float(row.get("rule_score")),
        "ml": _safe_float(row.get("ml_score")),
        "transformer": _safe_float(row.get("transformer_score")),
        "context": _safe_float(row.get("context_score")),
    }
    available = {key: value for key, value in layer_scores.items() if value is not None}
    if not available:
        return "none"
    return max(available.items(), key=lambda item: item[1])[0]


def _prediction_from_row(
    row: dict[str, str],
    *,
    use_ml: bool,
    use_transformer: bool,
    transformer_model: str,
    ml_model_type: str,
) -> dict[str, Any]:
    result = run_hybrid_detection(
        user_prompt=row["user_task"],
        user_task=row["user_task"],
        external_content=row["external_content"],
        ml_model_type=ml_model_type,
        transformer_model=transformer_model,
        use_ml=use_ml,
        use_transformer=use_transformer,
        use_cuda=False,
    )
    scores = result.get("model_scores", {})
    thresholds = result.get("threshold_used", {})
    return {
        "id": row["id"],
        "label": int(row["label"]),
        "predicted_label": int(result["label"]),
        "risk_level": result["risk_level"],
        "final_score": float(result["final_score"]),
        "rule_score": scores.get("rule_based"),
        "ml_score": scores.get("ml_model"),
        "transformer_score": scores.get("transformer"),
        "context_score": scores.get("context_aware"),
        "threshold_warn": thresholds.get("warn"),
        "threshold_block": thresholds.get("block"),
        "user_task": row["user_task"],
        "external_content": row["external_content"],
        "reasons": " | ".join(str(reason) for reason in result.get("reasons", [])),
        "attack_type": row.get("attack_type", ""),
        "source_task": row.get("source_task", ""),
        "difficulty": row.get("difficulty", ""),
    }


def _load_threshold_summary(output_dir: Path) -> dict[str, Any] | None:
    path = output_dir / "threshold_optimization" / "threshold_summary.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _format_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return "n/a"
    return str(value)


def _write_report(
    path: Path,
    *,
    input_path: Path,
    predictions: list[dict[str, Any]],
    metrics: dict[str, Any],
    source_task_metrics: dict[str, Any],
    difficulty_metrics: dict[str, Any],
    false_positives: list[dict[str, Any]],
    false_negatives: list[dict[str, Any]],
    split_threshold_summary: dict[str, Any] | None,
    threshold_summary: dict[str, Any] | None,
    use_ml: bool,
    use_transformer: bool,
) -> None:
    label_counts = _group_counts(predictions, "label")
    source_counts = _group_counts(predictions, "source_task")
    difficulty_counts = _group_counts(predictions, "difficulty")
    layer_counts = Counter(_strongest_layer(row) for row in predictions)

    lines = [
        "# BIPIA External Benchmark Evaluation",
        "",
        "## 1. Dataset overview",
        "",
        f"- Input: `{input_path}`",
        f"- Total samples: `{len(predictions)}`",
        f"- Label distribution: `{label_counts}`",
        f"- Source task distribution: `{source_counts}`",
        f"- Difficulty distribution: `{difficulty_counts}`",
        f"- ML enabled: `{use_ml}`",
        f"- Transformer enabled: `{use_transformer}`",
        "",
        "BIPIA is used here only as an external/OOD benchmark. It is not copied into training data.",
        "",
        "## 2. Overall results",
        "",
        f"- Accuracy: `{_format_metric(metrics['accuracy'])}`",
        f"- Precision: `{_format_metric(metrics['precision'])}`",
        f"- Recall: `{_format_metric(metrics['recall'])}`",
        f"- F1: `{_format_metric(metrics['f1'])}`",
        f"- F2: `{_format_metric(metrics['f2'])}`",
        f"- ROC-AUC: `{_format_metric(metrics.get('roc_auc'))}`",
        f"- PR-AUC: `{_format_metric(metrics.get('pr_auc'))}`",
        f"- Confusion matrix [[TN, FP], [FN, TP]]: `{metrics['confusion_matrix']}`",
        f"- False positives: `{metrics['false_positive_count']}`",
        f"- False negatives: `{metrics['false_negative_count']}`",
        "",
        "## 3. Layer contribution",
        "",
        f"- Strongest layer counts: `{dict(sorted(layer_counts.items()))}`",
        "",
        "This is an approximate contribution view based on the largest score among rule, ML, Transformer and context-aware signals.",
        "",
        "## 4. Results by source task",
        "",
    ]
    for group, group_metrics in source_task_metrics.items():
        lines.append(
            f"- `{group}`: accuracy={group_metrics['accuracy']:.4f}, "
            f"precision={group_metrics['precision']:.4f}, recall={group_metrics['recall']:.4f}, "
            f"F1={group_metrics['f1']:.4f}, cm={group_metrics['confusion_matrix']}"
        )
    lines.extend(["", "## 5. Results by difficulty", ""])
    for group, group_metrics in difficulty_metrics.items():
        lines.append(
            f"- `{group}`: accuracy={group_metrics['accuracy']:.4f}, "
            f"precision={group_metrics['precision']:.4f}, recall={group_metrics['recall']:.4f}, "
            f"F1={group_metrics['f1']:.4f}, cm={group_metrics['confusion_matrix']}"
        )

    lines.extend(["", "## 6. False positives", ""])
    if false_positives:
        lines.append("Safe samples were flagged because the detector saw assistant-directed or response-manipulation wording.")
        for row in false_positives[:10]:
            lines.append(
                f"- `{row['id']}` score={row['final_score']} source={row['source_task']} "
                f"difficulty={row['difficulty']} reasons={str(row['reasons'])[:220]}"
            )
    else:
        lines.append("No false positives in this run.")

    lines.extend(["", "## 7. False negatives", ""])
    if false_negatives:
        lines.append("Missed injections usually indicate task-hijacking phrased as a natural standalone request without classic keywords.")
        for row in false_negatives[:10]:
            lines.append(
                f"- `{row['id']}` score={row['final_score']} source={row['source_task']} "
                f"difficulty={row['difficulty']} reasons={str(row['reasons'])[:220]}"
            )
    else:
        lines.append("No false negatives in this run.")

    lines.extend(
        [
            "",
            "## 8. Threshold optimization",
            "",
            "Do not reuse the 24-sample indirect evaluation threshold for BIPIA. BIPIA threshold optimization must be run separately.",
        ]
    )
    if threshold_summary:
        selected = threshold_summary.get("recommended_threshold", {})
        lines.extend(
            [
                "",
                "Separate BIPIA threshold optimization found:",
                f"- Threshold: `{_format_metric(selected.get('threshold'))}`",
                f"- Precision: `{_format_metric(selected.get('precision'))}`",
                f"- Recall: `{_format_metric(selected.get('recall'))}`",
                f"- F1: `{_format_metric(selected.get('f1'))}`",
                f"- F2: `{_format_metric(selected.get('f2'))}`",
                f"- Confusion matrix: `{selected.get('confusion_matrix')}`",
            ]
        )
    else:
        lines.append("Run `scripts/optimize_threshold.py` on `bipia_predictions.csv` to compute a BIPIA-specific threshold.")

    if split_threshold_summary:
        lines.extend(
            [
                "",
                "Optional validation/test threshold split:",
                f"- Validation rows: `{split_threshold_summary['validation_rows']}`",
                f"- Test rows: `{split_threshold_summary['test_rows']}`",
                f"- Tuned threshold on validation: `{split_threshold_summary['threshold']:.4f}`",
                f"- Test F2 at tuned threshold: `{split_threshold_summary['test_metrics']['f2']:.4f}`",
                f"- Test confusion matrix: `{split_threshold_summary['test_metrics']['confusion_matrix']}`",
            ]
        )

    lines.extend(
        [
            "",
            "## 9. Data leakage checks",
            "",
            "- BIPIA is stored under `data/external_benchmark/bipia/`.",
            "- `prepare_bipia_benchmark.py` does not write to `data/raw`, `datasets/processed`, or model training folders.",
            "- Threshold optimization on BIPIA is reported as benchmark calibration, not as final held-out performance unless `--split-threshold` is used.",
            "",
            "## 10. Initial conclusion",
            "",
            "This benchmark primarily tests OOD indirect prompt injection. Strong performance indicates the detector is not only memorizing classic `ignore previous instructions` patterns. Failure cases should be used to improve context-aware/task-hijacking recognition, not to train directly on BIPIA unless a separate train/validation/test protocol is defined.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_bipia(
    *,
    input_path: Path,
    output_dir: Path,
    limit: int | None = None,
    use_ml: bool = False,
    use_transformer: bool = False,
    transformer_model: str = "roberta",
    ml_model_type: str = "logistic_regression",
    split_threshold: bool = False,
    seed: int = 2026,
) -> dict[str, Any]:
    rows = _load_rows(input_path, limit)
    predictions = [
        _prediction_from_row(
            row,
            use_ml=use_ml,
            use_transformer=use_transformer,
            transformer_model=transformer_model,
            ml_model_type=ml_model_type,
        )
        for row in rows
    ]

    y_true = [int(row["label"]) for row in predictions]
    y_pred = [int(row["predicted_label"]) for row in predictions]
    scores = [_safe_float(row["final_score"]) for row in predictions]
    metrics = _metrics_from_labels(y_true, y_pred, scores)
    metrics.update(
        {
            "input": str(input_path),
            "limit": limit,
            "use_ml": use_ml,
            "use_transformer": use_transformer,
            "transformer_model": transformer_model,
            "ml_model_type": ml_model_type,
            "source_task_counts": _group_counts(predictions, "source_task"),
            "difficulty_counts": _group_counts(predictions, "difficulty"),
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "bipia_predictions.csv"
    _write_csv(predictions_path, predictions, PREDICTION_COLUMNS)

    false_positives = [row for row in predictions if int(row["label"]) == 0 and int(row["predicted_label"]) == 1]
    false_negatives = [row for row in predictions if int(row["label"]) == 1 and int(row["predicted_label"]) == 0]
    hard_cases = [
        row
        for row in predictions
        if str(row.get("difficulty")) == "hard" or int(row["label"]) != int(row["predicted_label"])
    ]
    _write_csv(output_dir / "false_positives.csv", false_positives, PREDICTION_COLUMNS)
    _write_csv(output_dir / "false_negatives.csv", false_negatives, PREDICTION_COLUMNS)
    _write_csv(output_dir / "hard_cases.csv", hard_cases, PREDICTION_COLUMNS)

    split_threshold_summary = None
    if split_threshold and predictions:
        validation_rows, test_rows = _stratified_split(predictions, seed)
        selected = _choose_best_f2_threshold(validation_rows)
        test_metrics = _metrics_at_threshold(test_rows, float(selected["threshold"]))
        split_threshold_summary = {
            "validation_rows": len(validation_rows),
            "test_rows": len(test_rows),
            "threshold": float(selected["threshold"]),
            "validation_metrics": selected,
            "test_metrics": test_metrics,
            "warning": "Threshold tuned on validation split only; test metrics are reported separately.",
        }
        metrics["split_threshold"] = split_threshold_summary

    source_task_metrics = _group_metrics(predictions, "source_task")
    difficulty_metrics = _group_metrics(predictions, "difficulty")
    metrics["by_source_task"] = source_task_metrics
    metrics["by_difficulty"] = difficulty_metrics
    metrics_path = output_dir / "bipia_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_report(
        output_dir / "bipia_evaluation_report.md",
        input_path=input_path,
        predictions=predictions,
        metrics=metrics,
        source_task_metrics=source_task_metrics,
        difficulty_metrics=difficulty_metrics,
        false_positives=false_positives,
        false_negatives=false_negatives,
        split_threshold_summary=split_threshold_summary,
        threshold_summary=_load_threshold_summary(output_dir),
        use_ml=use_ml,
        use_transformer=use_transformer,
    )
    return metrics


def main() -> None:
    args = _parse_args()
    metrics = evaluate_bipia(
        input_path=args.input,
        output_dir=args.output_dir,
        limit=args.limit,
        use_ml=args.use_ml,
        use_transformer=args.use_transformer,
        transformer_model=args.transformer_model,
        ml_model_type=args.ml_model_type,
        split_threshold=args.split_threshold,
        seed=args.seed,
    )
    print("BIPIA evaluation complete")
    print(f"Rows: {metrics['rows']}")
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


if __name__ == "__main__":
    main()
