"""Strict validation/test threshold evaluation for direct external benchmarks.

This script does not call rule-based, context-aware, BIPIA, or indirect
pipelines. It uses model-only scores from existing direct benchmark predictions
when available, and only reruns model scoring if those predictions are missing.

Example:
    python scripts/evaluate_direct_benchmarks_strict.py --dataset all --model roberta --split-threshold
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_direct_benchmarks import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    MODEL_LABELS,
    PREDICTION_COLUMNS,
    _build_predictions,
    _format_metric,
    _metrics_from_scores,
    _score_model,
    _write_csv,
    DATASET_FILES,
)


STRICT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "strict"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict direct benchmark validation/test evaluation.")
    parser.add_argument("--dataset", choices=sorted(DATASET_FILES), required=True)
    parser.add_argument("--model", choices=sorted(MODEL_LABELS), required=True)
    parser.add_argument("--split-threshold", action="store_true", help="Required for strict protocol.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--force-rescore", action="store_true", help="Ignore existing predictions and score again.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=STRICT_OUTPUT_DIR)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _prediction_path(dataset: str, model: str) -> Path:
    model_dir = DEFAULT_OUTPUT_DIR / model
    if dataset == "all":
        return model_dir / "predictions.csv"
    nested = model_dir / dataset / "predictions.csv"
    if nested.exists():
        return nested
    return model_dir / "predictions.csv"


def _load_dataset_rows(dataset: str) -> list[dict[str, Any]]:
    rows = _read_csv(DATASET_FILES[dataset])
    normalized: list[dict[str, Any]] = []
    for row in rows:
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        try:
            label = int(float(str(row.get("label", "")).strip()))
        except ValueError:
            continue
        if label not in {0, 1}:
            continue
        payload = dict(row)
        payload["label"] = label
        normalized.append(payload)
    if not normalized:
        raise ValueError(f"No usable rows in {DATASET_FILES[dataset]}")
    return normalized


def _load_or_score_predictions(
    *,
    dataset: str,
    model: str,
    batch_size: int,
    max_length: int,
    use_cuda: bool,
    force_rescore: bool,
) -> list[dict[str, Any]]:
    path = _prediction_path(dataset, model)
    if path.exists() and not force_rescore:
        rows = _read_csv(path)
        if dataset != "all":
            rows = [row for row in rows if str(row.get("dataset_name")) == dataset]
        if rows:
            return rows

    dataset_rows = _load_dataset_rows(dataset)
    scores, metadata = _score_model(
        rows=dataset_rows,
        model_key=model,
        batch_size=batch_size,
        max_length=max_length,
        use_cuda=use_cuda,
    )
    return _build_predictions(dataset_rows, scores, threshold=0.5, row_metadata=metadata.get("row_metadata"))


def _split_indices(predictions: list[dict[str, Any]], dataset: str, seed: int) -> tuple[list[int], list[int]]:
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(predictions):
        label = int(float(row["label"]))
        group_dataset = str(row.get("dataset_name", dataset)) if dataset == "all" else dataset
        grouped[(group_dataset, label)].append(index)

    rng = random.Random(seed)
    validation: list[int] = []
    test: list[int] = []
    for indices in grouped.values():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        cut = max(1, int(round(len(shuffled) * 0.30))) if len(shuffled) > 1 else len(shuffled)
        validation.extend(shuffled[:cut])
        test.extend(shuffled[cut:])
    validation.sort()
    test.sort()
    return validation, test


def _rank_threshold(row: dict[str, Any]) -> tuple[float, float, float, int, float]:
    return (
        float(row["f2"]),
        float(row["recall"]),
        float(row["precision"]),
        -int(row["false_positive_count"]),
        float(row["threshold"]),
    )


def _choose_threshold(labels: list[int], scores: list[float]) -> dict[str, Any]:
    candidates = [_metrics_from_scores(labels, scores, threshold / 100) for threshold in range(1, 100)]
    selected = max(candidates, key=_rank_threshold)
    return {
        "threshold": float(selected["threshold"]),
        "selected_metrics": selected,
        "candidate_thresholds": "0.01..0.99 step 0.01",
        "selection_rule": "maximize F2, then recall, precision, fewer false positives, higher threshold",
    }


def _apply_threshold(rows: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for row in rows:
        payload = {column: row.get(column, "") for column in PREDICTION_COLUMNS}
        payload["label"] = int(float(row["label"]))
        payload["score"] = round(float(row["score"]), 8)
        payload["threshold"] = round(float(threshold), 4)
        payload["predicted_label"] = 1 if float(row["score"]) >= threshold else 0
        predictions.append(payload)
    return predictions


def _write_report(
    *,
    path: Path,
    dataset: str,
    model: str,
    threshold_summary: dict[str, Any],
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    false_positives: list[dict[str, Any]],
    false_negatives: list[dict[str, Any]],
) -> None:
    threshold = float(threshold_summary["threshold"])
    lines = [
        f"# Strict Direct Benchmark Evaluation - {dataset} / {model}",
        "",
        "## Protocol",
        "",
        "- Rule-based detector: `No`",
        "- Context-aware detector: `No`",
        "- BIPIA/indirect pipeline: `No`",
        "- Validation/test split: `30% / 70%`",
        "- Split grouping: `dataset_name + label` for `all`, otherwise `label`",
        "- Threshold is selected only on validation and then fixed for test.",
        "",
        "## Threshold",
        "",
        f"- Selected threshold on validation: `{threshold:.4f}`",
        f"- Validation F2: `{_format_metric(validation_metrics.get('f2'))}`",
        f"- Test F2: `{_format_metric(test_metrics.get('f2'))}`",
        "",
        "## Test metrics",
        "",
        f"- Rows: `{test_metrics['rows']}`",
        f"- Accuracy: `{_format_metric(test_metrics.get('accuracy'))}`",
        f"- Precision: `{_format_metric(test_metrics.get('precision'))}`",
        f"- Recall: `{_format_metric(test_metrics.get('recall'))}`",
        f"- F1: `{_format_metric(test_metrics.get('f1'))}`",
        f"- F2: `{_format_metric(test_metrics.get('f2'))}`",
        f"- ROC-AUC: `{_format_metric(test_metrics.get('roc_auc'))}`",
        f"- PR-AUC: `{_format_metric(test_metrics.get('pr_auc'))}`",
        f"- Confusion matrix [[TN, FP], [FN, TP]]: `{test_metrics['confusion_matrix']}`",
        f"- False positives: `{test_metrics['false_positive_count']}`",
        f"- False negatives: `{test_metrics['false_negative_count']}`",
        "",
        "## False positives",
        "",
    ]
    if false_positives:
        for row in false_positives[:10]:
            lines.append(f"- `{row['id']}` score={row['score']} text={str(row.get('text', ''))[:220]}")
    else:
        lines.append("No false positives.")
    lines.extend(["", "## False negatives", ""])
    if false_negatives:
        for row in false_negatives[:10]:
            lines.append(f"- `{row['id']}` score={row['score']} text={str(row.get('text', ''))[:220]}")
    else:
        lines.append("No false negatives.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report_vi(
    *,
    path: Path,
    dataset: str,
    model: str,
    threshold_summary: dict[str, Any],
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    false_positives: list[dict[str, Any]],
    false_negatives: list[dict[str, Any]],
) -> None:
    threshold = float(threshold_summary["threshold"])
    lines = [
        f"# Đánh giá strict direct benchmark - {dataset} / {model}",
        "",
        "## Protocol",
        "",
        "- Không dùng rule-based detector.",
        "- Không dùng context-aware detector.",
        "- Không dùng BIPIA/indirect pipeline.",
        "- Split validation/test: 30% / 70%.",
        "- Nếu dataset là `all`, split theo `dataset_name + label`; dataset riêng thì split theo `label`.",
        "- Threshold chỉ được chọn trên validation, sau đó giữ cố định để đánh giá test.",
        "",
        "## Threshold",
        "",
        f"- Threshold chọn trên validation: `{threshold:.4f}`.",
        f"- Validation F2: `{_format_metric(validation_metrics.get('f2'))}`.",
        f"- Test F2: `{_format_metric(test_metrics.get('f2'))}`.",
        "",
        "## Metrics trên test",
        "",
        f"- Rows: `{test_metrics['rows']}`.",
        f"- Accuracy: `{_format_metric(test_metrics.get('accuracy'))}`.",
        f"- Precision: `{_format_metric(test_metrics.get('precision'))}`.",
        f"- Recall: `{_format_metric(test_metrics.get('recall'))}`.",
        f"- F1: `{_format_metric(test_metrics.get('f1'))}`.",
        f"- F2: `{_format_metric(test_metrics.get('f2'))}`.",
        f"- ROC-AUC: `{_format_metric(test_metrics.get('roc_auc'))}`.",
        f"- PR-AUC: `{_format_metric(test_metrics.get('pr_auc'))}`.",
        f"- Confusion matrix [[TN, FP], [FN, TP]]: `{test_metrics['confusion_matrix']}`.",
        f"- False positives: `{test_metrics['false_positive_count']}`.",
        f"- False negatives: `{test_metrics['false_negative_count']}`.",
        "",
        "## False positives mẫu",
        "",
    ]
    if false_positives:
        for row in false_positives[:10]:
            lines.append(f"- `{row['id']}` score={row['score']} text={str(row.get('text', ''))[:220]}")
    else:
        lines.append("Không có false positive.")
    lines.extend(["", "## False negatives mẫu", ""])
    if false_negatives:
        for row in false_negatives[:10]:
            lines.append(f"- `{row['id']}` score={row['score']} text={str(row.get('text', ''))[:220]}")
    else:
        lines.append("Không có false negative.")
    lines.extend(
        [
            "",
            "## Kết luận",
            "",
            "Kết quả này là protocol chính thức hơn raw full-set threshold sweep vì không tune threshold trực tiếp trên test.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_strict(
    *,
    dataset: str,
    model: str,
    split_threshold: bool = True,
    batch_size: int = 16,
    max_length: int = 128,
    use_cuda: bool = True,
    force_rescore: bool = False,
    seed: int = 2026,
    output_dir: Path = STRICT_OUTPUT_DIR,
) -> dict[str, Any]:
    if not split_threshold:
        raise ValueError("Strict evaluation requires --split-threshold.")

    all_predictions = _load_or_score_predictions(
        dataset=dataset,
        model=model,
        batch_size=batch_size,
        max_length=max_length,
        use_cuda=use_cuda,
        force_rescore=force_rescore,
    )
    validation_indices, test_indices = _split_indices(all_predictions, dataset, seed)
    validation_raw = [all_predictions[index] for index in validation_indices]
    test_raw = [all_predictions[index] for index in test_indices]

    validation_labels = [int(float(row["label"])) for row in validation_raw]
    validation_scores = [float(row["score"]) for row in validation_raw]
    threshold_summary = _choose_threshold(validation_labels, validation_scores)
    threshold = float(threshold_summary["threshold"])

    validation_predictions = _apply_threshold(validation_raw, threshold)
    test_predictions = _apply_threshold(test_raw, threshold)

    validation_metrics = _metrics_from_scores(
        [int(row["label"]) for row in validation_predictions],
        [float(row["score"]) for row in validation_predictions],
        threshold,
    )
    test_metrics = _metrics_from_scores(
        [int(row["label"]) for row in test_predictions],
        [float(row["score"]) for row in test_predictions],
        threshold,
    )
    test_metrics.update(
        {
            "dataset": dataset,
            "model": model,
            "threshold_selected_on": "validation",
            "validation_rows": len(validation_predictions),
            "test_rows": len(test_predictions),
            "split_strategy": "stratified_by_dataset_name_and_label" if dataset == "all" else "stratified_by_label",
        }
    )
    threshold_summary["validation_metrics"] = validation_metrics
    threshold_summary["test_metrics_at_validation_threshold"] = test_metrics

    target_dir = output_dir / dataset / model
    target_dir.mkdir(parents=True, exist_ok=True)
    false_positives = [
        row for row in test_predictions if int(row["label"]) == 0 and int(row["predicted_label"]) == 1
    ]
    false_negatives = [
        row for row in test_predictions if int(row["label"]) == 1 and int(row["predicted_label"]) == 0
    ]
    _write_csv(target_dir / "validation_predictions.csv", validation_predictions, PREDICTION_COLUMNS)
    _write_csv(target_dir / "test_predictions.csv", test_predictions, PREDICTION_COLUMNS)
    _write_csv(target_dir / "false_positives.csv", false_positives, PREDICTION_COLUMNS)
    _write_csv(target_dir / "false_negatives.csv", false_negatives, PREDICTION_COLUMNS)
    (target_dir / "validation_threshold_summary.json").write_text(
        json.dumps(threshold_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (target_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_report(
        path=target_dir / "strict_evaluation_report.md",
        dataset=dataset,
        model=model,
        threshold_summary=threshold_summary,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )
    _write_report_vi(
        path=target_dir / "strict_evaluation_report_vi.md",
        dataset=dataset,
        model=model,
        threshold_summary=threshold_summary,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )
    return test_metrics


def main() -> int:
    args = _parse_args()
    metrics = evaluate_strict(
        dataset=args.dataset,
        model=args.model,
        split_threshold=args.split_threshold,
        batch_size=args.batch_size,
        max_length=args.max_length,
        use_cuda=not args.no_cuda,
        force_rescore=args.force_rescore,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print("Strict direct benchmark evaluation complete")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Test rows: {metrics['rows']}")
    print(f"Threshold: {metrics['threshold']:.4f}")
    print(f"Test F1: {metrics['f1']:.4f}")
    print(f"Test F2: {metrics['f2']:.4f}")
    print(f"Test ROC-AUC: {_format_metric(metrics.get('roc_auc'))}")
    print(f"Confusion matrix: {metrics['confusion_matrix']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
