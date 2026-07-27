"""Calibrate direct benchmark scores using validation-only calibration.

This script reads model-only predictions, splits them into validation/test using
the same stratification rule as the strict evaluator, fits calibration only on
validation, and evaluates calibrated scores on test.

Outputs:
    reports/direct_external_evaluation/calibration/{dataset}/{model}/
        calibration_metrics.json
        reliability_diagram.png
        calibrated_predictions.csv

It also refreshes:
    reports/direct_external_evaluation/threshold_calibration_summary.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_direct_benchmarks import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    MODEL_LABELS,
    _format_metric,
    _metrics_from_scores,
    _write_csv,
    DATASET_FILES,
)
from scripts.evaluate_direct_benchmarks_strict import (  # noqa: E402
    _apply_threshold,
    _choose_threshold,
    _load_or_score_predictions,
    _split_indices,
)


CALIBRATION_DIR = DEFAULT_OUTPUT_DIR / "calibration"
RUNTIME_CALIBRATION_DIR = PROJECT_ROOT / "models" / "calibration"
CALIBRATED_THRESHOLDS_PATH = PROJECT_ROOT / "models" / "calibrated_thresholds.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate direct benchmark model scores.")
    parser.add_argument("--dataset", choices=sorted(DATASET_FILES), required=True)
    parser.add_argument("--model", choices=sorted(MODEL_LABELS), required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--force-rescore", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=CALIBRATION_DIR)
    return parser.parse_args()


def _brier(labels: list[int], scores: list[float]) -> float:
    return sum((score - label) ** 2 for label, score in zip(labels, scores)) / len(labels)


def _calibration_errors(labels: list[int], scores: list[float], bins: int = 10) -> dict[str, Any]:
    total = len(labels)
    ece = 0.0
    mce = 0.0
    bin_rows: list[dict[str, Any]] = []
    for index in range(bins):
        start = index / bins
        end = (index + 1) / bins
        selected = [
            (label, score)
            for label, score in zip(labels, scores)
            if (start <= score < end) or (index == bins - 1 and start <= score <= end)
        ]
        if not selected:
            bin_rows.append(
                {
                    "bin_start": start,
                    "bin_end": end,
                    "count": 0,
                    "mean_score": None,
                    "fraction_positive": None,
                    "abs_gap": None,
                }
            )
            continue
        mean_score = sum(score for _, score in selected) / len(selected)
        fraction_positive = sum(label for label, _ in selected) / len(selected)
        gap = abs(mean_score - fraction_positive)
        ece += (len(selected) / total) * gap
        mce = max(mce, gap)
        bin_rows.append(
            {
                "bin_start": start,
                "bin_end": end,
                "count": len(selected),
                "mean_score": mean_score,
                "fraction_positive": fraction_positive,
                "abs_gap": gap,
            }
        )
    return {"ece": ece, "mce": mce, "bins": bin_rows}


def _score_stats(scores: list[float]) -> dict[str, float | None]:
    if not scores:
        return {"min": None, "max": None, "mean": None, "std": None}
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / len(scores)
    return {
        "min": min(scores),
        "max": max(scores),
        "mean": mean,
        "std": math.sqrt(variance),
    }


def _fit_calibrators(validation_labels: list[int], validation_scores: list[float]) -> tuple[str | None, Any | None, dict[str, Any]]:
    if len(set(validation_labels)) < 2:
        return None, None, {"error": "Calibration requires both classes in validation split."}
    candidates: dict[str, Any] = {}
    notes: list[str] = []

    try:
        from sklearn.linear_model import LogisticRegression

        platt = LogisticRegression(solver="lbfgs")
        platt.fit([[score] for score in validation_scores], validation_labels)
        platt_scores = [float(value[1]) for value in platt.predict_proba([[score] for score in validation_scores])]
        candidates["platt"] = {
            "model": platt,
            "validation_brier": _brier(validation_labels, platt_scores),
            "validation_ece": _calibration_errors(validation_labels, platt_scores)["ece"],
        }
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Platt scaling failed: {type(exc).__name__}: {exc}")

    if len(validation_labels) >= 100:
        try:
            from sklearn.isotonic import IsotonicRegression

            isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            isotonic.fit(validation_scores, validation_labels)
            isotonic_scores = [float(value) for value in isotonic.predict(validation_scores)]
            candidates["isotonic"] = {
                "model": isotonic,
                "validation_brier": _brier(validation_labels, isotonic_scores),
                "validation_ece": _calibration_errors(validation_labels, isotonic_scores)["ece"],
            }
        except Exception as exc:  # noqa: BLE001
            notes.append(f"Isotonic regression failed: {type(exc).__name__}: {exc}")
    else:
        notes.append("Isotonic regression skipped because validation set has fewer than 100 rows.")

    if not candidates:
        return None, None, {"error": "No calibration method could be fitted.", "notes": notes}
    selected_name, selected = min(candidates.items(), key=lambda item: item[1]["validation_brier"])
    metadata = {
        "selected_method": selected_name,
        "candidate_metrics": {
            name: {
                "validation_brier": values["validation_brier"],
                "validation_ece": values["validation_ece"],
            }
            for name, values in candidates.items()
        },
        "notes": notes,
        "temperature_scaling": {
            "attempted": False,
            "reason": "Predictions CSV contains probabilities/scores only; logits are not available for temperature scaling.",
        },
    }
    return selected_name, selected["model"], metadata


def _predict_calibrated(method: str | None, calibrator: Any | None, scores: list[float]) -> list[float]:
    if method is None or calibrator is None:
        return scores[:]
    if method == "platt":
        return [float(value[1]) for value in calibrator.predict_proba([[score] for score in scores])]
    if method == "isotonic":
        return [float(value) for value in calibrator.predict(scores)]
    raise ValueError(f"Unsupported calibration method: {method}")


def _metrics_payload(labels: list[int], scores: list[float], threshold: float) -> dict[str, Any]:
    metrics = _metrics_from_scores(labels, scores, threshold)
    calibration = _calibration_errors(labels, scores)
    metrics.update(
        {
            "brier": _brier(labels, scores),
            "ece": calibration["ece"],
            "mce": calibration["mce"],
            "calibration_bins": calibration["bins"],
        }
    )
    return metrics


def _plot_reliability(path: Path, raw_bins: list[dict[str, Any]], calibrated_bins: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def points(bins: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
        xs: list[float] = []
        ys: list[float] = []
        for row in bins:
            if row["mean_score"] is not None and row["fraction_positive"] is not None:
                xs.append(float(row["mean_score"]))
                ys.append(float(row["fraction_positive"]))
        return xs, ys

    raw_x, raw_y = points(raw_bins)
    cal_x, cal_y = points(calibrated_bins)
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    plt.plot(raw_x, raw_y, marker="o", label="raw")
    plt.plot(cal_x, cal_y, marker="o", label="calibrated")
    plt.xlabel("Mean predicted score")
    plt.ylabel("Fraction positive")
    plt.title("Reliability diagram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_single_reliability(path: Path, bins: list[dict[str, Any]], title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs: list[float] = []
    ys: list[float] = []
    for row in bins:
        if row["mean_score"] is not None and row["fraction_positive"] is not None:
            xs.append(float(row["mean_score"]))
            ys.append(float(row["fraction_positive"]))

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    plt.plot(xs, ys, marker="o", label=title)
    plt.xlabel("Mean predicted score")
    plt.ylabel("Fraction positive")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _calibrated_prediction_rows(
    rows: list[dict[str, Any]],
    split_name: str,
    raw_threshold: float,
    calibrated_threshold: float,
    calibrated_scores: list[float],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row, calibrated_score in zip(rows, calibrated_scores):
        raw_score = float(row["score"])
        output.append(
            {
                "id": row.get("id", ""),
                "dataset_name": row.get("dataset_name", ""),
                "split": split_name,
                "text": row.get("text", ""),
                "label": int(float(row["label"])),
                "raw_score": round(raw_score, 8),
                "raw_threshold": round(raw_threshold, 4),
                "raw_predicted_label": 1 if raw_score >= raw_threshold else 0,
                "calibrated_score": round(float(calibrated_score), 8),
                "calibrated_threshold": round(calibrated_threshold, 4),
                "calibrated_predicted_label": 1 if float(calibrated_score) >= calibrated_threshold else 0,
                "attack_type": row.get("attack_type", ""),
                "source_label": row.get("source_label", ""),
            }
        )
    return output


def _raw_test_prediction_rows(rows: list[dict[str, Any]], raw_threshold: float) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        raw_score = float(row["score"])
        output.append(
            {
                "id": row.get("id", ""),
                "dataset_name": row.get("dataset_name", ""),
                "text": row.get("text", ""),
                "label": int(float(row["label"])),
                "score_type": "raw_probability",
                "raw_score": round(raw_score, 8),
                "threshold": round(raw_threshold, 4),
                "predicted_label": 1 if raw_score >= raw_threshold else 0,
                "attack_type": row.get("attack_type", ""),
                "source_label": row.get("source_label", ""),
                "logit_safe": row.get("logit_safe", ""),
                "logit_injection": row.get("logit_injection", ""),
            }
        )
    return output


def _calibrated_test_prediction_rows(
    rows: list[dict[str, Any]],
    calibrated_scores: list[float],
    calibrated_threshold: float,
    method: str | None,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row, calibrated_score in zip(rows, calibrated_scores):
        output.append(
            {
                "id": row.get("id", ""),
                "dataset_name": row.get("dataset_name", ""),
                "text": row.get("text", ""),
                "label": int(float(row["label"])),
                "score_type": "calibrated_probability",
                "calibration_method": method,
                "raw_score": round(float(row["score"]), 8),
                "calibrated_score": round(float(calibrated_score), 8),
                "threshold": round(calibrated_threshold, 4),
                "predicted_label": 1 if float(calibrated_score) >= calibrated_threshold else 0,
                "attack_type": row.get("attack_type", ""),
                "source_label": row.get("source_label", ""),
                "logit_safe": row.get("logit_safe", ""),
                "logit_injection": row.get("logit_injection", ""),
            }
        )
    return output


def _delta(before: Any, after: Any) -> str:
    try:
        return _format_metric(float(after) - float(before))
    except (TypeError, ValueError):
        return "n/a"


def _write_calibration_report_vi(path: Path, metrics: dict[str, Any]) -> None:
    raw = metrics["raw_test_metrics"]
    calibrated = metrics["calibrated_test_metrics"]
    method = metrics.get("selected_calibration_method") or "không có"
    temperature = metrics.get("calibration_metadata", {}).get("temperature_scaling", {})
    lines = [
        f"# Báo cáo calibration - {metrics['dataset']} / {metrics['model']}",
        "",
        "## Protocol",
        "",
        "- Không dùng rule-based detector.",
        "- Không dùng context-aware detector.",
        "- Split: validation 30%, test 70%.",
        f"- Split strategy: `{metrics.get('split_strategy')}`.",
        "- Calibrator fit trên validation, sau đó apply lên test.",
        "- Threshold raw và calibrated đều được chọn trên validation, không tune trên test.",
        "",
        "## Calibration method",
        "",
        f"- Method được chọn: `{method}`.",
        f"- Temperature Scaling: `{temperature.get('attempted', False)}`.",
        f"- Lý do Temperature Scaling: {temperature.get('reason', 'n/a')}",
        "",
        "## Threshold",
        "",
        f"- Raw threshold: `{_format_metric(metrics.get('raw_threshold'))}`.",
        f"- Calibrated threshold: `{_format_metric(metrics.get('calibrated_threshold'))}`.",
        "",
        "## Metrics trên test",
        "",
        "| Metric | Raw | Calibrated | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, label in [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("f2", "F2"),
        ("roc_auc", "ROC-AUC"),
        ("pr_auc", "PR-AUC"),
        ("brier", "Brier Score"),
        ("ece", "ECE"),
    ]:
        lines.append(
            f"| {label} | {_format_metric(raw.get(key))} | {_format_metric(calibrated.get(key))} | {_delta(raw.get(key), calibrated.get(key))} |"
        )
    lines.extend(
        [
            "",
            "## Confusion matrix",
            "",
            f"- Raw [[TN, FP], [FN, TP]]: `{raw.get('confusion_matrix')}`.",
            f"- Calibrated [[TN, FP], [FN, TP]]: `{calibrated.get('confusion_matrix')}`.",
            "",
            "## Nhận xét",
            "",
            "- Calibration giúp score/probability dễ diễn giải hơn, đặc biệt khi raw threshold quá thấp.",
            "- Nếu F2 tăng sau calibration, mô hình giữ được mục tiêu ưu tiên recall nhưng threshold trở nên hợp lý hơn.",
            "- Nếu precision giảm nhưng recall tăng, đây là trade-off do selection metric F2 ưu tiên giảm false negative.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _runtime_dataset_name(dataset: str) -> str:
    return "direct_all" if dataset == "all" else f"direct_{dataset}"


def _save_runtime_calibrator(dataset: str, model: str, calibrator: Any | None, report_calibrator_path: Path) -> Path | None:
    if calibrator is None:
        return None
    runtime_dir = RUNTIME_CALIBRATION_DIR / _runtime_dataset_name(dataset) / model
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = runtime_dir / "probability_calibrator.joblib"
    joblib.dump(calibrator, runtime_path)
    # Also ensure the report artifact exists even when runtime save is the primary write.
    if not report_calibrator_path.exists():
        joblib.dump(calibrator, report_calibrator_path)
    return runtime_path


def _refresh_calibrated_thresholds() -> None:
    payload: dict[str, Any] = {}
    for metrics_path in sorted((CALIBRATION_DIR / "all").glob("*/calibration_metrics.json")):
        metrics = _read_json(metrics_path)
        if not metrics:
            continue
        model = str(metrics.get("model"))
        method = metrics.get("selected_calibration_method")
        payload[model] = {
            "dataset": "direct_all",
            "calibration_method": method,
            "score_type": "calibrated_probability",
            "threshold_eval": float(metrics.get("calibrated_threshold")),
            "threshold_warn": 0.50,
            "threshold_block": 0.80,
            "selection_metric": "F2",
            "validation_split": "30%",
            "test_split": "70%",
            "split_strategy": metrics.get("split_strategy"),
            "calibrator_path": str(RUNTIME_CALIBRATION_DIR / "direct_all" / model / "probability_calibrator.joblib"),
            "report_calibrator_path": str(metrics_path.parent / "probability_calibrator.joblib"),
            "notes": (
                "Threshold selected on validation, evaluated on test. "
                "threshold_eval is optimized for evaluation/F2; warn/block are stricter runtime policy thresholds to reduce false positives."
            ),
            "test_metrics": {
                "accuracy": metrics.get("calibrated_test_metrics", {}).get("accuracy"),
                "precision": metrics.get("calibrated_test_metrics", {}).get("precision"),
                "recall": metrics.get("calibrated_test_metrics", {}).get("recall"),
                "f1": metrics.get("calibrated_test_metrics", {}).get("f1"),
                "f2": metrics.get("calibrated_test_metrics", {}).get("f2"),
                "roc_auc": metrics.get("calibrated_test_metrics", {}).get("roc_auc"),
                "pr_auc": metrics.get("calibrated_test_metrics", {}).get("pr_auc"),
                "brier": metrics.get("calibrated_test_metrics", {}).get("brier"),
                "ece": metrics.get("calibrated_test_metrics", {}).get("ece"),
                "confusion_matrix": metrics.get("calibrated_test_metrics", {}).get("confusion_matrix"),
            },
        }
    CALIBRATED_THRESHOLDS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _stats_text(stats: dict[str, Any]) -> str:
    return (
        f"min={_format_metric(stats.get('min'))}, "
        f"max={_format_metric(stats.get('max'))}, "
        f"mean={_format_metric(stats.get('mean'))}, "
        f"std={_format_metric(stats.get('std'))}"
    )


def write_score_source_audit(path: Path = DEFAULT_OUTPUT_DIR / "score_source_audit.md") -> None:
    source_map = {
        "logistic_regression": "scikit-learn `predict_proba` positive class từ TF-IDF LogisticRegression; nếu không có predict_proba mới fallback sigmoid(decision_function).",
        "random_forest": "scikit-learn `predict_proba` positive class từ TF-IDF RandomForest; tương đương trung bình vote/probability của cây.",
        "roberta": "Transformer softmax probability của class INJECTION từ sequence-classification logits.",
        "xlm_roberta": "Transformer softmax probability của class INJECTION từ sequence-classification logits.",
    }
    rows: list[str] = []
    for model in ["logistic_regression", "random_forest", "roberta", "xlm_roberta"]:
        metrics = _read_json(CALIBRATION_DIR / "all" / model / "calibration_metrics.json") or {}
        method = metrics.get("selected_calibration_method")
        report_calibrator = CALIBRATION_DIR / "all" / model / "probability_calibrator.joblib"
        runtime_calibrator = RUNTIME_CALIBRATION_DIR / "direct_all" / model / "probability_calibrator.joblib"
        raw_val_stats = metrics.get("raw_validation_score_stats", {})
        raw_test_stats = metrics.get("raw_test_score_stats", {})
        warning = (
            "Đã có calibrator runtime/report."
            if runtime_calibrator.exists()
            else "Chưa có calibrator runtime; score raw có thể chưa calibrated."
        )
        rows.append(
            "| "
            + " | ".join(
                [
                    model,
                    source_map[model],
                    "Có" if method else "Không",
                    str(method or "N/A"),
                    "Có" if report_calibrator.exists() else "Không",
                    "Có" if runtime_calibrator.exists() else "Không",
                    _stats_text(raw_val_stats),
                    _stats_text(raw_test_stats),
                    warning,
                ]
            )
            + " |"
        )

    lines = [
        "# Audit nguồn score cho Direct External Evaluation",
        "",
        "Audit này chỉ xét model-only direct benchmark scoring. Không dùng rule-based, context-aware, BIPIA hoặc indirect pipeline.",
        "",
        "| Model | Score source | Đã calibrate? | Calibration method | Report calibrator | Runtime calibrator | Raw validation score stats | Raw test score stats | Warning |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        *rows,
        "",
        "## Ghi chú",
        "",
        "- Logistic Regression/Random Forest dùng `predict_proba` của scikit-learn.",
        "- RoBERTa/XLM-RoBERTa dùng softmax probability của class INJECTION; runtime Transformer cũng trả về logits.",
        "- Temperature Scaling chưa chạy trong calibration hiện tại nếu prediction CSV chỉ có probability/score và chưa có logits được lưu từ direct benchmark scorer.",
        "- Threshold/calibrator chính thức phải được fit/chọn trên validation và chỉ đánh giá trên test.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def calibrate_scores(
    *,
    dataset: str,
    model: str,
    batch_size: int = 16,
    max_length: int = 128,
    use_cuda: bool = True,
    force_rescore: bool = False,
    seed: int = 2026,
    output_dir: Path = CALIBRATION_DIR,
) -> dict[str, Any]:
    predictions = _load_or_score_predictions(
        dataset=dataset,
        model=model,
        batch_size=batch_size,
        max_length=max_length,
        use_cuda=use_cuda,
        force_rescore=force_rescore,
    )
    validation_indices, test_indices = _split_indices(predictions, dataset, seed)
    validation_rows = [predictions[index] for index in validation_indices]
    test_rows = [predictions[index] for index in test_indices]
    validation_labels = [int(float(row["label"])) for row in validation_rows]
    validation_scores = [float(row["score"]) for row in validation_rows]
    test_labels = [int(float(row["label"])) for row in test_rows]
    test_scores = [float(row["score"]) for row in test_rows]

    raw_threshold_summary = _choose_threshold(validation_labels, validation_scores)
    raw_threshold = float(raw_threshold_summary["threshold"])
    raw_validation_metrics = _metrics_payload(validation_labels, validation_scores, raw_threshold)
    raw_test_metrics = _metrics_payload(test_labels, test_scores, raw_threshold)

    method, calibrator, calibration_metadata = _fit_calibrators(validation_labels, validation_scores)
    calibrated_validation_scores = _predict_calibrated(method, calibrator, validation_scores)
    calibrated_test_scores = _predict_calibrated(method, calibrator, test_scores)
    calibrated_threshold_summary = _choose_threshold(validation_labels, calibrated_validation_scores)
    calibrated_threshold = float(calibrated_threshold_summary["threshold"])
    calibrated_validation_metrics = _metrics_payload(validation_labels, calibrated_validation_scores, calibrated_threshold)
    calibrated_test_metrics = _metrics_payload(test_labels, calibrated_test_scores, calibrated_threshold)

    target_dir = output_dir / dataset / model
    target_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.extend(
        _calibrated_prediction_rows(
            validation_rows,
            "validation",
            raw_threshold,
            calibrated_threshold,
            calibrated_validation_scores,
        )
    )
    rows.extend(
        _calibrated_prediction_rows(
            test_rows,
            "test",
            raw_threshold,
            calibrated_threshold,
            calibrated_test_scores,
        )
    )
    _write_csv(target_dir / "calibrated_predictions.csv", rows)
    _write_csv(target_dir / "raw_test_predictions.csv", _raw_test_prediction_rows(test_rows, raw_threshold))
    _write_csv(
        target_dir / "calibrated_test_predictions.csv",
        _calibrated_test_prediction_rows(test_rows, calibrated_test_scores, calibrated_threshold, method),
    )
    _plot_reliability(
        target_dir / "reliability_diagram.png",
        raw_test_metrics["calibration_bins"],
        calibrated_test_metrics["calibration_bins"],
    )
    _plot_single_reliability(
        target_dir / "reliability_diagram_raw.png",
        raw_test_metrics["calibration_bins"],
        "Raw reliability",
    )
    _plot_single_reliability(
        target_dir / "reliability_diagram_calibrated.png",
        calibrated_test_metrics["calibration_bins"],
        "Calibrated reliability",
    )
    report_calibrator_path = target_dir / "probability_calibrator.joblib"
    runtime_calibrator_path = _save_runtime_calibrator(dataset, model, calibrator, report_calibrator_path)

    metrics = {
        "dataset": dataset,
        "model": model,
        "validation_rows": len(validation_rows),
        "test_rows": len(test_rows),
        "split_strategy": "stratified_by_dataset_name_and_label" if dataset == "all" else "stratified_by_label",
        "calibration_fit_split": "validation",
        "selected_calibration_method": method,
        "calibration_metadata": calibration_metadata,
        "probability_calibrator_path": str(report_calibrator_path) if report_calibrator_path.exists() else None,
        "runtime_probability_calibrator_path": str(runtime_calibrator_path) if runtime_calibrator_path else None,
        "raw_validation_score_stats": _score_stats(validation_scores),
        "raw_test_score_stats": _score_stats(test_scores),
        "calibrated_validation_score_stats": _score_stats(calibrated_validation_scores),
        "calibrated_test_score_stats": _score_stats(calibrated_test_scores),
        "raw_threshold": raw_threshold,
        "raw_validation_metrics": raw_validation_metrics,
        "raw_test_metrics": raw_test_metrics,
        "calibrated_threshold": calibrated_threshold,
        "calibrated_validation_metrics": calibrated_validation_metrics,
        "calibrated_test_metrics": calibrated_test_metrics,
    }
    (target_dir / "calibration_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_calibration_report_vi(target_dir / "calibration_report_vi.md", metrics)
    _refresh_calibrated_thresholds()
    write_score_source_audit()
    write_threshold_calibration_summary()
    return metrics


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _summary_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(CALIBRATION_DIR.glob("*/*/calibration_metrics.json")):
        metrics = _read_json(metrics_path)
        if not metrics:
            continue
        raw_test = metrics.get("raw_test_metrics", {})
        calibrated_test = metrics.get("calibrated_test_metrics", {})
        rows.append(
            {
                "Dataset": metrics.get("dataset"),
                "Model": metrics.get("model"),
                "Raw Threshold": _format_metric(metrics.get("raw_threshold")),
                "Raw Test F1": _format_metric(raw_test.get("f1")),
                "Raw Test F2": _format_metric(raw_test.get("f2")),
                "Calibrated Threshold": _format_metric(metrics.get("calibrated_threshold")),
                "Calibrated Test F1": _format_metric(calibrated_test.get("f1")),
                "Calibrated Test F2": _format_metric(calibrated_test.get("f2")),
                "ROC-AUC": _format_metric(calibrated_test.get("roc_auc")),
                "PR-AUC": _format_metric(calibrated_test.get("pr_auc")),
                "Brier Before": _format_metric(raw_test.get("brier")),
                "Brier After": _format_metric(calibrated_test.get("brier")),
                "ECE Before": _format_metric(raw_test.get("ece")),
                "ECE After": _format_metric(calibrated_test.get("ece")),
                "_metrics": metrics,
            }
        )
    rows.sort(key=lambda row: (str(row["Dataset"]), str(row["Model"])))
    return rows


def write_threshold_calibration_summary(path: Path = DEFAULT_OUTPUT_DIR / "threshold_calibration_summary.md") -> None:
    rows = _summary_rows()
    columns = [
        "Dataset",
        "Model",
        "Raw Threshold",
        "Raw Test F1",
        "Raw Test F2",
        "Calibrated Threshold",
        "Calibrated Test F1",
        "Calibrated Test F2",
        "ROC-AUC",
        "PR-AUC",
        "Brier Before",
        "Brier After",
        "ECE Before",
        "ECE After",
    ]
    lines = [
        "# Direct Threshold and Calibration Summary",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    lines.extend(["", "## Analysis", ""])
    if rows:
        by_key = {(row["Dataset"], row["Model"]): row for row in rows}
        roberta_all = by_key.get(("all", "roberta"))
        xlm_all = by_key.get(("all", "xlm_roberta"))
        if roberta_all:
            metrics = roberta_all["_metrics"]
            lines.append(
                "1. RoBERTa low threshold: raw validation-selected threshold is "
                f"`{_format_metric(metrics.get('raw_threshold'))}` because F2 optimization favors recall and a non-trivial injection tail has low softmax scores."
            )
            lines.append(
                "2. RoBERTa calibration: calibrated threshold becomes "
                f"`{_format_metric(metrics.get('calibrated_threshold'))}`; compare Brier "
                f"`{_format_metric(metrics['raw_test_metrics'].get('brier'))}` -> "
                f"`{_format_metric(metrics['calibrated_test_metrics'].get('brier'))}`."
            )
        if xlm_all:
            metrics = xlm_all["_metrics"]
            lines.append(
                "3. XLM-RoBERTa over-sensitivity: raw threshold "
                f"`{_format_metric(metrics.get('raw_threshold'))}` with high recall but many false positives indicates score calibration/decision boundary issues."
            )
            lines.append(
                "4. XLM-RoBERTa calibration: Brier "
                f"`{_format_metric(metrics['raw_test_metrics'].get('brier'))}` -> "
                f"`{_format_metric(metrics['calibrated_test_metrics'].get('brier'))}`, "
                f"ECE `{_format_metric(metrics['raw_test_metrics'].get('ece'))}` -> "
                f"`{_format_metric(metrics['calibrated_test_metrics'].get('ece'))}`."
            )
        all_rows = [row for row in rows if row.get("Dataset") == "all"]
        best_all = (
            max(all_rows, key=lambda row: float(row["_metrics"]["calibrated_test_metrics"].get("f2", 0.0)))
            if all_rows
            else None
        )
        best = max(rows, key=lambda row: float(row["_metrics"]["calibrated_test_metrics"].get("f2", 0.0)))
        if best_all:
            lines.append(
                f"5. Best calibrated F2 on `all`: `{best_all['Model']}` with calibrated F2 `{best_all['Calibrated Test F2']}`."
            )
        lines.append(
            f"6. Best calibrated F2 among all completed calibration runs: `{best['Model']}` on `{best['Dataset']}` "
            f"with calibrated F2 `{best['Calibrated Test F2']}`."
        )
        lines.append(
            "7. Temperature scaling was not run here because saved predictions contain probabilities/scores, not logits. "
            "To do temperature scaling, rerun Transformer inference and persist logits on the validation split."
        )
    else:
        lines.append("No calibration runs found yet.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    metrics = calibrate_scores(
        dataset=args.dataset,
        model=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        use_cuda=not args.no_cuda,
        force_rescore=args.force_rescore,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print("Direct score calibration complete")
    print(f"Dataset: {metrics['dataset']}")
    print(f"Model: {metrics['model']}")
    print(f"Method: {metrics['selected_calibration_method']}")
    print(f"Raw threshold: {metrics['raw_threshold']:.4f}")
    print(f"Calibrated threshold: {metrics['calibrated_threshold']:.4f}")
    print(f"Raw test F2: {metrics['raw_test_metrics']['f2']:.4f}")
    print(f"Calibrated test F2: {metrics['calibrated_test_metrics']['f2']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
