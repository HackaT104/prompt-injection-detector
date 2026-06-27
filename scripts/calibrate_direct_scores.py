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
    _plot_reliability(
        target_dir / "reliability_diagram.png",
        raw_test_metrics["calibration_bins"],
        calibrated_test_metrics["calibration_bins"],
    )

    metrics = {
        "dataset": dataset,
        "model": model,
        "validation_rows": len(validation_rows),
        "test_rows": len(test_rows),
        "split_strategy": "stratified_by_dataset_name_and_label" if dataset == "all" else "stratified_by_label",
        "calibration_fit_split": "validation",
        "selected_calibration_method": method,
        "calibration_metadata": calibration_metadata,
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
