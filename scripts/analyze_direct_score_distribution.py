"""Analyze direct benchmark score distributions.

Outputs per dataset/model:
    reports/direct_external_evaluation/score_analysis/{dataset}/{model}/
        score_summary.json
        score_distribution.csv
        score_histogram.png
        score_density.png
        roc_curve.png
        pr_curve.png

The script also refreshes:
    reports/direct_external_evaluation/score_source_audit.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
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
    DATASET_FILES,
)


ANALYSIS_DIR = DEFAULT_OUTPUT_DIR / "score_analysis"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze direct benchmark score distribution.")
    parser.add_argument("--dataset", choices=sorted(DATASET_FILES), required=True)
    parser.add_argument("--model", choices=sorted(MODEL_LABELS), required=True)
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS_DIR)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _prediction_path(dataset: str, model: str) -> Path:
    if dataset == "all":
        return DEFAULT_OUTPUT_DIR / model / "predictions.csv"
    nested = DEFAULT_OUTPUT_DIR / model / dataset / "predictions.csv"
    if nested.exists():
        return nested
    return DEFAULT_OUTPUT_DIR / model / "predictions.csv"


def _load_predictions(dataset: str, model: str) -> list[dict[str, Any]]:
    path = _prediction_path(dataset, model)
    if not path.exists():
        raise FileNotFoundError(
            f"Predictions not found: {path}. Run scripts/evaluate_direct_benchmarks.py first."
        )
    rows = _read_csv(path)
    if dataset != "all":
        rows = [row for row in rows if str(row.get("dataset_name")) == dataset]
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        try:
            label = int(float(row["label"]))
            score = float(row["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if label not in {0, 1} or math.isnan(score):
            continue
        payload = dict(row)
        payload["label"] = label
        payload["score"] = score
        cleaned.append(payload)
    if not cleaned:
        raise ValueError(f"No valid scored rows in {path}")
    return cleaned


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def _stats(values: list[float], prefix: str) -> dict[str, Any]:
    if not values:
        return {
            f"{prefix}_score_min": None,
            f"{prefix}_score_max": None,
            f"{prefix}_score_mean": None,
            f"{prefix}_score_median": None,
        }
    payload = {
        f"{prefix}_score_min": min(values),
        f"{prefix}_score_max": max(values),
        f"{prefix}_score_mean": statistics.fmean(values),
        f"{prefix}_score_std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        f"{prefix}_score_median": _percentile(values, 0.50),
    }
    if prefix == "safe":
        payload.update(
            {
                "safe_score_p90": _percentile(values, 0.90),
                "safe_score_p95": _percentile(values, 0.95),
                "safe_score_p99": _percentile(values, 0.99),
            }
        )
    else:
        payload.update(
            {
                "injection_score_p01": _percentile(values, 0.01),
                "injection_score_p05": _percentile(values, 0.05),
                "injection_score_p10": _percentile(values, 0.10),
            }
        )
    return payload


def _write_distribution_csv(path: Path, safe_scores: list[float], injection_scores: list[float]) -> None:
    bins = [index / 20 for index in range(21)]
    rows: list[dict[str, Any]] = []
    for start, end in zip(bins[:-1], bins[1:]):
        safe_count = sum(
            1
            for score in safe_scores
            if (start <= score < end) or (end == 1.0 and start <= score <= end)
        )
        injection_count = sum(
            1
            for score in injection_scores
            if (start <= score < end) or (end == 1.0 and start <= score <= end)
        )
        rows.append(
            {
                "bin_start": round(start, 2),
                "bin_end": round(end, 2),
                "safe_count": safe_count,
                "injection_count": injection_count,
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["bin_start", "bin_end", "safe_count", "injection_count"])
        writer.writeheader()
        writer.writerows(rows)


def _plot_curves(target_dir: Path, labels: list[int], scores: list[float], safe_scores: list[float], injection_scores: list[float]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    plt.figure(figsize=(8, 5))
    plt.hist(safe_scores, bins=40, alpha=0.65, label="safe", color="#4c78a8")
    plt.hist(injection_scores, bins=40, alpha=0.65, label="injection", color="#f58518")
    plt.xlabel("Score")
    plt.ylabel("Count")
    plt.title("Score histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(target_dir / "score_histogram.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(safe_scores, bins=40, alpha=0.65, label="safe", color="#4c78a8", density=True)
    plt.hist(injection_scores, bins=40, alpha=0.65, label="injection", color="#f58518", density=True)
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.title("Score density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(target_dir / "score_density.png", dpi=150)
    plt.close()

    fpr, tpr, _ = roc_curve(labels, scores)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(target_dir / "roc_curve.png", dpi=150)
    plt.close()

    precision, recall, _ = precision_recall_curve(labels, scores)
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label="PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(target_dir / "pr_curve.png", dpi=150)
    plt.close()


def analyze_scores(dataset: str, model: str, output_dir: Path = ANALYSIS_DIR) -> dict[str, Any]:
    rows = _load_predictions(dataset, model)
    labels = [int(row["label"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    safe_scores = [score for label, score in zip(labels, scores) if label == 0]
    injection_scores = [score for label, score in zip(labels, scores) if label == 1]

    from sklearn.metrics import average_precision_score, roc_auc_score

    summary: dict[str, Any] = {
        "dataset": dataset,
        "model": model,
        "rows": len(rows),
        "safe_count": len(safe_scores),
        "injection_count": len(injection_scores),
        **_stats(safe_scores, "safe"),
        **_stats(injection_scores, "injection"),
        "roc_auc": float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None,
        "pr_auc": float(average_precision_score(labels, scores)) if len(set(labels)) == 2 else None,
        "score_source": SCORE_SOURCES[model]["source"],
        "posthoc_calibrated": SCORE_SOURCES[model]["posthoc_calibrated"],
        "calibration_warning": SCORE_SOURCES[model]["warning"],
    }

    target_dir = output_dir / dataset / model
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "score_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_distribution_csv(target_dir / "score_distribution.csv", safe_scores, injection_scores)
    _plot_curves(target_dir, labels, scores, safe_scores, injection_scores)
    write_score_source_audit()
    return summary


SCORE_SOURCES = {
    "logistic_regression": {
        "source": "scikit-learn predict_proba positive class from TF-IDF LogisticRegression; decision_function sigmoid fallback only if predict_proba is absent.",
        "posthoc_calibrated": False,
        "warning": "Model probability is not externally calibrated on the direct benchmark validation split.",
    },
    "random_forest": {
        "source": "scikit-learn predict_proba positive class from TF-IDF RandomForest; effectively tree vote/probability average.",
        "posthoc_calibrated": False,
        "warning": "Random Forest probabilities are often poorly calibrated without post-hoc calibration.",
    },
    "roberta": {
        "source": "Transformer softmax probability for INJECTION class from sequence-classification logits.",
        "posthoc_calibrated": False,
        "warning": "No probability_calibrator.joblib was used for roberta_v5_vi in this direct evaluation; softmax probabilities may be miscalibrated.",
    },
    "xlm_roberta": {
        "source": "Transformer softmax probability for INJECTION class from sequence-classification logits.",
        "posthoc_calibrated": False,
        "warning": "No probability_calibrator.joblib was used for xlm_roberta_v5_vi in this direct evaluation; softmax probabilities may be miscalibrated.",
    },
}


def _audit_row(model: str) -> dict[str, Any]:
    path = _prediction_path("all", model)
    rows = _read_csv(path) if path.exists() else []
    scores: list[float] = []
    for row in rows:
        try:
            scores.append(float(row["score"]))
        except (KeyError, TypeError, ValueError):
            continue
    source = SCORE_SOURCES[model]
    return {
        "model": model,
        "score_source": source["source"],
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "score_mean": statistics.fmean(scores) if scores else None,
        "score_std": statistics.pstdev(scores) if len(scores) > 1 else None,
        "posthoc_calibrated": source["posthoc_calibrated"],
        "warning": source["warning"],
    }


def write_score_source_audit(output_path: Path = DEFAULT_OUTPUT_DIR / "score_source_audit.md") -> None:
    rows = [_audit_row(model) for model in ["logistic_regression", "random_forest", "roberta", "xlm_roberta"]]
    lines = [
        "# Direct External Evaluation Score Source Audit",
        "",
        "This audit covers model-only direct benchmark scoring. It does not use rule-based, context-aware, BIPIA, or indirect pipelines.",
        "",
        "| Model | Score source | Min | Max | Mean | Std | Post-hoc calibrated? | Warning |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model"]),
                    str(row["score_source"]),
                    _format_metric(row["score_min"]),
                    _format_metric(row["score_max"]),
                    _format_metric(row["score_mean"]),
                    _format_metric(row["score_std"]),
                    str(row["posthoc_calibrated"]),
                    str(row["warning"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- RoBERTa/XLM-RoBERTa scores are softmax probabilities, not raw logits, sigmoid scores, or margins.",
            "- The direct evaluation scripts apply no external calibrator unless a model directory contains `probability_calibrator.joblib`; the current v5 RoBERTa/XLM-R runs reported no calibration method.",
            "- Low optimal thresholds can happen when F2 optimization prioritizes recall, when probabilities are compressed near zero, or when calibration is poor. Use strict validation/test and calibration reports before drawing final conclusions.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    summary = analyze_scores(args.dataset, args.model, args.output_dir)
    print("Score distribution analysis complete")
    print(f"Dataset: {summary['dataset']}")
    print(f"Model: {summary['model']}")
    print(f"Rows: {summary['rows']}")
    print(f"ROC-AUC: {_format_metric(summary.get('roc_auc'))}")
    print(f"PR-AUC: {_format_metric(summary.get('pr_auc'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
