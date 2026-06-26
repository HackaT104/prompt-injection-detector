"""Optimize binary decision thresholds from prediction CSV files.

Example:
    python scripts/optimize_threshold.py \
        --input reports/indirect_evaluation/predictions.csv \
        --label-col label \
        --score-col final_score \
        --beta 2
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "reports" / "indirect_evaluation" / "predictions.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "threshold_optimization"

LABEL_CANDIDATES = ["label", "true_label", "y_true", "target", "is_injection"]
SCORE_CANDIDATES = ["risk_score", "final_score", "model_score", "score", "probability", "injection_probability"]
TEXT_CANDIDATES = ["text", "prompt", "user_prompt", "user_task", "external_content", "content"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize prompt-injection decision thresholds.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="CSV containing labels and scores.")
    parser.add_argument("--label-col", default=None, help="Ground-truth label column. Auto-detected if omitted.")
    parser.add_argument("--score-col", default=None, help="Risk/probability score column. Auto-detected if omitted.")
    parser.add_argument("--text-col", default=None, help="Text column copied to FP/FN reports. Auto-detected if omitted.")
    parser.add_argument("--beta", type=float, default=1.0, help="F-beta value for the recommendation. Use 2 to reduce false negatives.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for threshold reports.")
    return parser.parse_args()


def _find_column(fieldnames: list[str], explicit: str | None, candidates: list[str], kind: str) -> str:
    if explicit:
        if explicit not in fieldnames:
            raise ValueError(f"{kind} column '{explicit}' not found. Available columns: {fieldnames}")
        return explicit
    lowered = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    raise ValueError(f"Could not auto-detect {kind} column. Available columns: {fieldnames}")


def _optional_column(fieldnames: list[str], explicit: str | None, candidates: list[str]) -> str | None:
    if explicit:
        return explicit if explicit in fieldnames else None
    lowered = {name.lower(): name for name in fieldnames}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _load_rows(path: Path, label_col: str | None, score_col: str | None, text_col: str | None) -> tuple[list[dict[str, Any]], str, str, str | None]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        fields = list(reader.fieldnames)
        resolved_label = _find_column(fields, label_col, LABEL_CANDIDATES, "label")
        resolved_score = _find_column(fields, score_col, SCORE_CANDIDATES, "score")
        resolved_text = _optional_column(fields, text_col, TEXT_CANDIDATES)
        rows: list[dict[str, Any]] = []
        for raw in reader:
            try:
                label = int(float(str(raw.get(resolved_label, "")).strip()))
                score = float(str(raw.get(resolved_score, "")).strip())
            except ValueError:
                continue
            if label not in {0, 1}:
                continue
            rows.append({"label": label, "score": score, "raw": raw})
    if not rows:
        raise ValueError("No valid rows with binary labels and numeric scores were found.")
    return rows, resolved_label, resolved_score, resolved_text


def _confusion(y_true: list[int], y_pred: list[int]) -> dict[str, int | list[list[int]]]:
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


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _fbeta(precision: float, recall: float, beta: float) -> float:
    beta_sq = beta * beta
    denominator = beta_sq * precision + recall
    if denominator == 0:
        return 0.0
    return (1 + beta_sq) * precision * recall / denominator


def _metrics_at(rows: list[dict[str, Any]], threshold: float, beta: float) -> dict[str, Any]:
    y_true = [int(row["label"]) for row in rows]
    y_pred = [1 if float(row["score"]) >= threshold else 0 for row in rows]
    counts = _confusion(y_true, y_pred)
    tn = int(counts["tn"])
    fp = int(counts["fp"])
    fn = int(counts["fn"])
    tp = int(counts["tp"])
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _fbeta(precision, recall, 1.0)
    f2 = _fbeta(precision, recall, 2.0)
    fbeta_value = _fbeta(precision, recall, beta)
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": _safe_div(tp + tn, len(rows)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "fbeta": fbeta_value,
        **counts,
    }


def _rank_key(row: dict[str, Any], metric: str) -> tuple[float, float, float, int, float]:
    return (
        float(row[metric]),
        float(row["recall"]),
        float(row["precision"]),
        -int(row["fp"]),
        float(row["threshold"]),
    )


def _write_error_rows(
    path: Path,
    rows: list[dict[str, Any]],
    threshold: float,
    desired_true: int,
    desired_pred: int,
) -> None:
    selected: list[dict[str, Any]] = []
    fieldnames: list[str] = []
    for row in rows:
        pred = 1 if float(row["score"]) >= threshold else 0
        if int(row["label"]) == desired_true and pred == desired_pred:
            payload = dict(row["raw"])
            payload["score_used_for_threshold"] = f"{float(row['score']):.6f}"
            payload["predicted_label_at_threshold"] = str(pred)
            selected.append(payload)
            for key in payload:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["empty"])
        writer.writeheader()
        writer.writerows(selected)


def optimize_threshold(
    *,
    input_path: Path,
    output_dir: Path,
    label_col: str | None = None,
    score_col: str | None = None,
    text_col: str | None = None,
    beta: float = 1.0,
) -> dict[str, Any]:
    rows, resolved_label, resolved_score, resolved_text = _load_rows(input_path, label_col, score_col, text_col)
    candidates = [_metrics_at(rows, threshold / 100, beta) for threshold in range(1, 100)]
    best_f1 = max(candidates, key=lambda row: _rank_key(row, "f1"))
    best_f2 = max(candidates, key=lambda row: _rank_key(row, "f2"))
    best_fbeta = max(candidates, key=lambda row: _rank_key(row, "fbeta"))
    selected = best_f2 if beta == 2 else best_fbeta

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "input": str(input_path),
        "rows": len(rows),
        "label_column": resolved_label,
        "score_column": resolved_score,
        "text_column": resolved_text,
        "beta": beta,
        "best_f1_threshold": best_f1,
        "best_f2_threshold": best_f2,
        "recommended_threshold": selected,
        "selection_reason": (
            "Recommended threshold maximizes F2 to reduce false negatives."
            if beta == 2
            else f"Recommended threshold maximizes F-beta with beta={beta:g}."
        ),
        "note": "Thresholds are selected from observed metrics, not by assuming 0.5.",
    }

    (output_dir / "threshold_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    recommendation = output_dir / "threshold_recommendation.md"
    recommendation.write_text(
        "\n".join(
            [
                "# Threshold Optimization Recommendation",
                "",
                f"- Input: `{input_path}`",
                f"- Rows: `{len(rows)}`",
                f"- Label column: `{resolved_label}`",
                f"- Score column: `{resolved_score}`",
                f"- Beta used for recommendation: `{beta:g}`",
                "",
                "## Recommended threshold",
                "",
                f"- Threshold: `{selected['threshold']:.4f}`",
                f"- Precision: `{selected['precision']:.4f}`",
                f"- Recall: `{selected['recall']:.4f}`",
                f"- F1: `{selected['f1']:.4f}`",
                f"- F2: `{selected['f2']:.4f}`",
                f"- Confusion matrix [[TN, FP], [FN, TP]]: `{selected['confusion_matrix']}`",
                "",
                "## Best F1 threshold",
                "",
                f"- Threshold: `{best_f1['threshold']:.4f}` with F1 `{best_f1['f1']:.4f}`",
                "",
                "## Best F2 threshold",
                "",
                f"- Threshold: `{best_f2['threshold']:.4f}` with F2 `{best_f2['f2']:.4f}`",
                "",
                "This recommendation is based on a threshold sweep from 0.01 to 0.99.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    threshold = float(selected["threshold"])
    _write_error_rows(output_dir / "false_positives.csv", rows, threshold, desired_true=0, desired_pred=1)
    _write_error_rows(output_dir / "false_negatives.csv", rows, threshold, desired_true=1, desired_pred=0)
    return summary


def main() -> None:
    args = _parse_args()
    summary = optimize_threshold(
        input_path=args.input,
        output_dir=args.output_dir,
        label_col=args.label_col,
        score_col=args.score_col,
        text_col=args.text_col,
        beta=args.beta,
    )
    selected = summary["recommended_threshold"]
    print("Threshold optimization complete")
    print(f"Input: {summary['input']}")
    print(f"Recommended threshold: {selected['threshold']:.4f}")
    print(f"Precision: {selected['precision']:.4f}")
    print(f"Recall: {selected['recall']:.4f}")
    print(f"F1: {selected['f1']:.4f}")
    print(f"F2: {selected['f2']:.4f}")
    print(f"Confusion matrix: {selected['confusion_matrix']}")


if __name__ == "__main__":
    main()
