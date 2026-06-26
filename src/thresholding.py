"""Threshold selection and score utilities for ML detectors."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    fbeta_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.preprocessing import clean_text


MIN_RUNTIME_WARN_THRESHOLD = 0.30
DEFAULT_RUNTIME_BLOCK_THRESHOLD = 0.80
THRESHOLD_GRID = [round(value / 100, 2) for value in range(1, 100)]


def get_positive_class_scores(model: Any, vectorizer: Any, texts: list[str]) -> np.ndarray:
    """Return probability-like scores for class 1 = malicious."""
    cleaned_texts = [clean_text(text) for text in texts]
    vectorized = vectorizer.transform(cleaned_texts)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vectorized)
        classes = list(getattr(model, "classes_", [0, 1]))
        positive_index = classes.index(1) if 1 in classes else 1
        return probabilities[:, positive_index].astype(float)

    if hasattr(model, "decision_function"):
        raw_scores = model.decision_function(vectorized)
        return np.array([1.0 / (1.0 + math.exp(-float(score))) for score in raw_scores], dtype=float)

    return model.predict(vectorized).astype(float)


def predict_with_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Convert positive-class scores to binary labels using a custom threshold."""
    return (scores >= threshold).astype(int)


def confusion_rates(y_true: list[int], y_pred: np.ndarray) -> dict[str, float | int | list[list[int]]]:
    """Calculate confusion-matrix-derived rates."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in cm.ravel()]
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    return {
        "confusion_matrix": cm.tolist(),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
    }


def metrics_at_threshold(
    y_true: list[int],
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float | int | list[list[int]]]:
    """Calculate classification metrics at a given threshold."""
    y_pred = predict_with_threshold(scores, threshold)
    rates = confusion_rates(y_true, y_pred)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, pos_label=1, zero_division=0)),
        **rates,
    }


def _candidate_thresholds(scores: np.ndarray | None = None) -> list[float]:
    """Use a stable 0.01..0.99 threshold grid for comparable reports."""
    return THRESHOLD_GRID.copy()


def _choose_evaluation_threshold(
    candidates: list[dict[str, Any]],
    optimization_metric: str,
    min_recall: float | None,
    min_precision: float | None,
) -> tuple[dict[str, Any], str, str]:
    metric = str(optimization_metric or "f1").strip().lower()
    if metric not in {"f1", "f2", "constraint"}:
        metric = "f1"

    if metric == "constraint" and min_recall is not None and min_precision is not None:
        viable = [
            row
            for row in candidates
            if float(row["recall"]) >= min_recall and float(row["precision"]) >= min_precision
        ]
        if viable:
            selected = sorted(
                viable,
                key=lambda row: (
                    float(row["threshold"]),
                    int(row["false_negative"]),
                    int(row["false_positive"]),
                ),
            )[0]
            return (
                selected,
                "constraint",
                f"Chon threshold nho nhat dat recall >= {min_recall:.2f} va precision >= {min_precision:.2f}.",
            )

    rank_metric = "f2" if metric == "f2" else "f1"
    selected = sorted(
        candidates,
        key=lambda row: (
            float(row[rank_metric]),
            float(row["recall"]),
            float(row["precision"]),
            -int(row["false_positive"]),
            -float(row["threshold"]),
        ),
        reverse=True,
    )[0]
    return selected, rank_metric, f"Chon threshold co {rank_metric.upper()} cao nhat tren validation/test set."


def _choose_precision_threshold(
    candidates: list[dict[str, Any]],
    min_precision: float = 0.95,
) -> float | None:
    viable = [
        row
        for row in candidates
        if float(row["precision"]) >= min_precision and int(row["true_positive"]) > 0
    ]
    if not viable:
        return None
    return float(sorted(viable, key=lambda row: (float(row["threshold"]), -float(row["recall"])))[0]["threshold"])


def derive_runtime_thresholds(
    evaluation_threshold: float,
    candidates: list[dict[str, Any]],
    precision_for_block: float = 0.95,
) -> tuple[float, float, str]:
    """Derive separated warn/block thresholds from validation behavior."""
    warn_threshold = max(MIN_RUNTIME_WARN_THRESHOLD, float(evaluation_threshold))
    precision_threshold = _choose_precision_threshold(candidates, min_precision=precision_for_block)
    if precision_threshold is not None:
        block_threshold = min(0.95, max(warn_threshold + 0.15, precision_threshold))
        reason = f"Block threshold uu tien threshold dat precision >= {precision_for_block:.2f}."
    else:
        block_threshold = min(0.95, warn_threshold + 0.20)
        reason = f"Khong tim thay threshold dat precision >= {precision_for_block:.2f}; dung warn+0.20 va cap 0.95."

    if block_threshold <= warn_threshold:
        warn_threshold = max(MIN_RUNTIME_WARN_THRESHOLD, min(warn_threshold, block_threshold - 0.01))
        reason += " Da giam warn threshold de dam bao warn_threshold < block_threshold."

    return round(float(warn_threshold), 4), round(float(block_threshold), 4), reason


def choose_threshold(
    y_validation: list[int],
    validation_scores: np.ndarray,
    min_recall: float | None = None,
    min_precision: float | None = None,
    optimization_metric: str = "f1",
) -> dict[str, Any]:
    """Choose evaluation, warn and block thresholds from validation scores."""
    candidates = [
        metrics_at_threshold(y_validation, validation_scores, threshold)
        for threshold in _candidate_thresholds(validation_scores)
    ]

    selected, best_metric, selection_reason = _choose_evaluation_threshold(
        candidates,
        optimization_metric=optimization_metric,
        min_recall=min_recall,
        min_precision=min_precision,
    )
    selected_threshold = float(selected["threshold"])
    runtime_warn_threshold, runtime_block_threshold, runtime_reason = derive_runtime_thresholds(
        selected_threshold,
        candidates,
    )

    return {
        "selected_threshold": selected_threshold,
        "evaluation_threshold": selected_threshold,
        "runtime_warn_threshold": float(runtime_warn_threshold),
        "runtime_block_threshold": float(runtime_block_threshold),
        "warn_threshold": float(runtime_warn_threshold),
        "block_threshold": float(runtime_block_threshold),
        "best_metric": best_metric,
        "min_recall": None if min_recall is None else float(min_recall),
        "min_precision": None if min_precision is None else float(min_precision),
        "selection_reason": selection_reason,
        "runtime_reason": runtime_reason,
        "runtime_policy": "risk < warn = allow, warn <= risk < block = warn, risk >= block = block.",
        "selected_metrics": selected,
        "candidate_metrics": candidates,
    }
