"""Model evaluation utilities."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.metrics import (
    auc,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

from src.preprocessing import clean_text
from src.thresholding import get_positive_class_scores, predict_with_threshold, confusion_rates
from src.file_utils import safe_write_text


REPORTS_DIR = PROJECT_ROOT / "reports"


def action_from_score(risk_score: float, warn_threshold: float = 0.80, block_threshold: float = 0.90) -> str:
    if risk_score >= block_threshold:
        return "block"
    if risk_score >= warn_threshold:
        return "warn"
    return "allow"


def plot_confusion_matrix(cm: list[list[int]], model_name: str, output_path: str | Path) -> None:
    """Plot a confusion matrix using matplotlib only."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    image = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(image, ax=ax)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Benign", "Malicious"],
        yticklabels=["Benign", "Malicious"],
        ylabel="Nhãn thật",
        xlabel="Nhãn dự đoán",
        title=f"Confusion Matrix - {model_name}",
    )

    threshold = max(max(row) for row in cm) / 2 if cm else 0
    for i in range(2):
        for j in range(2):
            value = cm[i][j]
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=12,
            )

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_roc_curve(y_test: list[int], scores: Any, model_name: str, output_path: str | Path) -> float:
    """Plot ROC curve and return ROC AUC."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_test, scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate / Recall",
        title=f"ROC Curve - {model_name}",
    )
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return float(roc_auc)


def plot_precision_recall_curve(y_test: list[int], scores: Any, model_name: str, output_path: str | Path) -> float:
    """Plot Precision-Recall curve and return average precision."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    precision, recall, _ = precision_recall_curve(y_test, scores)
    average_precision = average_precision_score(y_test, scores)

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    ax.plot(recall, precision, label=f"AP = {average_precision:.4f}")
    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        title=f"Precision-Recall Curve - {model_name}",
    )
    ax.legend(loc="lower left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return float(average_precision)


def evaluate_model(
    model: Any,
    vectorizer: Any,
    X_test: list[str],
    y_test: list[int],
    model_name: str,
    train_size: int | None = None,
    reports_dir: str | Path | None = None,
    threshold: float = 0.5,
    runtime_warn_threshold: float = 0.80,
    runtime_block_threshold: float = 0.90,
) -> dict[str, Any]:
    """Evaluate a trained model and return real metrics from test data."""
    cleaned_texts = [clean_text(text) for text in X_test]

    start_time = time.perf_counter()
    scores = get_positive_class_scores(model, vectorizer, cleaned_texts)
    y_pred = predict_with_threshold(scores, threshold)
    elapsed = time.perf_counter() - start_time

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["benign", "malicious"],
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["benign", "malicious"],
        zero_division=0,
    )

    safe_model_name = model_name.lower().replace(" ", "_").replace("-", "_")
    output_dir = Path(reports_dir) if reports_dir else REPORTS_DIR
    plot_path = output_dir / f"confusion_matrix_{safe_model_name}.png"
    roc_path = output_dir / f"roc_curve_{safe_model_name}.png"
    pr_path = output_dir / f"precision_recall_curve_{safe_model_name}.png"
    plot_confusion_matrix(cm, model_name, plot_path)
    roc_auc = plot_roc_curve(y_test, scores, model_name, roc_path)
    average_precision = plot_precision_recall_curve(y_test, scores, model_name, pr_path)
    rates = confusion_rates(y_test, y_pred)

    return {
        "model_name": model_name,
        "evaluation_threshold": float(threshold),
        "runtime_warn_threshold": float(runtime_warn_threshold),
        "runtime_block_threshold": float(runtime_block_threshold),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "false_positive_rate": float(rates["false_positive_rate"]),
        "false_negative_rate": float(rates["false_negative_rate"]),
        "roc_auc": float(roc_auc),
        "average_precision": float(average_precision),
        "confusion_matrix": cm,
        "classification_report": report_dict,
        "classification_report_text": report_text,
        "train_size": int(train_size) if train_size is not None else None,
        "test_size": int(len(y_test)),
        "total_prediction_time_seconds": float(elapsed),
        "avg_prediction_time_seconds": float(elapsed / max(len(y_test), 1)),
        "confusion_matrix_path": str(plot_path),
        "roc_curve_path": str(roc_path),
        "precision_recall_curve_path": str(pr_path),
    }


def save_metrics_json(metrics: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_write_text(path, json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def _format_distribution(distribution: dict[str, int]) -> str:
    return ", ".join(f"{label}: {count}" for label, count in distribution.items())


def build_metrics_markdown(metrics: dict[str, Any]) -> str:
    """Build a Vietnamese Markdown metrics report."""
    lines = [
        "# Báo cáo đánh giá mô hình",
        "",
        "> Tất cả chỉ số trong file này được sinh từ quá trình train/test thật.",
        "",
    ]

    dataset = metrics.get("dataset", {})
    split = metrics.get("split", {})
    if dataset:
        lines.extend(
            [
                "## Dataset",
                "",
                f"- Số mẫu: {dataset.get('num_rows')}",
                f"- Cột prompt: `{dataset.get('text_column')}`",
                f"- Cột nhãn: `{dataset.get('label_column')}`",
                f"- Phân phối nhãn: {_format_distribution(dataset.get('label_distribution', {}))}",
                "",
            ]
        )

    if split:
        lines.extend(
            [
                "## Chia train/test",
                "",
                f"- Train size: {split.get('train_size')}",
                f"- Test size: {split.get('test_size')}",
                f"- Test size ratio: {split.get('test_size_ratio')}",
                f"- Random state: {split.get('random_state')}",
                f"- Stratify theo nhãn: {split.get('stratify')}",
                f"- Phân phối train: {_format_distribution(split.get('train_label_distribution', {}))}",
                f"- Phân phối test: {_format_distribution(split.get('test_label_distribution', {}))}",
                "",
            ]
        )

    augmentation = metrics.get("augmentation", {})
    if augmentation:
        lines.extend(
            [
                "## Multilingual augmentation",
                "",
                f"- Bật augmentation: {augmentation.get('enabled')}",
                f"- File augmented dataset: `{augmentation.get('saved_path')}`",
                f"- Số dòng augmented đã lưu: {augmentation.get('total_augmented_rows_saved')}",
                f"- Số dòng augmented dùng trong train split: {augmentation.get('train_augmented_rows_used')}",
                f"- Ghi chú: {augmentation.get('note')}",
                "",
            ]
        )

    validation = metrics.get("validation", {})
    if validation:
        lines.extend(
            [
                "## Validation và threshold",
                "",
                f"- Validation size: {validation.get('validation_size')}",
                f"- Phân phối validation: {_format_distribution(validation.get('validation_label_distribution', {}))}",
                "- Threshold phân loại được chọn trên validation set, không dùng mặc định 0.5.",
                "- Runtime API dùng ngưỡng cảnh báo/chặn cao hơn để giảm false positive khi rule-based không khớp.",
                "",
            ]
        )

    lines.extend(
        [
            "## Ý nghĩa chỉ số",
            "",
            "- Accuracy: tỷ lệ dự đoán đúng trên toàn bộ tập test.",
            "- Precision: trong các prompt bị dự đoán là malicious, tỷ lệ thật sự malicious.",
            "- Recall: trong các prompt malicious thật, tỷ lệ model phát hiện được.",
            "- F1-score: trung bình điều hòa giữa precision và recall.",
            "- Confusion matrix: bảng so sánh nhãn thật và nhãn dự đoán.",
            "",
            "Trong bài toán prompt injection, recall của lớp malicious rất quan trọng vì bỏ sót prompt tấn công có thể nguy hiểm hơn cảnh báo nhầm.",
            "",
        ]
    )

    for model_key, model_metrics in metrics.get("models", {}).items():
        lines.extend(
            [
                f"## {model_metrics.get('model_name', model_key)}",
                "",
                f"- Accuracy: {model_metrics.get('accuracy', 0):.4f}",
                f"- Precision (positive label = malicious): {model_metrics.get('precision', 0):.4f}",
                f"- Recall (positive label = malicious): {model_metrics.get('recall', 0):.4f}",
                f"- F1-score: {model_metrics.get('f1', 0):.4f}",
                f"- Evaluation threshold: {model_metrics.get('evaluation_threshold', 0):.4f}",
                f"- Runtime warn threshold: {model_metrics.get('runtime_warn_threshold', 0):.4f}",
                f"- Runtime block threshold: {model_metrics.get('runtime_block_threshold', 0):.4f}",
                f"- False Positive Rate: {model_metrics.get('false_positive_rate', 0):.4f}",
                f"- False Negative Rate: {model_metrics.get('false_negative_rate', 0):.4f}",
                f"- ROC AUC: {model_metrics.get('roc_auc', 0):.4f}",
                f"- Average Precision: {model_metrics.get('average_precision', 0):.4f}",
                f"- Train size: {model_metrics.get('train_size')}",
                f"- Test size: {model_metrics.get('test_size')}",
                f"- Thời gian huấn luyện: {model_metrics.get('training_time_seconds', 0):.4f} giây",
                f"- Thời gian dự đoán trung bình: {model_metrics.get('avg_prediction_time_seconds', 0):.8f} giây/prompt",
                f"- File confusion matrix: `{model_metrics.get('confusion_matrix_path')}`",
                f"- File ROC curve: `{model_metrics.get('roc_curve_path')}`",
                f"- File Precision-Recall curve: `{model_metrics.get('precision_recall_curve_path')}`",
                "",
                "### Confusion Matrix",
                "",
                "| | Dự đoán benign | Dự đoán malicious |",
                "|---|---:|---:|",
                f"| Thật benign | {model_metrics.get('confusion_matrix', [[0, 0], [0, 0]])[0][0]} | {model_metrics.get('confusion_matrix', [[0, 0], [0, 0]])[0][1]} |",
                f"| Thật malicious | {model_metrics.get('confusion_matrix', [[0, 0], [0, 0]])[1][0]} | {model_metrics.get('confusion_matrix', [[0, 0], [0, 0]])[1][1]} |",
                "",
                "### Classification Report",
                "",
                "```text",
                str(model_metrics.get("classification_report_text", "")).strip(),
                "```",
                "",
            ]
        )

        feature_analysis = model_metrics.get("feature_analysis", {})
        if feature_analysis:
            lines.extend(["### Top malicious indicators", "", "| Feature | Weight |", "|---|---:|"])
            for item in feature_analysis.get("top_malicious_features", []):
                lines.append(f"| `{item.get('feature')}` | {item.get('weight', 0):.6f} |")
            lines.extend(["", "### Top benign indicators", "", "| Feature | Weight |", "|---|---:|"])
            for item in feature_analysis.get("top_benign_features", []):
                lines.append(f"| `{item.get('feature')}` | {item.get('weight', 0):.6f} |")
            lines.append("")

    best_model = metrics.get("best_model")
    if best_model:
        lines.extend(
            [
                "## Kết luận so sánh",
                "",
                f"- Model được khuyến nghị: `{best_model.get('name')}`",
                f"- Lý do: {best_model.get('reason')}",
                "",
            ]
        )

    return "\n".join(lines)


def save_metrics_markdown(metrics: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_write_text(path, build_metrics_markdown(metrics), encoding="utf-8")
