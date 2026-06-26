"""Compare trained model metrics and write model comparison reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.file_utils import safe_write_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"
DEFAULT_COMPARISON_JSON = PROJECT_ROOT / "reports" / "model_comparison.json"
DEFAULT_COMPARISON_MD = PROJECT_ROOT / "reports" / "model_comparison.md"


def _model_bonus(model_key: str) -> float:
    if model_key == "logistic_regression":
        return 0.03
    if model_key == "linear_svm":
        return 0.01
    if model_key == "random_forest":
        return 0.005
    return 0.0


def _reason_for_best(model_key: str) -> str:
    if model_key == "logistic_regression":
        return (
            "Logistic Regression có recall/F1 cạnh tranh, có predict_proba ổn định "
            "để tạo risk_score và dễ giải thích khi demo API."
        )
    if model_key == "linear_svm":
        return (
            "Linear SVM có recall/F1 tốt theo thực nghiệm hiện tại và mạnh với TF-IDF sparse vector. "
            "Model đã được calibration để có risk_score, nhưng khó giải thích hơn Logistic Regression."
        )
    if model_key == "random_forest":
        return (
            "Random Forest có recall/F1 tốt nhất theo metrics hiện tại. Tuy nhiên TF-IDF là vector sparse nhiều chiều, "
            "nên Random Forest cần được cân nhắc về tốc độ và khả năng tổng quát trước khi chọn làm model chính."
        )
    return "Model có recall/F1 tốt nhất theo metrics hiện tại."


def compare_models(metrics_path: str | Path = DEFAULT_METRICS_PATH) -> dict[str, Any]:
    """Compare models by recall, F1, false negative rate, precision and deployment fit."""
    path = Path(metrics_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy metrics.json: {path}")

    metrics = json.loads(path.read_text(encoding="utf-8"))
    models = metrics.get("models", {})
    if not models:
        raise ValueError("metrics.json không có dữ liệu model để so sánh.")

    rows = []
    for model_key, model_metrics in models.items():
        score_tuple = (
            float(model_metrics.get("recall", 0.0)),
            float(model_metrics.get("f1", 0.0)),
            -float(model_metrics.get("false_negative_rate", 1.0)),
            float(model_metrics.get("precision", 0.0)),
            -float(model_metrics.get("avg_prediction_time_seconds", 999.0)),
            _model_bonus(model_key),
        )
        rows.append(
            {
                "model_key": model_key,
                "model_name": model_metrics.get("model_name", model_key),
                "accuracy": model_metrics.get("accuracy"),
                "precision": model_metrics.get("precision"),
                "recall": model_metrics.get("recall"),
                "f1": model_metrics.get("f1"),
                "false_positive_rate": model_metrics.get("false_positive_rate"),
                "false_negative_rate": model_metrics.get("false_negative_rate"),
                "roc_auc": model_metrics.get("roc_auc"),
                "average_precision": model_metrics.get("average_precision"),
                "training_time_seconds": model_metrics.get("training_time_seconds"),
                "avg_prediction_time_seconds": model_metrics.get("avg_prediction_time_seconds"),
                "selection_score": score_tuple,
            }
        )

    rows = sorted(rows, key=lambda row: row["selection_score"], reverse=True)
    best = rows[0]
    return {
        "ranking": rows,
        "best_model": {
            "key": best["model_key"],
            "name": best["model_name"],
            "reason": _reason_for_best(best["model_key"]),
        },
        "selection_policy": [
            "Ưu tiên recall của lớp malicious.",
            "Sau đó xét F1-score.",
            "Sau đó xét False Negative Rate thấp.",
            "Sau đó xét precision hợp lý.",
            "Sau đó xét runtime/prediction speed.",
            "Sau đó xét khả năng giải thích, khả năng tạo risk_score và triển khai API ổn định.",
        ],
    }


def _format_float(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.4f}"


def _format_distribution(distribution: dict[str, Any]) -> str:
    if not distribution:
        return "N/A"
    return ", ".join(f"{key}: {value}" for key, value in distribution.items())


def build_detailed_comparison_markdown(metrics: dict[str, Any], comparison: dict[str, Any]) -> str:
    dataset = metrics.get("dataset", {})
    split = metrics.get("split", {})
    rows = comparison.get("ranking", [])
    best = comparison.get("best_model", {})

    lines = [
        "# So sánh mô hình phát hiện Prompt Injection",
        "",
        "## 1. Mục tiêu so sánh",
        "",
        "So sánh 3 mô hình ML dùng cùng dataset, cùng train/test split, cùng preprocessing và cùng TF-IDF representation:",
        "",
        "- TF-IDF + Logistic Regression",
        "- TF-IDF + Linear SVM",
        "- TF-IDF + Random Forest",
        "",
        "## 2. Dataset",
        "",
        f"- Tổng số dòng: {dataset.get('num_rows')}",
        f"- Train size: {split.get('train_size')}",
        f"- Test size: {split.get('test_size')}",
        f"- Label distribution: `{_format_distribution(dataset.get('label_distribution', {}))}`",
        f"- Train label distribution: `{_format_distribution(split.get('train_label_distribution', {}))}`",
        f"- Test label distribution: `{_format_distribution(split.get('test_label_distribution', {}))}`",
        f"- Language distribution: `{_format_distribution(dataset.get('language_distribution', {}))}`",
        f"- Source distribution: `{_format_distribution(dataset.get('source_distribution', {}))}`",
        "",
        "## 3. Bảng metrics",
        "",
        "| Model | Accuracy | Precision | Recall | F1 | FPR | FNR | ROC AUC | AP | Train time | Avg pred time |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model_name']} | {_format_float(row['accuracy'])} | {_format_float(row['precision'])} | "
            f"{_format_float(row['recall'])} | {_format_float(row['f1'])} | "
            f"{_format_float(row['false_positive_rate'])} | {_format_float(row['false_negative_rate'])} | "
            f"{_format_float(row['roc_auc'])} | {_format_float(row['average_precision'])} | "
            f"{_format_float(row['training_time_seconds'])} | {_format_float(row['avg_prediction_time_seconds'])} |"
        )

    lines.extend(
        [
            "",
            "## 4. Phân tích từng mô hình",
            "",
            "### Logistic Regression",
            "",
            "- Ưu điểm: nhanh, dễ giải thích, có `predict_proba` tự nhiên để tạo `risk_score`.",
            "- Nhược điểm: tuyến tính nên có thể bỏ lỡ quan hệ phi tuyến phức tạp.",
            "- Phù hợp làm model chính cho API vì cân bằng giữa hiệu năng, khả năng giải thích và triển khai ổn định.",
            "",
            "### Linear SVM",
            "",
            "- Ưu điểm: rất mạnh với TF-IDF sparse vector và text classification ngắn.",
            "- Nhược điểm: bản gốc không có xác suất trực tiếp; project dùng `CalibratedClassifierCV` để có risk_score.",
            "- Phù hợp làm model so sánh mạnh cho dữ liệu văn bản.",
            "",
            "### Random Forest",
            "",
            "- Random Forest là ensemble gồm nhiều Decision Tree. Mỗi cây đưa ra dự đoán, kết quả cuối lấy theo voting.",
            "- Ưu điểm: trực giác dễ hiểu, có feature importance, giảm overfitting so với một cây đơn.",
            "- Nhược điểm trong text classification: TF-IDF thường rất nhiều chiều và sparse, Random Forest có thể không tối ưu bằng Logistic Regression hoặc Linear SVM.",
            "- Dùng Random Forest để so sánh bổ sung; không mặc định là model chính nếu metrics không vượt trội rõ.",
            "",
            "## 5. Chọn model tối ưu",
            "",
            "Không chọn chỉ dựa trên accuracy. Thứ tự ưu tiên là recall malicious, F1, FNR thấp, precision hợp lý, tốc độ dự đoán, khả năng giải thích, risk_score và triển khai API.",
            "",
            f"- Model khuyến nghị theo metrics thật: `{best.get('name')}`",
            f"- Lý do: {best.get('reason')}",
            "",
        ]
    )
    return "\n".join(lines)


def save_model_comparison_reports(
    metrics: dict[str, Any],
    comparison: dict[str, Any],
    json_output: str | Path = DEFAULT_COMPARISON_JSON,
    md_output: str | Path = DEFAULT_COMPARISON_MD,
) -> None:
    payload = {
        "comparison": comparison,
        "dataset": metrics.get("dataset", {}),
        "split": metrics.get("split", {}),
    }
    safe_write_text(json_output, json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    safe_write_text(md_output, build_detailed_comparison_markdown(metrics, comparison), encoding="utf-8")


def build_comparison_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# So sánh mô hình",
        "",
        "Tiêu chí chọn model không chỉ dựa vào accuracy. Với prompt injection, bỏ sót malicious prompt nguy hiểm hơn cảnh báo nhầm.",
        "",
        "| Model | Accuracy | Precision | Recall | F1-score |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in comparison["ranking"]:
        lines.append(
            f"| {row['model_name']} | {_format_float(row['accuracy'])} | {_format_float(row['precision'])} | "
            f"{_format_float(row['recall'])} | {_format_float(row['f1'])} |"
        )

    best = comparison["best_model"]
    lines.extend(["", f"Model khuyến nghị: `{best['name']}`.", "", f"Lý do: {best['reason']}", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare prompt injection detection models.")
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS_PATH), help="Path to reports/metrics.json")
    parser.add_argument("--output", default="", help="Optional Markdown output path")
    args = parser.parse_args()

    comparison = compare_models(args.metrics)
    markdown = build_comparison_markdown(comparison)
    print(markdown)
    if args.output:
        safe_write_text(args.output, markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
