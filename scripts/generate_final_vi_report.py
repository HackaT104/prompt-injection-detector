"""Generate the final Vietnamese report package for lecturer-facing review."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "final_vi_report"
DIRECT = ROOT / "reports" / "direct_external_evaluation"
BIPIA = ROOT / "reports" / "bipia_evaluation"
AUDIT = ROOT / "reports" / "project_audit"
DATE = "2026-06-28"

MODELS = ["logistic_regression", "random_forest", "roberta", "xlm_roberta"]
MODEL_LABEL = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "roberta": "RoBERTa",
    "xlm_roberta": "XLM-RoBERTa",
}


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fmt(value: Any) -> str:
    if value is None or value == "":
        return "N/A"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_int(value: Any) -> str:
    if value is None or value == "":
        return "N/A"
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return str(value)


def table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |")
    return "\n".join(lines)


def write(name: str, content: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(content.strip() + "\n", encoding="utf-8")


def calibration_rows(dataset: str = "all") -> list[list[Any]]:
    rows = []
    for model in MODELS:
        metrics = read_json(DIRECT / "calibration" / dataset / model / "calibration_metrics.json", {})
        if not metrics:
            continue
        raw = metrics.get("raw_test_metrics", {})
        cal = metrics.get("calibrated_test_metrics", {})
        rows.append(
            [
                MODEL_LABEL[model],
                metrics.get("selected_calibration_method"),
                fmt(metrics.get("raw_threshold")),
                fmt(metrics.get("calibrated_threshold")),
                fmt(raw.get("f1")),
                fmt(cal.get("f1")),
                fmt(raw.get("f2")),
                fmt(cal.get("f2")),
                fmt(raw.get("brier")),
                fmt(cal.get("brier")),
                fmt(raw.get("ece")),
                fmt(cal.get("ece")),
                raw.get("confusion_matrix"),
                cal.get("confusion_matrix"),
            ]
        )
    return rows


def direct_strict_rows(dataset: str = "all") -> list[list[Any]]:
    rows = []
    for model in MODELS:
        metrics = read_json(DIRECT / "strict" / dataset / model / "test_metrics.json", {})
        if not metrics:
            continue
        rows.append(
            [
                MODEL_LABEL[model],
                fmt_int(metrics.get("rows")),
                fmt(metrics.get("threshold")),
                fmt(metrics.get("accuracy")),
                fmt(metrics.get("precision")),
                fmt(metrics.get("recall")),
                fmt(metrics.get("f1")),
                fmt(metrics.get("f2")),
                fmt(metrics.get("roc_auc")),
                fmt(metrics.get("pr_auc")),
                fmt_int(metrics.get("false_positive_count")),
                fmt_int(metrics.get("false_negative_count")),
                metrics.get("confusion_matrix"),
            ]
        )
    return rows


def bipia_rows() -> list[list[Any]]:
    rows = []
    for row in read_csv(BIPIA / "ablation_comparison.csv"):
        rows.append(
            [
                row.get("Model"),
                row.get("Context-aware"),
                fmt_int(row.get("Rows")),
                fmt(row.get("Threshold")),
                fmt(row.get("Accuracy")),
                fmt(row.get("Precision")),
                fmt(row.get("Recall")),
                fmt(row.get("F1")),
                fmt(row.get("F2")),
                fmt(row.get("ROC-AUC")),
                fmt(row.get("PR-AUC")),
                fmt_int(row.get("FP")),
                fmt_int(row.get("FN")),
            ]
        )
    return rows


def thresholds_rows() -> list[list[Any]]:
    payload = read_json(ROOT / "models" / "calibrated_thresholds.json", {})
    rows = []
    for model in MODELS:
        entry = payload.get(model, {})
        if not entry:
            continue
        rows.append(
            [
                MODEL_LABEL[model],
                entry.get("calibration_method"),
                fmt(entry.get("threshold_eval")),
                fmt(entry.get("threshold_warn")),
                fmt(entry.get("threshold_block")),
                entry.get("score_type"),
                entry.get("calibrator_path"),
            ]
        )
    return rows


def tong_quan_he_thong() -> str:
    return f"""
# Tổng quan hệ thống Prompt Injection Detection

Ngày cập nhật: {DATE}

Hệ thống hiện tại là một prototype nghiên cứu phát hiện prompt injection. Thành phần chính:

- Rule-based detector: phát hiện các mẫu lệnh nguy hiểm rõ ràng.
- Logistic Regression: baseline ML truyền thống với TF-IDF.
- Random Forest: baseline ML truyền thống để so sánh với Transformer.
- RoBERTa: Transformer hiện có kết quả tổng quát hóa tốt nhất trên direct external benchmark.
- XLM-RoBERTa: hướng multilingual, hiện cần cải thiện thêm.
- Context-aware detector: phân tích mâu thuẫn giữa instruction/context, đặc biệt hữu ích trên BIPIA subset.
- Explainable output: trả về score, action, threshold, rule/model signals.
- Threshold optimization: chọn threshold theo validation, ưu tiên F2 để giảm false negative.
- Calibration: dùng Isotonic Regression/Platt Scaling để làm probability dễ diễn giải hơn.

Luồng runtime chính:

Input → model/rule/context scores → calibrated probability nếu có → threshold policy → allow/warn/block → explanation.

Lưu ý: direct model calibration không dùng rule-based/context-aware để tránh trộn tín hiệu khi đánh giá model.
"""


def tien_do_2_tuan() -> str:
    return f"""
# Tiến độ 2 tuần gần đây

Ngày cập nhật: {DATE}

Đã hoàn thành:

- Thêm external direct benchmark: deepset, neuralchemy, cyberec, direct_all.
- Chạy strict validation/test protocol cho direct external benchmark.
- Chạy calibration analysis cho Logistic Regression, Random Forest, RoBERTa, XLM-RoBERTa.
- Lưu calibrated threshold và probability calibrator artifact.
- Chạy BIPIA subset và BIPIA ablation study.
- Phân tích nguyên nhân threshold thấp của RoBERTa/XLM-R.
- Tạo project audit bằng tiếng Việt.
- Tạo final Vietnamese report package.

Điểm quan trọng: hiện project không chỉ dừng ở việc train model, mà đã có workflow đánh giá nghiên cứu gồm external benchmark, ablation, strict split và calibration.
"""


def ket_qua_metrics_hien_tai() -> str:
    return f"""
# Kết quả metrics hiện tại

## Direct external strict test - dataset all

{table(["Model", "N", "Threshold", "Accuracy", "Precision", "Recall", "F1", "F2", "ROC-AUC", "PR-AUC", "FP", "FN", "Confusion Matrix"], direct_strict_rows("all"))}

## Calibration - dataset all

{table(["Model", "Method", "Raw threshold", "Cal threshold", "Raw F1", "Cal F1", "Raw F2", "Cal F2", "Brier raw", "Brier cal", "ECE raw", "ECE cal", "CM raw", "CM cal"], calibration_rows("all"))}

## Calibrated runtime thresholds

{table(["Model", "Method", "Eval", "Warn", "Block", "Score type", "Calibrator path"], thresholds_rows())}
"""


def giai_thich_threshold_calibration() -> str:
    return """
# Giải thích threshold và calibration

## Vì sao raw threshold thấp?

RoBERTa raw threshold = 0.01 trên direct external strict protocol không nhất thiết là bug. Nguyên nhân chính:

- Threshold được tối ưu theo F2, tức ưu tiên Recall để giảm false negative.
- Một phần injection samples có raw softmax score rất thấp, tạo “đuôi” injection gần 0.
- Softmax probability chưa chắc là xác suất đã calibrated.

XLM-RoBERTa có raw threshold 0.09 trên direct_all và 0.01 trên deepset vì phân phối score giữa benign/injection bị overlap mạnh hơn. Model có recall cao nhưng false positive nhiều.

## Vì sao calibration cần thiết?

Calibration không nhất thiết làm ROC-AUC tăng mạnh, nhưng giúp score dễ diễn giải hơn. Ví dụ RoBERTa sau isotonic calibration có threshold tăng từ khoảng 0.01 lên khoảng 0.26 trên direct_all, nghĩa là decision boundary trở nên hợp lý hơn về mặt xác suất.

## Vì sao không chọn threshold trên test?

Nếu tune threshold trực tiếp trên test rồi báo cáo final performance, kết quả sẽ bị optimistic. Protocol đúng là:

1. Split validation/test.
2. Fit calibrator trên validation.
3. Chọn threshold trên validation.
4. Fix calibrator + threshold.
5. Chỉ sau đó mới đánh giá test.
"""


def ket_qua_direct_external() -> str:
    return f"""
# Kết quả Direct External Benchmark

## Strict validation/test - dataset all

{table(["Model", "N", "Threshold", "Accuracy", "Precision", "Recall", "F1", "F2", "ROC-AUC", "PR-AUC", "FP", "FN", "Confusion Matrix"], direct_strict_rows("all"))}

Nhận xét:

- RoBERTa là model tốt nhất hiện tại trên direct external benchmark: F1/ROC-AUC cao và false positive thấp hơn nhiều so với ML truyền thống.
- Logistic Regression và Random Forest có recall rất cao nhưng false positive rất nhiều.
- XLM-RoBERTa chưa ổn định: recall cao nhưng precision thấp hơn, cần fine-tune lại bằng multilingual/hard negatives.
- Calibration giúp score dễ giải thích hơn, đặc biệt với RoBERTa.
"""


def ket_qua_bipia() -> str:
    return f"""
# Kết quả BIPIA Ablation

{table(["Model", "Context-aware", "N", "Threshold", "Accuracy", "Precision", "Recall", "F1", "F2", "ROC-AUC", "PR-AUC", "FP", "FN"], bipia_rows())}

Nhận xét:

- RoBERTa-only và XLM-R-only chưa đủ tốt trên BIPIA subset, đặc biệt RoBERTa-only bỏ sót nhiều injection.
- Context-aware fusion cải thiện rất mạnh trên BIPIA subset.
- Không nên kết luận context-aware đã có semantic reasoning đầy đủ, vì hiện logic vẫn còn heuristic/rule-heavy.
- Cần mở rộng full BIPIA và nâng context-aware bằng semantic/NLI/embedding.
"""


def diem_manh_yeu() -> str:
    return """
# Điểm mạnh và điểm yếu

## Điểm mạnh

- Có workflow nghiên cứu, không chỉ train model.
- Có external benchmark.
- Có ablation study.
- Có strict validation/test protocol.
- Có calibration với Brier Score và ECE.
- Có explainable output.
- Có context-aware layer.

## Điểm yếu

- XLM-R chưa đạt kỳ vọng multilingual.
- Context-aware còn heuristic-heavy.
- Calibration runtime mới chuẩn hóa cho direct_all, cần tiếp tục kiểm thử trong nhiều endpoint/use case.
- BIPIA hiện là subset, chưa phải full benchmark.
- Chưa có full RAG/PDF/HTML/email real-world benchmark.
- Chưa có semantic entailment/NLI.
"""


def de_xuat_tiep_theo() -> str:
    return """
# Đề xuất tiếp theo

## Priority 1

- Kiểm thử runtime calibrated thresholds trên API/frontend.
- Lưu logits cho Transformer trong các lần rescore direct benchmark để chạy Temperature Scaling.
- Fine-tune lại XLM-R bằng multilingual/hard negatives.

## Priority 2

- Nâng context-aware lên semantic bằng embedding hoặc NLI.
- Mở rộng BIPIA full nếu có đủ context files.
- Làm common-failure analysis giữa Random Forest, RoBERTa, XLM-RoBERTa.

## Priority 3

- Test RAG/PDF/HTML/email injection.
- Làm dashboard hiển thị metrics, score, threshold, explanation.
- Chuẩn bị slide/báo cáo bảo vệ.
"""


def bao_cao_tom_tat() -> str:
    return f"""
# Báo cáo tóm tắt cho giảng viên

## Mục tiêu đề tài

Xây dựng prototype nghiên cứu phát hiện prompt injection, kết hợp ML/Transformer, context-aware signals, threshold optimization, calibration và external benchmarks.

## Hệ thống hiện có

Hệ thống gồm rule-based detector, Logistic Regression, Random Forest, RoBERTa, XLM-RoBERTa, context-aware detector, explainable output và FastAPI/frontend demo.

## Việc đã hoàn thành trong 2 tuần

- Tích hợp direct external benchmark.
- Chạy strict validation/test evaluation.
- Chạy calibration đúng validation/test.
- Lưu probability calibrator và calibrated thresholds.
- Tích hợp runtime ưu tiên calibrated probability nếu có.
- Chạy BIPIA subset và ablation study.
- Tạo project audit và báo cáo tiếng Việt.

## Kết quả nổi bật

RoBERTa hiện là model tốt nhất trên direct external benchmark. Context-aware fusion cải thiện mạnh trên BIPIA subset. Calibration giúp giải thích threshold tốt hơn, đặc biệt RoBERTa raw threshold 0.01 sau calibration tăng lên khoảng 0.26.

## Vấn đề phát hiện được

XLM-RoBERTa chưa ổn định và cần fine-tune lại. Context-aware hiện còn heuristic. BIPIA mới là subset. Chưa nên claim hệ thống đã giải quyết hoàn toàn prompt injection.

## Kế hoạch tuần tới

Kiểm thử runtime calibration, fine-tune XLM-R với multilingual hard negatives, lưu logits để chạy Temperature Scaling, và nâng context-aware theo hướng semantic reasoning.
"""


def main() -> None:
    write("tong_quan_he_thong.md", tong_quan_he_thong())
    write("tien_do_2_tuan.md", tien_do_2_tuan())
    write("ket_qua_metrics_hien_tai.md", ket_qua_metrics_hien_tai())
    write("giai_thich_threshold_calibration.md", giai_thich_threshold_calibration())
    write("ket_qua_direct_external_benchmark.md", ket_qua_direct_external())
    write("ket_qua_bipia_ablation.md", ket_qua_bipia())
    write("diem_manh_diem_yeu.md", diem_manh_yeu())
    write("de_xuat_tiep_theo.md", de_xuat_tiep_theo())
    write("bao_cao_tom_tat_cho_giang_vien.md", bao_cao_tom_tat())
    print(f"Wrote final Vietnamese reports to {OUT}")


if __name__ == "__main__":
    main()
