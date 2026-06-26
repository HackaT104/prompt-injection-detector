# Quản lý report Batch Evaluation

## 1. Vị trí lưu report

Mỗi lần chạy `scripts/run_dataset_evaluation.py`, hệ thống tạo một thư mục riêng trong:

```text
reports/batch_evaluation/runs/YYYYMMDD_HHMMSS_dataset_name/
```

Ví dụ:

```text
reports/batch_evaluation/runs/20260608_231500_batch_test_sample/
```

Nếu dùng `--run-name`, phần cuối thư mục sẽ lấy theo tên run:

```bat
python scripts/run_dataset_evaluation.py --file "dataset.csv" --run-name "neuralchemy_test_500"
```

Kết quả:

```text
reports/batch_evaluation/runs/20260608_231500_neuralchemy_test_500/
```

Nếu dùng `--output-dir`, thư mục gốc sẽ thay đổi nhưng vẫn có cấu trúc `runs/...`:

```bat
python scripts/run_dataset_evaluation.py --file "dataset.csv" --output-dir "reports/my_custom_eval"
```

Kết quả:

```text
reports/my_custom_eval/runs/YYYYMMDD_HHMMSS_dataset/
```

## 2. File trong mỗi run folder

Mỗi run folder có các file:

- `metrics_summary.csv`: bảng metrics từng model, dễ đưa vào Excel hoặc báo cáo.
- `metrics_summary.json`: metrics dạng JSON kèm metadata.
- `predictions_full.csv`: toàn bộ prompt và prediction của từng model.
- `predictions_full.json`: toàn bộ response batch dạng JSON.
- `false_positives.csv`: các case safe nhưng model dự đoán injection.
- `false_negatives.csv`: các case injection nhưng model bỏ sót.
- `model_disagreements.csv`: các prompt mà các model không đồng thuận.
- `batch_evaluation_report.md`: báo cáo Markdown chính.
- `run_metadata.json`: metadata của lần chạy.

## 3. Đọc index.csv

File tổng hợp nằm tại:

```text
reports/batch_evaluation/index.csv
```

Mỗi dòng tương ứng một lần chạy, gồm:

- `run_id`: mã run theo timestamp và tên dataset/run.
- `timestamp`: thời gian chạy.
- `dataset_path`: đường dẫn dataset.
- `dataset_name`: tên file dataset.
- `total_rows`: số prompt.
- `has_ground_truth`: dataset có label thật hay không.
- `model_count`: số model đã chạy.
- `best_model_by_f1`: model có F1 cao nhất.
- `best_f1`: F1 cao nhất.
- `output_folder`: thư mục chứa report của run.

Dùng `index.csv` để so sánh nhanh nhiều lần chạy và tìm lại report tương ứng.

## 4. README.md tổng hợp

File:

```text
reports/batch_evaluation/README.md
```

README hiển thị danh sách các lần chạy gần nhất ở dạng bảng:

| Time | Dataset | Rows | Best Model | Best F1 | Folder |
| ---- | ------- | ---: | ---------- | ------: | ------ |

Đây là file nên mở đầu tiên khi cần xem lịch sử test nhiều dataset.

## 5. Xem false positives và false negatives

Trong từng run folder:

- Mở `false_positives.csv` để xem cảnh báo nhầm.
- Mở `false_negatives.csv` để xem các prompt injection bị bỏ sót.

Trong bài toán Prompt Injection, `false_negatives.csv` đặc biệt quan trọng vì bỏ sót prompt độc hại có thể làm detector không chặn tấn công.

## 6. File nên dùng trong báo cáo đồ án

Khuyến nghị:

- Dùng `batch_evaluation_report.md` làm báo cáo chính cho từng lần chạy.
- Dùng `metrics_summary.csv` để đưa bảng metrics vào Word/Excel.
- Dùng `false_positives.csv` và `false_negatives.csv` để phân tích lỗi.
- Dùng `predictions_full.csv` nếu cần chứng minh toàn bộ prompt đã được đánh nhãn.
- Dùng `index.csv` để tổng hợp nhiều dataset hoặc nhiều lần thử nghiệm.

## 7. So sánh nhiều dataset khác nhau

Quy trình gợi ý:

1. Chạy mỗi dataset bằng một `--run-name` rõ ràng.
2. Mở `reports/batch_evaluation/index.csv` để xem tổng quan.
3. So sánh `best_model_by_f1`, `best_f1`, `total_rows` và `has_ground_truth`.
4. Mở từng `batch_evaluation_report.md` để xem confusion matrix, FP/FN và disagreement.
5. Khi cần phân tích sâu, mở `predictions_full.csv` của từng run.

Ví dụ:

```bat
python scripts/run_dataset_evaluation.py --file "datasets/examples/batch_test_sample.csv" --run-name "sample_20"
python scripts/run_dataset_evaluation.py --file "datasets/unified/prompt_injection_ml_ready.csv" --run-name "unified_ml_ready"
```