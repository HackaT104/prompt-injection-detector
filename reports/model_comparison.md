# So sánh mô hình phát hiện Prompt Injection

## 1. Mục tiêu so sánh

So sánh 3 mô hình ML dùng cùng dataset, cùng train/test split, cùng preprocessing và cùng TF-IDF representation:

- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM
- TF-IDF + Random Forest

## 2. Dataset

- Tổng số dòng: 7275
- Train size: 5820
- Test size: 1455
- Label distribution: `0: 3222, 1: 4053`
- Train label distribution: `0: 2578, 1: 3242`
- Test label distribution: `0: 644, 1: 811`
- Language distribution: `N/A`
- Source distribution: `augmented_multilingual_dataset: 31, direct_merged: 1160, direct_ml_ready: 17, hard_negatives: 20, neuralchemy_core: 6027, role_override: 20`

## 3. Bảng metrics

| Model | Accuracy | Precision | Recall | F1 | FPR | FNR | ROC AUC | AP | Train time | Avg pred time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random_forest | 0.9416 | 0.9332 | 0.9642 | 0.9485 | 0.0870 | 0.0358 | 0.9883 | 0.9901 | 2.7550 | 0.0001 |
| linear_svm | 0.9354 | 0.9273 | 0.9593 | 0.9430 | 0.0947 | 0.0407 | 0.9880 | 0.9899 | 0.8318 | 0.0001 |
| logistic_regression | 0.9271 | 0.9201 | 0.9519 | 0.9358 | 0.1040 | 0.0481 | 0.9864 | 0.9891 | 0.6081 | 0.0001 |

## 4. Phân tích từng mô hình

### Logistic Regression

- Ưu điểm: nhanh, dễ giải thích, có `predict_proba` tự nhiên để tạo `risk_score`.
- Nhược điểm: tuyến tính nên có thể bỏ lỡ quan hệ phi tuyến phức tạp.
- Phù hợp làm model chính cho API vì cân bằng giữa hiệu năng, khả năng giải thích và triển khai ổn định.

### Linear SVM

- Ưu điểm: rất mạnh với TF-IDF sparse vector và text classification ngắn.
- Nhược điểm: bản gốc không có xác suất trực tiếp; project dùng `CalibratedClassifierCV` để có risk_score.
- Phù hợp làm model so sánh mạnh cho dữ liệu văn bản.

### Random Forest

- Random Forest là ensemble gồm nhiều Decision Tree. Mỗi cây đưa ra dự đoán, kết quả cuối lấy theo voting.
- Ưu điểm: trực giác dễ hiểu, có feature importance, giảm overfitting so với một cây đơn.
- Nhược điểm trong text classification: TF-IDF thường rất nhiều chiều và sparse, Random Forest có thể không tối ưu bằng Logistic Regression hoặc Linear SVM.
- Dùng Random Forest để so sánh bổ sung; không mặc định là model chính nếu metrics không vượt trội rõ.

## 5. Chọn model tối ưu

Không chọn chỉ dựa trên accuracy. Thứ tự ưu tiên là recall malicious, F1, FNR thấp, precision hợp lý, tốc độ dự đoán, khả năng giải thích, risk_score và triển khai API.

- Model khuyến nghị theo metrics thật: `random_forest`
- Lý do: Random Forest có recall/F1 tốt nhất theo metrics hiện tại. Tuy nhiên TF-IDF là vector sparse nhiều chiều, nên Random Forest cần được cân nhắc về tốc độ và khả năng tổng quát trước khi chọn làm model chính.
