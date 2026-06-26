# Báo cáo đánh giá mô hình

> Tất cả chỉ số trong file này được sinh từ quá trình train/test thật.

## Dataset

- Số mẫu: 7275
- Cột prompt: `text`
- Cột nhãn: `label`
- Phân phối nhãn: 0: 3222, 1: 4053

## Chia train/test

- Train size: 5820
- Test size: 1455
- Test size ratio: 0.2
- Random state: 42
- Stratify theo nhãn: True
- Phân phối train: 0: 2578, 1: 3242
- Phân phối test: 0: 644, 1: 811

## Multilingual augmentation

- Bật augmentation: True
- File augmented dataset: `None`
- Số dòng augmented đã lưu: 0
- Số dòng augmented dùng trong train split: 62
- Ghi chú: Augmented Vietnamese prompts are generated from the training split only for model fitting; the test split remains the original dataset. The saved augmented CSV is generated only when training from the original JSONL source.

## Validation và threshold

- Validation size: 1177
- Phân phối validation: 0: 523, 1: 654
- Threshold phân loại được chọn trên validation set, không dùng mặc định 0.5.
- Runtime API dùng ngưỡng cảnh báo/chặn cao hơn để giảm false positive khi rule-based không khớp.

## Ý nghĩa chỉ số

- Accuracy: tỷ lệ dự đoán đúng trên toàn bộ tập test.
- Precision: trong các prompt bị dự đoán là malicious, tỷ lệ thật sự malicious.
- Recall: trong các prompt malicious thật, tỷ lệ model phát hiện được.
- F1-score: trung bình điều hòa giữa precision và recall.
- Confusion matrix: bảng so sánh nhãn thật và nhãn dự đoán.

Trong bài toán prompt injection, recall của lớp malicious rất quan trọng vì bỏ sót prompt tấn công có thể nguy hiểm hơn cảnh báo nhầm.

## logistic_regression

- Accuracy: 0.9271
- Precision (positive label = malicious): 0.9201
- Recall (positive label = malicious): 0.9519
- F1-score: 0.9358
- Evaluation threshold: 0.4966
- Runtime warn threshold: 0.5000
- Runtime block threshold: 0.8000
- False Positive Rate: 0.1040
- False Negative Rate: 0.0481
- ROC AUC: 0.9864
- Average Precision: 0.9891
- Train size: 5882
- Test size: 1455
- Thời gian huấn luyện: 0.6081 giây
- Thời gian dự đoán trung bình: 0.00006644 giây/prompt
- File confusion matrix: `F:\Capstone Project\prompt-injection-detector\reports\confusion_matrix_logistic_regression.png`
- File ROC curve: `F:\Capstone Project\prompt-injection-detector\reports\roc_curve_logistic_regression.png`
- File Precision-Recall curve: `F:\Capstone Project\prompt-injection-detector\reports\precision_recall_curve_logistic_regression.png`

### Confusion Matrix

| | Dự đoán benign | Dự đoán malicious |
|---|---:|---:|
| Thật benign | 577 | 67 |
| Thật malicious | 39 | 772 |

### Classification Report

```text
precision    recall  f1-score   support

      benign       0.94      0.90      0.92       644
   malicious       0.92      0.95      0.94       811

    accuracy                           0.93      1455
   macro avg       0.93      0.92      0.93      1455
weighted avg       0.93      0.93      0.93      1455
```

### Top malicious indicators

| Feature | Weight |
|---|---:|
| `pwned` | 6.059017 |
| `run` | 4.013587 |
| `whoami` | 2.972029 |
| `say` | 2.698522 |
| `no` | 2.469967 |
| `all` | 2.469711 |
| `only` | 2.209563 |
| `ignore` | 2.057072 |
| `now` | 2.034245 |
| `p\w\n\e\d` | 2.014901 |
| `execute` | 1.965250 |
| `output` | 1.956043 |
| `instructions` | 1.904032 |
| `import` | 1.865373 |
| `sentence` | 1.821152 |
| `bypass` | 1.758546 |
| `print` | 1.718850 |
| `above` | 1.689893 |
| `following` | 1.687724 |
| `respond` | 1.578173 |

### Top benign indicators

| Feature | Weight |
|---|---:|
| `write` | -3.310287 |
| `largest` | -3.074132 |
| `compare` | -2.947353 |
| `create` | -2.860455 |
| `deutschland` | -2.648104 |
| `explain` | -2.548666 |
| `techniques` | -2.508026 |
| `generate` | -2.476451 |
| `capital` | -2.415425 |
| `germany` | -2.408464 |
| `cybersecurity` | -2.296700 |
| `computing` | -2.155663 |
| `learning` | -2.100229 |
| `history` | -2.034252 |
| `algorithms` | -2.000447 |
| `tips` | -1.983679 |
| `work` | -1.965409 |
| `development` | -1.963407 |
| `discussing` | -1.934939 |
| `storytelling` | -1.864710 |

## linear_svm

- Accuracy: 0.9354
- Precision (positive label = malicious): 0.9273
- Recall (positive label = malicious): 0.9593
- F1-score: 0.9430
- Evaluation threshold: 0.4440
- Runtime warn threshold: 0.5000
- Runtime block threshold: 0.8000
- False Positive Rate: 0.0947
- False Negative Rate: 0.0407
- ROC AUC: 0.9880
- Average Precision: 0.9899
- Train size: 5882
- Test size: 1455
- Thời gian huấn luyện: 0.8318 giây
- Thời gian dự đoán trung bình: 0.00006388 giây/prompt
- File confusion matrix: `F:\Capstone Project\prompt-injection-detector\reports\confusion_matrix_linear_svm.png`
- File ROC curve: `F:\Capstone Project\prompt-injection-detector\reports\roc_curve_linear_svm.png`
- File Precision-Recall curve: `F:\Capstone Project\prompt-injection-detector\reports\precision_recall_curve_linear_svm.png`

### Confusion Matrix

| | Dự đoán benign | Dự đoán malicious |
|---|---:|---:|
| Thật benign | 583 | 61 |
| Thật malicious | 33 | 778 |

### Classification Report

```text
precision    recall  f1-score   support

      benign       0.95      0.91      0.93       644
   malicious       0.93      0.96      0.94       811

    accuracy                           0.94      1455
   macro avg       0.94      0.93      0.93      1455
weighted avg       0.94      0.94      0.94      1455
```

### Top malicious indicators

| Feature | Weight |
|---|---:|
| `pwned` | 2.401530 |
| `run` | 1.419140 |
| `say` | 1.301250 |
| `forget` | 1.111517 |
| `sie` | 1.094852 |
| `everything` | 1.071793 |
| `no` | 1.048553 |
| `all` | 1.006963 |
| `whoami` | 0.976341 |
| `following` | 0.917059 |
| `forget everything` | 0.907264 |
| `show` | 0.896461 |
| `expert` | 0.890859 |
| `respond` | 0.858041 |
| `without` | 0.846842 |
| `now` | 0.835616 |
| `above` | 0.829463 |
| `only` | 0.822319 |
| `before` | 0.791265 |
| `act` | 0.781034 |

### Top benign indicators

| Feature | Weight |
|---|---:|
| `largest` | -1.627302 |
| `germany` | -1.429510 |
| `deutschland` | -1.410488 |
| `corona` | -1.308124 |
| `europa` | -1.229751 |
| `ukraine` | -1.228679 |
| `capital` | -1.224801 |
| `currency` | -1.200232 |
| `cybersecurity` | -1.139691 |
| `write` | -1.112245 |
| `describe` | -1.111905 |
| `europe` | -1.094132 |
| `algorithms` | -1.079334 |
| `history` | -1.078981 |
| `time` | -1.069127 |
| `ist` | -1.068258 |
| `israel` | -1.066047 |
| `best` | -1.062652 |
| `storytelling` | -1.040391 |
| `berlin` | -1.032453 |

## random_forest

- Accuracy: 0.9416
- Precision (positive label = malicious): 0.9332
- Recall (positive label = malicious): 0.9642
- F1-score: 0.9485
- Evaluation threshold: 0.4933
- Runtime warn threshold: 0.5000
- Runtime block threshold: 0.8000
- False Positive Rate: 0.0870
- False Negative Rate: 0.0358
- ROC AUC: 0.9883
- Average Precision: 0.9901
- Train size: 5882
- Test size: 1455
- Thời gian huấn luyện: 2.7550 giây
- Thời gian dự đoán trung bình: 0.00012649 giây/prompt
- File confusion matrix: `F:\Capstone Project\prompt-injection-detector\reports\confusion_matrix_random_forest.png`
- File ROC curve: `F:\Capstone Project\prompt-injection-detector\reports\roc_curve_random_forest.png`
- File Precision-Recall curve: `F:\Capstone Project\prompt-injection-detector\reports\precision_recall_curve_random_forest.png`

### Confusion Matrix

| | Dự đoán benign | Dự đoán malicious |
|---|---:|---:|
| Thật benign | 588 | 56 |
| Thật malicious | 29 | 782 |

### Classification Report

```text
precision    recall  f1-score   support

      benign       0.95      0.91      0.93       644
   malicious       0.93      0.96      0.95       811

    accuracy                           0.94      1455
   macro avg       0.94      0.94      0.94      1455
weighted avg       0.94      0.94      0.94      1455
```

### Top malicious indicators

| Feature | Weight |
|---|---:|

### Top benign indicators

| Feature | Weight |
|---|---:|

## Kết luận so sánh

- Model được khuyến nghị: `random_forest`
- Lý do: Random Forest có recall/F1 tốt nhất theo metrics hiện tại. Tuy nhiên TF-IDF là vector sparse nhiều chiều, nên Random Forest cần được cân nhắc về tốc độ và khả năng tổng quát trước khi chọn làm model chính.
