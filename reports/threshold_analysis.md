# Threshold analysis

Threshold được chọn trên validation set để giảm false positive nhưng vẫn giữ recall cao cho lớp malicious.

## logistic_regression

- Evaluation threshold: 0.4966
- Runtime warn threshold: 0.5000
- Runtime block threshold: 0.8000
- Validation precision: 0.9201
- Validation recall: 0.9511
- Validation F1-score: 0.9353
- Validation FPR: 0.1033
- Validation FNR: 0.0489
- Lý do chọn: Chọn threshold có precision cao nhất trong các ngưỡng đạt recall >= 0.95 và precision >= 0.9 trên validation set.

## linear_svm

- Evaluation threshold: 0.4440
- Runtime warn threshold: 0.5000
- Runtime block threshold: 0.8000
- Validation precision: 0.9219
- Validation recall: 0.9572
- Validation F1-score: 0.9392
- Validation FPR: 0.1013
- Validation FNR: 0.0428
- Lý do chọn: Chọn threshold có precision cao nhất trong các ngưỡng đạt recall >= 0.95 và precision >= 0.9 trên validation set.

## random_forest

- Evaluation threshold: 0.4933
- Runtime warn threshold: 0.5000
- Runtime block threshold: 0.8000
- Validation precision: 0.9094
- Validation recall: 0.9511
- Validation F1-score: 0.9297
- Validation FPR: 0.1185
- Validation FNR: 0.0489
- Lý do chọn: Chọn threshold có precision cao nhất trong các ngưỡng đạt recall >= 0.95 và precision >= 0.9 trên validation set.
