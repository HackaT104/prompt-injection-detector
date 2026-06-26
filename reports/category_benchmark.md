# Category Benchmark

Detection rate là tỷ lệ mẫu trong category bị model đánh dấu warn/block.
Với Benign, detection rate càng thấp càng tốt. Với các category tấn công, detection rate càng cao càng tốt.

- Raw CSV: `F:\Capstone Project\prompt-injection-detector\reports\quality\category_benchmark.csv`

| Model | Category | Count | Expected label | Detected count | Detection rate | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | Authority Escalation | 8 | 1 | 8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| logistic_regression | Benign | 50 | 0 | 16 | 0.3200 | 0.0000 | 0.0000 | 0.0000 |
| logistic_regression | Crescendo | 5 | 1 | 5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| logistic_regression | Direct Injection | 6 | 1 | 6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| logistic_regression | Indirect Injection | 7 | 1 | 7 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| logistic_regression | Jailbreak | 6 | 1 | 6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| logistic_regression | Obfuscated Injection | 5 | 1 | 5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| logistic_regression | Prompt Extraction | 6 | 1 | 6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| logistic_regression | Role Override | 7 | 1 | 7 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| linear_svm | Authority Escalation | 8 | 1 | 8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| linear_svm | Benign | 50 | 0 | 12 | 0.2400 | 0.0000 | 0.0000 | 0.0000 |
| linear_svm | Crescendo | 5 | 1 | 5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| linear_svm | Direct Injection | 6 | 1 | 6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| linear_svm | Indirect Injection | 7 | 1 | 5 | 0.7143 | 1.0000 | 0.7143 | 0.8333 |
| linear_svm | Jailbreak | 6 | 1 | 6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| linear_svm | Obfuscated Injection | 5 | 1 | 5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| linear_svm | Prompt Extraction | 6 | 1 | 5 | 0.8333 | 1.0000 | 0.8333 | 0.9091 |
| linear_svm | Role Override | 7 | 1 | 7 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| random_forest | Authority Escalation | 8 | 1 | 8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| random_forest | Benign | 50 | 0 | 16 | 0.3200 | 0.0000 | 0.0000 | 0.0000 |
| random_forest | Crescendo | 5 | 1 | 5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| random_forest | Direct Injection | 6 | 1 | 6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| random_forest | Indirect Injection | 7 | 1 | 6 | 0.8571 | 1.0000 | 0.8571 | 0.9231 |
| random_forest | Jailbreak | 6 | 1 | 6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| random_forest | Obfuscated Injection | 5 | 1 | 5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| random_forest | Prompt Extraction | 6 | 1 | 5 | 0.8333 | 1.0000 | 0.8333 | 0.9091 |
| random_forest | Role Override | 7 | 1 | 7 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| distilbert | Authority Escalation | 8 | 1 | 3 | 0.3750 | 1.0000 | 0.3750 | 0.5455 |
| distilbert | Benign | 50 | 0 | 11 | 0.2200 | 0.0000 | 0.0000 | 0.0000 |
| distilbert | Crescendo | 5 | 1 | 5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| distilbert | Direct Injection | 6 | 1 | 6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| distilbert | Indirect Injection | 7 | 1 | 7 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| distilbert | Jailbreak | 6 | 1 | 2 | 0.3333 | 1.0000 | 0.3333 | 0.5000 |
| distilbert | Obfuscated Injection | 5 | 1 | 5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| distilbert | Prompt Extraction | 6 | 1 | 4 | 0.6667 | 1.0000 | 0.6667 | 0.8000 |
| distilbert | Role Override | 7 | 1 | 6 | 0.8571 | 1.0000 | 0.8571 | 0.9231 |
| roberta | Authority Escalation | 8 | 1 | 8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| roberta | Benign | 50 | 0 | 16 | 0.3200 | 0.0000 | 0.0000 | 0.0000 |
| roberta | Crescendo | 5 | 1 | 5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| roberta | Direct Injection | 6 | 1 | 6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| roberta | Indirect Injection | 7 | 1 | 7 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| roberta | Jailbreak | 6 | 1 | 6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| roberta | Obfuscated Injection | 5 | 1 | 5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| roberta | Prompt Extraction | 6 | 1 | 5 | 0.8333 | 1.0000 | 0.8333 | 0.9091 |
| roberta | Role Override | 7 | 1 | 7 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid | Authority Escalation | 8 | 1 | 8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid | Benign | 50 | 0 | 23 | 0.4600 | 0.0000 | 0.0000 | 0.0000 |
| hybrid | Crescendo | 5 | 1 | 5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid | Direct Injection | 6 | 1 | 6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid | Indirect Injection | 7 | 1 | 7 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid | Jailbreak | 6 | 1 | 6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid | Obfuscated Injection | 5 | 1 | 5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid | Prompt Extraction | 6 | 1 | 6 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| hybrid | Role Override | 7 | 1 | 7 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
