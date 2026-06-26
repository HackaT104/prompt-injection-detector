# 100 Prompt Stress Test Results

- Runtime allow: risk < 0.50
- Runtime warn: 0.50 <= risk < 0.80
- Runtime block: risk >= 0.80
- Suite path: `F:\Capstone Project\prompt-injection-detector\reports\quality\stress_suite.csv`
- Prediction path: `F:\Capstone Project\prompt-injection-detector\reports\quality\stress_test_predictions.csv`

| Model | Accuracy | Precision | Recall | F1 | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.8400 | 0.7576 | 1.0000 | 0.8621 | 16 | 0 |
| linear_svm | 0.8500 | 0.7966 | 0.9400 | 0.8624 | 12 | 3 |
| random_forest | 0.8200 | 0.7500 | 0.9600 | 0.8421 | 16 | 2 |
| distilbert | 0.7700 | 0.7755 | 0.7600 | 0.7677 | 11 | 12 |
| roberta | 0.8300 | 0.7538 | 0.9800 | 0.8522 | 16 | 1 |
| hybrid | 0.7700 | 0.6849 | 1.0000 | 0.8130 | 23 | 0 |
