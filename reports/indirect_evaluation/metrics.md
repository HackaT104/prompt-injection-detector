# Indirect Prompt Injection Evaluation

- Dataset: `F:\Capstone Project\prompt-injection-detector\datasets\test\indirect_test_cases.csv`
- Model: `roberta`
- Rows: `24`
- Positive class: `1 = indirect injection`

| Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN | FP | FN | TP | Avg latency |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 12 | 0 | 0 | 12 | 509.59 ms |

Full predictions are in `predictions.csv`; error cases are in `false_positives.csv` and `false_negatives.csv`.
