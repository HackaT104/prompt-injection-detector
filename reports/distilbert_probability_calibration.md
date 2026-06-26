# DistilBERT Probability Calibration

- Model: `distilbert_v3`
- Dataset: `F:\Capstone Project\prompt-injection-detector\datasets\unified\prompt_injection_transformer_ready_v3.csv`
- Validation rows: `39563`
- Calibrator: `IsotonicRegression`
- Calibrator path: `F:\Capstone Project\prompt-injection-detector\models\transformers\distilbert_v3\probability_calibrator.joblib`

| Metric | Raw | Calibrated |
| --- | ---: | ---: |
| Brier score | 0.006129 | 0.005031 |
| Log loss | 0.045883 | 0.019785 |
| ROC-AUC | 0.999525 | 0.999562 |
| Average precision | 0.999574 | 0.999529 |
| Evaluation threshold | 0.3300 | 0.3900 |
| Warn threshold | 0.3300 | 0.3900 |
| Block threshold | 0.4800 | 0.5400 |

Runtime now uses calibrated probability for `distilbert_v3` when `probability_calibrator.joblib` is present.