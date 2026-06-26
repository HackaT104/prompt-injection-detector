# Threshold Optimization Summary

Validation dataset: `F:\Capstone Project\prompt-injection-detector\datasets\processed\vi_validation_processed.csv`
Test dataset: `F:\Capstone Project\prompt-injection-detector\datasets\processed\vi_test_processed.csv`

| Model | Selected Threshold | Method | Accuracy | Precision | Recall | F1 | F2 | ROC-AUC | FP | FN | Recommendation |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| logistic_regression | 0.78 | fallback_max_precision_with_recall_ge_0.75 | 0.7380 | 0.8233 | 0.5396 | 0.6519 | 0.5795 | 0.8682 | 79 | 314 | comparison_only |
| linear_svm | 0.95 | fallback_max_precision_with_recall_ge_0.75 | 0.7313 | 0.8394 | 0.5059 | 0.6313 | 0.5495 | 0.8346 | 66 | 337 | comparison_only |
| random_forest | 0.83 | fallback_max_precision_with_recall_ge_0.75 | 0.6953 | 0.6437 | 0.7390 | 0.6881 | 0.7177 | 0.7957 | 279 | 178 | comparison_only |
| roberta_v5_vi | 0.20 | production_precision_ge_0.80_recall_ge_0.80_min_fp | 0.9933 | 0.9927 | 0.9927 | 0.9927 | 0.9927 | 0.9997 | 5 | 5 | demo_and_primary_runtime |
| xlm_roberta_v5_vi | 0.77 | high_precision_block_precision_ge_0.90 | 0.7300 | 0.8901 | 0.4633 | 0.6095 | 0.5125 | 0.8703 | 39 | 366 | warning_only_high_precision_block_optional |
