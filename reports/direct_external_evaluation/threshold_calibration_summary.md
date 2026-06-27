# Direct Threshold and Calibration Summary

| Dataset | Model | Raw Threshold | Raw Test F1 | Raw Test F2 | Calibrated Threshold | Calibrated Test F1 | Calibrated Test F2 | ROC-AUC | PR-AUC | Brier Before | Brier After | ECE Before | ECE After |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | logistic_regression | 0.1600 | 0.7162 | 0.8628 | 0.2000 | 0.7163 | 0.8629 | 0.6236 | 0.6596 | 0.2464 | 0.2248 | 0.1293 | 0.0048 |
| all | random_forest | 0.1000 | 0.7190 | 0.8644 | 0.2000 | 0.7190 | 0.8644 | 0.6448 | 0.6749 | 0.2687 | 0.2210 | 0.1861 | 0.0107 |
| all | roberta | 0.0100 | 0.9389 | 0.9192 | 0.2600 | 0.9471 | 0.9616 | 0.9838 | 0.9865 | 0.0640 | 0.0397 | 0.0640 | 0.0026 |
| all | xlm_roberta | 0.0900 | 0.7767 | 0.8746 | 0.1800 | 0.7761 | 0.8747 | 0.8907 | 0.9038 | 0.1362 | 0.1305 | 0.0700 | 0.0066 |
| deepset | logistic_regression | 0.4600 | 0.8756 | 0.9224 | 0.3300 | 0.8756 | 0.9224 | 0.9662 | 0.9334 | 0.1125 | 0.0637 | 0.2110 | 0.0345 |
| deepset | random_forest | 0.5700 | 0.9596 | 0.9642 | 0.3300 | 0.9596 | 0.9642 | 0.9767 | 0.9613 | 0.0349 | 0.0313 | 0.0697 | 0.0285 |
| deepset | roberta | 0.0100 | 0.9398 | 0.9101 | 0.4000 | 0.9227 | 0.9331 | 0.9794 | 0.9722 | 0.0479 | 0.0372 | 0.0494 | 0.0215 |
| deepset | xlm_roberta | 0.0100 | 0.5893 | 0.7529 | 0.2300 | 0.5688 | 0.7673 | 0.7068 | 0.6189 | 0.2776 | 0.2033 | 0.2743 | 0.0477 |

## Analysis

1. RoBERTa low threshold: raw validation-selected threshold is `0.0100` because F2 optimization favors recall and a non-trivial injection tail has low softmax scores.
2. RoBERTa calibration: calibrated threshold becomes `0.2600`; compare Brier `0.0640` -> `0.0397`.
3. XLM-RoBERTa over-sensitivity: raw threshold `0.0900` with high recall but many false positives indicates score calibration/decision boundary issues.
4. XLM-RoBERTa calibration: Brier `0.1362` -> `0.1305`, ECE `0.0700` -> `0.0066`.
5. Best calibrated F2 on `all`: `roberta` with calibrated F2 `0.9616`.
6. Best calibrated F2 among all completed calibration runs: `random_forest` on `deepset` with calibrated F2 `0.9642`.
7. Temperature scaling was not run here because saved predictions contain probabilities/scores, not logits. To do temperature scaling, rerun Transformer inference and persist logits on the validation split.
