# Threshold Optimization Recommendation

- Input: `reports\bipia_evaluation\bipia_predictions.csv`
- Rows: `500`
- Label column: `label`
- Score column: `final_score`
- Beta used for recommendation: `2`

## Recommended threshold

- Threshold: `0.7400`
- Precision: `1.0000`
- Recall: `1.0000`
- F1: `1.0000`
- F2: `1.0000`
- Confusion matrix [[TN, FP], [FN, TP]]: `[[150, 0], [0, 350]]`

## Best F1 threshold

- Threshold: `0.7400` with F1 `1.0000`

## Best F2 threshold

- Threshold: `0.7400` with F2 `1.0000`

This recommendation is based on a threshold sweep from 0.01 to 0.99.
