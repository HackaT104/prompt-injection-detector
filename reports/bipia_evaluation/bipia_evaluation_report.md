# BIPIA External Benchmark Evaluation

## 1. Dataset overview

- Input: `F:\Capstone Project\prompt-injection-detector\data\external_benchmark\bipia\bipia_normalized.csv`
- Total samples: `500`
- Label distribution: `{'0': 150, '1': 350}`
- Source task distribution: `{'email': 173, 'table': 327}`
- Difficulty distribution: `{'easy': 154, 'hard': 140, 'medium': 206}`
- ML enabled: `False`
- Transformer enabled: `False`

BIPIA is used here only as an external/OOD benchmark. It is not copied into training data.

## 2. Overall results

- Accuracy: `0.9980`
- Precision: `0.9972`
- Recall: `1.0000`
- F1: `0.9986`
- F2: `0.9994`
- ROC-AUC: `1.0000`
- PR-AUC: `1.0000`
- Confusion matrix [[TN, FP], [FN, TP]]: `[[149, 1], [0, 350]]`
- False positives: `1`
- False negatives: `0`

## 3. Layer contribution

- Strongest layer counts: `{'context': 350, 'rule': 150}`

This is an approximate contribution view based on the largest score among rule, ML, Transformer and context-aware signals.

## 4. Results by source task

- `email`: accuracy=1.0000, precision=1.0000, recall=1.0000, F1=1.0000, cm=[[50, 0], [0, 123]]
- `table`: accuracy=0.9969, precision=0.9956, recall=1.0000, F1=0.9978, cm=[[99, 1], [0, 227]]

## 5. Results by difficulty

- `easy`: accuracy=0.9935, precision=0.8000, recall=1.0000, F1=0.8889, cm=[[149, 1], [0, 4]]
- `hard`: accuracy=1.0000, precision=1.0000, recall=1.0000, F1=1.0000, cm=[[0, 0], [0, 140]]
- `medium`: accuracy=1.0000, precision=1.0000, recall=1.0000, F1=1.0000, cm=[[0, 0], [0, 206]]

## 6. False positives

Safe samples were flagged because the detector saw assistant-directed or response-manipulation wording.
- `bipia-table-00043-safe` score=0.7 source=table difficulty=easy reasons=Matched keyword: external_content: bash | ML model scoring disabled by runtime option. | Transformer scoring disabled by runtime option.

## 7. False negatives

No false negatives in this run.

## 8. Threshold optimization

Do not reuse the 24-sample indirect evaluation threshold for BIPIA. BIPIA threshold optimization must be run separately.

Separate BIPIA threshold optimization found:
- Threshold: `0.7400`
- Precision: `1.0000`
- Recall: `1.0000`
- F1: `1.0000`
- F2: `1.0000`
- Confusion matrix: `[[150, 0], [0, 350]]`

Optional validation/test threshold split:
- Validation rows: `150`
- Test rows: `350`
- Tuned threshold on validation: `0.7400`
- Test F2 at tuned threshold: `1.0000`
- Test confusion matrix: `[[105, 0], [0, 245]]`

## 9. Data leakage checks

- BIPIA is stored under `data/external_benchmark/bipia/`.
- `prepare_bipia_benchmark.py` does not write to `data/raw`, `datasets/processed`, or model training folders.
- Threshold optimization on BIPIA is reported as benchmark calibration, not as final held-out performance unless `--split-threshold` is used.

## 10. Initial conclusion

This benchmark primarily tests OOD indirect prompt injection. Strong performance indicates the detector is not only memorizing classic `ignore previous instructions` patterns. Failure cases should be used to improve context-aware/task-hijacking recognition, not to train directly on BIPIA unless a separate train/validation/test protocol is defined.
