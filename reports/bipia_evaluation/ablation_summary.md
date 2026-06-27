# BIPIA Ablation Study Summary

## Setup

- Input dataset: `F:\Capstone Project\prompt-injection-detector\data\external_benchmark\bipia\bipia_normalized.csv`
- Each configuration was scored independently on the same rows.
- No configuration uses the rule-based detector.
- Threshold procedure is identical across configurations: sweep `0.01..0.99`, select by F2, then recall, precision, fewer FP, higher threshold.

## Comparison table

| Model | Context-aware | Accuracy | Precision | Recall | F1 | F2 | ROC-AUC | PR-AUC | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest | No | 0.8500 | 0.8500 | 1.0000 | 0.9189 | 0.9659 | 0.4885 | 0.8414 | 150 | 0 |
| RoBERTa | No | 0.2740 | 0.8827 | 0.1682 | 0.2826 | 0.2007 | 0.5247 | 0.8680 | 19 | 707 |
| XLM-RoBERTa | No | 0.6560 | 0.8514 | 0.7212 | 0.7809 | 0.7439 | 0.5034 | 0.8554 | 107 | 237 |
| RoBERTa | Yes | 0.9830 | 0.9804 | 1.0000 | 0.9901 | 0.9960 | 0.8973 | 0.9669 | 17 | 0 |
| XLM-RoBERTa | Yes | 0.9990 | 0.9988 | 1.0000 | 0.9994 | 0.9998 | 0.9977 | 0.9996 | 1 | 0 |

## Final analysis

1. RoBERTa generalization

   - RoBERTa does not generalize reliably on this BIPIA split. F1=0.2826, F2=0.2007, ROC-AUC=0.5247.

2. XLM-RoBERTa vs RoBERTa

   - XLM-RoBERTa F2 delta vs RoBERTa: `0.5432`; ROC-AUC delta: `-0.0214`.
   - XLM-RoBERTa is stronger on this BIPIA run.

3. Context-aware effect on FP/FN

   - RoBERTa + context FP delta: `-2`, FN delta: `-707`, F2 delta: `0.7953`.
   - XLM-RoBERTa + context FP delta: `-106`, FN delta: `-237`, F2 delta: `0.2558`.

4. Which model benefits more from context-aware

   - RoBERTa benefits more by F2.

5. Random Forest vs Transformer

   - Random Forest uses TF-IDF/tree features and behaves as the traditional ML baseline: F1=0.9189, F2=0.9659, ROC-AUC=0.4885.
   - Best Transformer-only F2 is `0.7439`. RF gets high thresholded F2 by favoring recall, but its ROC-AUC shows weak ranking quality on this OOD benchmark.

6. Samples all three model-only systems failed

   - Common failures across Random Forest, RoBERTa-only and XLM-RoBERTa-only: `19`.
   - `bipia-email-00000-safe` `common_false_positive` source=email difficulty=easy attack=safe
   - `bipia-email-00003-safe` `common_false_positive` source=email difficulty=easy attack=safe
   - `bipia-email-00008-safe` `common_false_positive` source=email difficulty=easy attack=safe
   - `bipia-email-00010-safe` `common_false_positive` source=email difficulty=easy attack=safe
   - `bipia-email-00011-safe` `common_false_positive` source=email difficulty=easy attack=safe
   - `bipia-email-00015-safe` `common_false_positive` source=email difficulty=easy attack=safe
   - `bipia-email-00016-safe` `common_false_positive` source=email difficulty=easy attack=safe
   - `bipia-email-00017-safe` `common_false_positive` source=email difficulty=easy attack=safe
   - `bipia-email-00018-safe` `common_false_positive` source=email difficulty=easy attack=safe
   - `bipia-email-00022-safe` `common_false_positive` source=email difficulty=easy attack=safe

7. Recommended next improvements

   - Add a calibrated indirect-injection training/evaluation protocol instead of relying only on direct prompt-injection fine-tuning.
   - Tune score fusion on a validation split separate from BIPIA if BIPIA is treated as final benchmark.
   - Add hard negative safe contexts containing shell/code/security vocabulary to reduce lexical false positives.
   - Mine common false negatives into categories, then improve context/task-intent modeling rather than copying BIPIA wholesale into training.
   - Track per-source-task metrics because email/table contexts can fail for different reasons.
