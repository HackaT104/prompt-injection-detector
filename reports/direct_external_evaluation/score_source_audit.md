# Direct External Evaluation Score Source Audit

This audit covers model-only direct benchmark scoring. It does not use rule-based, context-aware, BIPIA, or indirect pipelines.

| Model | Score source | Min | Max | Mean | Std | Post-hoc calibrated? | Warning |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| logistic_regression | scikit-learn predict_proba positive class from TF-IDF LogisticRegression; decision_function sigmoid fallback only if predict_proba is absent. | 0.0129 | 0.9985 | 0.6302 | 0.1957 | False | Model probability is not externally calibrated on the direct benchmark validation split. |
| random_forest | scikit-learn predict_proba positive class from TF-IDF RandomForest; effectively tree vote/probability average. | 0.0000 | 1.0000 | 0.6966 | 0.2501 | False | Random Forest probabilities are often poorly calibrated without post-hoc calibration. |
| roberta | Transformer softmax probability for INJECTION class from sequence-classification logits. | 0.0000 | 1.0000 | 0.4932 | 0.4995 | False | No probability_calibrator.joblib was used for roberta_v5_vi in this direct evaluation; softmax probabilities may be miscalibrated. |
| xlm_roberta | Transformer softmax probability for INJECTION class from sequence-classification logits. | 0.0000 | 1.0000 | 0.4631 | 0.3584 | False | No probability_calibrator.joblib was used for xlm_roberta_v5_vi in this direct evaluation; softmax probabilities may be miscalibrated. |

Notes:
- RoBERTa/XLM-RoBERTa scores are softmax probabilities, not raw logits, sigmoid scores, or margins.
- The direct evaluation scripts apply no external calibrator unless a model directory contains `probability_calibrator.joblib`; the current v5 RoBERTa/XLM-R runs reported no calibration method.
- Low optimal thresholds can happen when F2 optimization prioritizes recall, when probabilities are compressed near zero, or when calibration is poor. Use strict validation/test and calibration reports before drawing final conclusions.
