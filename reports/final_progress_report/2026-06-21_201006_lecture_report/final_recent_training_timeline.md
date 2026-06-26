# Timeline train/evaluate gần đây

| Time | Event | Model | Dataset | Result | Artifacts |
| --- | --- | --- | --- | --- | --- |
| 2026-05-31 20:31 | Train 3 traditional models | LR, SVM, RF | prompt_injection_ml_ready.csv | Artifacts + metrics + 5-fold CV | models/*_model.joblib; reports/metrics.json |
| 2026-06-13 23:40 | Complete v4 | XLM-R v4 | jayavibhav/prompt-injection | Valid checkpoint; head-only due 4GB VRAM OOM | models/transformers/xlm_roberta_v4 |
| 2026-06-14 03:41 | Complete v4 | RoBERTa v4 | jayavibhav/prompt-injection | Full fine-tune; HF test F1 0.9935 | models/transformers/roberta_v4 |
| 2026-06-14 05:06 | Complete v4 | DistilBERT v4 | jayavibhav/prompt-injection | Full fine-tune; later deprecated | models/deprecated/distilbert_20260620_014520 |
| 2026-06-20 03:20 | Continue fine-tune VI | RoBERTa v5 VI | replay + VI/mixed | Full encoder; VI test F1 0.9927, FP/FN 5/5 | models/transformers/roberta_v5_vi; reports/transformer_v5_vi_results.json |
| 2026-06-20 14:25 | Continue fine-tune VI | XLM-R v5 VI | replay + VI/mixed | Full encoder; old th 0.16 F1 0.7322 | models/transformers/xlm_roberta_v5_vi; logs/xlm_roberta_v5_vi_sgd_resume_20260620_130951.out.log |
| 2026-06-21 00:13 | Threshold sweep, no retrain | 5 current models | VI validation/test | Separated balanced evaluation and production block | models/thresholds.json; reports/threshold_optimization_summary.* |
