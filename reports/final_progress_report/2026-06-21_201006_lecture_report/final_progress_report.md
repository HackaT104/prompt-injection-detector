# Báo cáo tiến độ Prompt Injection Detector

Generated: `2026-06-21T20:10:06.117556+07:00`  
Project: `F:\Capstone Project\prompt-injection-detector`  
Mode: **artifact-only, không train lại**

# 1. Executive Summary

- Đã train và giữ artifact cho ba baseline TF-IDF, Transformer v4, RoBERTa v5 VI và XLM-R v5 VI.
- **RoBERTa v5 VI** là model tốt nhất: test VI/mixed 1.500 mẫu đạt Accuracy/F1 `0.9933`, Recall `0.9927`, ROC-AUC `0.9997`, FP/FN `5/5`.
- **XLM-R v5 VI** chỉ nên warning/auxiliary: balanced F1 `0.7802`; block 0.77 giảm FP nhưng tạo 366 FN.
- Ba traditional models dùng baseline/comparison. DistilBERT v4 có artifact nhưng đã deprecated.

# 2. Model Inventory

| Model | Type | Path | Parent | Train/Val/Test | Modified | Status | Config/tokenizer | Eval report |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | TF-IDF + LogisticRegression | `F:\Capstone Project\prompt-injection-detector\models\logistic_regression_model.joblib` | N/A | prompt_injection_ml_ready.csv: 7275, stratified 80/20 | 2026-05-31 20:31:49 | trained | N/A; TF-IDF vectorizer present | metrics.json; threshold_optimization_summary.json |
| Linear SVM | TF-IDF + calibrated LinearSVC | `F:\Capstone Project\prompt-injection-detector\models\linear_svm_model.joblib` | N/A | prompt_injection_ml_ready.csv: 7275, stratified 80/20 | 2026-05-31 20:31:50 | trained | N/A; TF-IDF vectorizer present | metrics.json; threshold_optimization_summary.json |
| Random Forest | TF-IDF + RandomForestClassifier | `F:\Capstone Project\prompt-injection-detector\models\random_forest_model.joblib` | N/A | prompt_injection_ml_ready.csv: 7275, stratified 80/20 | 2026-05-31 20:31:53 | trained | N/A; TF-IDF vectorizer present | metrics.json; threshold_optimization_summary.json |
| DistilBERT v4 | Transformer | `F:\Capstone Project\prompt-injection-detector\models\deprecated\distilbert_20260620_014520` | distilbert-base-uncased | jayavibhav: train 261588 / val 32699 / test 32699 | 2026-06-20 01:45:21 | trained; deprecated | yes | transformer_v4_training_summary.md |
| RoBERTa v4 | Transformer | `F:\Capstone Project\prompt-injection-detector\models\transformers\roberta_v4` | roberta-base | jayavibhav: train 261588 / val 32699 / test 32699 | 2026-06-14 05:43:21 | trained; full fine-tune | yes | transformer_v4_training_summary.md |
| RoBERTa v5 VI | Transformer | `F:\Capstone Project\prompt-injection-detector\models\transformers\roberta_v5_vi` | roberta_v4 | replay train 23400 / val 520 / VI test 1500 / EN test 32698 | 2026-06-20 03:20:29 | trained; continued full fine-tune | yes | transformer_v5_vi_results.json |
| XLM-RoBERTa v4 | Transformer | `F:\Capstone Project\prompt-injection-detector\models\transformers\xlm_roberta_v4` | xlm-roberta-base | jayavibhav: train 261588 / val 32699 / test 32699 | 2026-06-14 05:43:21 | partially trained/head-only due OOM | yes | transformer_v4_training_summary.md |
| XLM-RoBERTa v5 VI | Transformer | `F:\Capstone Project\prompt-injection-detector\models\transformers\xlm_roberta_v5_vi` | xlm_roberta_v4 | replay train 23400 / val 520 / VI test 1500 / EN test 32698 | 2026-06-20 14:25:20 | trained; continued full fine-tune | yes | transformer_v5_vi_results.json |

# 3. Latest Evaluation Results

> Dataset khác nhau được ghi rõ; không so metric cross-domain như cùng test set.

| Model | Dataset | Eval th | Selection method | Acc | Prec | Recall | F1 | F2 | AUC | AP | TN | FP | FN | TP | Recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic_regression | vi_test_processed.csv (test, 1500) | 0.63 | maximize_f1 | 0.7647 | 0.6790 | 0.9150 | 0.7795 | 0.8555 | 0.8682 | 0.8200 | 523 | 295 | 58 | 624 | comparison baseline |
| linear_svm | vi_test_processed.csv (test, 1500) | 0.81 | maximize_f1 | 0.7460 | 0.6947 | 0.7874 | 0.7381 | 0.7669 | 0.8346 | 0.8120 | 582 | 236 | 145 | 537 | comparison baseline |
| random_forest | vi_test_processed.csv (test, 1500) | 0.62 | maximize_f1 | 0.6427 | 0.5647 | 0.9340 | 0.7039 | 0.8260 | 0.7957 | 0.7654 | 327 | 491 | 45 | 637 | comparison baseline |
| roberta_v5_vi | vi_test_processed.csv (test, 1500) | 0.05 | maximize_f1 | 0.9933 | 0.9927 | 0.9927 | 0.9927 | 0.9927 | 0.9997 | 0.9996 | 813 | 5 | 5 | 677 | primary runtime/demo |
| xlm_roberta_v5_vi | vi_test_processed.csv (test, 1500) | 0.37 | maximize_f1 | 0.7813 | 0.7185 | 0.8534 | 0.7802 | 0.8225 | 0.8703 | 0.8427 | 590 | 228 | 100 | 582 | warning-only multilingual auxiliary |
| distilbert_v4 | hf_prompt_injection_test.csv (test, 32699) | 0.81 | historical maximize_f2 | 0.9939 | 0.9923 | 0.9955 | 0.9939 | 0.9948 | 0.9996 | 0.9997 | 16430 | 125 | 73 | 16071 | deprecated historical comparison |
| roberta_v4 | vi_test_processed.csv (test, 1500) | 0.18 | historical maximize_f2 | 0.8273 | 0.7794 | 0.8651 | 0.8200 | 0.8465 | 0.9392 | 0.9311 | 651 | 167 | 92 | 590 | historical parent checkpoint |
| xlm_roberta_v4 | vi_test_processed.csv (test, 1500) | 0.27 | historical maximize_f2 | 0.7107 | 0.6260 | 0.9032 | 0.7395 | 0.8297 | 0.8495 | 0.8180 | 450 | 368 | 66 | 616 | historical partial/head-only |

Baseline gốc trên `prompt_injection_ml_ready.csv` (test 1.455): random_forest: F1=0.9485, linear_svm: F1=0.9430, logistic_regression: F1=0.9358. Latest cross-domain VI evaluation của traditional models thấp hơn vì đổi miền dữ liệu, không phải do train lại.

# 4. Threshold Analysis

Xem [final_threshold_summary.md](final_threshold_summary.md). XLM-R dùng 0.37 cho evaluation/warning và 0.77 cho block bảo thủ; 0.16 chỉ phù hợp security/F2 study.

# 5. Best Model Recommendation

1. Primary: **RoBERTa v5 VI**, vì precision/recall/F1 gần 0.993, AUC gần 1 và artifact đầy đủ.
2. Auxiliary: **XLM-R v5 VI warning-only/multilingual backup**.
3. Baseline: Logistic Regression, Linear SVM, Random Forest.
4. DistilBERT v4: historical/deprecated.
5. Nên tiếp tục XLM-R với dữ liệu multilingual thật, hard negatives và từng nhóm code-switching/sandwich/indirect; threshold alone không đủ.

# 6. Error Analysis Summary

Xem [final_error_analysis.md](final_error_analysis.md). XLM block 0.77: FP `39`, FN `366`; FN gồm code-switching `143`, sandwich `130`, direct VI `47`, obfuscated `23`, indirect `11`, cultural `12`.

# 7. Recent Training Timeline

| Time | Event | Model | Dataset | Result | Artifacts |
| --- | --- | --- | --- | --- | --- |
| 2026-05-31 20:31 | Train 3 traditional models | LR, SVM, RF | prompt_injection_ml_ready.csv | Artifacts + metrics + 5-fold CV | models/*_model.joblib; reports/metrics.json |
| 2026-06-13 23:40 | Complete v4 | XLM-R v4 | jayavibhav/prompt-injection | Valid checkpoint; head-only due 4GB VRAM OOM | models/transformers/xlm_roberta_v4 |
| 2026-06-14 03:41 | Complete v4 | RoBERTa v4 | jayavibhav/prompt-injection | Full fine-tune; HF test F1 0.9935 | models/transformers/roberta_v4 |
| 2026-06-14 05:06 | Complete v4 | DistilBERT v4 | jayavibhav/prompt-injection | Full fine-tune; later deprecated | models/deprecated/distilbert_20260620_014520 |
| 2026-06-20 03:20 | Continue fine-tune VI | RoBERTa v5 VI | replay + VI/mixed | Full encoder; VI test F1 0.9927, FP/FN 5/5 | models/transformers/roberta_v5_vi; reports/transformer_v5_vi_results.json |
| 2026-06-20 14:25 | Continue fine-tune VI | XLM-R v5 VI | replay + VI/mixed | Full encoder; old th 0.16 F1 0.7322 | models/transformers/xlm_roberta_v5_vi; logs/xlm_roberta_v5_vi_sgd_resume_20260620_130951.out.log |
| 2026-06-21 00:13 | Threshold sweep, no retrain | 5 current models | VI validation/test | Separated balanced evaluation and production block | models/thresholds.json; reports/threshold_optimization_summary.* |

# 8. Files Generated for Reporting

- `final_progress_report.md`, `.html`, `.pdf`
- `final_model_comparison.csv`, `.json`
- `final_threshold_summary.md`
- `final_error_analysis.md`
- `final_model_inventory.csv`
- `final_recent_training_timeline.md`

# 9. Safety Check

- Không train, xóa hoặc ghi đè artifact cũ.
- Mọi model trong inventory được kiểm tra weights/model và vectorizer hoặc tokenizer/config.
- V5 metrics dùng VI test; v4 historical dùng HF test hoặc VI test được ghi rõ; baseline gốc dùng split 1.455 riêng.
- Transformer configs hiện hành map đúng `0=SAFE`, `1=INJECTION`.
- Transformer v5 có runtime thresholds mới; traditional detector còn đọc file threshold cũ và cần đồng bộ ở task riêng.

# 10. Kết luận 3 dòng

1. Model chính: **RoBERTa v5 VI**.
2. Model phụ: **XLM-RoBERTa v5 VI warning-only**.
3. Tiếp theo: **hợp nhất runtime threshold config và cải thiện multilingual hard negatives cho XLM-R**.
