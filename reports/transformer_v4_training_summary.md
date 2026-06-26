# Transformer v4 Training Summary

Generated at: `2026-06-14T05:39:25`

## Dataset mới

- Hugging Face dataset: `jayavibhav/prompt-injection`
- Text column: `text`
- Label column: `label`, trong đó `0 = SAFE/BENIGN`, `1 = PROMPT INJECTION`
- Rows loaded ban đầu: `327154`
- Rows sau khi lọc rỗng và deduplicate theo `text`: `326986`
- Split đã lưu:
  - Train: `261588`
  - Validation: `32699`
  - Test: `32699`
- Label distribution sau deduplicate: SAFE/0=`165542`, INJECTION/1=`161444`
- Test split label distribution: SAFE/0=`16555`, INJECTION/1=`16144`

## Backup và checkpoint

- Backup trước khi train v4: `F:\Capstone Project\prompt-injection-detector\models\backup_before_v4_20260613_213045`
- XLM-RoBERTa v4: `models/transformers/xlm_roberta_v4` và junction `models/xlm_roberta_v4`
- RoBERTa v4: `models/transformers/roberta_v4` và junction `models/roberta_v4`
- DistilBERT v4: `models/transformers/distilbert_v4` và junction `models/distilbert_v4`

## Cấu hình train thực tế

- XLM-RoBERTa v4 được ưu tiên train trước. Full fine-tune bị CUDA OOM trên RTX 3050 4GB, nên đã dùng partial fine-tune/head-only với toàn bộ encoder bị freeze để hoàn tất checkpoint hợp lệ.
- RoBERTa v4 fine-tune full từ `roberta-base`, 3 epoch, batch 16, max_length 128, learning_rate 2e-5, warmup_ratio 0.1, metric F2.
- DistilBERT v4 fine-tune full từ `distilbert-base-uncased`, 3 epoch, batch 32, max_length 128, learning_rate 3e-5, warmup_ratio 0.1, metric F2.

## Full Evaluation - HF Test Split

| Model | Accuracy | Precision | Recall | F1 | F2 | ROC-AUC | AP/PR-AUC | TP | FP | TN | FN | Eval Th | Warn | Block |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic_regression | 0.4941 | 0.4939 | 1.0000 | 0.6612 | 0.8299 | 0.5449 | 0.5378 | 16144 | 16543 | 12 | 0 | 0.1400 | 0.3000 | 0.5000 |
| linear_svm | 0.4941 | 0.4939 | 0.9999 | 0.6612 | 0.8299 | 0.5443 | 0.5348 | 16143 | 16542 | 13 | 1 | 0.0100 | 0.3000 | 0.5000 |
| random_forest | 0.4951 | 0.4944 | 0.9998 | 0.6616 | 0.8301 | 0.5578 | 0.5361 | 16141 | 16508 | 47 | 3 | 0.0900 | 0.3000 | 0.5000 |
| distilbert_v4 | 0.9939 | 0.9923 | 0.9955 | 0.9939 | 0.9948 | 0.9996 | 0.9997 | 16071 | 125 | 16430 | 73 | 0.8100 | 0.8100 | 0.9500 |
| roberta_v4 | 0.9936 | 0.9902 | 0.9970 | 0.9935 | 0.9956 | 0.9997 | 0.9997 | 16095 | 160 | 16395 | 49 | 0.1800 | 0.3000 | 0.4500 |
| xlm_roberta_v4 | 0.8526 | 0.7914 | 0.9526 | 0.8645 | 0.9153 | 0.9526 | 0.9525 | 15378 | 4053 | 12502 | 766 | 0.2700 | 0.3000 | 0.6800 |

## Evaluation trên dataset test cũ

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | AP/PR-AUC | TP | FP | TN | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic_regression | 0.8800 | 0.8065 | 1.0000 | 0.8929 | 0.9993 | 0.9993 | 250 | 60 | 190 | 0 |
| linear_svm | 0.9720 | 0.9470 | 1.0000 | 0.9728 | 0.9997 | 0.9996 | 250 | 14 | 236 | 0 |
| random_forest | 0.9760 | 0.9542 | 1.0000 | 0.9766 | 0.9997 | 0.9997 | 250 | 12 | 238 | 0 |
| distilbert_v4 | 0.9440 | 0.9081 | 0.9880 | 0.9464 | 0.9918 | 0.9911 | 247 | 25 | 225 | 3 |
| roberta_v4 | 0.9120 | 0.8576 | 0.9880 | 0.9182 | 0.9916 | 0.9920 | 247 | 41 | 209 | 3 |
| xlm_roberta_v4 | 0.7120 | 0.7070 | 0.7240 | 0.7154 | 0.8017 | 0.7667 | 181 | 75 | 175 | 69 |

## Nhận xét kỹ thuật

- RoBERTa v4 và DistilBERT v4 đạt mục tiêu trên HF test split mới: F1, F2, recall và ROC-AUC đều cao.
- XLM-RoBERTa v4 cải thiện rõ so với v3 cũ nhưng chưa đạt mục tiêu F1/F2 >= 0.95. Nguyên nhân nghi ngờ: full fine-tune bị giới hạn bởi 4GB VRAM, checkpoint hiện tại chỉ partial fine-tune/head-only; dataset cũng có xu hướng tiếng Anh nhiều nên lợi thế multilingual chưa phát huy đầy đủ.
- Trên dataset test cũ, các Transformer v4 vẫn có false positive ở prompt SAFE dạng toán, dịch và coding. Đây là tín hiệu cần bổ sung hard negative hoặc fine-tune thêm với tập SAFE kiểu tác vụ phổ thông.
- Ba model TF-IDF truyền thống không được train lại trong bước này. Khi evaluate trên HF test split mới, domain lệch làm precision rất thấp; không nên dùng số HF này để kết luận chúng kém trên dataset train cũ.

## File liên quan

- `reports/transformer_v4_full_evaluation.csv`
- `reports/error_cases_transformer_v4.csv`
- `reports/score_distribution_transformer_v4.csv`
- `reports/threshold_summary.md`
- `models/thresholds.json`
