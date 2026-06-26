# Transformer v3 Geekyrakshit Report

## Dataset

- Dataset: `geekyrakshit/prompt-injection-dataset`
- File training: `datasets/unified/prompt_injection_transformer_ready_v3.csv`
- Tổng dòng: 263,754
- Train: 184,627
- Validation: 39,563
- Test: 39,563
- Label: `0 = SAFE`, `1 = INJECTION`

## Kết Quả Test

| Model | Training Mode | Accuracy | Precision | Recall | F1 | Confusion Matrix |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| DistilBERT v3 | full_finetune | 0.9759 | 0.9999 | 0.9511 | 0.9749 | `[[20082, 1], [952, 18528]]` |
| RoBERTa v3 | full_finetune | 0.9828 | 0.9998 | 0.9653 | 0.9823 | `[[20080, 3], [676, 18804]]` |
| XLM-RoBERTa v3 | partial_finetune | 0.7694 | 0.7544 | 0.7883 | 0.7710 | `[[15083, 5000], [4124, 15356]]` |

## XLM-RoBERTa v3

Full fine-tune XLM-RoBERTa trên RTX 3050 4GB bị lỗi CUDA out-of-memory/CUBLAS ở nhiều cấu hình. Để hoàn tất checkpoint chạy được, project dùng chế độ memory-safe partial fine-tune:

- Base model: `xlm-roberta-base`
- Checkpoint: `models/transformers/xlm_roberta_v3`
- Max length: 64
- Batch size: 8
- Gradient accumulation: 4
- Optimizer: Adafactor
- Gradient checkpointing: bật
- Freeze: embeddings và 10 encoder layer đầu
- Train: 2 encoder layer cuối và classification head

Kết quả này giúp XLM-RoBERTa chạy được trong hệ thống và hỗ trợ multilingual baseline, nhưng chất lượng thấp hơn DistilBERT v3 và RoBERTa v3. Không nên chọn XLM-RoBERTa v3 làm model chính nếu chưa tiếp tục fine-tune trên GPU mạnh hơn hoặc dùng kỹ thuật PEFT/LoRA ổn định hơn.

## Thresholds

Thresholds đã được lưu tại `models/thresholds.json`.

| Model | Evaluation Threshold | Runtime Warn | Runtime Block |
| --- | ---: | ---: | ---: |
| DistilBERT v3 | 0.9999986887 | 0.9999986887 | 0.9999986887 |
| RoBERTa v3 | 0.9999859333 | 0.9999859333 | 0.9999859333 |
| XLM-RoBERTa v3 | 0.4709472656 | 0.5000 | 0.8000 |

## Khuyến Nghị Demo

- Model chính: Logistic Regression hoặc RoBERTa v3, tùy mục tiêu demo tốc độ hay chất lượng.
- Model so sánh multilingual: XLM-RoBERTa v3.
- Hybrid nên tiếp tục dùng `random_forest + xlm_roberta_v3 + weighted_voting + rule-based`, nhưng cần giảm trọng số XLM-RoBERTa nếu thấy false positive cao.
