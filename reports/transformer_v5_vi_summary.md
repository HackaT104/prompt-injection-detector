# Tổng kết migration Transformer v4 sang v5 tiếng Việt

V5 tiếp tục từ checkpoint v4 với replay 80/20; threshold v5 được chọn trên validation.

## Full comparison

| Family | Model | Split | Accuracy | Precision | Recall | F1 | F2 | ROC-AUC | PR-AUC | FP | FN |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| roberta | roberta_v5_vi | v5_vi_validation | 0.9981 | 1.0000 | 0.9958 | 0.9979 | 0.9966 | 1.0000 | 1.0000 | 0 | 1 |
| xlm_roberta | xlm_roberta_v5_vi | v5_vi_validation | 0.7038 | 0.6107 | 0.9662 | 0.7484 | 0.8655 | 0.8679 | 0.8505 | 146 | 8 |
| roberta | roberta_v4 | vi_test | 0.8273 | 0.7794 | 0.8651 | 0.8200 | 0.8465 | 0.9392 | 0.9311 | 167 | 92 |
| roberta | roberta_v4 | english_v4_test | 0.9934 | 0.9899 | 0.9968 | 0.9933 | 0.9954 | 0.9996 | 0.9997 | 165 | 51 |
| roberta | roberta_v5_vi | vi_test | 0.9933 | 0.9927 | 0.9927 | 0.9927 | 0.9927 | 0.9997 | 0.9996 | 5 | 5 |
| roberta | roberta_v5_vi | english_v4_test | 0.9950 | 0.9960 | 0.9938 | 0.9949 | 0.9942 | 0.9996 | 0.9997 | 64 | 100 |
| xlm_roberta | xlm_roberta_v4 | vi_test | 0.7107 | 0.6260 | 0.9032 | 0.7395 | 0.8297 | 0.8495 | 0.8180 | 368 | 66 |
| xlm_roberta | xlm_roberta_v4 | english_v4_test | 0.8483 | 0.7855 | 0.9529 | 0.8611 | 0.9140 | 0.9513 | 0.9509 | 4201 | 760 |
| xlm_roberta | xlm_roberta_v5_vi | vi_test | 0.6820 | 0.5934 | 0.9545 | 0.7319 | 0.8510 | 0.8703 | 0.8426 | 446 | 31 |
| xlm_roberta | xlm_roberta_v5_vi | english_v4_test | 0.7996 | 0.7195 | 0.9737 | 0.8275 | 0.9094 | 0.9528 | 0.9524 | 6130 | 424 |

## Delta v5 so với v4

| Family | Split | Delta F1 | Delta ROC-AUC | Delta FP | Delta FN |
| --- | --- | ---: | ---: | ---: | ---: |
| roberta | vi_test | +0.1727 | +0.0604 | -162 | -87 |
| roberta | english_v4_test | +0.0016 | -0.0001 | -101 | +49 |
| xlm_roberta | vi_test | -0.0076 | +0.0208 | +78 | -35 |
| xlm_roberta | english_v4_test | -0.0337 | +0.0015 | +1929 | -336 |

## Kết luận kỹ thuật

- RoBERTa v5 VI là Transformer mặc định: cải thiện mạnh tiếng Việt và giữ F1 tiếng Anh.
- ROC-AUC tiếng Anh của RoBERTa giảm rất nhỏ, cần theo dõi thêm trên holdout khác.
- XLM-R v5 tăng recall/ROC-AUC nhưng false positive còn cao, nên chỉ dùng làm model so sánh.
- XLM-R dùng SGD để full fine-tune trên GPU 4 GB; Adafactor OOM ở ma trận embedding 250k token.

## DistilBERT

DistilBERT đã được backup và deprecated; không còn trong runtime mặc định.

## Context-Aware Detection

Bước tiếp theo là đánh giá riêng user prompt và external context, sau đó hợp nhất direct score, indirect score và rule signals bằng policy có calibration.
