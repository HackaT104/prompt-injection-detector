# Threshold Transformer v5 tiếng Việt

Quét 0.01 đến 0.99 trên validation, chọn theo F2. Test không tham gia calibration.

| Model | Evaluation | Warn | Block | Precision | Recall | F1 | F2 | TN | FP | FN | TP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| roberta_v5_vi | 0.01 | 0.30 | 0.45 | 1.0000 | 0.9958 | 0.9979 | 0.9966 | 283 | 0 | 1 | 236 |
| xlm_roberta_v5_vi | 0.16 | 0.30 | 0.85 | 0.6107 | 0.9662 | 0.7484 | 0.8655 | 137 | 146 | 8 | 229 |

- evaluation_threshold: probability thành nhãn cho report.
- warn_threshold: runtime bắt đầu cảnh báo.
- block_threshold: luôn cao hơn warn và dùng để chặn.
