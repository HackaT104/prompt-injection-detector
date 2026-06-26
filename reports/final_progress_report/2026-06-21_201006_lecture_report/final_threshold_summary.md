# Tổng hợp threshold cuối cùng

Generated at: `2026-06-21T20:10:06.117556+07:00`

## Ý nghĩa

- `evaluation_threshold`: dùng chuyển probability thành nhãn khi tính metric, chọn theo F1 trên validation.
- `runtime_warn_threshold`: dùng cảnh báo, không đồng nghĩa block.
- `runtime_block_threshold`: dùng chặn thật, ưu tiên precision để giảm false positive.

| Model | Evaluation | Warn | Block | Method |
| --- | --- | --- | --- | --- |
| logistic_regression | 0.63 | 0.58 | 0.78 | maximize_f1 |
| linear_svm | 0.81 | 0.75 | 0.95 | maximize_f1 |
| random_forest | 0.62 | 0.63 | 0.83 | maximize_f1 |
| roberta_v5_vi | 0.05 | 0.05 | 0.20 | maximize_f1 |
| xlm_roberta_v5_vi | 0.37 | 0.37 | 0.77 | maximize_f1 |

## XLM-RoBERTa v5 VI

- Cũ `0.16`: precision `0.5933`, recall `0.9560`, F1 `0.7322`, FP `447`, FN `30`. Threshold này ưu tiên F2/Recall.
- Balanced `0.37`: precision `0.7185`, recall `0.8534`, F1 `0.7802`, FP `228`, FN `100`.
- Runtime block `0.77`: precision `0.8901`, recall `0.4633`, F1 `0.6095`, FP `39`, FN `366`.
- Block 0.77 giảm 408 FP nhưng tăng 336 FN so với 0.16. Kết luận: warning/auxiliary, chưa làm model block chính.

## Cảnh báo runtime config

- `models/thresholds.json` có threshold mới cho năm model hiện hành.
- Transformer v5 fallback đúng sang file này vì `models/transformer_thresholds.json` không có key v5.
- `src/detector.py` của model truyền thống vẫn ưu tiên `models/model_thresholds.json` cũ: LR `0.14/0.30/0.50`, SVM `0.01/0.30/0.50`, RF `0.09/0.30/0.50`.
- Source còn hard-coded `0.50/0.80` ở vai trò fallback/rule/hybrid gate. Báo cáo này chỉ ghi nhận, không sửa code.
