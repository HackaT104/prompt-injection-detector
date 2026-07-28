# Báo cáo baseline ba Transformer

> Đây là benchmark nghiên cứu bằng raw softmax probability. Báo cáo không tuyên bố model production, không calibration, không threshold sweep, không fine-tune và không thay đổi model runtime.

## Cấu hình thống nhất

- Dataset: `neuralchemy/Prompt-injection-dataset`
- Config: `full`
- Split duy nhất: `test`
- Số mẫu: `942`
- Phân phối nhãn: `{'SAFE': 390, 'INJECTION': 552}`
- Seed: `42`
- Max length: `256`
- Batch size: `16`
- Threshold baseline cố định: `0.5`
- Label mapping: `0 = SAFE`, `1 = INJECTION`
- Padding: dynamic theo từng batch
- Score: raw softmax probability của label 1
- PR-AUC: average precision trên raw injection score
- Inference time: tổng thời gian tokenize + transfer + forward của các batch sau warm-up; không gồm thời gian load model
- `latency_ms` của từng sample là batch latency chia đều cho số sample trong batch

## So sánh metrics

| Model | Accuracy | Bal. Acc. | Precision | Recall | Specificity | F1 | F2 | PR-AUC | ROC-AUC | FPR | FNR | MCC | Samples/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RoBERTa v4 | 0.845011 | 0.831633 | 0.839465 | 0.909420 | 0.753846 | 0.873043 | 0.894512 | 0.929334 | 0.914662 | 0.246154 | 0.090580 | 0.678516 | 481.05 |
| RoBERTa v5 VI | 0.867304 | 0.882260 | 0.973392 | 0.795290 | 0.969231 | 0.875374 | 0.825498 | 0.978863 | 0.973532 | 0.030769 | 0.204710 | 0.753810 | 569.36 |
| XLM-RoBERTa v5 VI | 0.594480 | 0.618241 | 0.736111 | 0.480072 | 0.756410 | 0.581140 | 0.515966 | 0.715567 | 0.611494 | 0.243590 | 0.519928 | 0.239711 | 537.82 |

## Chi tiết từng model

### RoBERTa v4

- Checkpoint: `/content/prompt-injection-detector/models/transformers/roberta_v4`
- Device: `cuda`
- FP16 autocast: `True`
- Sample count: `942`
- Inference time: `1.958211` giây
- Samples/second: `481.051309`
- Mean latency: `2.078780` ms/sample
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[294, 96], [50, 502]]`

### RoBERTa v5 VI

- Checkpoint: `/content/prompt-injection-detector/models/transformers/roberta_v5_vi`
- Device: `cuda`
- FP16 autocast: `True`
- Sample count: `942`
- Inference time: `1.654481` giây
- Samples/second: `569.362901`
- Mean latency: `1.756349` ms/sample
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[378, 12], [113, 439]]`

### XLM-RoBERTa v5 VI

- Checkpoint: `/content/prompt-injection-detector/models/transformers/xlm_roberta_v5_vi`
- Device: `cuda`
- FP16 autocast: `True`
- Sample count: `942`
- Inference time: `1.751528` giây
- Samples/second: `537.816149`
- Mean latency: `1.859371` ms/sample
- Confusion matrix `[[TN, FP], [FN, TP]]`: `[[295, 95], [287, 265]]`

## Recommended baseline model

- Model: **RoBERTa v4** (`roberta_v4`)
- Reason: RoBERTa v4 có F2 cao nhất (0.894512) tại raw softmax threshold 0.5; tie-break lần lượt dùng PR-AUC (0.929334), MCC (0.678516), balanced accuracy và throughput. Đây chỉ là baseline cho thí nghiệm tiếp theo, không phải khuyến nghị production.

Lựa chọn này chỉ dùng để xác định baseline thống nhất trước thí nghiệm tiếp theo. Không thay model runtime và không chứng minh mức sẵn sàng production.

## Artifact

- `model_comparison.csv`
- `model_comparison.json`
- `predictions_<model>.csv`
- `false_positives_<model>.csv`
- `false_negatives_<model>.csv`
- `confusion_matrix_<model>.png`
- `precision_recall_curve.png`
- `roc_curve.png`
