# Threshold recommendation for XLM-RoBERTa v5 VI

## Kết luận ngắn

Threshold cũ `0.16` được chọn vì tối ưu F2/Recall trên validation, nên recall cao nhưng false positive rất lớn. Ngưỡng này phù hợp cho nghiên cứu/security sweep, không phù hợp làm runtime block threshold.

## Threshold cũ và mới

| Loại | Cũ | Mới | Ghi chú |
| --- | ---: | ---: | --- |
| evaluation_threshold | 0.1600 | 0.3700 | Balanced mode: F1 cao nhất |
| runtime_warn_threshold | 0.3000 | 0.3700 | Cảnh báo, không block |
| runtime_block_threshold | 0.8500 | 0.7700 | Production mode: high_precision_block_precision_ge_0.90 |

## Metric trên test set

| Metric | Tại threshold cũ | Tại block threshold mới | Thay đổi |
| --- | ---: | ---: | ---: |
| accuracy | 0.6820 | 0.7300 | 0.0480 |
| precision | 0.5933 | 0.8901 | 0.2969 |
| recall | 0.9560 | 0.4633 | -0.4927 |
| f1 | 0.7322 | 0.6095 | -0.1227 |
| f2 | 0.8518 | 0.5125 | -0.3394 |
| roc_auc | 0.8703 | 0.8703 | 0.0000 |
| average_precision | 0.8427 | 0.8427 | 0.0000 |
| FP | 447 | 39 | -408 |
| FN | 30 | 366 | 336 |
| TP | 652 | 316 | -336 |
| TN | 371 | 779 | 408 |

## Ba chế độ chọn threshold trên validation

| Mode | Threshold | Precision | Recall | F1 | F2 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Security | 0.1600 | 0.6107 | 0.9662 | 0.7484 | 0.8655 | 146 | 8 |
| Balanced | 0.3700 | 0.7163 | 0.8523 | 0.7784 | 0.8211 | 80 | 35 |
| Production | 0.5000 | 0.7759 | 0.7595 | 0.7676 | 0.7627 | 52 | 57 |

## Khuyến nghị

- Không dùng XLM-RoBERTa v5 VI làm model chính để block tự động vì precision vẫn chưa đủ cao.
- Có thể dùng ở chế độ `warning only` hoặc model so sánh đa ngôn ngữ.
- Model chính cho demo/runtime vẫn nên là RoBERTa v5 VI.
