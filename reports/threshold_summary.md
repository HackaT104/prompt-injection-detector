# Threshold Calibration Summary

Report này được tạo bằng cách load model đã train/fine-tune, lấy risk_score của class attack = 1,
quét threshold từ 0.01 đến 0.99 với step 0.01 và chọn threshold theo metric cấu hình.

## Dataset

- Path: `F:\Capstone Project\prompt-injection-detector\datasets\processed\hf_prompt_injection_test.csv`
- Rows used: `32699` / total `32699`
- Text column: `text`
- Label column: `label`
- Label distribution: SAFE/0=`16555`, INJECTION/1=`16144`

## Threshold Table

| model | best_metric | evaluation_threshold | warn_threshold | block_threshold | precision | recall | f1 | f2 | tn | fp | fn | tp |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic_regression | f2 | 0.1400 | 0.3000 | 0.5000 | 0.4939 | 1.0000 | 0.6612 | 0.8299 | 12 | 16543 | 0 | 16144 |
| linear_svm | f2 | 0.0100 | 0.3000 | 0.5000 | 0.4939 | 0.9999 | 0.6612 | 0.8299 | 13 | 16542 | 1 | 16143 |
| random_forest | f2 | 0.0900 | 0.3000 | 0.5000 | 0.4944 | 0.9998 | 0.6616 | 0.8301 | 47 | 16508 | 3 | 16141 |
| distilbert_v4 | f2 | 0.8100 | 0.8100 | 0.9500 | 0.9923 | 0.9955 | 0.9939 | 0.9948 | 16430 | 125 | 73 | 16071 |
| roberta_v4 | f2 | 0.1800 | 0.3000 | 0.4500 | 0.9902 | 0.9970 | 0.9935 | 0.9956 | 16395 | 160 | 49 | 16095 |
| xlm_roberta_v4 | f2 | 0.2700 | 0.3000 | 0.6800 | 0.7914 | 0.9526 | 0.8645 | 0.9153 | 12502 | 4053 | 766 | 15378 |

## Runtime Rule

- `risk_score < warn_threshold` -> allow
- `warn_threshold <= risk_score < block_threshold` -> warn
- `risk_score >= block_threshold` -> block

## Notes

- `evaluation_threshold` dùng cho report/evaluate.
- `warn_threshold` và `block_threshold` dùng cho API/runtime.
- `warn_threshold` luôn nhỏ hơn `block_threshold`.
- File chi tiết theo từng threshold: `reports/threshold_search_report.csv`.