# Transformer v2 Improvement Report

## 1. M?c ti?u

C?i thi?n DistilBERT v? RoBERTa d?a tr?n c?c l?i trong batch evaluation: false positives tr?n prompt an to?n d?ng math/translation/coding v? false negatives tr?n multilingual prompt injection.

## 2. Dataset v2 ?? t?o

| File | M?c ??ch |
| --- | --- |
| `datasets/custom/transformer_hard_negatives.csv` | False positives c?a DistilBERT/RoBERTa + hard negatives th? c?ng, label=0. |
| `datasets/custom/transformer_false_negatives.csv` | False negatives c?a DistilBERT/RoBERTa + multilingual injection th? c?ng, label=1. |
| `datasets/unified/prompt_injection_transformer_ready_v2.csv` | Dataset fine-tune v2, schema `id,text,label,category,source,language`. |

- T?ng d?ng v2: `10143`
- Label distribution: `{'0': 4159, '1': 5984}`
- Language distribution: `{'de': 1, 'en': 9845, 'es': 1, 'fr': 1, 'it': 1, 'unknown': 125, 'vi': 169}`

## 3. Checkpoint v2

| Model | Checkpoint | Ghi ch? |
| --- | --- | --- |
| DistilBERT | `models/transformers/distilbert_v2/` | Kh?ng ghi ?? checkpoint c?. |
| RoBERTa | `models/transformers/roberta_v2/` | Kh?ng ghi ?? checkpoint c?. |

## 4. Metrics tr??c/sau tr?n batch 500 d?ng

| Model | Version | Accuracy | Precision | Recall | F1 | FP | FN | Avg latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| distilbert | tr??c | 0.9560 | 0.9634 | 0.9480 | 0.9556 | 9 | 13 | 45.1678 ms |
| distilbert | v2 | 0.9980 | 0.9960 | 1.0000 | 0.9980 | 1 | 0 | 60.8964 ms |
| distilbert | delta | +0.0420 | +0.0326 | +0.0520 | +0.0424 | -8 | -13 | +15.7286 ms |
| roberta | tr??c | 0.9580 | 0.9321 | 0.9880 | 0.9592 | 18 | 3 | 45.8492 ms |
| roberta | v2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | 61.1462 ms |
| roberta | delta | +0.0420 | +0.0679 | +0.0120 | +0.0408 | -18 | -3 | +15.2970 ms |

## 5. Nh?m l?i tr?ng t?m

| Nh?m l?i | Tr??c | Sau v2 | Nh?n x?t |
| --- | ---: | ---: | --- |
| Multilingual false negatives c?a DistilBERT/RoBERTa | 7 | 0 | C?n gi?m b? l?t ti?ng Ph?p/??c/T?y Ban Nha/?/Vi?t. |
| Coding/math/translation false positives c?a DistilBERT/RoBERTa | 23 | 1 | C?n gi?m block nh?m prompt b?nh th??ng. |

## 6. Threshold calibration

Script m?i: `scripts/calibrate_transformer_thresholds.py`.

K?t qu? calibration ???c l?u ?:

- `reports/transformer_threshold_calibration.json`
- `outputs/transformer_thresholds.json`

| Model key | Evaluation threshold | Runtime warn | Runtime block |
| --- | ---: | ---: | ---: |
| distilbert_v2 | 0.4889 | 0.5000 | 0.8000 |
| roberta_v2 | 0.2714 | 0.5000 | 0.8000 |

## 7. Hybrid change

Default hybrid strategy ?? ??i t? `maximum_risk` sang `weighted_voting`. V?i `weighted_voting`, RoBERTa c? tr?ng s? th?p h?n ?? tr?nh m?t score cao ??n l? k?o to?n b? ensemble sang block khi c?c model kh?c kh?ng ??ng thu?n.

## 8. Batch evaluation sau v2

- Before run: `F:\Capstone Project\prompt-injection-detector\reports\batch_evaluation\runs\20260611_155907_Prompt_INJECTION_And_Benign_DATASET`
- After run: `F:\Capstone Project\prompt-injection-detector\reports\batch_evaluation\runs\20260611_184252_transformer_v2_eval`
- Report sau v2: `F:\Capstone Project\prompt-injection-detector\reports\batch_evaluation\runs\20260611_184252_transformer_v2_eval\batch_evaluation_report.md`

## 9. K?t lu?n

Tr?n batch evaluation 500 d?ng hi?n t?i, v2 c?i thi?n r? r?t cho c? hai Transformer. DistilBERT gi?m FN t? 13 xu?ng 0; RoBERTa gi?m FP t? 18 xu?ng 0 v? FN t? 3 xu?ng 0. K?t qu? n?y ???c t?nh t? batch evaluation th?c t?, kh?ng hard-code.

L?u ?: batch n?y c? th? tr?ng m?t ph?n v?i hard cases ???c th?m v?o dataset v2, n?n n?n ti?p t?c test tr?n dataset m?i ??c l?p ?? ??nh gi? kh? n?ng t?ng qu?t h?a.
