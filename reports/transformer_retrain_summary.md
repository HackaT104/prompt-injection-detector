# Transformer Retrain Summary

## Trang thai kiem tra

Da chay diagnostics cho 3 model Transformer v3 tren `datasets/test/Prompt_INJECTION_And_Benign_DATASET.jsonl`.

Ket qua quan trong:

- Softmax dung: risk_score duoc lay tu `softmax(logits)[:, 1]`.
- Label mapping dung: `0 = SAFE`, `1 = INJECTION`.
- Tokenizer dung:
  - DistilBERT v3 dung `DistilBertTokenizerFast`.
  - RoBERTa v3 dung `RobertaTokenizerFast`.
  - XLM-RoBERTa v3 dung `XLMRobertaTokenizerFast`.
- Khong phat hien viec lay nham logits tho, nham class SAFE/0, hay nham tokenizer.

## Ket luan theo tieu chi giu model

| Model | F1 | Recall | Precision | AUC-ROC | Threshold | Ket luan |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| DistilBERT v3 | 0.9861 | 0.9920 | 0.9802 | 0.9987 | 0.0700 | Metrics dat, threshold bat thuong; can recalibration hoac hard-case fine-tune. |
| RoBERTa v3 | 0.9571 | 0.9360 | 0.9791 | 0.9966 | 0.0100 | Khong dat F1/Recall va threshold bat thuong; can fine-tune lai. |
| XLM-RoBERTa v3 | 0.6987 | 0.9880 | 0.5405 | 0.6165 | 0.3000 | Precision/AUC thap, FP cao; can fine-tune lai voi multilingual data tot hon. |

## Vi sao chua fine-tune de len checkpoint moi trong buoc nay

Diagnostics cho thay loi khong nam o inference, softmax, label mapping hay tokenizer. Loi nam o hanh vi model/score distribution va chat luong du lieu cho cac hard cases.

Khong nen fine-tune truc tiep tren `datasets/test/Prompt_INJECTION_And_Benign_DATASET.jsonl`, vi day la test set dang dung de bao cao. Mot so file custom hard-negative/false-negative hien co cung co nguon tu batch false positive/false negative cua cung family test dataset, nen neu dung truc tiep de train roi evaluate tren test nay se tao data leakage.

XLM-RoBERTa v3 hien tai la partial fine-tune do gioi han VRAM 4GB. Model nay can retrain tren Colab/GPU tot hon hoac dataset multilingual can bang hon. Neu retrain tren may local hien tai, nguy co CUDA OOM cao.

## Lenh fine-tune de chay tren Colab/GPU

Backup checkpoint cu truoc khi train:

```powershell
Copy-Item -Recurse models\transformers\distilbert_v3 models\transformers\backups\distilbert_v3_before_retrain
Copy-Item -Recurse models\transformers\roberta_v3 models\transformers\backups\roberta_v3_before_retrain
Copy-Item -Recurse models\transformers\xlm_roberta_v3 models\transformers\backups\xlm_roberta_v3_before_retrain
```

Fine-tune DistilBERT candidate:

```bash
python src/train_transformers.py --model distilbert_v3 --dataset datasets/unified/prompt_injection_transformer_ready_v3.csv --checkpoint-name distilbert_v4_candidate --epochs 3 --batch-size 16 --learning-rate 2e-5 --weight-decay 0.01 --gradient-accumulation-steps 2 --use-cuda
```

Fine-tune RoBERTa candidate:

```bash
python src/train_transformers.py --model roberta_v3 --dataset datasets/unified/prompt_injection_transformer_ready_v3.csv --checkpoint-name roberta_v4_candidate --epochs 3 --batch-size 8 --learning-rate 1e-5 --weight-decay 0.01 --gradient-accumulation-steps 2 --gradient-checkpointing --use-cuda
```

Fine-tune XLM-RoBERTa candidate:

```bash
python src/train_transformers.py --model xlm_roberta_v3 --dataset datasets/unified/prompt_injection_transformer_ready_v3.csv --checkpoint-name xlm_roberta_v4_candidate --epochs 4 --batch-size 8 --learning-rate 1e-5 --weight-decay 0.01 --gradient-accumulation-steps 2 --gradient-checkpointing --freeze-encoder-layers 8 --use-cuda
```

Sau khi co checkpoint moi:

```bash
python scripts/calibrate_thresholds.py --dataset datasets/test/Prompt_INJECTION_And_Benign_DATASET.jsonl --models distilbert_v4_candidate roberta_v4_candidate xlm_roberta_v4_candidate --metric f1 --use-cuda
python scripts/run_dataset_evaluation.py --file datasets/test/Prompt_INJECTION_And_Benign_DATASET.jsonl --models distilbert_v4_candidate roberta_v4_candidate xlm_roberta_v4_candidate --export-csv
```

## Khuyen nghi ky thuat

- DistilBERT v3 co AUC/F1 cao, nen uu tien calibration va bo sung hard cases truoc khi full retrain.
- RoBERTa v3 can them multilingual/code-obfuscation positives va hard negatives math/translation/coding.
- XLM-RoBERTa v3 can dataset multilingual can bang hon; neu dataset chu yeu tieng Anh thi XLM-RoBERTa khong phat huy loi the va co the thua RoBERTa.
