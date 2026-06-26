# XLM-RoBERTa và Geekyrakshit Dataset

## Dataset được dùng

Dataset Hugging Face:

```python
from datasets import load_dataset
ds = load_dataset("geekyrakshit/prompt-injection-dataset")
```

Dataset gốc có cột chính:

- `prompt`
- `label`

Project chuẩn hóa thành schema:

```csv
id,text,label,category,source,language
```

Quy ước nhãn:

- `0 = SAFE`
- `1 = INJECTION`

File được tạo:

- `datasets/processed/geekyrakshit_prompt_injection.csv`
- `datasets/unified/prompt_injection_transformer_ready_v3.csv`

## Vì sao thêm XLM-RoBERTa

DistilBERT và RoBERTa-base chủ yếu là model tiếng Anh. Prompt injection thực tế có thể xuất hiện bằng tiếng Việt, Pháp, Đức, Tây Ban Nha, Ý hoặc trộn nhiều ngôn ngữ. XLM-RoBERTa là Transformer multilingual, phù hợp hơn cho mục tiêu phát hiện prompt injection đa ngôn ngữ.

## Chuẩn hóa dataset

Chạy:

```powershell
cd /d "F:\Capstone Project\prompt-injection-detector"
.\.venv\Scripts\activate
python scripts\prepare_geekyrakshit_dataset.py
```

Script sẽ:

- tải `geekyrakshit/prompt-injection-dataset`
- map `prompt -> text`
- giữ `label` theo chuẩn `0/1`
- đặt `source = geekyrakshit/prompt-injection-dataset`
- đặt `category = unknown`
- tự detect `language`, nếu lỗi thì `unknown`

## Train 3 model Transformer v3

Các checkpoint mới được lưu riêng, không ghi đè checkpoint cũ:

- `models/transformers/distilbert_v3/`
- `models/transformers/roberta_v3/`
- `models/transformers/xlm_roberta_v3/`

Lệnh train:

```powershell
python src\train_transformers.py --model distilbert --dataset datasets\unified\prompt_injection_transformer_ready_v3.csv --epochs 3

python src\train_transformers.py --model roberta --dataset datasets\unified\prompt_injection_transformer_ready_v3.csv --epochs 3

python src\train_transformers.py --model xlm_roberta --dataset datasets\unified\prompt_injection_transformer_ready_v3.csv --epochs 3
```

Với GPU 4GB VRAM, nếu bị CUDA out of memory:

```powershell
python src\train_transformers.py --model xlm_roberta --dataset datasets\unified\prompt_injection_transformer_ready_v3.csv --epochs 3 --batch-size 2 --max-length 96
```

Dataset v3 hiện có thể rất lớn, nên full training 3 epochs có thể mất nhiều giờ.

## Metadata và metrics

Mỗi checkpoint sau khi train sẽ có:

- `training_metadata.json`
- `metrics.json`

Nội dung gồm:

- model name
- base model
- dataset name/path
- thời điểm train
- số epoch
- label mapping
- số dòng train/validation/test
- metrics thực tế trên test split

## Calibrate threshold

Không dùng threshold cứng cho Transformer v3. Chạy:

```powershell
python scripts\calibrate_thresholds.py --dataset datasets\unified\prompt_injection_transformer_ready_v3.csv --models distilbert_v3 roberta_v3 xlm_roberta_v3
```

Script quét threshold từ `0.05` đến `0.95`.

Chọn:

- `warn_threshold`: threshold có `recall >= 0.95` và F1 cao nhất
- `block_threshold`: threshold có `precision >= 0.95` và F1 cao nhất

Output:

- `models/thresholds.json`
- `reports/threshold_calibration_v3.json`

## Chạy evaluation

Deepset test:

```powershell
python scripts\run_dataset_evaluation.py --file datasets\custom\deepset_test.csv --models random_forest distilbert roberta xlm_roberta distilbert_v3 roberta_v3 xlm_roberta_v3 hybrid --run-name deepset_transformer_v3_eval --export-csv
```

Dataset v3:

```powershell
python scripts\run_dataset_evaluation.py --file datasets\unified\prompt_injection_transformer_ready_v3.csv --models random_forest distilbert roberta xlm_roberta distilbert_v3 roberta_v3 xlm_roberta_v3 hybrid --run-name geekyrakshit_v3_eval --export-csv
```

Dataset JSONL cũ nếu có:

```powershell
python scripts\run_dataset_evaluation.py --file "F:\Tải về\archive (1)\Prompt_INJECTION_And_Benign_DATASET.jsonl" --models random_forest distilbert roberta xlm_roberta distilbert_v3 roberta_v3 xlm_roberta_v3 hybrid --run-name transformer_v3_jsonl_eval --export-csv
```

## Chọn model trên UI

Advanced demo:

```text
http://127.0.0.1:8000/advanced-demo
```

Các model mới:

- XLM-RoBERTa
- DistilBERT v3
- RoBERTa v3
- XLM-RoBERTa v3

Hybrid mặc định khuyến nghị:

```json
{
  "traditional_model": "random_forest",
  "transformer_model": "xlm_roberta_v3",
  "strategy": "weighted_vote",
  "use_rule_based": true
}
```

## So sánh model cũ và model mới

Các model nên đưa vào report:

- DistilBERT cũ
- RoBERTa cũ
- DistilBERT v3
- RoBERTa v3
- XLM-RoBERTa v3
- Random Forest
- Hybrid

Chỉ số cần so sánh:

- Accuracy
- Precision
- Recall
- F1
- TP, FP, TN, FN
- Average latency
- Metrics theo ngôn ngữ nếu dataset có cột `language`

## Lưu ý

Không xóa DistilBERT/RoBERTa hiện có. XLM-RoBERTa được thêm như model mới để tăng khả năng multilingual detection. Full training trên dataset lớn nên chạy khi máy có đủ thời gian và nguồn điện ổn định.
