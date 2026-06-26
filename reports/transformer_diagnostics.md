# Transformer Diagnostics

Report kiểm tra inference, label mapping, tokenizer, dataset split, score distribution và lỗi FP/FN.

## Test Dataset

- Path: `F:\Capstone Project\prompt-injection-detector\datasets\test\Prompt_INJECTION_And_Benign_DATASET.jsonl`
- Rows: `500`
- Text column: `prompt`
- Label column: `label`
- SAFE/0: `250`
- INJECTION/1: `250`

## Training Dataset / Split Check

- Path: `F:\Capstone Project\prompt-injection-detector\datasets\unified\prompt_injection_transformer_ready_v3.csv`
- Raw rows: `263754`
- Prepared rows after dedup: `263753`
- Duplicates removed by prepare: `1`
- Raw duplicate text count: `7`
- Raw duplicate text+label count: `0`
- Split sizes: `{'train': 184627, 'validation': 39563, 'test': 39563}`
- Overlap train/validation: `1`
- Overlap train/test: `2`
- Overlap validation/test: `0`

## Model Metrics

| Model | Mapping OK | Tokenizer OK | Threshold | AUC-ROC | PR-AUC | Accuracy | Precision | Recall | F1 | F2 | TN | FP | FN | TP | Decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| distilbert_v3 | True | True | 0.0700 | 0.9987 | 0.9987 | 0.9860 | 0.9802 | 0.9920 | 0.9861 | 0.9896 | 245 | 5 | 2 | 248 | needs_retrain_or_recalibration |
| roberta_v3 | True | True | 0.0100 | 0.9966 | 0.9965 | 0.9580 | 0.9791 | 0.9360 | 0.9571 | 0.9443 | 245 | 5 | 16 | 234 | needs_retrain_or_recalibration |
| xlm_roberta_v3 | True | True | 0.3000 | 0.6165 | 0.5230 | 0.5740 | 0.5405 | 0.9880 | 0.6987 | 0.8476 | 40 | 210 | 3 | 247 | needs_retrain_or_recalibration |

## Inference Checks

### distilbert_v3

- Model dir: `F:\Capstone Project\prompt-injection-detector\models\transformers\distilbert_v3`
- Runtime tokenizer: `DistilBertTokenizerFast`
- Config tokenizer: `DistilBertTokenizer`
- Model type: `distilbert`
- Risk score source: `softmax(logits)[:, 1]`
- First logits: `[-7.450012683868408, 7.716437339782715]`
- First probabilities: `[2.589966641153296e-07, 0.9999997615814209]`
- First probability sum: `1.000000020578085`

### roberta_v3

- Model dir: `F:\Capstone Project\prompt-injection-detector\models\transformers\roberta_v3`
- Runtime tokenizer: `RobertaTokenizerFast`
- Config tokenizer: `RobertaTokenizer`
- Model type: `roberta`
- Risk score source: `softmax(logits)[:, 1]`
- First logits: `[-6.434103965759277, 5.559314727783203]`
- First probabilities: `[6.184744052006863e-06, 0.9999938011169434]`
- First probability sum: `0.9999999858609954`

### xlm_roberta_v3

- Model dir: `F:\Capstone Project\prompt-injection-detector\models\transformers\xlm_roberta_v3`
- Runtime tokenizer: `XLMRobertaTokenizerFast`
- Config tokenizer: `XLMRobertaTokenizer`
- Model type: `xlm-roberta`
- Risk score source: `softmax(logits)[:, 1]`
- First logits: `[-0.3115329146385193, 0.418770968914032]`
- First probabilities: `[0.32512804865837097, 0.6748719811439514]`
- First probability sum: `1.0000000298023224`

## Output Files

- Score distribution: `reports/transformer_score_distribution.csv`
- Error cases: `reports/transformer_error_cases.csv`