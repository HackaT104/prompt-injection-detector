# Direct Prompt Injection External Benchmark Summary

## Dataset preparation

- `deepset`: status=`success`, rows=`662`, labels=`{'0': 399, '1': 263}`.
- `neuralchemy`: status=`success`, rows=`9145`, labels=`{'1': 5670, '0': 3475}`.
- `rogue_security`: status=`error`, rows=`0`, labels=`{}`. error=Could not load rogue-security/prompt-injections-benchmark with configs ['<default>']: DatasetNotFoundError: Dataset 'rogue-security/prompt-injections-benchmark' is a gated dataset on the Hub. You must be authenticated to access it.
- `cyberec`: status=`success`, rows=`54978`, labels=`{'0': 26410, '1': 28568}`.
- `all`: rows=`64785`, dataset_distribution=`{'cyberec': 54978, 'deepset': 662, 'neuralchemy': 9145}`, labels=`{'0': 30284, '1': 34501}`.

## Comparison table

| Dataset | Model | Threshold | Accuracy | Precision | Recall | F1 | F2 | ROC-AUC | PR-AUC | FP | FN | Rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all | Logistic Regression | 0.1800 | 0.5790 | 0.5586 | 0.9989 | 0.7165 | 0.8629 | 0.6230 | 0.6764 | 27236 | 37 | 64785 |
| all | Random Forest | 0.1000 | 0.5839 | 0.5614 | 0.9993 | 0.7190 | 0.8645 | 0.6475 | 0.6925 | 26933 | 23 | 64785 |
| all | RoBERTa | 0.0100 | 0.9364 | 0.9732 | 0.9055 | 0.9381 | 0.9182 | 0.9834 | 0.9878 | 861 | 3262 | 64785 |
| all | XLM-RoBERTa | 0.0900 | 0.7086 | 0.6553 | 0.9554 | 0.7774 | 0.8752 | 0.8914 | 0.9078 | 17342 | 1539 | 64785 |
| deepset | Logistic Regression | 0.4500 | 0.8897 | 0.8006 | 0.9620 | 0.8739 | 0.9247 | 0.9697 | 0.9511 | 63 | 10 | 662 |
| deepset | Random Forest | 0.3300 | 0.9592 | 0.9126 | 0.9924 | 0.9508 | 0.9753 | 0.9911 | 0.9866 | 25 | 2 | 662 |
| deepset | RoBERTa | 0.0100 | 0.9592 | 0.9917 | 0.9049 | 0.9463 | 0.9211 | 0.9834 | 0.9844 | 2 | 25 | 662 |
| deepset | XLM-RoBERTa | 0.0100 | 0.4728 | 0.4243 | 0.9163 | 0.5800 | 0.7438 | 0.7134 | 0.6768 | 327 | 22 | 662 |

## Analysis

1. Logistic Regression generalization: on `all`, F1=`0.7165`, F2=`0.8629`, ROC-AUC=`0.6230`. It catches injections aggressively but has high FP on the combined benchmark.
2. Random Forest vs Logistic Regression: Random Forest is slightly better on `all` by F2 (`0.8645` vs `0.8629`) and ROC-AUC (`0.6475` vs `0.6230`).
3. RoBERTa vs traditional ML: RoBERTa is much better by F1/ROC-AUC and has far fewer FP on `all`; F1=`0.9381`, ROC-AUC=`0.9834`.
4. XLM-RoBERTa vs RoBERTa: XLM-RoBERTa has higher recall on `all` (`0.9554` vs `0.9055`), but RoBERTa is stronger by F1/F2/precision and FP count.
5. Highest recall on `all`: `Random Forest` with recall `0.9993`.
6. Lowest false positives on `all`: `RoBERTa` with FP `861`.
7. Most missed prompt injections on `all`: `RoBERTa` with FN `3262`.
8. Memorization/dataset-overlap signal: neuralchemy is included because the user requested it, but it may overlap with prior project training/evaluation sources. Treat high neuralchemy performance as possible leakage unless a separate provenance audit confirms no overlap.
9. Hardest source dataset by F1 under the `all` threshold:
   - Logistic Regression: deepset (F1=0.5930, F2=0.7846, FP=361, FN=0)
   - Random Forest: cyberec (F1=0.6846, F2=0.8441, FP=26288, FN=21)
   - RoBERTa: neuralchemy (F1=0.9186, F2=0.8888, FP=137, FN=737)
   - XLM-RoBERTa: deepset (F1=0.6030, F2=0.6085, FP=110, FN=102)
10. Recommendation: combine multiple Direct datasets for continued fine-tuning only after de-duplication/provenance checks and a strict held-out external split. RoBERTa is the best candidate to continue fine-tuning; XLM-RoBERTa needs FP calibration and additional hard negatives.

Overall best on `all` by F1: `RoBERTa` with F1 `0.9381`.
Overall best on `all` by F2: `RoBERTa` with F2 `0.9182`.
Highest recall across all completed rows: `Random Forest` on `all` with recall `0.9993`.
Lowest FP across all completed rows: `RoBERTa` on `deepset` with FP `2`.
Most FN across all completed rows: `RoBERTa` on `all` with FN `3262`.

Threshold note: rows with `threshold_mode=auto_test_sweep` use threshold sweeping on the test/evaluation set and should not be interpreted as strict held-out performance.
