# Final Project Readiness

## Current strengths
- Có nhiều lớp detector: rule-based, TF-IDF ML, Transformer và hybrid.
- Runtime action đã thống nhất theo risk score 0.5/0.8.
- Có diagnostics endpoint để kiểm tra từng model độc lập.
- Có benchmark theo category và stress test 100 prompt.

## Current weaknesses
- RoBERTa cần được fine-tune lại với role-override và hard-negative data mới để giảm lệch dự đoán.
- Benchmark hiện là bộ kiểm thử thủ công, không thay thế evaluation trên test split lớn.
- OCR ảnh và PDF extraction chưa phải trọng tâm reliability.

## Dataset summary
- Unified rows: `10148`
- ML-ready rows: `7275`
- Transformer-ready rows: `10148`
- Label distribution: `{0: 4159, 1: 5989}`

## Model summary
- Best stress-test model by recall/F1: `logistic_regression`
- Recommended deployment model from current stress test: `logistic_regression`.
- Hybrid remains useful for high-recall blocking mode, but current max-risk strategy increases false positives.
- Future improvement: tune hybrid weighted voting and continue expanding hard-negative/role-override data.

## Report paths
- RoBERTa diagnostics: `F:\Capstone Project\prompt-injection-detector\reports\roberta_diagnostics.md`
- Category benchmark: `F:\Capstone Project\prompt-injection-detector\reports\category_benchmark.md`
- Stress test: `F:\Capstone Project\prompt-injection-detector\reports\stress_test_results.md`
