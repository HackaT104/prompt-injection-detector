# Context-Aware Hybrid Prompt Injection Detection Upgrade

## 1. Mục tiêu nâng cấp

Nâng cấp runtime detector từ phát hiện theo văn bản đơn lẻ sang Hybrid Context-Aware Detection. Hệ thống mới đánh giá cả nội dung prompt, nội dung bên ngoài không đáng tin cậy và mức độ lệch ngữ cảnh giữa `user_task` và `external_content`.

Mục tiêu chính:

- Phát hiện direct và indirect prompt injection.
- Nhận diện instruction ẩn trong tài liệu, email, website hoặc dữ liệu ngoài.
- Tránh block nhầm nội dung học thuật chỉ nhắc đến các cụm như `ignore previous instructions`.
- Trả output giải thích được với score, threshold, lý do và khuyến nghị.
- Không xóa model cũ và không phá pipeline train/test hiện có.

## 2. Kiến trúc mới

Luồng runtime mới nằm trong `src/detection/pipeline.py`:

```text
user_prompt + user_task + optional external_content
-> rule-based detection
-> optional ML model scoring
-> optional Transformer scoring
-> context-aware detection
-> final_score = max(rule_score, ml_score, transformer_score, context_risk_score)
-> compare warn/block threshold
-> explainable output
```

Các module mới:

- `src/detection/context_aware_detector.py`
- `src/detection/pipeline.py`
- `src/detection/__init__.py`

Endpoint `/detect-context-aware` đã được chuyển sang pipeline explainable mới. Endpoint upload vẫn giữ pipeline indirect cũ vì pipeline đó đang xử lý bytes/chunk/source metadata tốt hơn.

## 3. Context-aware layer

Module `ContextAwareDetector` nhận:

- `user_task`
- `external_content`
- `model_score`
- `rule_hits`

Output chính:

```json
{
  "context_mismatch": true,
  "detected_instruction": "ignore previous instructions",
  "reason": "Context mismatch: external content contains assistant-directed instruction(s) unrelated to the user task.",
  "context_risk_score": 0.94
}
```

Layer mới phát hiện các nhóm tín hiệu:

- instruction override
- system/developer/hidden instruction extraction
- data exfiltration
- jailbreak
- tool or command abuse
- biến thể tiếng Việt
- obfuscation nhẹ như spacing, underscore, leetspeak và ký tự cách quãng

Layer này cũng có benign-reference guard để tránh false positive khi nội dung chỉ là phân tích, giải thích, bài học, glossary hoặc ví dụ học thuật.

## 4. External benchmark

Benchmark mới nằm tại:

`data/external_benchmark/external_prompt_injection_benchmark.csv`

Tổng số mẫu: 105.

Các nhóm dữ liệu:

- direct injection
- indirect injection
- Vietnamese prompt injection
- English prompt injection
- obfuscated attack
- benign but suspicious
- safe normal text
- multilingual mixed attack

Các cột:

- `id`
- `language`
- `attack_type`
- `user_task`
- `external_content`
- `label`
- `expected_behavior`
- `difficulty`

Benchmark có nhóm benign chứa từ nhạy cảm để kiểm tra false positive, ví dụ nội dung học thuật nhắc đến `ignore previous instructions` nhưng không yêu cầu hệ thống làm theo.

## 5. Threshold optimization

Script mới:

`scripts/optimize_threshold.py`

Chức năng:

- Đọc CSV prediction có label và score.
- Tự nhận diện hoặc nhận tham số cột label/score.
- Quét threshold từ 0.01 đến 0.99.
- Tính accuracy, precision, recall, F1, F2, F-beta.
- Xuất confusion matrix, false positives và false negatives.
- Không chọn threshold cảm tính 0.5.

Lệnh đã chạy:

```powershell
.\.venv\Scripts\python.exe scripts\optimize_threshold.py --input reports\indirect_evaluation\predictions.csv --label-col label --score-col final_score --beta 2
```

Output:

- `reports/threshold_optimization/threshold_summary.json`
- `reports/threshold_optimization/threshold_recommendation.md`
- `reports/threshold_optimization/false_positives.csv`
- `reports/threshold_optimization/false_negatives.csv`

Kết quả trên indirect predictions hiện có:

- recommended threshold: `0.5100`
- precision: `1.0000`
- recall: `1.0000`
- F1: `1.0000`
- F2: `1.0000`
- confusion matrix: `[[12, 0], [0, 12]]`

## 6. Explainable output

Runtime pipeline mới trả format:

```json
{
  "risk_level": "safe/warn/block",
  "final_score": 0.94,
  "model_scores": {
    "rule_based": 0.8,
    "ml_model": 0.6189,
    "transformer": null,
    "context_aware": 0.94
  },
  "threshold_used": {
    "warn": 0.3,
    "block": 0.5
  },
  "reasons": [
    "Matched keyword: external_content: ignore previous instructions",
    "Context mismatch: external content contains assistant-directed instruction(s) unrelated to the user task."
  ],
  "recommendation": "Block this input because external content contains an instruction unrelated to the user task."
}
```

Nếu safe, output vẫn giữ `risk_level`, `final_score`, `model_scores`, `threshold_used` và lý do ngắn.

## 7. Kết quả test

Lệnh benchmark:

```powershell
.\.venv\Scripts\python.exe scripts\test_context_aware_detection.py
```

Kết quả deterministic benchmark trên 105 mẫu:

- accuracy: `1.0000`
- precision: `1.0000`
- recall: `1.0000`
- F1: `1.0000`
- confusion matrix: `[[30, 0], [0, 75]]`

Lệnh pytest:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Kết quả:

- `82 passed`

## 8. Hạn chế hiện tại

- Transformer scoring là optional vì chậm hơn và phụ thuộc checkpoint/runtime.
- ML truyền thống hiện có threshold rất thiên về recall nên khi bật `--use-ml` trên benchmark mới có thể tăng false positive. Cần calibrate lại ML trên benchmark/context-aware dataset nếu muốn dùng ML làm tín hiệu quyết định chính.
- Benchmark 105 mẫu là synthetic benchmark để kiểm tra hành vi; cần bổ sung dữ liệu thực tế từ email, HTML, PDF, RAG chunk và tài liệu người dùng.
- Context-aware layer hiện dùng deterministic pattern matching, chưa có semantic entailment giữa task và instruction.

## 9. Hướng cải thiện tiếp theo

- Sinh prediction CSV từ benchmark mới rồi chạy `scripts/optimize_threshold.py` riêng cho context-aware runtime.
- Calibrate lại Logistic Regression/Linear SVM/Random Forest với hard negatives và benign suspicious samples.
- Thêm Transformer evaluation batch cho benchmark mới.
- Tích hợp context-aware score vào dashboard để hiển thị rõ lý do block/warn.
- Thêm corpus indirect injection thực tế từ HTML/email/RAG tài liệu.
