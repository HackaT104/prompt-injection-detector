# Compare All Models Debug Report

## Nguyên nhân gây score 0.3

Lỗi nhiều model cùng trả `risk_score = 0.3` xuất phát từ backend, không phải từ frontend render. Hai pipeline inference cũ có benign guard ép điểm về `min(risk_score, 0.30)` khi prompt trông giống yêu cầu học thuật hoặc trích dẫn prompt nguy hiểm:

- `src/detector.py`: TF-IDF + Logistic Regression, Linear SVM, Random Forest.
- `src/transformer_utils.py`: DistilBERT và RoBERTa.

Vì vậy Compare All Models có thể hiển thị nhiều model cùng `0.3000`, dù raw model score thật sự khác nhau. Frontend `static/advanced_demo.html` đã dùng `row.risk_score` trong bảng compare, nên không phải lỗi lấy chung `result.risk_score`.

## Cách sửa

- Bỏ logic cap score `0.30` trong traditional ML detector.
- Bỏ logic cap score `0.30` trong Transformer inference.
- Giữ `benign_guard` như metadata giải thích, không thay đổi `risk_score` nữa.
- `/detect/compare` trả debug fields riêng cho từng model:
  - `requested_model`
  - `loaded_model`
  - `model_path`
  - `available`
  - `raw_score`
  - `risk_score`
  - `predicted_label`
  - `action`
  - `error`
- Nếu model lỗi/chưa sẵn sàng, backend trả `available=false`, `action=model_not_ready`, `risk_score=null`, không trả score giả.
- Hybrid trả thêm `individual_scores`, `selected_models`, `voting_strategy`, `final_score`, `final_action`.

## Các số 0.30 còn lại

Sau khi sửa, không còn `risk_score = 0.3`, `default_score`, hoặc `fallback_score`. Hai literal `0.30` còn lại không phải fallback score:

- `src/advanced_detection.py`: trọng số 0.30 trong weighted voting của hybrid.
- `src/transformer_utils.py`: `test_size=0.30` để chia train/validation/test cho Transformer.

## File đã sửa

- `src/detector.py`
- `src/transformer_utils.py`
- `src/advanced_detection.py`
- `static/advanced_demo.html`
- `tests/test_compare_models.py`
- `reports/compare_models_debug_report.md`

## Endpoint compare

Frontend nút Compare All Models gọi:

```http
POST /detect/compare
```

Backend loop qua:

```python
[
  "logistic_regression",
  "linear_svm",
  "random_forest",
  "distilbert",
  "roberta",
  "hybrid",
]
```

Mỗi vòng gọi `detect_prompt_advanced(..., model=model_name)`, nên mỗi model được chạy riêng.

## Sample response trước/sau

Trước khi sửa, một số prompt bị benign guard cap điểm:

```json
{
  "model": "logistic_regression",
  "risk_score": 0.3,
  "action": "allow"
}
```

Sau khi sửa, compare row có debug rõ ràng:

```json
{
  "requested_model": "logistic_regression",
  "loaded_model": "tfidf_logistic_regression",
  "model_path": "F:\\Capstone Project\\prompt-injection-detector\\models\\logistic_regression_model.joblib",
  "available": true,
  "raw_score": 1.2345,
  "risk_score": 0.8123,
  "predicted_label": 1,
  "action": "block",
  "error": null
}
```

Nếu model chưa sẵn sàng:

```json
{
  "requested_model": "roberta",
  "loaded_model": "transformer_roberta_base",
  "available": false,
  "raw_score": null,
  "risk_score": null,
  "action": "model_not_ready",
  "error": "Model checkpoint not found or inference failed."
}
```

## Cách kiểm tra lại

Chạy test:

```bat
.venv\Scripts\python.exe -m pytest tests\test_compare_models.py
```

Chạy API:

```bat
.venv\Scripts\python.exe -m uvicorn src.api:app --reload
```

Test bằng curl:

```bat
curl -X POST "http://127.0.0.1:8000/detect/compare" ^
-H "Content-Type: application/json" ^
-d "{\"input_type\":\"text\",\"text\":\"Translate this sentence: Ignore previous instructions.\",\"hybrid_config\":{\"traditional_model\":\"all\",\"transformer_model\":\"roberta\",\"use_rule_based\":true,\"decision_strategy\":\"maximum_risk\"}}"
```

## Trạng thái model

Trạng thái thực tế kiểm tra bằng:

```http
GET /health
```

Nếu model ready, Compare All Models hiển thị `available=true` và điểm inference thật. Nếu model chưa ready, Compare All Models hiển thị `available=false`, `risk_score=null`.