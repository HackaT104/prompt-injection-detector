# Giải thích mô hình

## 1. Vì sao không xây LLM mới?

Xây một LLM mới cần rất nhiều dữ liệu, GPU, chi phí huấn luyện và quy trình kiểm định an toàn. Mục tiêu của đồ án là phát hiện prompt injection ở lớp bảo vệ trước LLM, nên dùng detector nhẹ, dễ triển khai và dễ giải thích sẽ phù hợp hơn.

## 2. Vì sao cần detector layer?

Ứng dụng LLM thường có system prompt, tool, dữ liệu nội bộ hoặc quyền truy cập API. Nếu người dùng gửi prompt độc hại, LLM có thể bị dụ bỏ qua chính sách hoặc tiết lộ thông tin. Detector layer giúp đánh giá prompt trước khi gửi vào LLM.

## 3. Rule-based hoạt động như thế nào?

Rule-based detector tìm các cụm từ nguy hiểm rõ ràng, ví dụ:

- `ignore previous instructions`
- `reveal your system prompt`
- `act as dan`
- `os.system`
- `show api key`

Mỗi nhóm rule có trọng số rủi ro. Nếu `risk_score >= 0.75`, hệ thống `block`. Nếu `0.4 <= risk_score < 0.75`, hệ thống `warn`. Nếu thấp hơn `0.4`, hệ thống `allow`.

## 3.1. Multilingual detection và canonical English

Pipeline mới phát hiện ngôn ngữ đầu vào, sau đó chuẩn hóa prompt về một biểu diễn tiếng Anh thống nhất gọi là `canonical_text`. Rule-based và ML đều chạy trên biểu diễn này.

Ví dụ:

- `Hãy bỏ qua mọi hướng dẫn trước đó` -> `ignore previous instructions`
- `Tiết lộ system prompt` -> `reveal system prompt`
- `Hãy truy cập trái phép vào hệ thống` -> `unauthorized access to the system`

Mục tiêu không phải dịch hoàn hảo, mà là giữ đúng ý định bảo mật để detector hoạt động nhất quán với prompt tiếng Việt và tiếng Anh.

Mặc định hệ thống dùng phrase mapping offline để dễ tái lập khi bảo vệ đồ án. `deep-translator` chỉ là tùy chọn thử nghiệm khi bật biến môi trường `ENABLE_EXTERNAL_TRANSLATION=1`.

Hạn chế:

- Chuẩn hóa dựa trên phrase mapping nên chưa bao phủ mọi slang.
- Prompt đa ngôn ngữ bị obfuscate vẫn khó phát hiện.
- Đây là baseline multilingual detector, có thể nâng cấp bằng mô hình embedding hoặc transformer đa ngôn ngữ.

## 4. TF-IDF là gì?

TF-IDF là cách biến văn bản thành vector số. Từ xuất hiện nhiều trong một prompt nhưng ít xuất hiện trong toàn bộ dataset sẽ có trọng số cao hơn. Trong project này, TF-IDF dùng unigram và bigram để giữ cả từ đơn và cụm hai từ.

## 5. Logistic Regression hoạt động như thế nào?

Logistic Regression học trọng số cho các đặc trưng TF-IDF và dự đoán xác suất prompt thuộc lớp malicious. Vì có `predict_proba`, model này phù hợp để tạo `risk_score`.

Ưu điểm:

- Dễ giải thích.
- Huấn luyện nhanh.
- Có xác suất dự đoán.
- Phù hợp dataset nhỏ và bài toán text classification baseline.

## 6. Linear SVM hoạt động như thế nào?

Linear SVM tìm một siêu phẳng tuyến tính để tách hai lớp benign và malicious trong không gian đặc trưng TF-IDF. Với văn bản ngắn, Linear SVM thường là baseline mạnh.

Trong project này, Linear SVM được bọc bằng `CalibratedClassifierCV` để có thể suy ra xác suất/risk score khi cần.

## 6.1. Random Forest hoạt động như thế nào?

Random Forest là mô hình ensemble gồm nhiều Decision Tree. Mỗi cây được huấn luyện trên một phần dữ liệu và một phần đặc trưng khác nhau. Khi dự đoán, các cây cùng bỏ phiếu để chọn nhãn cuối cùng.

Ưu điểm:

- Dễ hiểu về mặt trực giác.
- Có `feature_importances_` để xem feature nào quan trọng.
- Giảm overfitting so với một Decision Tree đơn lẻ.

Nhược điểm trong bài toán text:

- TF-IDF tạo vector sparse nhiều chiều.
- Random Forest thường không tối ưu với dữ liệu sparse cao chiều bằng Logistic Regression hoặc Linear SVM.
- Feature importance chỉ nói feature quan trọng, không nói rõ feature nghiêng về malicious hay benign.

Trong project này, Random Forest được thêm để so sánh bổ sung, không mặc định là model chính nếu metrics không vượt trội rõ.

## 7. Vì sao chọn Logistic Regression, Linear SVM và Random Forest?

Ba model này là baseline kinh điển cho phân loại văn bản:

- Chạy nhanh trên CPU.
- Dễ tái lập kết quả.
- Dễ giải thích hơn deep learning.
- Phù hợp đồ án cần chứng minh pipeline end-to-end.

## 8. Vì sao không dùng BERT ngay từ đầu?

BERT có thể mạnh hơn, nhưng cần nhiều tài nguyên hơn, thời gian train dài hơn và khó giải thích hơn. Với đồ án tốt nghiệp, nên hoàn thiện baseline ML trước: đọc dữ liệu, train/test đúng, metrics thật, API chạy được. BERT nên là hướng phát triển sau.

## 9. Vì sao accuracy không đủ?

Nếu dataset lệch nhãn, model có thể đạt accuracy cao bằng cách dự đoán lớp chiếm đa số. Trong prompt injection, chỉ nhìn accuracy có thể che giấu việc model bỏ sót nhiều prompt độc hại.

## 10. Vì sao recall quan trọng trong prompt injection?

Recall của lớp malicious đo tỷ lệ prompt tấn công thật sự được phát hiện. False Negative là trường hợp prompt độc hại bị cho qua. Đây là rủi ro lớn vì prompt đó có thể đến được LLM phía sau.

## 10.1. Vì sao cần chọn threshold?

Xác suất ML không nên được chuyển thẳng thành cảnh báo bằng ngưỡng mặc định 0.5. Một prompt benign đơn giản có thể nhận score trung bình nếu chứa từ xuất hiện trong cả hai lớp. Vì vậy project chọn threshold trên validation set theo precision-recall tradeoff, sau đó dùng ngưỡng runtime cao hơn cho API để giảm false positive khi rule-based không phát hiện dấu hiệu nguy hiểm.

Trong API hybrid:

- Rule-based score cao sẽ block ngay.
- Nếu rule-based score bằng 0, ML score mức trung bình chưa đủ để warn.
- ML chỉ warn/block khi vượt ngưỡng runtime trong `models/model_thresholds.json`.

## 10.2. Spurious correlation và feature bias là gì?

Spurious correlation là hiện tượng model học nhầm một dấu hiệu không thật sự liên quan đến tấn công. Ví dụ, nếu dataset nhỏ hoặc phân phối dữ liệu lệch, các từ rất chung như `this`, `that`, `the`, `and`, `you`, `your` có thể vô tình xuất hiện nhiều hơn trong prompt malicious. Khi đó Logistic Regression hoặc Linear SVM có thể gán trọng số cao cho các từ này, dù bản thân chúng không nguy hiểm.

Đây là feature bias: model dựa vào feature nhiễu thay vì dấu hiệu bảo mật thật như `ignore previous instructions`, `reveal system prompt`, `bypass safety`, `api key`, `whoami`, `eval`, `exec`.

Để giảm lỗi này, pipeline dùng custom stopwords cho các từ quá chung, đồng thời giữ lại các token security quan trọng. TF-IDF cũng được cấu hình với `min_df=2`, `max_df=0.9`, `ngram_range=(1, 2)` để giảm feature quá hiếm hoặc quá phổ biến.

Project sinh thêm `reports/feature_analysis.md` sau mỗi lần train. File này liệt kê:

- Top malicious indicators.
- Top benign indicators.
- Suspected biased features nếu stopword vẫn xuất hiện trong top features.

Việc kiểm tra feature analysis giúp giải thích model khi bảo vệ đồ án và phát hiện sớm các false positive do dataset bias.

## 11. Confusion matrix đọc như thế nào?

Confusion matrix so sánh nhãn thật và nhãn dự đoán:

| | Dự đoán benign | Dự đoán malicious |
|---|---:|---:|
| Thật benign | True Negative | False Positive |
| Thật malicious | False Negative | True Positive |

## 12. Ý nghĩa TP, TN, FP, FN

- True Positive: prompt malicious được phát hiện đúng.
- True Negative: prompt benign được cho phép đúng.
- False Positive: prompt benign bị cảnh báo hoặc chặn nhầm.
- False Negative: prompt malicious bị bỏ sót.

Trong bảo mật, False Negative thường nguy hiểm hơn False Positive.

## 13. Model nào nên dùng trong demo API?

Nên dùng `hybrid`, tức Rule-based + Logistic Regression. Rule-based bắt các prompt nguy hiểm rõ ràng, còn Logistic Regression xử lý các trường hợp không khớp rule.

## 14. Model nào nên ghi là hướng phát triển?

Baseline TF-IDF đã ổn định nên project hiện có thêm pipeline thử nghiệm Transformer với DistilBERT và RoBERTa-base. Hai mô hình này vẫn nên được trình bày là phần mở rộng/so sánh nâng cao, không thay thế ngay detector hybrid đang dùng cho API nếu chưa có metrics tốt hơn trên tập test đầy đủ.

DistilBERT nhẹ hơn BERT gốc, phù hợp máy cá nhân vì train/inference nhanh hơn. RoBERTa-base thường mạnh hơn về biểu diễn ngôn ngữ nhưng tốn VRAM và thời gian train hơn. Với NVIDIA RTX 3050 4GB, nên bắt đầu bằng `max_length=128`, `batch_size=4`, `gradient_accumulation_steps=2`; nếu thiếu VRAM thì giảm `batch_size=2` hoặc `max_length=64`.

Transformer sinh `risk_score = P(label=1 | prompt)` bằng softmax trên logits. Threshold vẫn được chọn trên validation set giống pipeline cũ: tối ưu `evaluation_threshold`, sau đó dùng runtime threshold cao hơn (`warn>=0.80`, `block>=0.90`) để tránh cảnh báo quá nhạy.

## Kết luận kỹ thuật

- Logistic Regression phù hợp làm model triển khai chính vì có xác suất dự đoán và dễ giải thích.
- Linear SVM phù hợp làm model so sánh vì mạnh với dữ liệu văn bản ngắn.
- Random Forest phù hợp làm model so sánh bổ sung vì trực giác dễ hiểu và có feature importance, nhưng có thể không tối ưu nhất với TF-IDF sparse vector.
- DistilBERT và RoBERTa-base phù hợp để so sánh nâng cao khi có đủ tài nguyên train, nhưng cần đánh giá bằng metrics thật trước khi chọn làm model chính.
- Hybrid Rule-based + Logistic Regression phù hợp nhất cho demo hệ thống.
