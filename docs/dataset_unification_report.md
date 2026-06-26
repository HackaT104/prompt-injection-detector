# Báo cáo chuẩn hóa và gom dataset

## Mục tiêu

Hai file yêu cầu không trùng nội dung. File thứ nhất tập trung vào kiểm tra dataset `neuralchemy/Prompt-injection-dataset`, checkpoint Transformer và logic inference. File thứ hai mở rộng thêm chuẩn hóa toàn bộ nguồn dữ liệu, báo lỗi model chưa sẵn sàng và tạo unified dataset. Vì vậy project đã được xử lý theo hướng: dataset nào trùng nội dung thì khử trùng, dataset nào khác nguồn/nội dung thì được gom vào bộ dữ liệu thống nhất.

## Schema chuẩn

Tất cả nguồn dữ liệu được chuẩn hóa về các cột:

```text
text,label,category,source,severity,split,language,augmented,dataset_config
```

Quy ước nhãn:

```text
0 = SAFE / benign
1 = INJECTION / malicious
```

## Nguồn dữ liệu đã gom

Theo file `datasets/reports/unification_summary.json`, các nguồn hiện có:

| Nguồn | Số dòng sau chuẩn hóa |
|---|---:|
| `project_raw_jsonl` | 500 |
| `direct_ml_ready` | 1160 |
| `direct_merged` | 1160 |
| `augmented_multilingual_dataset` | 41 |
| `neuralchemy_core` | 6274 |
| `neuralchemy_full` | 15919 |

Ghi chú: direct và indirect vẫn được giữ tách riêng theo yêu cầu. File unified ở trên dùng cho direct/classical ML và Transformer direct detector. Dataset BIPIA indirect nếu được tải sẽ tiếp tục nằm ở `datasets/processed/bipia_indirect.csv` và model indirect nằm trong `models/indirect/`, không trộn vào file direct unified.

## Kết quả khử trùng

| Chỉ số | Giá trị |
|---|---:|
| Tổng dòng sau chuẩn hóa ban đầu | 25054 |
| Dòng unified sau khử trùng và loại conflict | 10108 |
| Dòng dùng cho ML cổ điển | 7235 |
| Dòng dùng cho Transformer | 10108 |
| Dòng duplicate đã loại | 14936 |
| Dòng label conflict | 10 |

## Phân phối nhãn unified

| Label | Ý nghĩa | Số dòng |
|---:|---|---:|
| 0 | SAFE / benign | 4139 |
| 1 | INJECTION / malicious | 5969 |

## Phân phối split

| Split | Số dòng |
|---|---:|
| train | 8291 |
| validation | 906 |
| test | 911 |

## File output

```text
datasets/unified/prompt_injection_unified.csv
datasets/unified/prompt_injection_ml_ready.csv
datasets/unified/prompt_injection_transformer_ready.csv
datasets/reports/label_conflicts.csv
datasets/reports/unification_summary.json
```

## Lệnh tạo lại dataset unified

```powershell
python scripts/build_unified_dataset.py
```

## Ghi chú kỹ thuật

- `core` và `full` của `neuralchemy/Prompt-injection-dataset` không giống nhau hoàn toàn. `full` có tập train lớn hơn, còn validation/test được dùng để kiểm tra nhất quán.
- Các dòng có cùng text nhưng nhãn khác nhau không được đưa vào dataset train chính; chúng được lưu riêng trong `datasets/reports/label_conflicts.csv`.
- Dataset cho ML cổ điển và Transformer được tách riêng vì Transformer nên dùng thêm dữ liệu `full`, còn ML cổ điển ưu tiên dữ liệu đã khử trùng và phù hợp với baseline TF-IDF.
