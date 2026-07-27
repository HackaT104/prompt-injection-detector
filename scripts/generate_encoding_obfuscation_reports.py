"""Generate Vietnamese reports from executed benchmark, ablation and test artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "encoding_obfuscation"


def pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def write(name: str, content: str) -> None:
    (REPORT / name).write_text(content.strip() + "\n", encoding="utf-8")


def main() -> int:
    before = json.loads((REPORT / "metrics_before.json").read_text(encoding="utf-8"))
    after = json.loads((REPORT / "metrics_after.json").read_text(encoding="utf-8"))
    tests = json.loads((REPORT / "test_results.json").read_text(encoding="utf-8"))
    smoke = tests.get("http_smoke", {})
    with (REPORT / "ablation_results.csv").open(encoding="utf-8-sig", newline="") as handle:
        ablations = list(csv.DictReader(handle))
    audit = (REPORT / "current_pipeline_audit.md").read_text(encoding="utf-8")
    shutil.copyfile(ROOT / "configs" / "security_runtime.yaml", REPORT / "runtime_config_snapshot.yaml")

    write("01_current_pipeline_audit.md", audit + "\n\n## Trạng thái\n\nAudit này được chụp trước khi thay đổi runtime. Kết quả sau triển khai nằm trong các báo cáo 02-19.")
    write("02_supported_techniques.md", """
# Kỹ thuật được hỗ trợ

## Encoding

| Nhóm | Trạng thái | Bằng chứng |
|---|---|---|
| Base64 standard, URL-safe, thiếu padding, inline, nhiều payload, xuống dòng | Hoàn thành | `test_variant_extractor.py`, fixture Base64 |
| Nested Base64 và Base64 + URL | Hoàn thành, depth tối đa 2 | fixture `base64_nested.jsonl` |
| URL percent encoding và double encoding | Hoàn thành | `url_encoding.jsonl` |
| Hex liền, cách byte, `\\xNN`, `0xNN` | Hoàn thành | `hex_encoding.jsonl` |
| Unicode escape, HTML entity | Hoàn thành | fixture tương ứng |
| ROT13 có heuristic | Hoàn thành trong phạm vi heuristic | unit/fixture ROT13 |
| ASCII decimal, binary 8-bit | Hoàn thành có candidate guard | unit/fixture numeric |

## Obfuscation

Đã kiểm thử zero-width, homoglyph Latin/Cyrillic/Greek có kiểm soát, mixed script, bidi controls, whitespace split, punctuation split, leetspeak theo ngữ cảnh, bộ typo giới hạn, case alternation và repeated characters. Mixed-language dùng rule/context hiện tại cùng variant đã chuẩn hóa; chưa phải bộ dịch/ngữ nghĩa đa ngôn ngữ tổng quát.

Phát hiện encoding chỉ tạo tín hiệu kỹ thuật. Nội dung sau biến đổi phải được Rule, RoBERTa và context xác nhận trước khi policy block.
""")
    write("03_variant_extraction_design.md", """
# Thiết kế Variant Extraction

Module trung tâm: `src/security/variant_extractor.py`.

Pipeline dùng BFS có giới hạn. Root `v0` giữ nguyên input; mỗi variant lưu `variant_id`, `parent_variant_id`, transform chain, depth, SHA-256, printable ratio, readability, confidence và metadata. Hash dedupe ngăn variant trùng. Text chỉ được đưa vào RoBERTa sau khi qua UTF-8, printable, readability, length và expansion guard.

Runtime public/audit chỉ trả metadata và preview dạng `<redacted:hash:len>`, không trả decoded secret. `preprocessing.py` là compatibility facade để các module cũ tiếp tục dùng `analysis_text`, `decoded_variants` và `detected_encodings`.
""")
    write("04_candidate_detection.md", """
# Candidate Detection

- Base64: regex riêng cho token, chuỗi tách khoảng trắng và chuỗi wrap; kiểm tra alphabet, padding bổ sung, UTF-8, printable/readability. JWT được nhận diện theo ba segment và bỏ qua.
- Hex: phân biệt continuous/spaced/escaped/prefixed; yêu cầu số nibble chẵn và UTF-8 đọc được. Hash thuần hex, MAC, UUID, memory address và color không được decode như payload.
- URL/HTML/Unicode: chỉ chạy khi có escape/entity hợp lệ.
- ROT13: chỉ tạo variant khi có chỉ dấu ROT13 hoặc decoded text làm lộ marker có nghĩa.
- ASCII decimal: yêu cầu context decode/ASCII hoặc decoded marker. Binary yêu cầu nhóm 8-bit hợp lệ.
- Character obfuscation: chỉ collapse các cấu trúc bounded; leetspeak có guard cho serial/version/hash.

Fixture false-positive chứa UUID, SHA-256, MD5, JWT, Git hash, serial, màu, MAC, IP, image fragment, code escape, URL và chuỗi ngẫu nhiên.
""")
    write("05_safe_decoding_limits.md", """
# Giới hạn giải mã an toàn

```yaml
max_decode_depth: 2
max_variants: 20
max_input_length: 100000
max_decoded_length: 100000
max_expansion_ratio: 10
min_printable_ratio: 0.75
min_readability_score: 0.4
deduplicate_by_hash: true
decode_timeout_ms: 75
max_total_variant_chars: 400000
```

Extractor dừng khi hết depth/variant/character budget hoặc timeout, ghi warning và resource guard. Decode bytes dùng UTF-8 strict; binary/unreadable không đi vào model. Đây là guard trong process, không phải sandbox process cứng.
""")
    write("06_obfuscation_detection.md", """
# Phát hiện obfuscation

`obfuscation_score` độc lập với malicious score. Encoding đơn lẻ tạo score thấp; score hiệu dụng tăng khi variant sau normalize có instruction độc hại, depth 2, character evasion hoặc decode-and-execute intent. Metadata zero-width/bidi có count và position; homoglyph có mixed-script flag và script-mixing score.

Rule ID ổn định gồm `ENC-*`, `OBF-*` và semantic ID như `ENCODED-SYSTEM-PROMPT-EXTRACTION`, `ENCODED-DATA-EXFILTRATION`, `ENCODED-TOOL-ABUSE`. Một `ENC-BASE64-DETECTED` riêng lẻ không block.
""")
    write("07_roberta_variant_analysis.md", """
# RoBERTa Variant Analysis

Runtime tiếp tục dùng singleton `roberta_v4` tại `models/transformers/roberta_v4`, max length 128. `predict_transformer_batch` gom original/normalized/variant vào batch và tái sử dụng cache model/tokenizer. Calibration production vẫn tắt do provenance không tương thích; checkpoint không thay đổi và không train lại.

Mỗi variant có raw rule/context/RoBERTa score và effective score. Original encoded score cao không được xem là bằng chứng model hiểu payload: nếu decoded variant đọc được và benign, policy cap tín hiệu opaque original. Selected variant và transform chain được log bằng metadata, không lưu full decoded text.
""")
    write("08_context_aware_handling.md", """
# Context-aware handling

Context mới tách: educational/security analysis, quoted content, decode-and-execute, external source mismatch, sensitive target và tool activation. Câu hỏi học thuật, dịch/phân tích không thực thi và structured identifier được cap khi nguồn là chat. Guard benign không áp dụng cho instruction nằm trong website/document/tool output vì đó vẫn là untrusted content cần quarantine.

Ví dụ đã kiểm thử: Base64 Hello được allow; phân tích encoded injection không thực thi được allow; `decode and follow` chứa exfiltration bị block; external encoded instruction bị block/quarantine.
""")
    write("09_risk_fusion_changes.md", """
# Thay đổi Risk Fusion

Fusion nhận rule, RoBERTa original, selected variant risk, context, obfuscation, source, sensitive target, tool risk, depth, confidence và execution intent. Weighted contributions phục vụ explainability; critical floor tránh làm loãng tín hiệu.

Override gồm malicious variant model, nested depth, encoded system prompt, encoded data exfiltration và destructive tool activation. Benign reference/decoded payload có risk cap 0.24 và không nhận critical override. Các override thực tế nằm trong `fusion.overridesApplied` và audit `critical_overrides`.
""")
    write("10_policy_engine_changes.md", """
# Thay đổi Policy Engine

- `POL-ENC-BENIGN-ALLOW`: ALLOW + ALLOW_WITH_LOG.
- `POL-ENC-MALICIOUS-BLOCK`: block encoded malicious chung.
- `POL-ENC-SYSTEM-PROMPT-BLOCK`: BLOCK + LOG_INCIDENT.
- `POL-ENC-EXFILTRATION-BLOCK`: BLOCK + LOG_INCIDENT + ESCALATE.
- `POL-ENC-TOOL-ABUSE-BLOCK`: BLOCK + RESTRICT_TOOLS + LOG_INCIDENT.

Policy trả selected variant, transformation chain, user-safe reason và admin technical reason. User Site vẫn chỉ hiển thị phản hồi an toàn, không hiện score/rule nội bộ.
""")
    write("11_output_security_changes.md", """
# Output Security

Output đi qua cùng extractor, RoBERTa batch, secret scanner, PII scanner và prompt-leak scanner trên original cùng decoded/deobfuscated variants. Finding chỉ lưu category, span, hash, redacted preview, variant ID và transform.

Plain secret có thể redact. Encoded secret/PII dùng SAFE_FALLBACK vì không thể bảo đảm sửa đúng byte span trong original encoded text. Encoded prompt leak yêu cầu regeneration hoặc fallback. Fixture thực tế: Base64 secret và Hex prompt leak bị chặn; Base64 Hello được allow. Test bổ sung URL-encoded secret trong Markdown và zero-width API key.
""")
    write("12_database_and_admin_changes.md", """
# Database và Admin

Repository không có ORM/database migration; Detection Logs hiện dùng `data/audit_log.jsonl`. Vì vậy thay đổi được thực hiện trên schema record JSONL thay vì bịa migration SQL.

Log bổ sung encoding/obfuscation types, decode success/depth, variant count, selected ID/preview masked, transform chain, original/selected score, obfuscation score, override, warning và preprocessing/model/total latency. Variant graph chỉ có hash/metadata/score, không có decoded text.

Admin Audit thêm counters encoded input, decoded malicious, nested encoding, encoded output stopped và cột selected variant score/transform/depth. Detail JSON hiển thị đầy đủ metadata kỹ thuật cho admin.
""")
    write("13_test_execution_report.md", f"""
# Báo cáo chạy test

Lệnh:

```powershell
{tests['command']}
```

| Passed | Failed | Skipped | Duration |
|---:|---:|---:|---:|
| {tests['passed']} | {tests['failed']} | {tests['skipped']} | {tests['duration_seconds']:.2f}s |

JUnit: `{tests['junit_report']}`. Test bao phủ decoder/normalizer, guards, fusion/policy, output scan, audit serialization và encoded chunk trong TXT/DOCX/PDF.

HTTP smoke trên backend restart: `/health` = `{smoke.get('health')}`, `/chat` = {smoke.get('chat_page_status')}; benign Base64 = `{smoke.get('benign_base64_decision')}`, malicious Base64 = `{smoke.get('malicious_base64_decision')}` với `{smoke.get('malicious_policy')}`; encoded output = `{smoke.get('encoded_output_action')}`. Admin API không role trả {smoke.get('admin_without_role_status')}, có role trả {smoke.get('admin_with_role_status')}. Audit có variant metadata và không lưu decoded payload.
""")

    before_c, after_c = before["classification"], after["classification"]
    write("14_benchmark_before_after.md", f"""
# Benchmark trước và sau

Đây là fixed internal diagnostic gồm {after['input_fixture_count']} input và {after['output_fixture_count']} output fixture do repository sở hữu; không phải đánh giá production độc lập.

| Metric | Before | After |
|---|---:|---:|
| Encoding detection recall | {pct(before['encoding_detection_recall'])} | {pct(after['encoding_detection_recall'])} |
| Obfuscation detection recall | {pct(before['obfuscation_detection_recall'])} | {pct(after['obfuscation_detection_recall'])} |
| Malicious decoded recall | {pct(before['malicious_decoded_content_recall'])} | {pct(after['malicious_decoded_content_recall'])} |
| Benign encoded pass rate | {pct(before['benign_encoded_pass_rate'])} | {pct(after['benign_encoded_pass_rate'])} |
| False-positive rate | {pct(before_c['false_positive_rate'])} | {pct(after_c['false_positive_rate'])} |
| False-negative rate | {pct(before_c['false_negative_rate'])} | {pct(after_c['false_negative_rate'])} |
| Nested malicious recall | {pct(before['nested_encoding_recall'])} | {pct(after['nested_encoding_recall'])} |
| Output encoded leak prevention | {pct(before['output_security']['encoded_leak_prevention_rate'])} | {pct(after['output_security']['encoded_leak_prevention_rate'])} |
| Average latency | {before['latency']['average_ms']:.2f} ms | {after['latency']['average_ms']:.2f} ms |
| P95 latency | {before['latency']['p95_ms']:.2f} ms | {after['latency']['p95_ms']:.2f} ms |
| Average variants | {before['variants']['average_count']:.2f} | {after['variants']['average_count']:.2f} |

Baseline average bị ảnh hưởng bởi cold model-load outlier (average lớn hơn P95). Sau nâng cấp P95 tăng do phân tích nhiều variant; preprocessing trung bình tăng từ {before['latency']['preprocessing_average_ms']:.3f} ms lên {after['latency']['preprocessing_average_ms']:.3f} ms.
""")

    ablation_lines = ["| Mode | Precision | Recall | F1 | FP | FN | Encoding | Obfuscation | Avg ms |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in ablations:
        ablation_lines.append(
            f"| {row['ablation']} | {pct(row['precision'])} | {pct(row['recall'])} | {pct(row['f1'])} | {row['false_positive']} | {row['false_negative']} | {pct(row['encoding_recall'])} | {pct(row['obfuscation_recall'])} | {float(row['average_latency_ms']):.2f} |"
        )
    write("15_ablation_study.md", "# Ablation A1-A6\n\n" + "\n".join(ablation_lines) + "\n\nA3 tăng coverage nhưng max model score làm tăng FP. A5 đạt recall cao nhưng context chưa tự giải quyết hard-negative. A6 cho thấy adaptive fusion + policy/context guard là phần cần thiết trên fixture này. Không dùng ablation để train hoặc chỉnh checkpoint.")
    write("16_false_positive_analysis.md", f"""
# Phân tích False Positive

Before có {before_c['false_positive']} FP/{before_c['true_negative'] + before_c['false_positive']} benign (FPR {pct(before_c['false_positive_rate'])}). Các nhóm chính là opaque identifier/image fragment, URL hợp lệ, câu hỏi encoding và sample credential.

After có {after_c['false_positive']} FP trên cùng fixture (FPR {pct(after_c['false_positive_rate'])}). Cải thiện đến từ candidate guard và context theo cặp tín hiệu, không phải hạ threshold. `false_positives.csv` hiện chỉ có header. Kết quả 0 FP chỉ đúng cho fixture nội bộ nhỏ này.
""")
    write("17_false_negative_analysis.md", f"""
# Phân tích False Negative

Before và after đều có {after_c['false_negative']} FN theo nhãn quyết định trên fixture. Baseline malicious recall cao một phần vì detector cũ cảnh báo rộng, trong khi encoding detection recall chỉ {pct(before['encoding_detection_recall'])} và nested recall {pct(before['nested_encoding_recall'])}; do đó không thể kết luận baseline đã hiểu payload.

After có encoding/obfuscation/nested recall {pct(after['encoding_detection_recall'])}/{pct(after['obfuscation_detection_recall'])}/{pct(after['nested_encoding_recall'])}. `false_negatives.csv` hiện chỉ có header. Vẫn cần external adversarial holdout lớn hơn để ước lượng FN production.
""")
    write("18_known_limitations.md", """
# Known Limitations

1. Benchmark chỉ có 55 input và 3 output fixture nội bộ; có nguy cơ coverage bias.
2. Indirect slice có 4 positive và không có negative nên specificity của slice không xác định (machine report ghi 0 theo convention).
3. Typoglycemia dùng danh sách typo bounded, chưa phải fuzzy/edit-distance tổng quát.
4. Homoglyph map có chủ đích và hữu hạn; không bao phủ toàn Unicode confusables.
5. Mixed-language detection dựa trên detector/rule hiện hữu và fixture giới hạn.
6. PDF scan ảnh chưa có OCR; archive/compressed attachment chưa được hỗ trợ.
7. Decode timeout là cooperative in-process guard, không phải process isolation.
8. Encoded output leak dùng safe fallback thay vì surgical redaction.
9. JSONL là audit store hiện hữu; chưa có relational `security_variants` table hay retention/index policy.
10. Chưa chạy load/concurrency benchmark dài hạn và chưa có independent red-team holdout.
""")
    write("19_final_completion_report.md", f"""
# Final Completion Report

## 1. Trước khi sửa

Preprocessing chỉ decode một lớp, không có variant graph/limits tập trung; Rule chạy trên text ghép nhưng RoBERTa chỉ chạy original; document cleaning làm hỏng case-sensitive Base64; output scanner chỉ bảo vệ plain text đầy đủ; audit thiếu provenance variant.

## 2. Kỹ thuật bổ sung

Base64/URL/Hex/Unicode/HTML/ROT13/ASCII/binary; nested depth 2; zero-width/homoglyph/mixed-script/bidi/whitespace/punctuation/leetspeak/typo/case/repeated-character; original + variant Rule/RoBERTa/context; document chunk và output variant scan.

## 3. File mới chính

`src/security/variant_extractor.py`, `src/security/variant_analysis.py`, hai test module variant, 17 JSONL fixture, benchmark/ablation/report scripts và thư mục báo cáo này.

## 4. File sửa chính

`configs/security_runtime.yaml`, `src/security/preprocessing.py`, `src/transformer_utils.py`, `src/roberta_runtime.py`, `src/context_runtime.py`, `src/official_runtime.py`, `src/indirect_pipeline.py`, `src/document_runtime.py`, `src/security/output_security.py`, `src/api.py`, `src/audit_log.py`, `static/admin_audit.html`, `README.md`.

## 5. Pipeline cuối

Request -> validate -> normalize -> bounded variant graph -> Rule + batched RoBERTa + context -> adaptive fusion/critical override -> policy -> secure prompt/LLM -> output variant scan -> JSONL audit -> User/Admin.

## 6-9. Model, threshold, fusion, policy

RoBERTa production: `roberta_v4` (`models/transformers/roberta_v4`), max length 128, input warn/block 0.30/0.45, output warn/block 0.30/0.85. Checkpoint và threshold không đổi; không train. Fusion nhận original/selected/context/obfuscation/source/sensitive/depth/confidence/intent. Policy ID encoded riêng cho benign, system prompt, exfiltration và tool abuse.

## 10-11. Test

Executed: {tests['passed']} passed, {tests['failed']} failed, {tests['skipped']} skipped trong {tests['duration_seconds']:.2f}s.

## 12-15. Benchmark và latency

Encoding recall {pct(before['encoding_detection_recall'])} -> {pct(after['encoding_detection_recall'])}; obfuscation {pct(before['obfuscation_detection_recall'])} -> {pct(after['obfuscation_detection_recall'])}; benign pass {pct(before['benign_encoded_pass_rate'])} -> {pct(after['benign_encoded_pass_rate'])}; FPR {pct(before_c['false_positive_rate'])} -> {pct(after_c['false_positive_rate'])}; FNR after {pct(after_c['false_negative_rate'])}. Average latency {before['latency']['average_ms']:.2f} -> {after['latency']['average_ms']:.2f} ms; P95 {before['latency']['p95_ms']:.2f} -> {after['latency']['p95_ms']:.2f} ms. Baseline average có cold-start outlier, vì vậy không diễn giải là speedup.

## 16-17. Hạn chế và remaining work

Xem `18_known_limitations.md`. Ưu tiên tiếp theo: independent red-team holdout, negative indirect corpus, load test, Unicode confusables mở rộng, OCR và audit database có retention/index.

## 18. Lệnh chạy

```powershell
.\\.venv\\Scripts\\python.exe -m pytest -q
.\\.venv\\Scripts\\python.exe scripts\\run_encoding_obfuscation_benchmark.py --phase after
.\\.venv\\Scripts\\python.exe scripts\\run_encoding_obfuscation_ablation.py
.\\.venv\\Scripts\\python.exe scripts\\generate_encoding_obfuscation_reports.py
.\\.venv\\Scripts\\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Các metric là diagnostic nội bộ, không phải chứng nhận an toàn production.
""")

    print("ENCODING AND OBFUSCATION SECURITY UPGRADE SUMMARY")
    print("\nRoBERTa model: roberta_v4")
    print("Input threshold: warn 0.30 / block 0.45")
    print("Output threshold: warn 0.30 / block 0.85")
    print("\nSupported encodings: Base64, URL, Hex, Unicode escape, HTML entity, ROT13, ASCII decimal, binary")
    print("Supported obfuscations: zero-width, homoglyph, mixed-script, bidi, whitespace/punctuation split, leetspeak, bounded typo, case, repeated character")
    print("Max decode depth: 2")
    print("Max variants: 20")
    print("Max expansion ratio: 10")
    print(f"\nEncoding detection recall: {pct(after['encoding_detection_recall'])}")
    print(f"Obfuscation detection recall: {pct(after['obfuscation_detection_recall'])}")
    print(f"Benign encoded pass rate: {pct(after['benign_encoded_pass_rate'])}")
    print(f"False positive rate: {pct(after_c['false_positive_rate'])}")
    print(f"False negative rate: {pct(after_c['false_negative_rate'])}")
    print(f"Nested encoding recall: {pct(after['nested_encoding_recall'])}")
    print(f"Output encoded leak prevention: {pct(after['output_security']['encoded_leak_prevention_rate'])}")
    print(f"\nTests passed: {tests['passed']}")
    print(f"Tests failed: {tests['failed']}")
    print(f"Tests skipped: {tests['skipped']}")
    print(f"\nAverage latency before: {before['latency']['average_ms']:.2f} ms")
    print(f"Average latency after: {after['latency']['average_ms']:.2f} ms")
    print(f"P95 latency after: {after['latency']['p95_ms']:.2f} ms")
    print(f"\nReports directory: {REPORT}")
    print("Known limitations: internal fixture set, bounded Unicode/typo coverage, no OCR or process-isolated decoder")
    print("Remaining work: independent holdout, negative indirect corpus, load test, audit DB retention/index")
    return 0


if __name__ == "__main__":
    sys.exit(main())
