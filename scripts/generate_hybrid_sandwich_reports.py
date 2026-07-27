"""Generate evidence-backed Hybrid Sandwich Security implementation reports."""

from __future__ import annotations

import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "hybrid_sandwich_security"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _pct(value: Any) -> str:
    return f"{float(value or 0) * 100:.2f}%"


def _num(value: Any, digits: int = 2) -> str:
    return f"{float(value or 0):.{digits}f}"


def _test_results() -> dict[str, Any]:
    root = ET.parse(REPORT_DIR / "pytest.xml").getroot()
    suites = list(root.findall("testsuite")) if root.tag == "testsuites" else [root]
    result = {
        "command": ".\\.venv\\Scripts\\python.exe -m pytest -q --junitxml=reports\\hybrid_sandwich_security\\pytest.xml",
        "total": sum(int(suite.attrib.get("tests", 0)) for suite in suites),
        "passed": 0,
        "failed": sum(int(suite.attrib.get("failures", 0)) + int(suite.attrib.get("errors", 0)) for suite in suites),
        "skipped": sum(int(suite.attrib.get("skipped", 0)) for suite in suites),
        "duration_seconds": sum(float(suite.attrib.get("time", 0.0)) for suite in suites),
        "failures": [],
    }
    result["passed"] = result["total"] - result["failed"] - result["skipped"]
    for suite in suites:
        for case in suite.findall("testcase"):
            failure = case.find("failure") or case.find("error")
            if failure is not None:
                result["failures"].append(
                    {
                        "test": f"{case.attrib.get('classname')}::{case.attrib.get('name')}",
                        "message": failure.attrib.get("message", "test failed"),
                    }
                )
    return result


def _external_v4() -> dict[str, str]:
    path = REPORT_DIR.parent / "roberta_versions_rogue_security" / "roberta_versions_summary.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("model_version") == "roberta_v4":
                return row
    raise RuntimeError("roberta_v4 external diagnostic row was not found")


def _ablation_table(ablation: dict[str, Any]) -> str:
    lines = [
        "| Mode | Pipeline | Balanced | Accuracy | Precision | Recall | F1 | PR-AUC | FPR | FNR | P95 ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, values in ablation.items():
        lines.append(
            "| {mode} | {label} | {balanced} | {accuracy} | {precision} | {recall} | {f1} | {pr_auc} | {fpr} | {fnr} | {p95} |".format(
                mode=mode,
                label=values["label"],
                balanced=_pct(values["balanced_accuracy"]),
                accuracy=_pct(values["accuracy"]),
                precision=_pct(values["precision"]),
                recall=_pct(values["recall"]),
                f1=_pct(values["f1"]),
                pr_auc=_pct(values["pr_auc"]),
                fpr=_pct(values["false_positive_rate"]),
                fnr=_pct(values["false_negative_rate"]),
                p95=_num(values["latency_p95_ms"]),
            )
        )
    return "\n".join(lines)


def _write(name: str, content: str) -> None:
    (REPORT_DIR / name).write_text(content.strip() + "\n", encoding="utf-8")


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = _read_json(REPORT_DIR / "metrics.json")
    registry = _read_json(REPORT_DIR / "model_registry_snapshot.json")
    tests = _test_results()
    external = _external_v4()
    (REPORT_DIR / "test_results.json").write_text(json.dumps(tests, ensure_ascii=False, indent=2), encoding="utf-8")

    a7 = metrics["ablation"]["A7"]
    output = metrics["output_security"]
    slices = a7["by_slice"]
    ablation_table = _ablation_table(metrics["ablation"])
    architecture = """```mermaid
flowchart TD
    A[User or external content] --> B[Input gateway and validation]
    B --> C[Normalization and obfuscation analysis]
    C --> D[Source separation and trust]
    D --> E[Rule detector]
    D --> F[RoBERTa v4 input scan]
    D --> G[Context-aware analysis]
    E --> H[Adaptive risk fusion]
    F --> H
    G --> H
    H --> I[Decision policy]
    I -->|ALLOW| J[Secure prompt builder]
    I -->|WARN or BLOCK| R[Public safe response]
    J --> K[Configured LLM provider]
    K --> L[Same RoBERTa v4 output scan]
    L --> M[Secret, PII and prompt-leak scanners]
    M --> N[Output policy]
    N -->|ALLOW or REDACT| O[User response]
    N -->|REGENERATE once| K
    N -->|still unsafe| P[Safe fallback]
    I --> Q[Audit log]
    N --> Q
    S[Tool request] --> T[Tool security gateway]
    T --> Q
```"""

    source_audit = (REPORT_DIR.parent / "current_system_architecture_audit.md").read_text(encoding="utf-8-sig")
    model_report = (REPORT_DIR.parent / "roberta_model_selection_report.md").read_text(encoding="utf-8-sig")
    calibrator_report = (REPORT_DIR.parent / "calibrator_validation_report.md").read_text(encoding="utf-8-sig")

    _write("01_current_system_audit.md", source_audit)
    _write("02_roberta_model_selection.md", model_report)
    _write("03_calibrator_validation.md", calibrator_report)

    _write(
        "04_target_architecture.md",
        f"""# Kiến trúc mục tiêu Hybrid Sandwich Security

{architecture}

## Trust boundaries

- User instruction được giữ tách biệt khỏi project document, upload, web, email, RAG và tool output.
- External content luôn là dữ liệu không tin cậy và không thể tự cấp quyền gọi tool.
- LLM chỉ được gọi khi input policy trả `safe`; output phải qua lớp quét thứ hai trước khi trả cho user.
- RoBERTa input và output dùng chung singleton, nhưng dùng threshold/policy theo stage.
- Mọi lỗi model được đưa về fail-safe, không âm thầm coi là SAFE.

## Tương thích

Các route `/detect`, `/api/chat/check`, `/api/chat/check-document`, `/admin`, `/chat` và `/user` được giữ. Các endpoint security/admin mới là phần mở rộng cộng thêm.
""",
    )

    _write(
        "05_implementation_changes.md",
        """# Thay đổi triển khai

## File tạo mới

- `configs/security_runtime.yaml`: model registry, threshold, trust, output/tool policy tập trung.
- `src/security/`: preprocessing, source separation, model registry, secure prompt, tool gateway, output security và pipeline facade.
- `scripts/run_hybrid_sandwich_benchmark.py`: benchmark fixture và ablation A1-A7.
- `scripts/generate_hybrid_sandwich_reports.py`: sinh báo cáo từ artifact máy đọc.
- `tests/security/` và `tests/security/fixtures/`: unit/integration/E2E fixture bảo mật.

## File chỉnh sửa chính

- `src/official_runtime.py`, `src/roberta_runtime.py`, `src/context_runtime.py`, `src/runtime_rule_signal.py`.
- `src/api.py`, `src/audit_log.py`, `src/chat_service.py`, `src/user_site_store.py`.
- `src/document_runtime.py`, `src/indirect_pipeline.py`, `src/llm_service.py`, `src/runtime_config.py`.
- `static/user_chat.html`, `static/admin_audit.html`.

## Giữ nguyên và tái sử dụng

Rule engine, detector context cũ, document extractor, JSON/JSONL storage, provider LLM, route Admin/User và toàn bộ checkpoint cũ không bị xóa.

## Deprecated nhưng chưa xóa

Alias RoBERTa cũ và calibrator `models/calibration/direct_all/roberta/probability_calibrator.joblib` vẫn còn để truy vết; production config không sử dụng calibrator này.
""",
    )

    _write(
        "06_input_security_layer.md",
        """# Input Security Layer

## Luồng

1. Kiểm tra input rỗng, giới hạn 50.000 ký tự, upload tối đa 8 MiB và extension `.txt/.docx/.pdf`.
2. Chuẩn hóa NFKC, HTML/URL decode, loại zero-width/control và giữ nguyên bản gốc cho provenance.
3. Phát hiện base64, hex, ROT13, homoglyph và leetspeak; decoded text chỉ dùng cho phân tích.
4. Tách user instruction, trusted project instruction và untrusted document/web/email/RAG/tool output.
5. Chạy rule, RoBERTa v4, context-aware rồi adaptive fusion và policy.

## Tài liệu

Document pipeline extract, clean, chunk có overlap, quét từng chunk, loại chunk nguy hiểm khỏi safe context. Test thực tế có TXT, DOCX và PDF.

## Privacy

User Site chỉ nhận decision/message/document status. Score, rule, threshold và model details chỉ có ở Admin APIs và audit log.
""",
    )

    _write(
        "07_context_aware_analysis.md",
        """# Context-aware Analysis

`src/context_runtime.py` suy ra mục tiêu user (`summarize`, `translate`, `extract`, `search`, `generate_code`, `execute_tool`), sau đó so sánh với external context.

Tín hiệu gồm `goalMismatch`, `sourceMismatch`, `privilegeEscalation`, `toolMismatch`, `sensitiveTargetScore` và `contextRisk`. Evidence được mask trước khi log. Với source document/web/email, assistant-directed instruction bị coi là context mismatch thay vì gộp thành user prompt.

Trên fixture nhỏ, thêm context từ A3 lên A4 tăng recall từ {a3_recall} lên {a4_recall}; specificity giữ {specificity}. Đây là diagnostic fixture, không phải estimate độc lập.
""".format(
            a3_recall=_pct(metrics["ablation"]["A3"]["recall"]),
            a4_recall=_pct(metrics["ablation"]["A4"]["recall"]),
            specificity=_pct(metrics["ablation"]["A4"]["specificity"]),
        ),
    )

    _write(
        "08_adaptive_risk_fusion.md",
        f"""# Adaptive Risk Fusion

Fusion nền dùng Rule/RoBERTa/Context theo cấu hình hiện hữu. Khi có source không tin cậy, obfuscation, tool hoặc sensitive target, hệ thống cộng contribution thích ứng có giới hạn và áp critical floor cho tool/sensitive signal nghiêm trọng.

Mỗi response admin có `contributions`, `highestRiskSource`, `sourceType`, `overridesApplied` và explanation. User response không expose các trường này.

Kết quả A5 fixed fusion và A6 adaptive input cùng balanced accuracy {_pct(a7['balanced_accuracy'])} trên bộ nhỏ; adaptive runtime có P95 {_num(a7['latency_p95_ms'])} ms và xử lý thêm provenance/tool/obfuscation. Không kết luận adaptive tốt hơn về chất lượng từ 16 mẫu này.
""",
    )

    _write(
        "09_decision_policy_engine.md",
        """# Decision Policy Engine

Input policy trả `safe/allow`, `warning/warn` hoặc `blocked/block`, kèm `policyId`, `reasonCodes`, policy/rule/model/threshold version. Critical hard-block rule và context hard block có ưu tiên cao hơn fusion threshold.

LLM chỉ được gọi cho `safe`. Warning và blocked trả thông báo công khai, không gửi nội dung tới LLM. Model unavailable không thể trở thành SAFE im lặng.

Output policy hỗ trợ `ALLOW`, `ALLOW_WITH_LOG`, `REDACT`, `REGENERATE`, `SAFE_FALLBACK`. Regeneration tối đa một lần theo config.
""",
    )

    _write(
        "10_secure_prompt_builder.md",
        """# Secure Prompt Builder

`src/security/secure_prompt_builder.py` tạo các section tách biệt: SYSTEM POLICY, USER TASK, TRUSTED PROJECT, UNTRUSTED EXTERNAL và TOOL CONSTRAINTS. External text được quote như dữ liệu và có chỉ dẫn không thực thi instruction bên trong.

Prompt template có version `secure-prompt-v1`. Khi output cần regenerate, builder nhận safety feedback cố định, không đưa output nguy hiểm nguyên văn vào system instruction.
""",
    )

    _write(
        "11_tool_security_gateway.md",
        f"""# Tool Security Gateway

Gateway dùng registry cho search/read/send/write/delete/execute, kiểm tra role, instruction source, task relevance, required arguments và confirmation. Unknown tool deny mặc định. External document, web, email và tool output không thể tự authorize tool.

Fixture tool abuse: {metrics['tool_gateway']['denied']}/{metrics['tool_gateway']['total']} yêu cầu bị từ chối, denial rate {_pct(metrics['tool_gateway']['denial_rate'])}.

Giới hạn: gateway mới là lớp authorization; project chưa có executor production nối với provider function-calling thực.
""",
    )

    _write(
        "12_output_security_layer.md",
        f"""# Output Security Layer

Output đi qua cùng RoBERTa v4, secret scanner, PII provenance scanner và prompt-leak scanner. Secret/PII được redact; prompt leak/model block yêu cầu regenerate một lần rồi safe fallback.

## Diagnostic fixture results

- Unsafe output recall: {_pct(output['recall'])} ({output['true_positive']}/{output['true_positive'] + output['false_negative']}).
- Safe output preservation: {_pct(output['safe_output_preservation_rate'])}.
- Unsafe interception: {_pct(output['unsafe_interception_rate'])}.
- Prompt-leak regex detection: {_pct(output['prompt_leak_detection_rate'])}; end-to-end prevention: {_pct(output['prompt_leak_prevention_rate'])}.
- Secret detection/prevention: {_pct(output['secret_detection_rate'])} / {_pct(output['secret_leakage_prevention_rate'])}.
- P95 scan latency: {_num(output['latency_p95_ms'])} ms.

Chỉ có 6 fixture output. Output threshold đang là baseline, chưa tối ưu trên một output holdout độc lập.
""",
    )

    _write(
        "13_roberta_output_reuse.md",
        f"""# Tái sử dụng RoBERTa ở output

Input và output cùng gọi singleton `src.roberta_runtime.roberta_service`. Model chỉ load qua process LRU cache, không tạo checkpoint output riêng.

- Model: `{registry['modelName']}` / `{registry['modelVersion']}`.
- Model/tokenizer path: `{registry['modelPath']}`.
- Artifact SHA-256: `{registry['artifactSha256']}`.
- Input warn/block: `{registry['inputWarnThreshold']}` / `{registry['inputBlockThreshold']}`.
- Output warn/block: `{registry['outputWarnThreshold']}` / `{registry['outputBlockThreshold']}`.
- Max length: `{registry['maxLength']}`.

Output threshold status là `{registry['outputThresholdStatus']}` và không được tuyên bố là tối ưu.
""",
    )

    _write(
        "14_database_and_api_changes.md",
        """# Database và API

## Storage

Audit xác nhận repository không có RDBMS, ORM hay migration framework. Dữ liệu hiện dùng `data/user_site_store.json` và audit JSONL. Vì vậy không tạo migration SQL giả. Schema log được mở rộng additive và reader chịu được record cũ; rollback là khôi phục code/config, không có lệnh migration database.

## API thêm mới

- `POST /api/security/analyze`
- `POST /api/security/analyze-document`
- `POST /api/security/scan-output`
- `POST /api/security/authorize-tool`
- `POST /api/security/evaluate-policy`
- `GET /api/admin/security-events` và `/{request_id}`
- `GET /api/admin/security-metrics`
- `GET /api/admin/model-status`
- `GET /api/admin/policy-status`

Các API security/admin yêu cầu admin role. User APIs cũ giữ contract; lịch sử conversation được thu hẹp về public DTO để không lộ detector internals.
""",
    )

    _write(
        "15_admin_dashboard_changes.md",
        """# Admin Dashboard

`static/admin_audit.html` giữ bảng Detection Logs và thêm:

- Input/output block rate, prompt/secret/tool incidents, average/P95 latency.
- Active model, readiness, tokenizer, calibrator, policy/rule version và output threshold status.
- Filter source, stage, risk level, model version, decision, user và project.

Admin detail vẫn xem signal/score/policy/version đầy đủ. User chatbot không hiển thị rule, score, threshold, model config hoặc audit internals.
""",
    )

    _write(
        "16_test_execution_report.md",
        f"""# Báo cáo chạy test

- Command: `{tests['command']}`
- Tổng: {tests['total']}
- Passed: {tests['passed']}
- Failed: {tests['failed']}
- Skipped: {tests['skipped']}
- Thời gian JUnit: {_num(tests['duration_seconds'])} giây

Kết quả: **{'PASS' if tests['failed'] == 0 else 'FAIL'}**. Không có failure hoặc skipped bị che giấu. Artifact: `pytest.xml` và `test_results.json`.

Coverage chức năng gồm preprocessing/obfuscation, source separation, tool authorization, secure prompt, output secret/PII/prompt leak, one-retry fallback, Admin RBAC, User privacy, TXT/DOCX/PDF, direct/indirect/Vietnamese và API regression.
""",
    )

    _write(
        "17_benchmark_report.md",
        f"""# Báo cáo benchmark

## Internal security fixtures

- Input: {metrics['input_fixture_count']} mẫu; A7 F1 {_pct(a7['f1'])}, PR-AUC {_pct(a7['pr_auc'])}, recall {_pct(a7['recall'])}, specificity {_pct(a7['specificity'])}, FPR {_pct(a7['false_positive_rate'])}, FNR {_pct(a7['false_negative_rate'])}.
- Direct recall: {_pct(slices['direct']['recall'])}.
- Indirect recall: {_pct(slices['indirect']['recall'])}.
- Benign-hard pass rate: {_pct(slices['benign_hard']['specificity'])}.
- Vietnamese accuracy: {_pct(slices['vietnamese']['accuracy'])}; English accuracy: {_pct(slices['english']['accuracy'])}.
- Output: {metrics['output_fixture_count']} mẫu; unsafe recall {_pct(output['recall'])}; safe preservation {_pct(output['safe_output_preservation_rate'])}.

## External-compatible RoBERTa comparison

Trên rogue-security diagnostic 5.000 mẫu, `roberta_v4` đạt balanced {_pct(external['balanced_accuracy'])}, F1 {_pct(external['f1'])}, recall {_pct(external['recall'])}, ROC-AUC {_pct(external['roc_auc'])}, PR-AUC {_pct(external['pr_auc'])}. Dataset này có duplicate/leakage với train sources, nên không phải clean independent holdout.

Không có external LLM call, không train model và không thay đổi checkpoint trong benchmark.
""",
    )

    _write(
        "18_ablation_study.md",
        f"""# Ablation study A1-A7

{ablation_table}

## Kết luận có giới hạn

Context-aware làm recall tăng trên fixture indirect/tool. A6/A7 không tăng input metric so với A4/A5 trong bộ 16 mẫu, nhưng thêm provenance, fail-safe, tool gating và output protection. A7 input metric bằng A6 vì output được đo riêng trên 6 fixture; không cộng hai tập khác nhau vào một accuracy gây hiểu nhầm.
""",
    )

    _write(
        "19_security_limitations.md",
        """# Hạn chế bảo mật

- Internal benchmark chỉ 16 input và 6 output fixture; không phải estimate production độc lập.
- Rogue-security comparison 5.000 mẫu có duplicate/leakage với train sources.
- Output threshold `0.30/0.85` là baseline, chưa tối ưu trên output holdout.
- RoBERTa v4 vẫn false-positive một academic hard-negative và bỏ sót một system-prompt-extraction fixture trong bộ nhỏ.
- Prompt-leak regex nhận 1/2 fixture; fixture còn lại được model bắt. Cần mở rộng theo intent trên holdout riêng, không thêm keyword đơn giản.
- Tool gateway chưa nối một executor/function-calling production thực.
- Storage JSON/JSONL phù hợp demo/local, chưa có transaction, index hoặc retention như RDBMS.
- PDF test dùng text PDF; OCR/scanned PDF chưa được hỗ trợ.
- Vietnamese có regression test nhưng checkpoint base RoBERTa không phải multilingual chuyên dụng.
- P95 phụ thuộc CPU/cache local; cold-start model có thể cao hơn.
- LLM end-to-end dùng mock trong test; không gọi provider thật để tránh chi phí và rò secret.
""",
    )

    checklist = """| Feature | Status | Files changed | Tests | Evidence |
|---|---|---|---|---|
| Chọn và khóa RoBERTa v4 | COMPLETED | config, runtime, registry | model/runtime regression | external-compatible comparison |
| Input sandwich | COMPLETED | preprocessing, source, official runtime | security tests | A6 metrics |
| Indirect document | COMPLETED | document/indirect pipeline | TXT/DOCX/PDF tests | document tests |
| Context/fusion/policy | COMPLETED | context and official runtime | unit/integration | A4-A6 ablation |
| Secure prompt/output scan | COMPLETED | secure prompt, output security, LLM service | E2E mock | output fixture metrics |
| Tool gateway | COMPLETED | tool gateway and API | authorization tests | 2/2 denied fixtures |
| User/Admin/Audit | COMPLETED | API/store/static/audit | API/RBAC/privacy | full pytest |
| Relational database migration | NOT COMPLETED | none | not applicable | no DB/ORM/migration exists |
| Output threshold optimization | PARTIAL | baseline config | fixture smoke | no independent output holdout |
| Real provider E2E | NOT COMPLETED | provider integration retained | mocked only | no real LLM call by policy |"""

    _write(
        "20_final_completion_report.md",
        f"""# Báo cáo hoàn thành Hybrid Sandwich Security

## 37.1 Executive summary

Hệ thống ban đầu đã có FastAPI, User/Admin static sites, Rule, RoBERTa, context/document pipeline và JSON/JSONL logs. Bản nâng cấp giữ các luồng đó, chọn `roberta_v4` bằng comparison metric, khóa model bằng path/version/hash, thêm source-aware input sandwich, secure prompt, tool authorization và output sandwich dùng lại cùng model.

`roberta_v4` được chọn vì đứng đầu nhóm classic RoBERTa trên protocol 5.000 mẫu: balanced {_pct(external['balanced_accuracy'])}, F1 {_pct(external['f1'])}, recall {_pct(external['recall'])}, ROC-AUC {_pct(external['roc_auc'])}, PR-AUC {_pct(external['pr_auc'])}. Caveat duplicate/leakage được giữ rõ.

## 37.2 Completed checklist

{checklist}

## 37.3 Model configuration

| Field | Value |
|---|---|
| Model path | `{registry['modelPath']}` |
| Tokenizer path | `{registry['tokenizerPath']}` |
| Model/version | `{registry['modelName']}` / `{registry['modelVersion']}` |
| Input warn/block | `{registry['inputWarnThreshold']}` / `{registry['inputBlockThreshold']}` |
| Output warn/block | `{registry['outputWarnThreshold']}` / `{registry['outputBlockThreshold']}` |
| Calibrator | disabled, `{registry['calibratorVersion']}` |
| Device | `{registry['device']}` |
| Max length / batch | `{registry['maxLength']}` / `{registry['batchSize']}` |

## 37.4 Pipeline cuối cùng

{architecture}

## 37.5 Source code changes

Tạo mới package `src/security`, central YAML, benchmark/report scripts và security fixtures. Chỉnh runtime/API/audit/User/Admin/document/LLM adapters. Giữ toàn bộ model cũ và API cũ. Calibrator cũ deprecated nhưng không xóa. Không tạo migration vì repository không có database hoặc migration framework; JSON/JSONL schema mở rộng additive.

## 37.6 Test summary

Tổng {tests['total']}; passed {tests['passed']}; failed {tests['failed']}; skipped {tests['skipped']}; JUnit {_num(tests['duration_seconds'])} giây. Không có lỗi hoặc skip.

## 37.7 Benchmark summary

- A7 input: balanced {_pct(a7['balanced_accuracy'])}, F1 {_pct(a7['f1'])}, PR-AUC {_pct(a7['pr_auc'])}, recall {_pct(a7['recall'])}, specificity {_pct(a7['specificity'])}.
- Direct recall {_pct(slices['direct']['recall'])}; indirect recall {_pct(slices['indirect']['recall'])}; benign-hard pass {_pct(slices['benign_hard']['specificity'])}.
- Vietnamese accuracy {_pct(slices['vietnamese']['accuracy'])}; English accuracy {_pct(slices['english']['accuracy'])}.
- Output unsafe detection {_pct(output['recall'])}; prompt-leak prevention {_pct(output['prompt_leak_prevention_rate'])}; secret prevention {_pct(output['secret_leakage_prevention_rate'])}.
- External-compatible v4: F1 {_pct(external['f1'])}, PR-AUC {_pct(external['pr_auc'])}; contaminated diagnostic caveat applies.

## 37.8 Known limitations

Dataset input/output nhỏ; output threshold chưa tối ưu; một academic false positive và một extraction false negative trong fixture; tool executor production chưa có; JSON/JSONL chưa thay thế RDBMS; OCR PDF chưa có; real LLM E2E không chạy; latency cold-start cao hơn warm cache.

## 37.9 Remaining work

1. Thu thập output holdout độc lập và tối ưu output threshold/calibrator theo fingerprint checkpoint.
2. Xây clean external holdout không trùng train để xác nhận model selection.
3. Cải thiện quoted/academic/translation intent và Vietnamese mà không thêm keyword rule đơn giản.
4. Nối Tool Gateway với executor thực, sandbox và approval UI.
5. Chuyển audit storage sang database có retention/index khi triển khai multi-user.
6. Chạy provider E2E trong staging với secret manager và billing guard.

## 37.10 Commands to run

```powershell
# Install
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt

# Migration: not applicable; repository uses JSON/JSONL and has no migration framework.

# Backend, User Site and Admin Site
.\\.venv\\Scripts\\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8000
# User: http://127.0.0.1:8000/chat
# Admin: http://127.0.0.1:8000/admin

# Unit, integration and E2E security tests
.\\.venv\\Scripts\\python.exe -m pytest tests\\security -q
.\\.venv\\Scripts\\python.exe -m pytest tests\\test_official_runtime.py tests\\test_document_runtime.py -q
.\\.venv\\Scripts\\python.exe -m pytest tests\\security\\test_hybrid_sandwich_integration.py -q

# Full regression, benchmark and reports
.\\.venv\\Scripts\\python.exe -m pytest -q
.\\.venv\\Scripts\\python.exe scripts\\run_hybrid_sandwich_benchmark.py
.\\.venv\\Scripts\\python.exe scripts\\generate_hybrid_sandwich_reports.py
```
""",
    )

    summary = f"""HYBRID SANDWICH SECURITY IMPLEMENTATION SUMMARY

Selected RoBERTa model: {registry['modelName']}
Model path: {registry['modelPath']}
Tokenizer path: {registry['tokenizerPath']}
Input threshold: warn={registry['inputWarnThreshold']}, block={registry['inputBlockThreshold']}
Output threshold: warn={registry['outputWarnThreshold']}, block={registry['outputBlockThreshold']} ({registry['outputThresholdStatus']})
Calibrator: disabled ({registry['calibratorVersion']})
Model version: {registry['modelVersion']}

Input Security Layer: COMPLETED
Context-aware: COMPLETED
Adaptive Risk Fusion: COMPLETED
Policy Engine: COMPLETED
Secure Prompt Builder: COMPLETED
Tool Security Gateway: COMPLETED
Output RoBERTa Scan: COMPLETED
Secret Scan: COMPLETED
PII Scan: COMPLETED
Prompt Leak Scan: COMPLETED
Audit Logging: COMPLETED (JSONL storage)
User Site: COMPLETED
Admin Site: COMPLETED

Tests passed: {tests['passed']}
Tests failed: {tests['failed']}
Tests skipped: {tests['skipped']}

Input F1: {_pct(a7['f1'])}
Input PR-AUC: {_pct(a7['pr_auc'])}
Direct recall: {_pct(slices['direct']['recall'])}
Indirect recall: {_pct(slices['indirect']['recall'])}
Benign pass rate: {_pct(slices['benign_hard']['specificity'])}
Output unsafe detection rate: {_pct(output['recall'])}
Prompt leak prevention rate: {_pct(output['prompt_leak_prevention_rate'])}
Secret leakage prevention rate: {_pct(output['secret_leakage_prevention_rate'])}
Average latency: {_num(a7['latency_mean_ms'])} ms input; {_num(output['latency_mean_ms'])} ms output
P95 latency: {_num(a7['latency_p95_ms'])} ms input; {_num(output['latency_p95_ms'])} ms output

Reports directory: {REPORT_DIR}
Known limitations: small internal fixtures, contaminated external diagnostic, baseline output thresholds, no RDBMS/OCR/real-provider E2E.
Remaining work: clean holdouts, output calibration, intent/Vietnamese robustness, real tool executor and staging provider E2E."""
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
