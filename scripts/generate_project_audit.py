"""Sinh bộ Project Audit bằng tiếng Việt.

Script này chỉ đọc source code, datasets, models và reports đã có trong repo.
Không train lại model, không chạy lại benchmark, không sửa checkpoint.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "project_audit"
AUDIT_DATE = "2026-06-28"

MODEL_NAME = {
    "logistic_regression": "Logistic Regression",
    "linear_svm": "Linear SVM",
    "random_forest": "Random Forest",
    "distilbert_v3": "DistilBERT v3",
    "distilbert_v4": "DistilBERT v4",
    "roberta": "RoBERTa",
    "roberta_v3": "RoBERTa v3",
    "roberta_v4": "RoBERTa v4",
    "roberta_v5_vi": "RoBERTa v5 Vietnamese",
    "xlm_roberta": "XLM-RoBERTa",
    "xlm_roberta_v3": "XLM-RoBERTa v3",
    "xlm_roberta_v4": "XLM-RoBERTa v4",
    "xlm_roberta_v5_vi": "XLM-RoBERTa v5 Vietnamese",
}

PRIMARY_MODELS = ["logistic_regression", "random_forest", "roberta", "xlm_roberta"]


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_report(filename: str, content: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / filename).write_text(content.strip() + "\n", encoding="utf-8")


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value: Any) -> int | None:
    number = to_float(value)
    if number is None:
        return None
    return int(round(number))


def safe_div(num: float | int | None, den: float | int | None) -> float | None:
    if num is None or den in (None, 0):
        return None
    return float(num) / float(den)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "N/A"
    number = to_float(value)
    if number is None:
        return str(value)
    return f"{number:.{digits}f}"


def fmt_int(value: Any) -> str:
    number = to_int(value)
    if number is None:
        return "N/A"
    return f"{number:,}"


def md_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(md_cell(v) for v in row) + " |")
    return "\n".join(lines)


def file_count(path: Path, suffixes: tuple[str, ...] | None = None) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return 1
    return sum(
        1
        for item in path.rglob("*")
        if item.is_file() and (suffixes is None or item.suffix.lower() in suffixes)
    )


def list_files(path: Path, suffixes: tuple[str, ...] | None = None) -> list[str]:
    if not path.exists():
        return []
    files: list[str] = []
    for item in sorted(path.rglob("*")):
        if "__pycache__" in item.parts:
            continue
        if item.is_file() and (suffixes is None or item.suffix.lower() in suffixes):
            files.append(rel(item))
    return files


def csv_profile(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    rows = 0
    labels: Counter[str] = Counter()
    datasets: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        for row in reader:
            rows += 1
            if row.get("label") not in (None, ""):
                labels[str(row["label"])] += 1
            if row.get("dataset_name"):
                datasets[str(row["dataset_name"])] += 1
            if row.get("language"):
                languages[str(row["language"])] += 1
    return {
        "exists": True,
        "rows": rows,
        "columns": columns,
        "labels": dict(labels),
        "datasets": dict(datasets),
        "languages": dict(languages),
    }


def git_log_since() -> list[list[str]]:
    try:
        result = subprocess.run(
            [
                "git",
                "log",
                "--since=2026-06-14",
                "--pretty=format:%h|%ad|%s",
                "--date=short",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        return []
    rows = []
    for line in result.stdout.splitlines():
        parts = line.strip().split("|", 2)
        if len(parts) == 3:
            rows.append(parts)
    return rows


def display_model(model: str) -> str:
    return MODEL_NAME.get(model, model.replace("_", " ").title())


def direct_label_counts(dataset: str) -> dict[str, int]:
    meta = read_json(ROOT / "data" / "external_benchmark" / "direct" / "direct_benchmark_metadata.json", {})
    if dataset == "all":
        dist = meta.get("combined", {}).get("label_distribution", {})
    else:
        dist = meta.get("datasets", {}).get(dataset, {}).get("label_distribution", {})
    return {"0": int(dist.get("0", 0)), "1": int(dist.get("1", 0))}


def bipia_label_counts() -> dict[str, int]:
    meta = read_json(ROOT / "data" / "external_benchmark" / "bipia" / "bipia_normalized.metadata.json", {})
    return {"0": int(meta.get("safe_rows", 0)), "1": int(meta.get("injection_rows", 0))}


def enrich_metrics(raw: dict[str, Any], labels: dict[str, int] | None = None) -> dict[str, Any]:
    """Bổ sung TP/TN/TNR/FPR/FNR/BalAcc/MCC từ confusion matrix hoặc label counts."""
    data = dict(raw)

    cm = data.get("confusion_matrix")
    if isinstance(cm, list) and len(cm) == 2 and len(cm[0]) == 2 and len(cm[1]) == 2:
        data.setdefault("tn", cm[0][0])
        data.setdefault("fp", cm[0][1])
        data.setdefault("fn", cm[1][0])
        data.setdefault("tp", cm[1][1])

    fp = to_int(data.get("fp", data.get("false_positive_count")))
    fn = to_int(data.get("fn", data.get("false_negative_count")))
    tp = to_int(data.get("tp"))
    tn = to_int(data.get("tn"))

    if labels:
        negatives = labels.get("0")
        positives = labels.get("1")
        if fp is not None and negatives is not None and tn is None:
            tn = negatives - fp
        if fn is not None and positives is not None and tp is None:
            tp = positives - fn

    data["tn"], data["fp"], data["fn"], data["tp"] = tn, fp, fn, tp
    data["specificity"] = safe_div(tn, (tn or 0) + (fp or 0)) if tn is not None and fp is not None else None
    data["fpr"] = safe_div(fp, (fp or 0) + (tn or 0)) if tn is not None and fp is not None else None
    data["fnr"] = safe_div(fn, (fn or 0) + (tp or 0)) if fn is not None and tp is not None else None
    recall = to_float(data.get("recall"))
    if recall is None and fn is not None and tp is not None:
        recall = safe_div(tp, tp + fn)
        data["recall"] = recall
    specificity = to_float(data.get("specificity"))
    data["balanced_accuracy"] = (
        (recall + specificity) / 2 if recall is not None and specificity is not None else None
    )

    if None not in (tp, tn, fp, fn):
        denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        data["mcc"] = ((tp * tn) - (fp * fn)) / denominator if denominator else None
        data["confusion_matrix_compact"] = f"[[{tn}, {fp}], [{fn}, {tp}]]"
    else:
        data["mcc"] = None
        data["confusion_matrix_compact"] = data.get("confusion_matrix", "N/A")

    return data


def metric_table_row(label: str, data: dict[str, Any]) -> list[Any]:
    return [
        label,
        fmt_int(data.get("rows") or data.get("test_rows")),
        fmt(data.get("threshold")),
        fmt(data.get("accuracy")),
        fmt(data.get("precision")),
        fmt(data.get("recall")),
        fmt(data.get("specificity")),
        fmt(data.get("fpr")),
        fmt(data.get("fnr")),
        fmt(data.get("f1")),
        fmt(data.get("f2")),
        fmt(data.get("roc_auc")),
        fmt(data.get("pr_auc")),
        fmt(data.get("brier")),
        fmt(data.get("ece")),
        fmt(data.get("balanced_accuracy")),
        fmt(data.get("mcc")),
        fmt_int(data.get("tn")),
        fmt_int(data.get("fp")),
        fmt_int(data.get("fn")),
        fmt_int(data.get("tp")),
        data.get("confusion_matrix_compact", "N/A"),
    ]


METRIC_HEADERS = [
    "Model",
    "N",
    "Threshold",
    "Accuracy",
    "Precision",
    "Recall/TPR",
    "Specificity/TNR",
    "FPR",
    "FNR",
    "F1",
    "F2",
    "ROC-AUC",
    "PR-AUC",
    "Brier",
    "ECE",
    "Balanced Acc",
    "MCC",
    "TN",
    "FP",
    "FN",
    "TP",
    "Confusion Matrix",
]


def load_strict_metrics(dataset: str, model: str) -> dict[str, Any]:
    path = ROOT / "reports" / "direct_external_evaluation" / "strict" / dataset / model / "test_metrics.json"
    return read_json(path, {})


def load_calibration_metrics(dataset: str, model: str) -> dict[str, Any]:
    path = (
        ROOT
        / "reports"
        / "direct_external_evaluation"
        / "calibration"
        / dataset
        / model
        / "calibration_metrics.json"
    )
    return read_json(path, {})


def project_tree() -> str:
    exclude_names = {".git", ".venv", "__pycache__", ".pytest_cache", ".matplotlib", ".mypy_cache"}
    exclude_prefixes = (".tmp_pytest",)

    def skip(path: Path) -> bool:
        return path.name in exclude_names or any(path.name.startswith(prefix) for prefix in exclude_prefixes)

    def sorted_children(path: Path) -> list[Path]:
        return sorted([p for p in path.iterdir() if not skip(p)], key=lambda p: (not p.is_dir(), p.name.lower()))

    lines = [f"{ROOT.name}/"]

    def render(path: Path, prefix: str = "") -> None:
        items = sorted_children(path)
        for idx, item in enumerate(items):
            connector = "└── " if idx == len(items) - 1 else "├── "
            suffix = "/" if item.is_dir() else ""
            size = ""
            if item.is_file() and item.stat().st_size > 5_000_000:
                size = f" ({item.stat().st_size / 1024 / 1024:.1f} MB)"
            lines.append(f"{prefix}{connector}{item.name}{suffix}{size}")
            if item.is_dir():
                render(item, prefix + ("    " if idx == len(items) - 1 else "│   "))

    render(ROOT)
    return "\n".join(lines)


def project_structure_report() -> str:
    src_files = list_files(ROOT / "src", (".py",))
    script_files = list_files(ROOT / "scripts", (".py",))
    training_files = list_files(ROOT / "training", (".py",))
    test_files = list_files(ROOT / "tests", (".py",))
    report_files = list_files(ROOT / "reports", (".md", ".json", ".csv"))
    model_files = list_files(ROOT / "models")
    dataset_files = list_files(ROOT / "data", (".csv", ".json", ".jsonl")) + list_files(
        ROOT / "datasets", (".csv", ".json", ".jsonl")
    )

    top_rows = []
    for folder, purpose in [
        ("src", "Runtime chính: API, detector, rule-based, ML/Transformer wrapper, thresholding, indirect detection."),
        ("src/detection", "Hybrid/context-aware detector mới và pipeline score fusion."),
        ("scripts", "Chuẩn bị dataset, benchmark, ablation, calibration, threshold optimization, diagnostics."),
        ("training", "Script chuẩn bị/train dataset/model, gồm indirect detector."),
        ("models", "Model ML truyền thống, vectorizer, checkpoint Transformer, threshold, backup/deprecated checkpoint."),
        ("data", "Raw/processed data và benchmark ngoài như BIPIA/direct benchmark."),
        ("datasets", "Dataset custom, processed, unified, external, test phục vụ train/evaluate."),
        ("reports", "Toàn bộ kết quả evaluation, ablation, calibration, diagnostics, audit."),
        ("tests", "Pytest cho API, detector, Transformer utils, context-aware, indirect, model readiness."),
        ("static", "Frontend HTML demo/chat/batch evaluation."),
        ("configs", "File cấu hình runtime/evaluation."),
        ("docs", "Tài liệu kỹ thuật về dataset, UI, batch evaluation, XLM-R training."),
    ]:
        path = ROOT / folder
        top_rows.append([folder, "Có" if path.exists() else "Thiếu", file_count(path), purpose])

    module_rows = [
        ["src/api.py", "FastAPI backend, health/detect endpoints, static serving."],
        ["src/detector.py", "Orchestrator runtime cho rule/ML/Transformer signals và thresholds."],
        ["src/rule_based.py", "Detector pattern/rule-based cho direct prompt injection."],
        ["src/advanced_detection.py", "Logic detection nâng cao/hybrid và explainable output."],
        ["src/transformer_utils.py", "Load checkpoint Transformer, inference, probability extraction."],
        ["src/thresholding.py", "Policy/helper cho threshold evaluation/runtime."],
        ["src/detection/context_aware_detector.py", "Phát hiện mâu thuẫn instruction-context, role boundary, context risk."],
        ["src/detection/pipeline.py", "Pipeline hybrid: model score + context score + optional rule score → fusion → output."],
        ["src/indirect_pipeline.py", "Pipeline cho indirect prompt injection trong external/untrusted content."],
        ["src/indirect_rule_detector.py", "Rule-based indirect injection detector."],
        ["src/indirect_context_detector.py", "Heuristic context-aware cho indirect injection."],
        ["src/external_content.py", "Parse/validate external content."],
        ["src/safe_context_builder.py", "Tạo context boundary an toàn hơn cho untrusted content."],
        ["src/language_utils.py", "Helper ngôn ngữ/Vietnamese."],
        ["src/train_models.py", "Training pipeline ML truyền thống."],
        ["src/train_transformers.py", "Fine-tune Transformer."],
        ["src/train_transformers_continue.py", "Resume/continue Transformer training."],
        ["src/batch_evaluation.py", "Batch evaluation và report generation."],
        ["src/compare_models.py", "So sánh model."],
        ["src/dataset_unification.py", "Merge/unify datasets."],
        ["src/data_loader.py", "Load/validate dataset."],
        ["src/data_augmentation.py", "Augmentation, gồm multilingual/Vietnamese prompts."],
        ["src/model_diagnostics.py", "Diagnostics score/prediction."],
        ["src/quality_reports.py", "Stress/category reports."],
        ["src/benign_intent.py", "Heuristic giảm false positive với benign intent."],
        ["src/preprocessing.py", "Tiền xử lý text."],
        ["src/evaluate.py", "Evaluation metrics/report utilities."],
        ["src/file_utils.py", "Filesystem helpers."],
    ]

    script_rows = [
        ["Benchmark", "prepare_bipia_benchmark.py, evaluate_bipia.py, evaluate_bipia_ablation.py, prepare_direct_benchmarks.py, evaluate_direct_benchmarks.py, evaluate_direct_benchmarks_strict.py"],
        ["Calibration/threshold", "calibrate_direct_scores.py, analyze_direct_score_distribution.py, calibrate_thresholds.py, calibrate_transformer_thresholds.py, optimize_threshold.py, optimize_thresholds.py"],
        ["Dataset preparation", "build_unified_dataset.py, build_transformer_v2_dataset.py, build_transformer_v4_dataset.py, prepare_geekyrakshit_dataset.py, prepare_jayavibhav_prompt_injection.py"],
        ["Training/diagnostics", "run_transformer_v3_full_training.py, run_xlm_roberta_v3_recovery_training.py, finalize_xlm_roberta_v3_checkpoint.py, diagnose_transformer_pipeline.py, run_dataset_evaluation.py"],
    ]

    transformer_rows = []
    transformer_root = ROOT / "models" / "transformers"
    if transformer_root.exists():
        for model_dir in sorted([p for p in transformer_root.iterdir() if p.is_dir()], key=lambda p: p.name):
            has_weights = any((model_dir / name).exists() for name in ["model.safetensors", "pytorch_model.bin"])
            transformer_rows.append(
                [
                    model_dir.name,
                    "Có" if has_weights else "Thiếu",
                    "Có" if (model_dir / "metrics.json").exists() else "Không",
                    "Có" if (model_dir / "training_metadata.json").exists() else "Không",
                    "Có" if (model_dir / "probability_calibrator.joblib").exists() else "Không",
                ]
            )

    direct_meta = read_json(ROOT / "data" / "external_benchmark" / "direct" / "direct_benchmark_metadata.json", {})
    bipia_meta = read_json(ROOT / "data" / "external_benchmark" / "bipia" / "bipia_normalized.metadata.json", {})
    dataset_rows = []
    for name, meta in direct_meta.get("datasets", {}).items():
        dataset_rows.append(
            [
                f"direct/{name}",
                meta.get("status", "N/A"),
                fmt_int(meta.get("rows_after_normalization")),
                meta.get("label_distribution", {}),
                "Benchmark ngoài, không dùng train",
            ]
        )
    if direct_meta.get("combined"):
        dataset_rows.append(
            [
                "direct/all",
                direct_meta["combined"].get("status", "N/A"),
                fmt_int(direct_meta["combined"].get("rows_after_normalization")),
                direct_meta["combined"].get("label_distribution", {}),
                "Benchmark ngoài tổng hợp",
            ]
        )
    if bipia_meta:
        dataset_rows.append(
            [
                "BIPIA normalized",
                "success-with-warnings" if bipia_meta.get("warnings") else "success",
                fmt_int(bipia_meta.get("rows")),
                {"0": bipia_meta.get("safe_rows"), "1": bipia_meta.get("injection_rows")},
                "Benchmark ngoài; hiện là subset",
            ]
        )

    profiles = []
    for dataset in [
        "datasets/unified/prompt_injection_ml_ready.csv",
        "datasets/unified/prompt_injection_transformer_ready_v4.csv",
        "datasets/processed/hf_prompt_injection_test.csv",
        "datasets/processed/vi_train_processed.csv",
        "data/processed/augmented_multilingual_dataset.csv",
    ]:
        profile = csv_profile(ROOT / dataset)
        if profile.get("exists"):
            profiles.append([dataset, fmt_int(profile["rows"]), profile["labels"], profile["languages"] or "không có field language"])

    return f"""
# Audit cấu trúc dự án

Ngày sinh báo cáo: {AUDIT_DATE}

Phạm vi: repo local tại `{ROOT}`. Các thư mục cache như `.git`, `.venv`, `__pycache__`, pytest tmp không được tính như module chức năng. Snapshot cây thư mục nằm ở `project_tree.txt`.

## 1. Inventory cấp cao

{md_table(["Folder", "Trạng thái", "Số file", "Vai trò"], top_rows)}

## 2. Module source code

Số Python source file dưới `src/`: {len(src_files)}.

{md_table(["Module", "Chức năng"], module_rows)}

## 3. Scripts

Số script Python dưới `scripts/`: {len(script_files)}. Số training helper dưới `training/`: {len(training_files)}.

{md_table(["Nhóm", "Scripts"], script_rows)}

## 4. Models/checkpoints

Artifact ML truyền thống cấp root gồm Logistic Regression, Linear SVM, Random Forest, vectorizer tương ứng, `model_thresholds.json`, `thresholds.json`, `transformer_thresholds.json`.

{md_table(["Transformer directory", "Weights", "metrics.json", "training_metadata.json", "probability calibrator"], transformer_rows)}

Tổng số file scan dưới `models/`: {len(model_files)}. Con số này gồm active checkpoints, backup và deprecated checkpoints.

## 5. Dataset/benchmark

Số file dataset/benchmark scan dưới `data/` và `datasets/`: {len(dataset_files)}.

{md_table(["Dataset", "Trạng thái", "Rows", "Label distribution", "Mục đích"], dataset_rows)}

Một số dataset train/evaluate nội bộ:

{md_table(["Path", "Rows", "Label distribution", "Language field"], profiles)}

## 6. Reports/evaluation artifacts

Số report artifact scan dưới `reports/`: {len(report_files)}.

- `reports/bipia_evaluation/`: BIPIA evaluation, ablation, predictions, FP/FN.
- `reports/direct_external_evaluation/`: direct external benchmark, strict split, calibration, score analysis.
- `reports/batch_evaluation/`: batch runs, predictions, metrics, disagreement.
- `reports/threshold_optimization/`: threshold recommendations.
- `reports/indirect_evaluation/`: indirect detector metrics/predictions.
- `reports/quality/`: stress/category diagnostics.
- Root report files: model comparison, transformer diagnostics, calibration, readiness, error analysis.

## 7. API, frontend, pipeline

- API: `src/api.py`.
- Frontend: `static/advanced_demo.html`, `static/batch_evaluation.html`, `static/chat_simulation.html`.
- Direct runtime pipeline: input → rule/ML/Transformer scoring → threshold → action/explanation.
- Hybrid context-aware pipeline: input → model score + context score (+ optional rule score) → fusion → threshold → explainable output.
- Indirect pipeline: external/untrusted content → rule/context indirect detection → safety decision.

## 8. Tests

Số test file: {len(test_files)}. Test bao phủ API schema, detectors, Random Forest, Transformer utils, inference consistency, context-aware pipeline, indirect detection, Vietnamese indirect detection, model readiness, batch evaluation, dataset unification.
"""


def completed_work_report() -> str:
    direct_meta = read_json(ROOT / "data" / "external_benchmark" / "direct" / "direct_benchmark_metadata.json", {})
    bipia_meta = read_json(ROOT / "data" / "external_benchmark" / "bipia" / "bipia_normalized.metadata.json", {})

    direct_rows = [
        [
            name,
            meta.get("status", "N/A"),
            fmt_int(meta.get("rows_after_normalization")),
            meta.get("label_distribution", {}),
            meta.get("error", ""),
        ]
        for name, meta in direct_meta.get("datasets", {}).items()
    ]
    if direct_meta.get("combined"):
        direct_rows.append(
            [
                "combined/direct_all",
                direct_meta["combined"].get("status", "N/A"),
                fmt_int(direct_meta["combined"].get("rows_after_normalization")),
                direct_meta["combined"].get("label_distribution", {}),
                "Gộp deepset + neuralchemy + cyberec",
            ]
        )

    return f"""
# Tổng hợp công việc đã hoàn thành trong 2 tuần gần đây

Ngày sinh báo cáo: {AUDIT_DATE}

Báo cáo này dựa trên Git history từ 2026-06-14 và artifacts hiện có trong repo.

## 1. Mốc commit

{md_table(["Commit", "Ngày", "Nội dung"], git_log_since())}

## 2. Dataset

Đã hoàn thành:

- Duy trì dataset nội bộ/unified cho ML và Transformer.
- Thêm BIPIA benchmark tại `data/external_benchmark/bipia/bipia_normalized.csv`.
- Thêm direct external benchmark tại `data/external_benchmark/direct/`.
- Giữ benchmark ngoài tách khỏi training data.
- Có dataset custom/hard-negative và dataset tiếng Việt processed.

Direct external benchmark:

{md_table(["Dataset", "Trạng thái", "Rows", "Label distribution", "Ghi chú"], direct_rows)}

BIPIA:

- Source: `{bipia_meta.get("source", "N/A")}`, commit `{bipia_meta.get("source_commit", "N/A")}`.
- Rows: {fmt_int(bipia_meta.get("rows"))}; safe: {fmt_int(bipia_meta.get("safe_rows"))}; injection: {fmt_int(bipia_meta.get("injection_rows"))}.
- Task thực tế trong file normalized: `{bipia_meta.get("source_task_counts", {})}`.
- Difficulty: `{bipia_meta.get("difficulty_counts", {})}`.
- Warning: {bipia_meta.get("warnings", []) or "Không có"}.

Lưu ý quan trọng: BIPIA hiện là subset 1,000 dòng; QA/abstract bị skip vì thiếu context raw. Không nên trình bày là full BIPIA benchmark.

## 3. Model

Đã có:

- Logistic Regression, Linear SVM, Random Forest và vectorizer.
- Transformer checkpoints: DistilBERT, RoBERTa, XLM-RoBERTa, các phiên bản v3/v4/v5 Vietnamese.
- Threshold snapshots: `models/model_thresholds.json`, `models/transformer_thresholds.json`, `models/thresholds.json`.
- DistilBERT v3 có `probability_calibrator.joblib`.
- RoBERTa/XLM-R direct calibration hiện nằm ở report-level artifacts, chưa persist ngược thành runtime model calibrator.

## 4. Detection pipeline

Đã hoàn thành:

- Rule-based direct detector.
- Traditional ML detector.
- Transformer detector.
- Context-aware detector.
- Hybrid score fusion pipeline.
- Explainable output.
- Threshold optimization.
- Indirect prompt injection pipeline.

## 5. Evaluation

{md_table(
    ["Evaluation", "Artifact", "Cấu hình/model", "Scope"],
    [
        ["BIPIA ablation", "reports/bipia_evaluation/", "RF, RoBERTa, XLM-R, RoBERTa+Context, XLM-R+Context", "1,000 rows"],
        ["Direct external raw", "reports/direct_external_evaluation/", "LR, RF, RoBERTa, XLM-R", "all=64,785; deepset=662"],
        ["Direct strict split", "reports/direct_external_evaluation/strict/", "LR, RF, RoBERTa, XLM-R", "validation 30% / test 70%"],
        ["Score analysis", "reports/direct_external_evaluation/score_analysis/", "LR, RF, RoBERTa, XLM-R", "score distribution + ROC/PR plots"],
        ["Calibration", "reports/direct_external_evaluation/calibration/", "LR, RF, RoBERTa, XLM-R", "Brier/ECE + Platt/isotonic"],
        ["Indirect evaluation", "reports/indirect_evaluation/", "Indirect detector", "local indirect cases"],
        ["Batch evaluation", "reports/batch_evaluation/", "multi-model batch runs", "nhiều run theo thời gian"],
    ],
)}
"""


def current_metrics_summary_report() -> str:
    direct_raw_rows = read_csv(ROOT / "reports" / "direct_external_evaluation" / "direct_ablation_summary.csv")
    bipia_rows_raw = read_csv(ROOT / "reports" / "bipia_evaluation" / "ablation_comparison.csv")

    direct_all, direct_deepset = [], []
    for row in direct_raw_rows:
        labels = direct_label_counts(row.get("Dataset", ""))
        data = enrich_metrics(
            {
                "rows": row.get("Rows"),
                "threshold": row.get("Threshold"),
                "accuracy": row.get("Accuracy"),
                "precision": row.get("Precision"),
                "recall": row.get("Recall"),
                "f1": row.get("F1"),
                "f2": row.get("F2"),
                "roc_auc": row.get("ROC-AUC"),
                "pr_auc": row.get("PR-AUC"),
                "fp": row.get("FP"),
                "fn": row.get("FN"),
            },
            labels,
        )
        target = direct_all if row.get("Dataset") == "all" else direct_deepset
        target.append(metric_table_row(row.get("Model", "N/A"), data))

    strict_all, strict_deepset = [], []
    raw_cal_all, cal_all, raw_cal_deepset, cal_deepset = [], [], [], []
    for dataset in ["all", "deepset"]:
        for model in PRIMARY_MODELS:
            strict = load_strict_metrics(dataset, model)
            if strict:
                target = strict_all if dataset == "all" else strict_deepset
                target.append(metric_table_row(display_model(model), enrich_metrics(strict)))

            cal = load_calibration_metrics(dataset, model)
            if cal:
                raw_target = raw_cal_all if dataset == "all" else raw_cal_deepset
                cal_target = cal_all if dataset == "all" else cal_deepset
                raw_target.append(
                    metric_table_row(
                        f"{display_model(model)} raw",
                        enrich_metrics(cal.get("raw_test_metrics", {})),
                    )
                )
                cal_target.append(
                    metric_table_row(
                        f"{display_model(model)} calibrated ({cal.get('selected_calibration_method', 'N/A')})",
                        enrich_metrics(cal.get("calibrated_test_metrics", {})),
                    )
                )

    bipia_labels = bipia_label_counts()
    bipia_rows = []
    for row in bipia_rows_raw:
        data = enrich_metrics(
            {
                "rows": row.get("Rows"),
                "threshold": row.get("Threshold"),
                "accuracy": row.get("Accuracy"),
                "precision": row.get("Precision"),
                "recall": row.get("Recall"),
                "f1": row.get("F1"),
                "f2": row.get("F2"),
                "roc_auc": row.get("ROC-AUC"),
                "pr_auc": row.get("PR-AUC"),
                "fp": row.get("FP"),
                "fn": row.get("FN"),
            },
            bipia_labels,
        )
        bipia_rows.append([row.get("Context-aware", "N/A"), *metric_table_row(row.get("Model", "N/A"), data)])

    threshold_rows = []
    ml_thresholds = read_json(ROOT / "models" / "model_thresholds.json", {})
    for name, data in ml_thresholds.items():
        threshold_rows.append(
            [
                display_model(name),
                fmt(data.get("evaluation_threshold")),
                fmt(data.get("runtime_warn_threshold")),
                fmt(data.get("runtime_block_threshold")),
                data.get("best_metric"),
                fmt(data.get("precision")),
                fmt(data.get("recall")),
                fmt(data.get("f1")),
                fmt(data.get("f2")),
                f"[[{data.get('tn')}, {data.get('fp')}], [{data.get('fn')}, {data.get('tp')}]]",
                rel(Path(data.get("dataset", ""))) if data.get("dataset") else "N/A",
            ]
        )
    transformer_thresholds = read_json(ROOT / "models" / "transformer_thresholds.json", {}).get("models", {})
    for name, data in transformer_thresholds.items():
        threshold_rows.append(
            [
                display_model(name),
                fmt(data.get("evaluation_threshold")),
                fmt(data.get("runtime_warn_threshold")),
                fmt(data.get("runtime_block_threshold")),
                data.get("best_metric"),
                fmt(data.get("precision")),
                fmt(data.get("recall")),
                fmt(data.get("f1")),
                fmt(data.get("f2")),
                f"[[{data.get('tn')}, {data.get('fp')}], [{data.get('fn')}, {data.get('tp')}]]",
                rel(Path(data.get("dataset", ""))) if data.get("dataset") else "N/A",
            ]
        )

    return f"""
# Tổng hợp metrics hiện tại

Ngày sinh báo cáo: {AUDIT_DATE}

Bảng này mở rộng thêm các chỉ số ngoài Accuracy/Precision/Recall/F1/F2:

- `Specificity/TNR`: tỷ lệ benign được nhận diện đúng.
- `FPR`: false positive rate.
- `FNR`: false negative rate.
- `Balanced Acc`: trung bình của Recall/TPR và Specificity/TNR.
- `MCC`: Matthews Correlation Coefficient, hữu ích khi label mất cân bằng.
- `TN/FP/FN/TP` và Confusion Matrix.
- `Brier/ECE` chỉ xuất hiện ở calibration reports; các benchmark không có calibration thì để `N/A`.

## 1. Direct external benchmark - raw full-dataset threshold optimization

Nguồn: `reports/direct_external_evaluation/direct_ablation_summary.csv`.

### Dataset: all direct external benchmarks

{md_table(METRIC_HEADERS, direct_all)}

### Dataset: deepset only

{md_table(METRIC_HEADERS, direct_deepset)}

## 2. Direct external benchmark - strict validation/test protocol

Nguồn: `reports/direct_external_evaluation/strict/*/*/test_metrics.json`. Threshold được chọn trên validation rồi mới áp dụng trên test.

### Strict dataset: all

{md_table(METRIC_HEADERS, strict_all)}

### Strict dataset: deepset

{md_table(METRIC_HEADERS, strict_deepset)}

## 3. Direct external calibration

Nguồn: `reports/direct_external_evaluation/calibration/*/*/calibration_metrics.json`. Calibration fit trên validation và đánh giá trên test. Temperature scaling chưa chạy được vì prediction CSV hiện chỉ lưu score/probability, không lưu logits.

### Raw test metrics có Brier/ECE - all

{md_table(METRIC_HEADERS, raw_cal_all)}

### Calibrated test metrics - all

{md_table(METRIC_HEADERS, cal_all)}

### Raw test metrics có Brier/ECE - deepset

{md_table(METRIC_HEADERS, raw_cal_deepset)}

### Calibrated test metrics - deepset

{md_table(METRIC_HEADERS, cal_deepset)}

## 4. BIPIA ablation study

Nguồn: `reports/bipia_evaluation/ablation_comparison.csv`. BIPIA không có Brier/ECE vì chưa chạy calibration riêng cho ablation này.

{md_table(["Context-aware", *METRIC_HEADERS], bipia_rows)}

## 5. Internal/runtime threshold snapshots

Nguồn: `models/model_thresholds.json` và `models/transformer_thresholds.json`. Các threshold này không so sánh trực tiếp với direct/BIPIA vì protocol và dataset khác nhau.

{md_table(
    ["Model", "Eval threshold", "Warn threshold", "Block threshold", "Best metric", "Precision", "Recall", "F1", "F2", "Confusion Matrix", "Dataset"],
    threshold_rows,
)}

## 6. Ghi chú về khác biệt/mâu thuẫn số liệu

- Direct raw và direct strict khác nhau vì raw tối ưu threshold trên toàn evaluation set, còn strict chọn threshold trên validation rồi report test.
- BIPIA context-aware không phải là bằng chứng pure model generalization; đó là bằng chứng đóng góp của context-aware fusion.
- BIPIA hiện là subset 1,000 dòng, chưa phải full benchmark.
- RoBERTa/XLM-R có threshold thấp không chỉ do bug sweep; strict validation/test vẫn chọn threshold thấp. Calibration giúp threshold dễ diễn giải hơn.
- Brier/ECE chỉ có trong calibration reports, không có ở BIPIA ablation.
"""


def research_contributions_report() -> str:
    rows = [
        ["Hybrid detection pipeline", "Engineering + research", "Kết hợp rule/model/context score và explainable output; có giá trị nếu gắn với ablation."],
        ["Context-aware detector", "Research direction nhưng còn heuristic", "BIPIA tăng mạnh, nhưng logic hiện còn rule/context pattern nhiều hơn semantic reasoning."],
        ["Explainable output", "Applied research + engineering", "Hữu ích cho security interpretability; chưa có đánh giá chất lượng explanation."],
        ["Threshold optimization", "Engineering + evaluation methodology", "Quan trọng cho deployment/fair comparison, không mới nếu đứng riêng."],
        ["Strict validation/test thresholding", "Research/evaluation contribution", "Giảm leakage/overfitting threshold, làm claim generalization đáng tin hơn."],
        ["Calibration analysis", "Research/evaluation contribution", "Brier/ECE cho thấy score tốt không đồng nghĩa probability đáng tin."],
        ["BIPIA ablation study", "Research contribution", "Tách đóng góp RF/RoBERTa/XLM-R/context-aware."],
        ["Direct external benchmark", "Research/evaluation contribution", "Thêm OOD evidence ngoài internal train/test."],
        ["Multilingual/Vietnamese path", "Research direction đang mở", "Có dữ liệu/checkpoint, nhưng XLM-R hiện chưa đủ mạnh để claim robust multilingual."],
        ["Frontend/API demo", "Engineering", "Cần cho demo, không phải novelty chính."],
    ]
    return f"""
# Đóng góp nghiên cứu hiện tại

Ngày sinh báo cáo: {AUDIT_DATE}

## 1. Phân loại đóng góp

{md_table(["Đóng góp", "Phân loại", "Đánh giá"], rows)}

## 2. Phần có giá trị nghiên cứu mạnh nhất

- Evaluation-driven security pipeline: external benchmarks, ablation, strict thresholding, calibration.
- Context-aware fusion: có evidence rõ trên BIPIA, nhưng cần nâng cấp semantic để claim mạnh hơn.
- Calibration: giúp chỉ ra vấn đề reliability của probability/softmax score.
- OOD analysis: RoBERTa và XLM-R có hành vi rất khác nhau trên direct external benchmark.

## 3. Phần chủ yếu là engineering

- API/frontend.
- Batch/report plumbing.
- Dataset format normalization.
- Threshold sweep thông thường.
- Wrapper training/evaluation.

## 4. Claim nên dùng và không nên dùng

Nên dùng:

> Dự án xây dựng và đánh giá một hệ thống hybrid prompt-injection detection kết hợp fine-tuned classifiers, context-aware signals, threshold optimization, calibration analysis và external benchmark ablations.

Chưa nên dùng:

> Dự án đã giải quyết hoàn toàn multilingual semantic prompt injection detection.
"""


def roadmap_report() -> str:
    rows = [
        ["Dataset preparation", "Hoàn thành phần lớn", "data/, datasets/, metadata", "Thêm dataset cards + hash"],
        ["ML baseline", "Hoàn thành", "LR/SVM/RF joblib + metrics", "Giữ RF làm baseline truyền thống"],
        ["Transformer training", "Hoàn thành nhưng cần cải thiện", "RoBERTa/XLM-R checkpoints", "Fine-tune XLM-R + hard negatives"],
        ["Threshold optimization", "Hoàn thành", "threshold JSON + strict reports", "Đặt strict protocol làm default"],
        ["Calibration", "Đang thực hiện", "Brier/ECE reports", "Persist calibrator/threshold runtime"],
        ["Context-aware", "Đang thực hiện", "src/detection + BIPIA ablation", "Nâng cấp semantic context-aware"],
        ["BIPIA", "Đang thực hiện", "1,000-row subset", "Mở rộng/ghi rõ giới hạn full BIPIA"],
        ["OOD/direct benchmark", "Hoàn thành bản đầu", "direct_all 64,785 rows", "Per-source + common failure analysis"],
        ["Explainable AI", "Đang thực hiện", "explainable output", "Đánh giá chất lượng explanation"],
        ["Frontend/API", "Có", "src/api.py + static/", "Hiển thị calibrated/context-aware results"],
        ["Indirect protocol", "Chưa đủ", "indirect detector có nhưng eval còn hạn chế", "Tách direct/indirect training/eval"],
    ]
    return f"""
# Roadmap tiếp theo

Ngày sinh báo cáo: {AUDIT_DATE}

## Priority 1 - cần làm trước khi claim nghiên cứu mạnh

1. Chuẩn hóa strict external evaluation làm protocol chính.
2. Tích hợp calibration vào runtime/evaluation pipeline.
3. Lưu logits ở future predictions để test Temperature Scaling.
4. Fine-tune XLM-R trên multilingual/hard-negative dataset.
5. Nâng context-aware từ heuristic lên semantic boundary reasoning.
6. Viết dataset cards/hash cho BIPIA subset, direct_all, internal unified, Vietnamese datasets.

## Priority 2 - tăng sức nặng cho bảo vệ đồ án

1. OOD benchmark theo từng source, không chỉ pooled metrics.
2. Hard-negative mining từ false positives/false negatives.
3. Separate indirect injection protocol với RAG/email/webpage contexts.
4. Đánh giá explanation quality.
5. One-command benchmark manifest: dataset hash, model path, threshold source, calibration state.

## Priority 3 - mở rộng sau khi core ổn định

1. RAG/tool-agent attack simulation.
2. HTML/PDF/email injection extraction.
3. Dashboard visualization cho ROC/PR/calibration/errors.
4. Monitoring drift.
5. Learned fusion/ensemble thay vì fixed fusion.

## Checklist roadmap

{md_table(["Hạng mục", "Trạng thái", "Evidence", "Next action"], rows)}
"""


def project_progress_report() -> str:
    scores = [
        ["Dataset", "8.0/10", "Có internal, external direct, BIPIA subset, Vietnamese/custom datasets; cần dataset cards/hash."],
        ["Machine Learning", "7.5/10", "LR/RF/SVM baseline đủ; RF yếu trên pooled direct external nhưng có giá trị baseline."],
        ["Transformer", "7.0/10", "RoBERTa mạnh; XLM-R yếu; cần calibration runtime và multilingual retraining."],
        ["Security", "7.0/10", "Có direct/indirect/context-aware/BIPIA; cần RAG/tool/PDF/HTML threat coverage."],
        ["Research", "7.5/10", "Ablation + strict external + calibration là nền tốt; cần semantic context và hypothesis rõ hơn."],
        ["Novelty", "6.5/10", "Hybrid/context-aware hứa hẹn nhưng hiện còn heuristic/engineering nhiều."],
        ["Evaluation", "8.0/10", "Metrics phong phú, ablation, strict split, calibration; cần common-failure analysis."],
        ["Explainability", "6.5/10", "Có explainable output nhưng chưa đánh giá chất lượng explanation."],
        ["Overall", "7.4/10", "Prototype capstone mạnh, evaluation tốt; chưa phải research system hoàn chỉnh."],
    ]
    return f"""
# Tiến độ dự án

Ngày sinh báo cáo: {AUDIT_DATE}

## Current Progress - dùng để báo cáo giảng viên tuần tới

### 1. Hệ thống hiện đã làm được gì?

- Detect direct prompt injection bằng rule-based, ML truyền thống và Transformer.
- Chạy hybrid context-aware detection với explainable output.
- Evaluate trên internal datasets, BIPIA subset và direct external benchmarks.
- Chạy ablation cho RF, RoBERTa, XLM-R, RoBERTa+Context, XLM-R+Context.
- Chạy strict validation/test threshold evaluation.
- Chạy calibration analysis với Brier Score và ECE.
- Có FastAPI backend và static demo/batch evaluation UI.

### 2. Hệ thống hoạt động như thế nào?

Runtime: input text/context → detector scores → score fusion/threshold → label/action → explanation.

Evaluation: dataset → model scorer → threshold selection → metrics/errors → report artifacts.

Khi báo cáo, nên nhấn mạnh strict protocol: threshold chọn trên validation, sau đó mới đánh giá test.

### 3. Đã chứng minh được điều gì?

- RoBERTa generalize tốt hơn XLM-R trên pooled direct external benchmark hiện tại.
- Context-aware fusion cải thiện mạnh trên BIPIA subset.
- Random Forest là baseline truyền thống hữu ích nhưng không bắt boundary OOD tốt.
- Probability/softmax score chưa chắc calibrated; cần Brier/ECE.
- Threshold protocol ảnh hưởng lớn tới kết luận.

### 4. Chưa chứng minh được điều gì?

- Chưa chứng minh full BIPIA coverage.
- Chưa chứng minh robust multilingual detection.
- Chưa chứng minh semantic context understanding.
- Chưa productionize runtime calibration đầy đủ.
- Chưa có RAG/indirect benchmark đủ rộng.

### 5. Rủi ro lớn nhất

Rủi ro lớn nhất là over-claiming. Claim an toàn là hybrid detector + external evaluation + calibration analysis. Không nên claim đã giải quyết hoàn toàn multilingual semantic prompt injection.

### 6. Tuần tới làm gì?

1. Đưa strict validation/test thành benchmark mặc định.
2. Persist calibrated thresholds/artifacts.
3. Fine-tune XLM-R với multilingual hard negatives.
4. Nâng context-aware lên semantic features.
5. Làm common-failure analysis giữa RF/RoBERTa/XLM-R.

## Điểm hoàn thành khách quan

{md_table(["Mảng", "Điểm", "Lý do"], scores)}
"""


def overall_report() -> str:
    direct = read_csv(ROOT / "reports" / "direct_external_evaluation" / "direct_ablation_summary.csv")
    bipia = read_csv(ROOT / "reports" / "bipia_evaluation" / "ablation_comparison.csv")

    def direct_value(dataset: str, model: str, field: str) -> str:
        for row in direct:
            if row.get("Dataset") == dataset and row.get("Model") == model:
                return fmt(row.get(field))
        return "N/A"

    def bipia_value(model: str, context: str, field: str) -> str:
        for row in bipia:
            if row.get("Model") == model and row.get("Context-aware") == context:
                if field in {"FP", "FN", "Rows"}:
                    return fmt_int(row.get(field))
                return fmt(row.get(field))
        return "N/A"

    return f"""
# Project Audit tổng thể

Ngày sinh báo cáo: {AUDIT_DATE}

## Tóm tắt kỹ thuật

Dự án hiện là một prototype Prompt Injection Detection khá đầy đủ: có baseline ML truyền thống, Transformer checkpoints, rule-based/context-aware logic, indirect detection, FastAPI/frontend, và hệ thống evaluation đang phát triển theo hướng nghiên cứu.

Tiến bộ quan trọng nhất trong 2 tuần gần đây là chuyển từ “model chạy được trên internal data” sang “model được kiểm tra bằng external benchmarks, ablation, strict thresholding và calibration”. Đây là hướng đúng để bảo vệ capstone theo hướng research.

## Evidence chính

- Git history từ 2026-06-14 có 6 mốc lớn: initial upload, context-aware hybrid detection, BIPIA benchmark, BIPIA ablation, direct external benchmark, strict threshold/calibration analysis.
- Có BIPIA ablation cho RF, RoBERTa, XLM-R, RoBERTa+Context, XLM-R+Context.
- Có direct external benchmark gồm deepset, neuralchemy, cyberec, direct_all.
- Có strict validation/test benchmark; đây nên là protocol đáng tin hơn raw full-set threshold sweep.
- Có calibration artifacts với Brier Score, ECE, selected calibration method.

## Snapshot metrics quan trọng

{md_table(
    ["Câu hỏi", "Evidence"],
    [
        ["Direct external all tốt nhất theo raw F1", f"RoBERTa F1={direct_value('all', 'RoBERTa', 'F1')}, ROC-AUC={direct_value('all', 'RoBERTa', 'ROC-AUC')}"],
        ["XLM-R direct external all", f"XLM-R F1={direct_value('all', 'XLM-RoBERTa', 'F1')}, ROC-AUC={direct_value('all', 'XLM-RoBERTa', 'ROC-AUC')}"],
        ["BIPIA RoBERTa only", f"F1={bipia_value('RoBERTa', 'No', 'F1')}, FN={bipia_value('RoBERTa', 'No', 'FN')}"],
        ["BIPIA XLM-R only", f"F1={bipia_value('XLM-RoBERTa', 'No', 'F1')}, FN={bipia_value('XLM-RoBERTa', 'No', 'FN')}"],
        ["BIPIA RoBERTa + Context", f"F1={bipia_value('RoBERTa', 'Yes', 'F1')}, FP={bipia_value('RoBERTa', 'Yes', 'FP')}, FN={bipia_value('RoBERTa', 'Yes', 'FN')}"],
        ["BIPIA XLM-R + Context", f"F1={bipia_value('XLM-RoBERTa', 'Yes', 'F1')}, FP={bipia_value('XLM-RoBERTa', 'Yes', 'FP')}, FN={bipia_value('XLM-RoBERTa', 'Yes', 'FN')}"],
    ],
)}

## Trạng thái hiện tại

{md_table(
    ["Thành phần", "Trạng thái", "Kết quả"],
    [
        ["Dataset", "Mạnh nhưng cần document", "Có internal, unified, Vietnamese, direct external, BIPIA subset."],
        ["Traditional ML", "Baseline hoàn thành", "LR/RF/SVM có đủ; RF là baseline nhưng yếu trên pooled direct external."],
        ["RoBERTa", "Transformer tốt nhất hiện tại", "Generalize tốt trên direct external; calibration cải thiện diễn giải threshold."],
        ["XLM-RoBERTa", "Cần cải thiện", "BIPIA context fusion mạnh nhưng direct external generalization yếu."],
        ["Context-aware", "Có triển vọng", "Giảm FP/FN mạnh trên BIPIA; vẫn heuristic-heavy."],
        ["Calibration", "Đã có analysis", "Brier/ECE có trong reports; runtime chưa tích hợp đầy đủ."],
        ["BIPIA", "Partial benchmark", "1,000-row subset; QA/abstract skipped."],
        ["Direct benchmark", "Mạnh", "64,785 rows; strict split reports có sẵn."],
        ["API/frontend", "Functional", "FastAPI + static pages."],
        ["Explainability", "Có nhưng chưa đánh giá", "Output giải thích có; chưa có explanation-quality metric."],
    ],
)}

## Điểm yếu cần nói thẳng

1. XLM-R chưa đủ mạnh để claim multilingual robust detector.
2. Context-aware hiện còn heuristic, chưa chứng minh semantic reasoning.
3. Calibration chưa productionize vào runtime toàn bộ.
4. BIPIA coverage mới là subset.
5. Direct raw threshold sweep có thể optimistic nếu không tách strict protocol.
6. Chưa có common-failure report đầy đủ giữa RF/RoBERTa/XLM-R.
7. Indirect/RAG evaluation chưa trưởng thành bằng direct evaluation.

## Framing nên dùng

> Dự án nghiên cứu một hệ thống hybrid prompt-injection detection kết hợp fine-tuned classifiers, context-aware signals, threshold optimization, calibration analysis và external benchmark ablations.

## Không nên claim

> Dự án đã giải quyết hoàn toàn multilingual semantic prompt injection detection.

## File trong audit package

- `project_structure.md`
- `project_tree.txt`
- `completed_work.md`
- `current_metrics_summary.md`
- `research_contributions.md`
- `roadmap_next_steps.md`
- `project_progress.md`
- `overall_project_audit.md`
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_report("project_structure.md", project_structure_report())
    write_report("project_tree.txt", project_tree())
    write_report("completed_work.md", completed_work_report())
    write_report("current_metrics_summary.md", current_metrics_summary_report())
    write_report("research_contributions.md", research_contributions_report())
    write_report("roadmap_next_steps.md", roadmap_report())
    write_report("project_progress.md", project_progress_report())
    write_report("overall_project_audit.md", overall_report())
    print(f"Wrote Vietnamese audit reports to {OUT}")


if __name__ == "__main__":
    main()
