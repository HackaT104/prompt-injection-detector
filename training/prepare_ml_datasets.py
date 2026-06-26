"""Prepare separate ML-ready datasets for direct and indirect detectors.

Outputs:
    datasets/processed/direct_ml_ready.csv
    datasets/processed/indirect_ml_ready.csv  (only when bipia_indirect.csv exists)
    reports/training_data_preparation_report.json
    reports/training_data_preparation_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.file_utils import safe_write_text  # noqa: E402
from src.preprocessing import prepare_text_for_detection  # noqa: E402


DEFAULT_DIRECT_SOURCE = PROJECT_ROOT / "datasets" / "processed" / "direct_merged.csv"
DEFAULT_INDIRECT_SOURCE = PROJECT_ROOT / "datasets" / "processed" / "bipia_indirect.csv"
DEFAULT_DIRECT_OUTPUT = PROJECT_ROOT / "datasets" / "processed" / "direct_ml_ready.csv"
DEFAULT_INDIRECT_OUTPUT = PROJECT_ROOT / "datasets" / "processed" / "indirect_ml_ready.csv"
DEFAULT_REPORT_JSON = PROJECT_ROOT / "reports" / "training_data_preparation_report.json"
DEFAULT_REPORT_MD = PROJECT_ROOT / "reports" / "training_data_preparation_report.md"


def _distribution(values: pd.Series) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(Counter(values.tolist()).items(), key=lambda item: str(item[0]))
    }


def _text_stats(values: pd.Series) -> dict[str, float | int]:
    lengths = values.fillna("").astype(str).str.len()
    if lengths.empty:
        return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    return {
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
        "median": float(lengths.median()),
    }


def _summary(name: str, source_path: Path, output_path: Path, df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "name": name,
        "status": "created",
        "source_path": str(source_path),
        "output_path": str(output_path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "label_distribution": _distribution(df["label"]),
        "language_distribution": _distribution(df["detected_language"].fillna("unknown")),
        "ml_text_length": _text_stats(df["ml_text"]),
    }
    if "source" in df.columns:
        summary["source_distribution"] = _distribution(df["source"].fillna("unknown"))
    return summary


def _missing_summary(name: str, source_path: Path, output_path: Path, instruction: str) -> dict[str, Any]:
    return {
        "name": name,
        "status": "missing_source",
        "source_path": str(source_path),
        "output_path": str(output_path),
        "rows": 0,
        "label_distribution": {},
        "language_distribution": {},
        "instruction": instruction,
    }


def prepare_direct_dataset(source_path: str | Path, output_path: str | Path) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    source = Path(source_path)
    output = Path(output_path)
    if not source.exists():
        summary = _missing_summary(
            "direct",
            source,
            output,
            "Hãy tạo direct_merged.csv bằng: python training/merge_direct_deepset_dataset.py",
        )
        return None, summary

    raw_df = pd.read_csv(source, encoding="utf-8-sig")
    required_columns = {"text", "label"}
    missing = required_columns - set(raw_df.columns)
    if missing:
        raise ValueError(f"Direct source thiếu cột: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for index, row in raw_df.iterrows():
        prepared = prepare_text_for_detection(str(row["text"]))
        label = int(row["label"])
        rows.append(
            {
                "sample_id": f"direct_{index}",
                "task_type": "direct",
                "label": label,
                "original_text": str(row["text"]),
                "detected_language": prepared["detected_language"],
                "canonical_text": prepared["cleaned_text"],
                "ml_text": prepared["cleaned_text"],
                "user_prompt": str(row["text"]),
                "context": "",
                "source": row.get("source", "direct"),
                "source_split": row.get("source_split", "unknown"),
            }
        )

    prepared_df = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    prepared_df.to_csv(output, index=False, encoding="utf-8-sig")
    return prepared_df, _summary("direct", source, output, prepared_df)


def prepare_indirect_dataset(source_path: str | Path, output_path: str | Path) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    source = Path(source_path)
    output = Path(output_path)
    if not source.exists():
        summary = _missing_summary(
            "indirect",
            source,
            output,
            "Chưa có bipia_indirect.csv local. Hãy đặt file này vào datasets/processed/ hoặc chạy loader BIPIA khi có mạng.",
        )
        return None, summary

    raw_df = pd.read_csv(source, encoding="utf-8-sig")
    required_columns = {"user_intent", "context", "label"}
    missing = required_columns - set(raw_df.columns)
    if missing:
        raise ValueError(f"Indirect source thiếu cột: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for index, row in raw_df.iterrows():
        user_prompt = str(row["user_intent"])
        context = str(row["context"])
        prepared_user = prepare_text_for_detection(user_prompt)
        prepared_context = prepare_text_for_detection(context)
        ml_text = (
            f"USER_INTENT: {prepared_user['cleaned_text']}\n"
            f"CONTEXT: {prepared_context['cleaned_text']}"
        )
        rows.append(
            {
                "sample_id": f"indirect_{index}",
                "task_type": "indirect",
                "label": int(row["label"]),
                "original_text": ml_text,
                "detected_language": prepared_context["detected_language"],
                "canonical_text": ml_text,
                "ml_text": ml_text,
                "user_prompt": user_prompt,
                "context": context,
                "source": "bipia_indirect",
                "source_split": row.get("source_split", "unknown"),
                "user_prompt_language": prepared_user["detected_language"],
                "context_language": prepared_context["detected_language"],
                "canonical_user_prompt": prepared_user["cleaned_text"],
                "canonical_context": prepared_context["cleaned_text"],
            }
        )

    prepared_df = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    prepared_df.to_csv(output, index=False, encoding="utf-8-sig")
    return prepared_df, _summary("indirect", source, output, prepared_df)


def _write_reports(report: dict[str, Any], json_path: str | Path, md_path: str | Path) -> None:
    safe_write_text(json_path, json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Training Data Preparation Report",
        "",
        "Báo cáo này tóm tắt dữ liệu đã chuẩn hóa cho machine learning. Direct và indirect được giữ thành hai file riêng biệt.",
        "",
    ]
    for item in report["datasets"]:
        lines.extend(
            [
                f"## {item['name']}",
                "",
                f"- Status: `{item['status']}`",
                f"- Source: `{item['source_path']}`",
                f"- Output: `{item['output_path']}`",
                f"- Rows: {item['rows']}",
                f"- Label distribution: `{item.get('label_distribution', {})}`",
                f"- Language distribution: `{item.get('language_distribution', {})}`",
            ]
        )
        if item.get("source_distribution"):
            lines.append(f"- Source distribution: `{item['source_distribution']}`")
        if item.get("ml_text_length"):
            lines.append(f"- ML text length: `{item['ml_text_length']}`")
        if item.get("instruction"):
            lines.append(f"- Instruction: {item['instruction']}")
        lines.append("")

    safe_write_text(md_path, "\n".join(lines), encoding="utf-8")


def prepare_ml_datasets(
    direct_source: str | Path = DEFAULT_DIRECT_SOURCE,
    indirect_source: str | Path = DEFAULT_INDIRECT_SOURCE,
    direct_output: str | Path = DEFAULT_DIRECT_OUTPUT,
    indirect_output: str | Path = DEFAULT_INDIRECT_OUTPUT,
    report_json: str | Path = DEFAULT_REPORT_JSON,
    report_md: str | Path = DEFAULT_REPORT_MD,
) -> dict[str, Any]:
    direct_df, direct_summary = prepare_direct_dataset(direct_source, direct_output)
    indirect_df, indirect_summary = prepare_indirect_dataset(indirect_source, indirect_output)

    report = {
        "outputs": {
            "direct_ml_ready": str(direct_output),
            "indirect_ml_ready": str(indirect_output) if indirect_df is not None else None,
            "report_json": str(report_json),
            "report_md": str(report_md),
        },
        "datasets": [direct_summary, indirect_summary],
        "notes": [
            "Direct và indirect là hai file riêng biệt để tránh trộn hai bài toán ML khác nhau.",
            "Cột ml_text là cột đã chuẩn hóa dùng để train/evaluate ML.",
            "Không có metrics giả; report chỉ thống kê từ dữ liệu thực tế đang có trong workspace.",
        ],
    }
    if direct_df is not None and indirect_df is not None:
        report["total_rows"] = int(len(direct_df) + len(indirect_df))
    elif direct_df is not None:
        report["total_rows"] = int(len(direct_df))
    elif indirect_df is not None:
        report["total_rows"] = int(len(indirect_df))
    else:
        report["total_rows"] = 0

    _write_reports(report, report_json, report_md)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare separate ML-ready direct and indirect datasets.")
    parser.add_argument("--direct-source", default=str(DEFAULT_DIRECT_SOURCE))
    parser.add_argument("--indirect-source", default=str(DEFAULT_INDIRECT_SOURCE))
    parser.add_argument("--direct-output", default=str(DEFAULT_DIRECT_OUTPUT))
    parser.add_argument("--indirect-output", default=str(DEFAULT_INDIRECT_OUTPUT))
    parser.add_argument("--report-json", default=str(DEFAULT_REPORT_JSON))
    parser.add_argument("--report-md", default=str(DEFAULT_REPORT_MD))
    args = parser.parse_args()

    prepare_ml_datasets(
        direct_source=args.direct_source,
        indirect_source=args.indirect_source,
        direct_output=args.direct_output,
        indirect_output=args.indirect_output,
        report_json=args.report_json,
        report_md=args.report_md,
    )


if __name__ == "__main__":
    main()
