"""Dataset loading and schema detection for JSONL prompt datasets."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from src.file_utils import safe_write_text


TEXT_COLUMN_CANDIDATES = ["ml_text", "text", "canonical_text", "prompt", "user_prompt", "input", "content", "original_text"]
LABEL_COLUMN_CANDIDATES = ["label", "category", "type", "is_malicious"]

LABEL_MAPPING = {
    "benign": 0,
    "safe": 0,
    "normal": 0,
    "clean": 0,
    "allowed": 0,
    "allow": 0,
    "0": 0,
    "false": 0,
    "no": 0,
    "malicious": 1,
    "injection": 1,
    "prompt_injection": 1,
    "prompt injection": 1,
    "jailbreak": 1,
    "jailbreaking": 1,
    "unsafe": 1,
    "attack": 1,
    "blocked": 1,
    "block": 1,
    "1": 1,
    "true": 1,
    "yes": 1,
}


def load_jsonl_dataset(path: str | Path) -> pd.DataFrame:
    """Load a JSONL dataset into a pandas DataFrame."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset: {dataset_path}")

    records: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Dòng {line_number} không phải JSON hợp lệ: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Dòng {line_number} phải là JSON object.")
            records.append(record)

    if not records:
        raise ValueError("Dataset rỗng hoặc không có dòng JSONL hợp lệ.")

    return pd.json_normalize(records)


def _case_insensitive_lookup(columns: list[str], candidates: list[str]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def auto_detect_text_column(df: pd.DataFrame) -> str:
    """Detect the most likely prompt/text column."""
    columns = list(df.columns)
    direct_match = _case_insensitive_lookup(columns, TEXT_COLUMN_CANDIDATES)
    if direct_match:
        return direct_match

    for column in columns:
        normalized = column.lower()
        if any(candidate in normalized for candidate in TEXT_COLUMN_CANDIDATES):
            return column

    object_columns = [
        column
        for column in columns
        if df[column].dtype == "object" and column.lower() not in LABEL_COLUMN_CANDIDATES
    ]
    if not object_columns:
        raise ValueError("Không tự nhận diện được cột chứa prompt/text.")

    return max(object_columns, key=lambda column: df[column].astype(str).str.len().mean())


def auto_detect_label_column(df: pd.DataFrame) -> str:
    """Detect the most likely label column."""
    columns = list(df.columns)
    direct_match = _case_insensitive_lookup(columns, LABEL_COLUMN_CANDIDATES)
    if direct_match:
        return direct_match

    for column in columns:
        normalized = column.lower()
        if any(candidate in normalized for candidate in LABEL_COLUMN_CANDIDATES):
            return column

    best_column: str | None = None
    best_score = -1.0
    for column in columns:
        values = df[column].dropna()
        if values.empty:
            continue
        unique_values = values.astype(str).str.lower().str.strip().unique()
        mapped_count = sum(value in LABEL_MAPPING for value in unique_values)
        score = mapped_count / max(len(unique_values), 1)
        if score > best_score and mapped_count > 0:
            best_score = score
            best_column = column

    if best_column:
        return best_column

    raise ValueError("Không tự nhận diện được cột nhãn.")


def _normalize_single_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
        return value
    if isinstance(value, float) and value in (0.0, 1.0):
        return int(value)

    normalized = str(value).strip().lower()
    if normalized in LABEL_MAPPING:
        return LABEL_MAPPING[normalized]

    raise ValueError(f"Nhãn không hỗ trợ: {value!r}")


def normalize_labels(
    df: pd.DataFrame,
    label_column: str | None = None,
    output_column: str = "label_normalized",
) -> pd.DataFrame:
    """Normalize labels to 0 = benign and 1 = malicious."""
    detected_label_column = label_column or auto_detect_label_column(df)
    normalized_df = df.copy()
    try:
        normalized_df[output_column] = normalized_df[detected_label_column].apply(_normalize_single_label)
    except ValueError as exc:
        unique_values = normalized_df[detected_label_column].dropna().unique().tolist()
        raise ValueError(
            f"Không normalize được cột nhãn '{detected_label_column}'. "
            f"Giá trị nhãn đang có: {unique_values}"
        ) from exc
    return normalized_df


def _distribution_as_dict(values: pd.Series) -> dict[str, int]:
    counter = Counter(values.tolist())
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def build_dataset_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Build a structured summary for reporting."""
    text_column = auto_detect_text_column(df)
    label_column = auto_detect_label_column(df)
    normalized_df = df if "label_normalized" in df.columns else normalize_labels(df, label_column)

    summary: dict[str, Any] = {
        "num_rows": int(len(normalized_df)),
        "num_columns": int(len(normalized_df.columns)),
        "columns": list(normalized_df.columns),
        "text_column": text_column,
        "label_column": label_column,
        "label_distribution": _distribution_as_dict(normalized_df["label_normalized"]),
        "original_label_distribution": _distribution_as_dict(normalized_df[label_column]),
    }

    if "attack_type" in normalized_df.columns:
        summary["attack_type_distribution"] = _distribution_as_dict(normalized_df["attack_type"].fillna("unknown"))
    if "detected_language" in normalized_df.columns:
        summary["language_distribution"] = _distribution_as_dict(normalized_df["detected_language"].fillna("unknown"))
    if "source" in normalized_df.columns:
        summary["source_distribution"] = _distribution_as_dict(normalized_df["source"].fillna("unknown"))

    prompt_lengths = normalized_df[text_column].fillna("").astype(str).str.len()
    summary["prompt_length"] = {
        "min": int(prompt_lengths.min()),
        "max": int(prompt_lengths.max()),
        "mean": float(prompt_lengths.mean()),
        "median": float(prompt_lengths.median()),
    }
    return summary


def print_dataset_summary(df: pd.DataFrame) -> None:
    """Print a concise dataset summary to stdout."""
    summary = build_dataset_summary(df)
    print("=== Dataset Summary ===")
    print(f"Số dòng: {summary['num_rows']}")
    print(f"Số cột: {summary['num_columns']}")
    print(f"Cột prompt: {summary['text_column']}")
    print(f"Cột nhãn: {summary['label_column']}")
    print(f"Phân phối nhãn chuẩn hóa: {summary['label_distribution']}")
    if "attack_type_distribution" in summary:
        print(f"Phân phối attack_type: {summary['attack_type_distribution']}")


def save_dataset_summary(df: pd.DataFrame, output_path: str | Path) -> dict[str, Any]:
    """Save dataset summary as a Vietnamese Markdown report."""
    summary = build_dataset_summary(df)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Tóm tắt dataset",
        "",
        f"- Số dòng: {summary['num_rows']}",
        f"- Số cột: {summary['num_columns']}",
        f"- Cột prompt được dùng: `{summary['text_column']}`",
        f"- Cột nhãn được dùng: `{summary['label_column']}`",
        "",
        "## Danh sách cột",
        "",
    ]
    lines.extend(f"- `{column}`" for column in summary["columns"])
    lines.extend(
        [
            "",
            "## Phân phối nhãn chuẩn hóa",
            "",
            "| Nhãn | Ý nghĩa | Số mẫu |",
            "|---|---|---:|",
            f"| 0 | benign/safe | {summary['label_distribution'].get('0', 0)} |",
            f"| 1 | malicious/prompt injection | {summary['label_distribution'].get('1', 0)} |",
            "",
            "## Phân phối nhãn gốc",
            "",
            "| Giá trị nhãn gốc | Số mẫu |",
            "|---|---:|",
        ]
    )
    for label, count in summary["original_label_distribution"].items():
        lines.append(f"| `{label}` | {count} |")

    if "attack_type_distribution" in summary:
        lines.extend(["", "## Phân phối attack_type", "", "| attack_type | Số mẫu |", "|---|---:|"])
        for attack_type, count in summary["attack_type_distribution"].items():
            lines.append(f"| `{attack_type}` | {count} |")

    length_stats = summary["prompt_length"]
    lines.extend(
        [
            "",
            "## Độ dài prompt",
            "",
            f"- Ngắn nhất: {length_stats['min']} ký tự",
            f"- Dài nhất: {length_stats['max']} ký tự",
            f"- Trung bình: {length_stats['mean']:.2f} ký tự",
            f"- Trung vị: {length_stats['median']:.2f} ký tự",
            "",
            "> Báo cáo này được sinh tự động từ dataset thật, không ghi tay số liệu.",
            "",
        ]
    )

    safe_write_text(path, "\n".join(lines), encoding="utf-8")
    return summary
