"""Dataset standardization and unification utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_loader import load_jsonl_dataset, normalize_labels
from src.language_utils import detect_language
from src.preprocessing import clean_text
from src.transformer_utils import (
    DEFAULT_DATASET_NAME,
    import_huggingface_load_dataset,
    load_cached_neuralchemy_arrow_dataframe,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATASETS_PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
DATASETS_CUSTOM_DIR = PROJECT_ROOT / "datasets" / "custom"
UNIFIED_DIR = PROJECT_ROOT / "datasets" / "unified"
DATASET_REPORTS_DIR = PROJECT_ROOT / "datasets" / "reports"
STANDARD_COLUMNS = [
    "text",
    "label",
    "category",
    "source",
    "severity",
    "split",
    "language",
    "augmented",
    "dataset_config",
]


def _default_category(label: int, raw_category: Any = None) -> str:
    if raw_category is not None and str(raw_category).strip():
        value = str(raw_category).strip().lower()
        if value in {"benign", "safe", "normal"}:
            return "benign"
        return value
    return "benign" if int(label) == 0 else "direct_injection"


def _normalize_severity(value: Any) -> str:
    if value is None or pd.isna(value) or not str(value).strip():
        return "unknown"
    normalized = str(value).strip().lower()
    return normalized if normalized in {"low", "medium", "high", "critical", "unknown"} else "unknown"


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def standardize_dataframe(
    df: pd.DataFrame,
    source: str,
    dataset_config: str,
    default_split: str = "train",
) -> pd.DataFrame:
    text_column = "text" if "text" in df.columns else ("prompt" if "prompt" in df.columns else None)
    if text_column is None:
        for candidate in [
            "ml_text",
            "model_text",
            "canonical_text",
            "augmented_prompt",
            "original_prompt",
            "user_prompt",
            "input",
            "content",
        ]:
            if candidate in df.columns:
                text_column = candidate
                break
    if text_column is None:
        raise ValueError(f"Cannot find text column for source={source}. columns={list(df.columns)}")

    label_column = "label" if "label" in df.columns else ("label_normalized" if "label_normalized" in df.columns else None)
    if label_column is None:
        raise ValueError(f"Cannot find label column for source={source}. columns={list(df.columns)}")

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        text = "" if pd.isna(row[text_column]) else str(row[text_column]).strip()
        if not text:
            continue
        raw_label = row[label_column]
        if isinstance(raw_label, str):
            lowered = raw_label.strip().lower()
            label = 0 if lowered in {"0", "benign", "safe", "normal", "clean"} else 1
        else:
            label = int(raw_label)
        if label not in {0, 1}:
            continue
        split = row.get("split", row.get("hf_split", default_split))
        rows.append(
            {
                "text": text,
                "label": label,
                "category": _default_category(label, row.get("category", row.get("attack_type", None))),
                "source": source,
                "severity": _normalize_severity(row.get("severity", None)),
                "split": str(split or default_split).strip().lower(),
                "language": row.get("language", detect_language(text)),
                "augmented": _normalize_bool(row.get("augmented", False)),
                "dataset_config": dataset_config,
            }
        )
    return pd.DataFrame(rows, columns=STANDARD_COLUMNS)


def _load_neuralchemy_config(config_name: str, prefer_cached: bool = True) -> pd.DataFrame:
    if prefer_cached:
        try:
            return load_cached_neuralchemy_arrow_dataframe(config_name)
        except FileNotFoundError:
            pass
    load_dataset = import_huggingface_load_dataset()
    dataset = load_dataset(DEFAULT_DATASET_NAME, config_name)
    frames = []
    for split, split_dataset in dataset.items():
        frame = split_dataset.to_pandas()
        frame["hf_split"] = split
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def collect_standardized_sources(include_hf_full: bool = True) -> tuple[list[pd.DataFrame], list[dict[str, Any]]]:
    sources: list[pd.DataFrame] = []
    metadata: list[dict[str, Any]] = []

    raw_jsonl = RAW_DIR / "Prompt_INJECTION_And_Benign_DATASET.jsonl"
    if raw_jsonl.exists():
        raw_df = normalize_labels(load_jsonl_dataset(raw_jsonl))
        standardized = standardize_dataframe(raw_df, "project_raw_jsonl", "custom", default_split="train")
        sources.append(standardized)
        metadata.append({"source": "project_raw_jsonl", "path": str(raw_jsonl), "rows": len(standardized), "columns": list(raw_df.columns)})

    for csv_path in [
        DATASETS_PROCESSED_DIR / "direct_ml_ready.csv",
        DATASETS_PROCESSED_DIR / "direct_merged.csv",
        PROJECT_ROOT / "data" / "processed" / "augmented_multilingual_dataset.csv",
    ]:
        if csv_path.exists():
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            standardized = standardize_dataframe(df, csv_path.stem, "custom", default_split="train")
            sources.append(standardized)
            metadata.append({"source": csv_path.stem, "path": str(csv_path), "rows": len(standardized), "columns": list(df.columns)})

    if DATASETS_CUSTOM_DIR.exists():
        for csv_path in sorted(DATASETS_CUSTOM_DIR.glob("*.csv")):
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            dataset_config = str(df.get("dataset_config", pd.Series(["custom_quality_v2"])).iloc[0])
            standardized = standardize_dataframe(df, csv_path.stem, dataset_config, default_split="train")
            sources.append(standardized)
            metadata.append(
                {
                    "source": csv_path.stem,
                    "path": str(csv_path),
                    "rows": len(standardized),
                    "columns": list(df.columns),
                }
            )

    for config_name in ["core", "full"] if include_hf_full else ["core"]:
        try:
            df = _load_neuralchemy_config(config_name, prefer_cached=True)
        except Exception as exc:
            metadata.append({"source": f"neuralchemy_{config_name}", "error": str(exc), "rows": 0, "columns": []})
            continue
        standardized = standardize_dataframe(df, f"neuralchemy_{config_name}", config_name)
        sources.append(standardized)
        metadata.append({"source": f"neuralchemy_{config_name}", "rows": len(standardized), "columns": list(df.columns)})

    return sources, metadata


def build_unified_datasets(include_hf_full: bool = True) -> dict[str, Any]:
    UNIFIED_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    sources, metadata = collect_standardized_sources(include_hf_full=include_hf_full)
    if not sources:
        raise RuntimeError("No dataset sources found.")

    combined = pd.concat(sources, ignore_index=True)
    combined["normalized_text"] = combined["text"].map(clean_text)
    combined = combined[combined["normalized_text"].str.len() > 0].copy()

    conflict_mask = combined.groupby("normalized_text")["label"].transform("nunique") > 1
    conflicts = combined[conflict_mask].copy()
    conflicts.to_csv(DATASET_REPORTS_DIR / "label_conflicts.csv", index=False, encoding="utf-8-sig")

    non_conflict = combined[~conflict_mask].copy()
    before_dedup = len(non_conflict)
    deduped = (
        non_conflict.sort_values(by=["augmented", "source"], ascending=[True, True])
        .drop_duplicates(subset=["normalized_text"], keep="first")
        .drop(columns=["normalized_text"])
        .reset_index(drop=True)
    )
    duplicate_removed = before_dedup - len(deduped)

    unified_path = UNIFIED_DIR / "prompt_injection_unified.csv"
    ml_ready_path = UNIFIED_DIR / "prompt_injection_ml_ready.csv"
    transformer_ready_path = UNIFIED_DIR / "prompt_injection_transformer_ready.csv"

    deduped.to_csv(unified_path, index=False, encoding="utf-8-sig")
    ml_ready = deduped[(deduped["augmented"] == False) | (deduped["dataset_config"].isin(["core", "custom", "merged"]))].copy()
    transformer_ready = deduped.copy()
    ml_ready.to_csv(ml_ready_path, index=False, encoding="utf-8-sig")
    transformer_ready.to_csv(transformer_ready_path, index=False, encoding="utf-8-sig")

    report = {
        "sources": metadata,
        "standard_schema": STANDARD_COLUMNS,
        "raw_rows_after_standardization": int(len(combined)),
        "unified_rows": int(len(deduped)),
        "ml_ready_rows": int(len(ml_ready)),
        "transformer_ready_rows": int(len(transformer_ready)),
        "duplicates_removed": int(duplicate_removed),
        "label_conflicts": int(len(conflicts)),
        "label_distribution": deduped["label"].value_counts().sort_index().to_dict(),
        "category_distribution": deduped["category"].value_counts().to_dict(),
        "language_distribution": deduped["language"].value_counts().to_dict(),
        "split_distribution": deduped["split"].value_counts().to_dict(),
        "paths": {
            "unified": str(unified_path),
            "ml_ready": str(ml_ready_path),
            "transformer_ready": str(transformer_ready_path),
            "label_conflicts": str(DATASET_REPORTS_DIR / "label_conflicts.csv"),
        },
    }
    (DATASET_REPORTS_DIR / "unification_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report
