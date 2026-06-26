"""Merge the existing direct dataset with deepset/prompt-injections.

This script does not touch the indirect/BIPIA dataset. It creates:
    datasets/processed/direct_merged.csv

Usage:
    python training/merge_direct_deepset_dataset.py
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (  # noqa: E402
    auto_detect_label_column,
    auto_detect_text_column,
    load_jsonl_dataset,
    normalize_labels,
)
from src.file_utils import safe_write_text  # noqa: E402


DEFAULT_DIRECT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Prompt_INJECTION_And_Benign_DATASET.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "datasets" / "processed" / "direct_merged.csv"
DEEPSET_DATASET_NAME = "deepset/prompt-injections"


def _normalize_deepset_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
        return int(value)
    if isinstance(value, float) and value in (0.0, 1.0):
        return int(value)

    normalized = str(value).strip().lower()
    if normalized in {"0", "benign", "safe", "normal", "false"}:
        return 0
    if normalized in {"1", "injection", "prompt_injection", "malicious", "unsafe", "true"}:
        return 1
    raise ValueError(f"Unsupported deepset label: {value!r}")


def _load_existing_direct_dataset(path: str | Path) -> pd.DataFrame:
    df = load_jsonl_dataset(path)
    text_column = auto_detect_text_column(df)
    label_column = auto_detect_label_column(df)
    normalized_df = normalize_labels(df, label_column=label_column)
    return pd.DataFrame(
        {
            "text": normalized_df[text_column].fillna("").astype(str),
            "label": normalized_df["label_normalized"].astype(int),
            "source": "existing_direct_jsonl",
            "source_split": "all",
        }
    )


def _load_deepset_prompt_injections(dataset_name: str = DEEPSET_DATASET_NAME) -> pd.DataFrame:
    load_dataset = _import_huggingface_load_dataset()

    dataset = load_dataset(dataset_name)
    rows: list[pd.DataFrame] = []
    for split_name, split_dataset in dataset.items():
        split_df = split_dataset.to_pandas()
        if "text" not in split_df.columns or "label" not in split_df.columns:
            raise ValueError(
                f"Expected columns ['text', 'label'] in split '{split_name}', got {list(split_df.columns)}"
            )
        rows.append(
            pd.DataFrame(
                {
                    "text": split_df["text"].fillna("").astype(str),
                    "label": split_df["label"].apply(_normalize_deepset_label).astype(int),
                    "source": "deepset_prompt_injections",
                    "source_split": split_name,
                }
            )
        )

    if not rows:
        raise ValueError(f"No splits found in Hugging Face dataset: {dataset_name}")
    return pd.concat(rows, ignore_index=True)


def _import_huggingface_load_dataset() -> Any:
    """Import Hugging Face datasets despite the local datasets/ data directory."""
    project_root = PROJECT_ROOT.resolve()
    original_sys_path = list(sys.path)
    local_datasets_module = sys.modules.get("datasets")
    if local_datasets_module is not None and getattr(local_datasets_module, "__file__", None) is None:
        sys.modules.pop("datasets", None)

    try:
        sys.path = [path for path in sys.path if Path(path or ".").resolve() != project_root]
        datasets_module = importlib.import_module("datasets")
    finally:
        sys.path = original_sys_path

    load_dataset = getattr(datasets_module, "load_dataset", None)
    if load_dataset is None:
        raise ImportError(
            "Không import được Hugging Face datasets.load_dataset. "
            "Hãy kiểm tra package 'datasets' trong môi trường Python."
        )
    return load_dataset


def merge_direct_datasets(
    direct_data_path: str | Path = DEFAULT_DIRECT_DATA_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    deepset_dataset_name: str = DEEPSET_DATASET_NAME,
) -> pd.DataFrame:
    existing_df = _load_existing_direct_dataset(direct_data_path)
    deepset_df = _load_deepset_prompt_injections(deepset_dataset_name)
    merged_df = pd.concat([existing_df, deepset_df], ignore_index=True)
    merged_df = merged_df.dropna(subset=["text", "label"])
    merged_df["text"] = merged_df["text"].astype(str)
    merged_df["label"] = merged_df["label"].astype(int)
    merged_df = merged_df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output, index=False, encoding="utf-8-sig")

    metadata = {
        "existing_direct_rows": int(len(existing_df)),
        "deepset_rows": int(len(deepset_df)),
        "merged_rows_after_dedup": int(len(merged_df)),
        "label_distribution": {
            str(label): int(count)
            for label, count in merged_df["label"].value_counts().sort_index().items()
        },
        "output_path": str(output),
        "deepset_dataset": deepset_dataset_name,
        "note": "This merged file is for the direct detector only. BIPIA remains separate for indirect detection.",
    }
    safe_write_text(
        output.with_suffix(".metadata.json"),
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return merged_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge existing direct dataset with deepset/prompt-injections.")
    parser.add_argument("--direct-data", default=str(DEFAULT_DIRECT_DATA_PATH), help="Existing direct JSONL dataset.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Merged direct CSV output.")
    parser.add_argument("--hf-dataset", default=DEEPSET_DATASET_NAME, help="Hugging Face dataset name.")
    args = parser.parse_args()

    merge_direct_datasets(args.direct_data, args.output, args.hf_dataset)


if __name__ == "__main__":
    main()
