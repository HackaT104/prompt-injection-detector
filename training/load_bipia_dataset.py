"""Load and normalize the BIPIA indirect prompt injection dataset.

Usage:
    python training/load_bipia_dataset.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "datasets" / "processed" / "bipia_indirect.csv"
DEFAULT_DATASET_NAME = "MAlmasabi/Indirect-Prompt-Injection-BIPIA-GPT"

USER_INTENT_CANDIDATES = [
    "user_intent",
    "user_prompt",
    "prompt",
    "instruction",
    "query",
    "question",
    "task",
]
CONTEXT_CANDIDATES = [
    "context",
    "external_context",
    "retrieved_context",
    "document_content",
    "document",
    "content",
    "webpage",
    "email",
    "text",
]
LABEL_CANDIDATES = [
    "label",
    "is_malicious",
    "malicious",
    "is_attack",
    "attack",
    "class",
    "category",
    "type",
]

LABEL_MAPPING = {
    "0": 0,
    "false": 0,
    "benign": 0,
    "safe": 0,
    "normal": 0,
    "none": 0,
    "clean": 0,
    "1": 1,
    "true": 1,
    "malicious": 1,
    "attack": 1,
    "injection": 1,
    "indirect": 1,
    "prompt_injection": 1,
    "unsafe": 1,
}


def _detect_column(columns: list[str], candidates: list[str], required_name: str) -> str:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    for column in columns:
        normalized = column.lower()
        if any(candidate in normalized for candidate in candidates):
            return column
    raise ValueError(f"Cannot detect {required_name} column from columns: {columns}")


def _normalize_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
        return int(value)
    if isinstance(value, float) and value in (0.0, 1.0):
        return int(value)

    normalized = str(value).strip().lower()
    if normalized in LABEL_MAPPING:
        return LABEL_MAPPING[normalized]
    raise ValueError(f"Unsupported BIPIA label value: {value!r}")


def _dataset_to_dataframe(dataset_name: str, split: str) -> pd.DataFrame:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name)
    if split in dataset:
        selected_split = dataset[split]
    else:
        first_split_name = next(iter(dataset.keys()))
        selected_split = dataset[first_split_name]
    return selected_split.to_pandas()


def load_and_process_bipia(
    dataset_name: str = DEFAULT_DATASET_NAME,
    split: str = "train",
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Load BIPIA from Hugging Face and save a normalized CSV."""
    df = _dataset_to_dataframe(dataset_name, split)
    columns = list(df.columns)
    user_intent_column = _detect_column(columns, USER_INTENT_CANDIDATES, "user intent")
    context_column = _detect_column(columns, CONTEXT_CANDIDATES, "context")
    label_column = _detect_column(columns, LABEL_CANDIDATES, "label")

    processed = pd.DataFrame(
        {
            "user_intent": df[user_intent_column].fillna("").astype(str),
            "context": df[context_column].fillna("").astype(str),
            "label": df[label_column].apply(_normalize_label).astype(int),
        }
    )
    processed["combined_text"] = (
        "USER_INTENT: "
        + processed["user_intent"]
        + "\nCONTEXT: "
        + processed["context"]
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output, index=False, encoding="utf-8-sig")

    metadata = {
        "dataset_name": dataset_name,
        "split_requested": split,
        "source_columns": columns,
        "user_intent_column": user_intent_column,
        "context_column": context_column,
        "label_column": label_column,
        "rows": int(len(processed)),
        "label_distribution": {
            str(label): int(count)
            for label, count in processed["label"].value_counts().sort_index().items()
        },
        "output_path": str(output),
    }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Load BIPIA indirect prompt injection dataset.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_NAME, help="Hugging Face dataset name.")
    parser.add_argument("--split", default="train", help="Dataset split to load.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH), help="Processed CSV output path.")
    args = parser.parse_args()

    load_and_process_bipia(args.dataset, args.split, args.output)


if __name__ == "__main__":
    main()

