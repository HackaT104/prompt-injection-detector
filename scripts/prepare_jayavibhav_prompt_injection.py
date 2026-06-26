"""Prepare jayavibhav/prompt-injection for Transformer v4 fine-tuning."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import clean_text  # noqa: E402
from src.transformer_utils import import_huggingface_load_dataset, normalize_transformer_label  # noqa: E402


DATASET_NAME = "jayavibhav/prompt-injection"
OUTPUT_DIR = PROJECT_ROOT / "datasets" / "processed"
TRAIN_PATH = OUTPUT_DIR / "hf_prompt_injection_train.csv"
VAL_PATH = OUTPUT_DIR / "hf_prompt_injection_val.csv"
TEST_PATH = OUTPUT_DIR / "hf_prompt_injection_test.csv"
COMBINED_PATH = OUTPUT_DIR / "hf_prompt_injection_transformer_ready.csv"
SUMMARY_PATH = OUTPUT_DIR / "hf_prompt_injection_summary.json"


def _read_hf_dataset() -> pd.DataFrame:
    load_dataset = import_huggingface_load_dataset()
    dataset = load_dataset(DATASET_NAME)
    frames: list[pd.DataFrame] = []
    for split_name, split_dataset in dataset.items():
        frame = split_dataset.to_pandas()
        frame["original_split"] = split_name
        frames.append(frame)
    if not frames:
        raise ValueError(f"No splits found in Hugging Face dataset: {DATASET_NAME}")
    return pd.concat(frames, ignore_index=True)


def _standardize(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"text", "label"}
    missing = sorted(required - set(raw_df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}. Columns: {list(raw_df.columns)}")

    working = raw_df.copy()
    before_rows = len(working)
    working["text"] = working["text"].fillna("").astype(str)
    working = working[working["text"].str.strip().str.len() > 0].copy()
    after_empty_filter = len(working)
    working["label"] = working["label"].map(normalize_transformer_label).astype(int)
    working["text_clean"] = working["text"].map(clean_text)
    working = working[working["text_clean"].str.len() > 0].copy()
    before_dedup = len(working)
    working = working.drop_duplicates(subset=["text_clean"]).reset_index(drop=True)

    output = pd.DataFrame(
        {
            "id": [f"jayavibhav_{idx:06d}" for idx in range(len(working))],
            "text": working["text"],
            "label": working["label"],
            "category": working.get("category", "hf_prompt_injection"),
            "source": DATASET_NAME,
            "language": working.get("language", "en"),
            "original_split": working.get("original_split", "train"),
        }
    )
    output["category"] = output["category"].fillna("hf_prompt_injection").astype(str)
    output["language"] = output["language"].fillna("en").astype(str).str.lower()
    output.loc[~output["language"].isin(["en", "vi", "mixed", "unknown"]), "language"] = "en"
    output = output.sample(frac=1.0, random_state=42).reset_index(drop=True)

    summary = {
        "dataset_name": DATASET_NAME,
        "loaded_rows": int(before_rows),
        "rows_after_empty_filter": int(after_empty_filter),
        "rows_before_dedup": int(before_dedup),
        "duplicate_text_removed": int(before_dedup - len(output)),
        "rows_after_dedup": int(len(output)),
        "label_distribution_after_dedup": {
            str(key): int(value) for key, value in output["label"].value_counts().sort_index().items()
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    return output, summary


def main() -> int:
    raw_df = _read_hf_dataset()
    df, summary = _standardize(raw_df)
    if set(df["label"].unique()) != {0, 1}:
        raise ValueError(f"Dataset must contain both labels 0 and 1. Found: {sorted(df['label'].unique())}")

    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["label"],
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "validation"
    test_df["split"] = "test"
    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False, encoding="utf-8-sig")
    val_df.to_csv(VAL_PATH, index=False, encoding="utf-8-sig")
    test_df.to_csv(TEST_PATH, index=False, encoding="utf-8-sig")
    combined.to_csv(COMBINED_PATH, index=False, encoding="utf-8-sig")

    summary.update(
        {
            "train_path": str(TRAIN_PATH),
            "validation_path": str(VAL_PATH),
            "test_path": str(TEST_PATH),
            "combined_path": str(COMBINED_PATH),
            "split_distribution": {
                "train": int(len(train_df)),
                "validation": int(len(val_df)),
                "test": int(len(test_df)),
            },
            "split_label_distribution": {
                "train": {str(k): int(v) for k, v in train_df["label"].value_counts().sort_index().items()},
                "validation": {str(k): int(v) for k, v in val_df["label"].value_counts().sort_index().items()},
                "test": {str(k): int(v) for k, v in test_df["label"].value_counts().sort_index().items()},
            },
            "label_mapping_verified": {"0": "SAFE/BENIGN", "1": "PROMPT_INJECTION"},
        }
    )
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
