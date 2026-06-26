"""Prepare geekyrakshit/prompt-injection-dataset for Transformer training.

Outputs:
    datasets/processed/geekyrakshit_prompt_injection.csv
    datasets/unified/prompt_injection_transformer_ready_v3.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.language_utils import detect_language  # noqa: E402
from src.transformer_utils import import_huggingface_load_dataset  # noqa: E402

DATASET_NAME = "geekyrakshit/prompt-injection-dataset"
PROCESSED_PATH = PROJECT_ROOT / "datasets" / "processed" / "geekyrakshit_prompt_injection.csv"
UNIFIED_V3_PATH = PROJECT_ROOT / "datasets" / "unified" / "prompt_injection_transformer_ready_v3.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare geekyrakshit prompt injection dataset.")
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--processed-output", default=str(PROCESSED_PATH))
    parser.add_argument("--unified-output", default=str(UNIFIED_V3_PATH))
    parser.add_argument("--max-rows", type=int, default=0, help="Optional small subset for smoke tests.")
    return parser.parse_args()


def normalize_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in {0, 1}:
        return int(value)
    if isinstance(value, float) and value in {0.0, 1.0}:
        return int(value)

    text = str(value).strip().lower()
    mapping = {
        "0": 0,
        "safe": 0,
        "benign": 0,
        "normal": 0,
        "clean": 0,
        "1": 1,
        "malicious": 1,
        "injection": 1,
        "prompt_injection": 1,
        "prompt injection": 1,
        "attack": 1,
        "unsafe": 1,
        "jailbreak": 1,
    }
    if text not in mapping:
        raise ValueError(f"Unsupported label value: {value!r}")
    return mapping[text]


def detect_language_safe(text: str) -> str:
    try:
        return detect_language(text) or "unknown"
    except Exception:
        return "unknown"


def load_dataset_frame(dataset_name: str) -> pd.DataFrame:
    load_dataset = import_huggingface_load_dataset()
    dataset = load_dataset(dataset_name)
    frames: list[pd.DataFrame] = []
    for split_name, split_dataset in dataset.items():
        frame = split_dataset.to_pandas()
        frame["source_split"] = split_name
        frames.append(frame)
    if not frames:
        raise ValueError(f"Dataset {dataset_name} has no splits.")
    return pd.concat(frames, ignore_index=True)


def prepare_frame(raw_df: pd.DataFrame, dataset_name: str, max_rows: int = 0) -> pd.DataFrame:
    columns = {str(column).strip().lower(): column for column in raw_df.columns}
    if "prompt" not in columns:
        raise ValueError(f"Missing required prompt column. Columns: {list(raw_df.columns)}")
    if "label" not in columns:
        raise ValueError(f"Missing required label column. Columns: {list(raw_df.columns)}")

    prompt_col = columns["prompt"]
    label_col = columns["label"]
    rows: list[dict[str, Any]] = []
    for index, row in raw_df.iterrows():
        text = "" if pd.isna(row[prompt_col]) else str(row[prompt_col]).strip()
        if not text:
            continue
        try:
            label = normalize_label(row[label_col])
        except ValueError:
            continue
        rows.append(
            {
                "id": f"geekyrakshit_{row.get('source_split', 'split')}_{index + 1:06d}",
                "text": text,
                "label": int(label),
                "category": "unknown",
                "source": dataset_name,
                "language": detect_language_safe(text),
            }
        )

    prepared = pd.DataFrame(rows)
    if prepared.empty:
        raise ValueError("Prepared geekyrakshit dataset is empty.")
    prepared["dedupe_key"] = (
        prepared["text"].str.lower().str.replace(r"\s+", " ", regex=True)
        + "|"
        + prepared["label"].astype(str)
    )
    prepared = prepared.drop_duplicates(subset=["dedupe_key"], keep="first").drop(columns=["dedupe_key"])
    if max_rows and max_rows > 0:
        prepared = prepared.sample(frac=1.0, random_state=42).head(max_rows)
    return prepared[["id", "text", "label", "category", "source", "language"]].reset_index(drop=True)


def write_outputs(frame: pd.DataFrame, processed_path: Path, unified_path: Path, dataset_name: str) -> None:
    for path in [processed_path, unified_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8-sig")

    summary = {
        "dataset_name": dataset_name,
        "rows": int(len(frame)),
        "processed_output": str(processed_path),
        "unified_output": str(unified_path),
        "label_distribution": {
            str(label): int(count)
            for label, count in frame["label"].value_counts().sort_index().items()
        },
        "language_distribution": {
            str(language): int(count)
            for language, count in frame["language"].value_counts().sort_index().items()
        },
        "schema": ["id", "text", "label", "category", "source", "language"],
    }
    summary_path = unified_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> int:
    args = parse_args()
    raw_df = load_dataset_frame(args.dataset_name)
    prepared = prepare_frame(raw_df, dataset_name=args.dataset_name, max_rows=args.max_rows)
    write_outputs(
        prepared,
        processed_path=Path(args.processed_output).resolve(),
        unified_path=Path(args.unified_output).resolve(),
        dataset_name=args.dataset_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
