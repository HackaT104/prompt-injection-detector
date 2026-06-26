from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transformer_utils import (
    DEFAULT_DATASET_NAME,
    import_huggingface_load_dataset,
    load_cached_neuralchemy_arrow_dataframe,
)


def load_config_dataframe(config: str):
    try:
        return load_cached_neuralchemy_arrow_dataframe(config)
    except FileNotFoundError:
        load_dataset = import_huggingface_load_dataset()
        dataset = load_dataset(DEFAULT_DATASET_NAME, config)
        frames = []
        for split, split_dataset in dataset.items():
            frame = split_dataset.to_pandas()
            frame["hf_split"] = split
            frames.append(frame)
        import pandas as pd

        return pd.concat(frames, ignore_index=True)


def distribution(frame, column: str) -> dict:
    if column not in frame.columns:
        return {}
    values = frame[column].tolist()
    result = {}
    for value in values:
        key = str(value)
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items(), key=lambda item: item[0]))


def main() -> None:
    datasets = {config: load_config_dataframe(config) for config in ["core", "full"]}

    for config, frame in datasets.items():
        print(f"\nCONFIG: {config}")
        for split, split_frame in frame.groupby("hf_split"):
            print(f"split={split} rows={len(split_frame)}")
            print("label_distribution:", distribution(split_frame, "label"))
            print("category_distribution:", distribution(split_frame, "category"))
            print("severity_distribution:", distribution(split_frame, "severity"))
            print("augmented_distribution:", distribution(split_frame, "augmented"))

    for split in ["validation", "test"]:
        core_texts = datasets["core"].loc[datasets["core"]["hf_split"] == split, "text"].tolist()
        full_texts = datasets["full"].loc[datasets["full"]["hf_split"] == split, "text"].tolist()
        print(f"{split}_same_between_core_full:", core_texts == full_texts)

    core_train_augmented = distribution(datasets["core"].loc[datasets["core"]["hf_split"] == "train"], "augmented")
    full_train_augmented = distribution(datasets["full"].loc[datasets["full"]["hf_split"] == "train"], "augmented")
    print("core_train_augmented:", core_train_augmented)
    print("full_train_augmented:", full_train_augmented)
    core_train_rows = len(datasets["core"].loc[datasets["core"]["hf_split"] == "train"])
    full_train_rows = len(datasets["full"].loc[datasets["full"]["hf_split"] == "train"])
    print("full_train_larger_than_core:", full_train_rows > core_train_rows)


if __name__ == "__main__":
    main()
