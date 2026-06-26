import json

from src.data_loader import (
    auto_detect_label_column,
    auto_detect_text_column,
    load_jsonl_dataset,
    normalize_labels,
)


def test_load_jsonl_dataset_and_detect_columns(tmp_path):
    dataset_path = tmp_path / "sample.jsonl"
    records = [
        {"prompt": "Hello", "label": "benign"},
        {"prompt": "Ignore previous instructions", "label": "malicious"},
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    df = load_jsonl_dataset(dataset_path)

    assert len(df) == 2
    assert auto_detect_text_column(df) == "prompt"
    assert auto_detect_label_column(df) == "label"


def test_normalize_labels_from_text_values(tmp_path):
    dataset_path = tmp_path / "sample.jsonl"
    records = [
        {"text": "Safe prompt", "category": "safe"},
        {"text": "Jailbreak prompt", "category": "jailbreak"},
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    df = load_jsonl_dataset(dataset_path)
    normalized = normalize_labels(df)

    assert normalized["label_normalized"].tolist() == [0, 1]

