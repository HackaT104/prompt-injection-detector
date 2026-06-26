from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src import advanced_detection, transformer_utils
from src.train_transformers_continue import (
    _assert_full_encoder_trainable,
    prepare_datasets,
    resolve_parent_checkpoint,
)


def _write_checkpoint(path: Path, model_type: str) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text(
        json.dumps({
            "model_type": model_type,
            "id2label": {"0": "SAFE", "1": "INJECTION"},
            "label2id": {"SAFE": 0, "INJECTION": 1},
        }),
        encoding="utf-8",
    )
    (path / "pytorch_model.bin").write_bytes(b"test")
    (path / "training_metadata.json").write_text(
        json.dumps({"fine_tuned": True}), encoding="utf-8"
    )


def test_default_runtime_prefers_v5_and_excludes_distilbert(monkeypatch, tmp_path: Path) -> None:
    _write_checkpoint(tmp_path / "roberta_v5_vi", "roberta")
    monkeypatch.setattr(transformer_utils, "TRANSFORMER_MODELS_DIR", tmp_path)

    assert transformer_utils.resolve_transformer_model_dir("roberta").name == "roberta_v5_vi"
    assert advanced_detection._normalize_transformer_model(None) == "roberta-base"
    assert "distilbert" not in advanced_detection.COMPARISON_MODELS


def test_parent_checkpoint_priority_keeps_v4(monkeypatch, tmp_path: Path) -> None:
    from src import train_transformers_continue as module

    v4 = tmp_path / "roberta_v4"
    v4_colab = tmp_path / "roberta_v4_colab"
    _write_checkpoint(v4, "roberta")
    _write_checkpoint(v4_colab, "roberta")
    spec = dict(module.MODEL_SPECS["roberta"])
    spec["source_candidates"] = [v4, v4_colab]
    monkeypatch.setitem(module.MODEL_SPECS, "roberta", spec)

    selected, is_v4 = resolve_parent_checkpoint("roberta")
    assert Path(selected) == v4.resolve()
    assert is_v4 is True


def test_dataset_processing_uses_stratified_vi_split_and_replay(tmp_path: Path, monkeypatch) -> None:
    from src import train_transformers_continue as module

    vi_rows = [
        {"id": f"vi_{index}", "text": f"Vietnamese sample {index}", "label": index % 2,
         "language": "vi" if index % 3 else "mixed"}
        for index in range(100)
    ]
    test_rows = [
        {"id": f"test_{index}", "text": f"Vietnamese test {index}", "label": index % 2,
         "language": "vi"}
        for index in range(20)
    ]
    replay_rows = [
        {"id": f"old_{index}", "text": f"English replay {index}", "label": index % 2,
         "language": "en"}
        for index in range(500)
    ]
    vi_train_path, vi_test_path, replay_path = (
        tmp_path / "vi_train.csv",
        tmp_path / "vi_test.csv",
        tmp_path / "replay.csv",
    )
    pd.DataFrame(vi_rows).to_csv(vi_train_path, index=False)
    pd.DataFrame(test_rows).to_csv(vi_test_path, index=False)
    pd.DataFrame(replay_rows).to_csv(replay_path, index=False)

    monkeypatch.setattr(module, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(module, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(module, "VI_TRAIN_OUTPUT", tmp_path / "vi_train_processed.csv")
    monkeypatch.setattr(module, "VI_VALIDATION_OUTPUT", tmp_path / "vi_validation_processed.csv")
    monkeypatch.setattr(module, "VI_TEST_OUTPUT", tmp_path / "vi_test_processed.csv")
    monkeypatch.setattr(module, "REPLAY_OUTPUT", tmp_path / "replay_processed.csv")
    monkeypatch.setattr(module, "DATASET_REPORT_PATH", tmp_path / "dataset_summary.json")

    summary = prepare_datasets(vi_train_path, vi_test_path, replay_path, replay_ratio=0.8)
    assert summary["splits"]["vi_validation"]["rows"] == 10
    assert summary["splits"]["mixed_replay_train"]["rows"] == 450
    assert summary["actual_replay_ratio"] == pytest.approx(0.8)
    assert summary["splits"]["vi_validation"]["labels"] == {"0": 5, "1": 5}


def test_full_encoder_guard_unfreezes_every_parameter() -> None:
    torch = pytest.importorskip("torch")

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2)
            self.classifier = torch.nn.Linear(2, 2)
            self.base_model = self.encoder

    model = FakeModel()
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    result = _assert_full_encoder_trainable(model)
    assert result["trainable_ratio"] == 1.0
    assert all(parameter.requires_grad for parameter in model.parameters())

