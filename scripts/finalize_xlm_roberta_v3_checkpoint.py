"""Finalize the saved XLM-RoBERTa v3 checkpoint.

The recovery training can save model weights successfully but hang while writing
final metadata on small Windows/CUDA setups. This script treats the saved
checkpoint as the source of truth, evaluates it, and writes the missing project
metadata/metrics artifacts.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import plot_confusion_matrix
from src.file_utils import safe_write_text
from src.thresholding import choose_threshold
from src.train_transformers import (
    DEFAULT_COMPARISON_PATH,
    DEFAULT_CONFUSION_DIR,
    DEFAULT_PROCESSED_DATA_PATH,
    DEFAULT_RESULTS_PATH,
    DEFAULT_THRESHOLDS_PATH,
    _write_transformer_comparison,
)
from src.transformer_utils import (
    ID2LABEL,
    LABEL2ID,
    MODELS_THRESHOLDS_PATH,
    evaluate_scores,
    import_optional,
    prepare_transformer_dataframe,
    softmax_positive_scores,
    split_transformer_dataframe_by_source,
    update_json_object,
)


DATASET_PATH = PROJECT_ROOT / "datasets" / "unified" / "prompt_injection_transformer_ready_v3.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "transformers" / "xlm_roberta_v3"
MAX_LENGTH = 64
BATCH_SIZE = 16


def _predict_scores(df: pd.DataFrame, tokenizer: Any, model: Any, torch: Any, device: Any) -> tuple[np.ndarray, float]:
    texts = df["model_text"].astype(str).tolist()
    scores: list[float] = []
    start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for start_index in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[start_index : start_index + BATCH_SIZE]
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                outputs = model(**encoded)
            batch_scores = softmax_positive_scores(outputs.logits.detach().cpu().numpy())
            scores.extend(float(value) for value in batch_scores)
    return np.asarray(scores, dtype=float), time.perf_counter() - start


def main() -> int:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    if not (MODEL_DIR / "pytorch_model.bin").exists() and not (MODEL_DIR / "model.safetensors").exists():
        raise FileNotFoundError(f"Checkpoint weights not found: {MODEL_DIR}")

    torch = import_optional("torch")
    transformers = import_optional("transformers")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_DIR)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.config.id2label = ID2LABEL
    model.config.label2id = LABEL2ID
    model.to(device)

    raw_df = pd.read_csv(DATASET_PATH, encoding="utf-8-sig")
    prepared_df = prepare_transformer_dataframe(raw_df)
    train_df, validation_df, test_df = split_transformer_dataframe_by_source(prepared_df)
    DEFAULT_PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    prepared_df.to_csv(
        PROJECT_ROOT / "outputs" / "transformer_processed_dataset_prompt_injection_transformer_ready_v3.csv",
        index=False,
        encoding="utf-8-sig",
    )

    validation_scores, validation_prediction_time = _predict_scores(validation_df, tokenizer, model, torch, device)
    validation_labels = validation_df["label"].astype(int).tolist()
    threshold_analysis = choose_threshold(validation_labels, validation_scores)
    thresholds = {
        "evaluation_threshold": float(threshold_analysis["selected_threshold"]),
        "runtime_warn_threshold": float(max(0.50, threshold_analysis["selected_threshold"])),
        "runtime_block_threshold": float(max(0.80, threshold_analysis["selected_threshold"])),
    }

    test_scores, test_prediction_time = _predict_scores(test_df, tokenizer, model, torch, device)
    test_labels = test_df["label"].astype(int).tolist()
    test_metrics = evaluate_scores(test_labels, test_scores, thresholds["evaluation_threshold"])
    test_metrics["total_prediction_time_seconds"] = float(test_prediction_time)
    test_metrics["avg_prediction_time_seconds"] = float(test_prediction_time / max(len(test_labels), 1))

    DEFAULT_CONFUSION_DIR.mkdir(parents=True, exist_ok=True)
    confusion_path = DEFAULT_CONFUSION_DIR / "xlm_roberta_v3_confusion_matrix.png"
    plot_confusion_matrix(test_metrics["confusion_matrix"], "xlm_roberta_v3", confusion_path)
    test_metrics["confusion_matrix_path"] = str(confusion_path)

    trainable_config = {
        "freeze_base_model": False,
        "freeze_encoder_layers": 10,
        "frozen_layers": ["embeddings", *[f"encoder.layer.{index}" for index in range(10)]],
        "training_note": "Memory-safe partial fine-tuning: embeddings and first 10 encoder layers frozen.",
    }
    trained_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "fine_tuned": True,
        "model_name": "xlm_roberta_v3",
        "model_alias": "xlm_roberta_v3",
        "base_model": "xlm-roberta-base",
        "dataset_name": "geekyrakshit/prompt-injection-dataset",
        "dataset_config": "full",
        "dataset_source": str(DATASET_PATH),
        "dataset_path": str(DATASET_PATH),
        "trained_at": trained_at,
        "epochs": 3,
        "label_mapping": {"0": "SAFE", "1": "INJECTION"},
        "training_mode": "partial_finetune",
        "trainable_config": trainable_config,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "test_rows": int(len(test_df)),
        "train_size": int(len(train_df)),
        "validation_size": int(len(validation_df)),
        "test_size": int(len(test_df)),
        "metrics": test_metrics,
        "created_by": "scripts.finalize_xlm_roberta_v3_checkpoint",
    }
    safe_write_text(MODEL_DIR / "training_metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2))
    safe_write_text(
        MODEL_DIR / "metrics.json",
        json.dumps(
            {
                "model_name": "xlm_roberta_v3",
                "base_model": "xlm-roberta-base",
                "dataset_name": "geekyrakshit/prompt-injection-dataset",
                "dataset_path": str(DATASET_PATH),
                "trained_at": trained_at,
                "epochs": 3,
                "label_mapping": {"0": "SAFE", "1": "INJECTION"},
                "training_mode": "partial_finetune",
                "trainable_config": trainable_config,
                "train_rows": int(len(train_df)),
                "validation_rows": int(len(validation_df)),
                "test_rows": int(len(test_df)),
                "thresholds": thresholds,
                "threshold_analysis": threshold_analysis,
                "validation_prediction_time_seconds": float(validation_prediction_time),
                "metrics": test_metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
    )

    result = {
        "model_name": "xlm-roberta-base",
        "model_alias": "xlm_roberta_v3",
        "model_dir": str(MODEL_DIR),
        "dataset_name": "geekyrakshit/prompt-injection-dataset",
        "dataset_config": "full",
        "dataset_source": str(DATASET_PATH),
        "local_csv": str(DATASET_PATH),
        "processed_dataset_path": str(
            PROJECT_ROOT / "outputs" / "transformer_processed_dataset_prompt_injection_transformer_ready_v3.csv"
        ),
        "train_size": int(len(train_df)),
        "validation_size": int(len(validation_df)),
        "test_size": int(len(test_df)),
        "label_distribution": {
            str(label): int(count)
            for label, count in prepared_df["label"].value_counts().sort_index().items()
        },
        "training_config": {
            "epochs": 3,
            "batch_size": 8,
            "gradient_accumulation_steps": 4,
            "max_length": MAX_LENGTH,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "fp16": bool(device.type == "cuda"),
            "device": str(device),
            "gradient_checkpointing": True,
            "freeze_encoder_layers": 10,
            "optim": "adafactor",
        },
        "training_mode": "partial_finetune",
        "trainable_config": trainable_config,
        "validation_prediction_time_seconds": float(validation_prediction_time),
        "threshold_analysis": threshold_analysis,
        "thresholds": thresholds,
        "test_metrics": test_metrics,
    }
    results_payload = update_json_object(DEFAULT_RESULTS_PATH, "xlm_roberta_v3", result)
    update_json_object(DEFAULT_THRESHOLDS_PATH, "xlm_roberta_v3", thresholds)
    update_json_object(MODELS_THRESHOLDS_PATH, "xlm_roberta_v3", thresholds)
    _write_transformer_comparison(results_payload, DEFAULT_COMPARISON_PATH)

    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
