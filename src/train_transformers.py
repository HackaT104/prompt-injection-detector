"""Fine-tune Transformer models for prompt injection detection.

Examples:
    python src/train_transformers.py --model distilbert --dataset-config full
    python src/train_transformers.py --model roberta --dataset-config full
    python src/train_transformers.py --model distilbert --dataset datasets/unified/prompt_injection_transformer_ready.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
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
from src.transformer_utils import (
    DEFAULT_DATASET_NAME,
    DEFAULT_TRANSFORMER_BLOCK_THRESHOLD,
    DEFAULT_TRANSFORMER_DATASET_CONFIG,
    DEFAULT_TRANSFORMER_WARN_THRESHOLD,
    ID2LABEL,
    LABEL2ID,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    SUPPORTED_TRANSFORMER_MODELS,
    TRANSFORMER_MODELS_DIR,
    evaluate_scores,
    import_optional,
    load_cached_neuralchemy_arrow_dataframe,
    load_neuralchemy_dataframe,
    prepare_transformer_dataframe,
    resolve_transformer_model_name,
    safe_model_dir_name,
    softmax_positive_scores,
    split_transformer_dataframe_by_source,
    update_json_object,
)


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
DEFAULT_RUNS_DIR = OUTPUTS_DIR / "transformer_runs"
DEFAULT_CONFUSION_DIR = OUTPUTS_DIR / "transformer_confusion_matrices"
DEFAULT_RESULTS_PATH = OUTPUTS_DIR / "transformer_results.json"
DEFAULT_THRESHOLDS_PATH = OUTPUTS_DIR / "transformer_thresholds.json"
DEFAULT_COMPARISON_PATH = OUTPUTS_DIR / "transformer_comparison.csv"
DEFAULT_PROCESSED_DATA_PATH = OUTPUTS_DIR / "transformer_processed_dataset.csv"


class TextClassificationDataset:
    def __init__(self, texts: list[str], labels: list[int], tokenizer: Any, max_length: int) -> None:
        torch = import_optional("torch")

        class _Dataset(torch.utils.data.Dataset):
            def __init__(self, outer: "TextClassificationDataset") -> None:
                self.outer = outer

            def __len__(self) -> int:
                return len(self.outer.texts)

            def __getitem__(self, index: int) -> dict[str, Any]:
                encoded = self.outer.tokenizer(
                    self.outer.texts[index],
                    truncation=True,
                    padding="max_length",
                    max_length=self.outer.max_length,
                    return_tensors="pt",
                )
                item = {key: value.squeeze(0) for key, value in encoded.items()}
                item["labels"] = self.outer.torch.tensor(self.outer.labels[index], dtype=self.outer.torch.long)
                return item

        self.torch = torch
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset = _Dataset(self)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.dataset[index]


def _training_arguments(transformers: Any, **kwargs: Any) -> Any:
    try:
        return transformers.TrainingArguments(evaluation_strategy="epoch", **kwargs)
    except TypeError:
        return transformers.TrainingArguments(eval_strategy="epoch", **kwargs)


def _compute_metrics(eval_pred: Any) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, fbeta_score, f1_score, precision_score, recall_score

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, pos_label=1, zero_division=0)),
        "recall": float(recall_score(labels, predictions, pos_label=1, zero_division=0)),
        "f1": float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
        "f2": float(fbeta_score(labels, predictions, beta=2, pos_label=1, zero_division=0)),
    }


def _predict_scores(trainer: Any, dataset: Any) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    predictions = trainer.predict(dataset)
    elapsed = time.perf_counter() - start
    scores = softmax_positive_scores(np.asarray(predictions.predictions))
    return scores, elapsed


def _load_existing_classical_rows() -> list[dict[str, Any]]:
    metrics_path = PROJECT_ROOT / "reports" / "metrics.json"
    if not metrics_path.exists():
        return []
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for model_key, model_metrics in metrics.get("models", {}).items():
        rows.append(
            {
                "model": model_key,
                "model_family": "classical_ml",
                "accuracy": model_metrics.get("accuracy"),
                "precision": model_metrics.get("precision"),
                "recall": model_metrics.get("recall"),
                "f1": model_metrics.get("f1"),
                "roc_auc": model_metrics.get("roc_auc"),
                "average_precision": model_metrics.get("average_precision"),
                "training_time_seconds": model_metrics.get("training_time_seconds"),
                "avg_prediction_time_seconds": model_metrics.get("avg_prediction_time_seconds"),
                "evaluation_threshold": model_metrics.get("evaluation_threshold"),
                "source": "reports/metrics.json",
            }
        )
    return rows


def _write_transformer_comparison(
    transformer_results: dict[str, Any],
    output_path: str | Path = DEFAULT_COMPARISON_PATH,
) -> None:
    rows = _load_existing_classical_rows()
    for model_key, result in transformer_results.get("models", {}).items():
        test_metrics = result.get("test_metrics", {})
        thresholds = result.get("thresholds", {})
        rows.append(
            {
                "model": model_key,
                "model_family": "transformer",
                "accuracy": test_metrics.get("accuracy"),
                "precision": test_metrics.get("precision"),
                "recall": test_metrics.get("recall"),
                "f1": test_metrics.get("f1"),
                "roc_auc": test_metrics.get("roc_auc"),
                "average_precision": test_metrics.get("average_precision"),
                "training_time_seconds": result.get("training_time_seconds"),
                "avg_prediction_time_seconds": test_metrics.get("avg_prediction_time_seconds"),
                "evaluation_threshold": thresholds.get("evaluation_threshold"),
                "source": "outputs/transformer_results.json",
            }
        )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "model_family",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
        "training_time_seconds",
        "avg_prediction_time_seconds",
        "evaluation_threshold",
        "source",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_fp16(torch: Any, requested_cuda: bool) -> bool:
    return bool(requested_cuda and torch.cuda.is_available())


def _iter_encoder_layers(base_model: Any) -> list[Any]:
    encoder = getattr(base_model, "encoder", None)
    if encoder is not None and hasattr(encoder, "layer"):
        return list(encoder.layer)
    transformer = getattr(base_model, "transformer", None)
    if transformer is not None and hasattr(transformer, "layer"):
        return list(transformer.layer)
    return []


def _configure_trainable_parameters(
    model: Any,
    freeze_base_model: bool = False,
    freeze_encoder_layers: int = 0,
) -> dict[str, Any]:
    """Optionally freeze Transformer layers to make large models fit small GPUs."""
    base_model = getattr(model, "base_model", None)
    frozen_parameter_count = 0
    frozen_layer_names: list[str] = []

    if freeze_base_model and base_model is not None:
        for parameter in base_model.parameters():
            if parameter.requires_grad:
                frozen_parameter_count += parameter.numel()
            parameter.requires_grad = False
        frozen_layer_names.append("base_model")
    elif freeze_encoder_layers > 0 and base_model is not None:
        embeddings = getattr(base_model, "embeddings", None)
        if embeddings is not None:
            for parameter in embeddings.parameters():
                if parameter.requires_grad:
                    frozen_parameter_count += parameter.numel()
                parameter.requires_grad = False
            frozen_layer_names.append("embeddings")

        layers = _iter_encoder_layers(base_model)
        freeze_count = min(max(int(freeze_encoder_layers), 0), len(layers))
        for layer_index, layer in enumerate(layers[:freeze_count]):
            for parameter in layer.parameters():
                if parameter.requires_grad:
                    frozen_parameter_count += parameter.numel()
                parameter.requires_grad = False
            frozen_layer_names.append(f"encoder.layer.{layer_index}")

    trainable_parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return {
        "freeze_base_model": bool(freeze_base_model),
        "freeze_encoder_layers": int(freeze_encoder_layers),
        "frozen_layers": frozen_layer_names,
        "frozen_parameter_count": int(frozen_parameter_count),
        "trainable_parameter_count": int(trainable_parameter_count),
        "total_parameter_count": int(total_parameter_count),
        "trainable_parameter_ratio": float(trainable_parameter_count / max(total_parameter_count, 1)),
    }


def train_transformer_model(
    model_name: str,
    dataset_config: str = DEFAULT_TRANSFORMER_DATASET_CONFIG,
    dataset_name: str = DEFAULT_DATASET_NAME,
    epochs: int = 3,
    batch_size: int = 4,
    max_length: int = 128,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.0,
    gradient_accumulation_steps: int = 2,
    output_dir: str | Path = DEFAULT_RUNS_DIR,
    use_cuda: bool = True,
    max_samples: int | None = None,
    local_csv: str | Path | None = None,
    prefer_cached_arrow: bool = False,
    checkpoint_name: str | None = None,
    gradient_checkpointing: bool = False,
    freeze_base_model: bool = False,
    freeze_encoder_layers: int = 0,
    optim: str = "adamw_torch",
    early_stopping_patience: int = 2,
    metric_for_best_model: str = "f1",
) -> dict[str, Any]:
    if model_name not in SUPPORTED_TRANSFORMER_MODELS:
        raise ValueError(f"model phải là một trong: {sorted(SUPPORTED_TRANSFORMER_MODELS)}")

    resolved_model_name = resolve_transformer_model_name(model_name)
    torch = import_optional("torch")
    transformers = import_optional("transformers")
    import_optional("accelerate")

    if use_cuda and not torch.cuda.is_available():
        print("CUDA is not available; training will run on CPU and may be very slow.")

    if not local_csv and dataset_config != DEFAULT_TRANSFORMER_DATASET_CONFIG:
        print(
            "WARNING: Transformer fine-tuning should use dataset_config='full'. "
            f"Current dataset_config='{dataset_config}'."
        )

    if local_csv:
        local_csv_path = Path(local_csv)
        if not local_csv_path.exists():
            raise FileNotFoundError(f"Không tìm thấy local CSV: {local_csv_path}")
        raw_df = pd.read_csv(local_csv_path, encoding="utf-8-sig")
        dataset_source = str(local_csv_path)
        processed_suffix = local_csv_path.stem
    elif prefer_cached_arrow:
        raw_df = load_cached_neuralchemy_arrow_dataframe(dataset_config=dataset_config)
        dataset_source = f"{dataset_name}/{dataset_config} (cached arrow)"
        processed_suffix = f"{dataset_config}_cached"
    else:
        raw_df = load_neuralchemy_dataframe(dataset_config=dataset_config, dataset_name=dataset_name)
        dataset_source = f"{dataset_name}/{dataset_config}"
        processed_suffix = dataset_config

    prepared_df = prepare_transformer_dataframe(raw_df, max_samples=max_samples)
    train_df, validation_df, test_df = split_transformer_dataframe_by_source(prepared_df)

    print("Dataset name:", dataset_name)
    print("Dataset config:", dataset_config)
    print("Train rows:", len(train_df))
    print("Validation rows:", len(validation_df))
    print("Test rows:", len(test_df))
    print("Label distribution:", prepared_df["label"].value_counts().sort_index().to_dict())
    print("Columns:", list(raw_df.columns))

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    processed_path = OUTPUTS_DIR / f"transformer_processed_dataset_{processed_suffix}.csv"
    prepared_df.to_csv(processed_path, index=False, encoding="utf-8-sig")

    base_safe_name = safe_model_dir_name(model_name)
    local_stem = str(Path(local_csv).stem) if local_csv else ""
    checkpoint_suffix = ""
    if local_stem.endswith("_v3"):
        checkpoint_suffix = "_v3"
    elif local_stem.endswith("_v2"):
        checkpoint_suffix = "_v2"
    safe_name = checkpoint_name or (f"{base_safe_name}{checkpoint_suffix}" if checkpoint_suffix else base_safe_name)
    safe_name = safe_name.strip().replace("/", "__").replace("\\", "__")
    run_dir = Path(output_dir) / safe_name
    model_dir = TRANSFORMER_MODELS_DIR / safe_name
    tokenizer = transformers.AutoTokenizer.from_pretrained(resolved_model_name)
    fp16_enabled = _resolve_fp16(torch, use_cuda)
    load_model_in_half = bool(fp16_enabled)
    model_kwargs = {
        "num_labels": 2,
        "id2label": ID2LABEL,
        "label2id": LABEL2ID,
    }
    if load_model_in_half:
        model_kwargs["torch_dtype"] = torch.float16
    model = transformers.AutoModelForSequenceClassification.from_pretrained(resolved_model_name, **model_kwargs)
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    trainable_config = _configure_trainable_parameters(
        model,
        freeze_base_model=freeze_base_model,
        freeze_encoder_layers=freeze_encoder_layers,
    )
    train_dataset = TextClassificationDataset(
        train_df["model_text"].tolist(),
        train_df["label"].astype(int).tolist(),
        tokenizer,
        max_length,
    )
    validation_dataset = TextClassificationDataset(
        validation_df["model_text"].tolist(),
        validation_df["label"].astype(int).tolist(),
        tokenizer,
        max_length,
    )
    test_dataset = TextClassificationDataset(
        test_df["model_text"].tolist(),
        test_df["label"].astype(int).tolist(),
        tokenizer,
        max_length,
    )

    args = _training_arguments(
        transformers,
        output_dir=str(run_dir),
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        fp16=bool(fp16_enabled and not load_model_in_half),
        logging_steps=25,
        report_to=[],
        seed=42,
        save_safetensors=False,
        save_total_limit=2,
        gradient_checkpointing=bool(gradient_checkpointing),
        optim=optim,
    )
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": validation_dataset,
        "compute_metrics": _compute_metrics,
    }
    callbacks = []
    if early_stopping_patience and early_stopping_patience > 0:
        callbacks.append(transformers.EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))
    if callbacks:
        trainer_kwargs["callbacks"] = callbacks
    try:
        trainer = transformers.Trainer(tokenizer=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = transformers.Trainer(processing_class=tokenizer, **trainer_kwargs)

    start = time.perf_counter()
    try:
        trainer.train()
    except RuntimeError as exc:
        message = str(exc).lower()
        if "out of memory" in message and batch_size > 2:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("CUDA out of memory. Retrying with batch_size=2.")
            return train_transformer_model(
                model_name=model_name,
                dataset_config=dataset_config,
                dataset_name=dataset_name,
                epochs=epochs,
                batch_size=2,
                max_length=max_length,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                warmup_ratio=warmup_ratio,
                gradient_accumulation_steps=gradient_accumulation_steps,
                output_dir=output_dir,
                use_cuda=use_cuda,
                max_samples=max_samples,
                local_csv=local_csv,
                prefer_cached_arrow=prefer_cached_arrow,
                checkpoint_name=checkpoint_name,
                gradient_checkpointing=gradient_checkpointing,
                freeze_base_model=freeze_base_model,
                freeze_encoder_layers=freeze_encoder_layers,
                optim=optim,
                early_stopping_patience=early_stopping_patience,
                metric_for_best_model=metric_for_best_model,
            )
        raise
    training_time = time.perf_counter() - start

    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    model.config.id2label = ID2LABEL
    model.config.label2id = LABEL2ID
    model.config.save_pretrained(str(model_dir))
    validation_scores, validation_prediction_time = _predict_scores(trainer, validation_dataset)
    validation_labels = validation_df["label"].astype(int).tolist()
    threshold_metric = "f2" if metric_for_best_model in {"f2", "recall"} else "f1"
    threshold_analysis = choose_threshold(
        validation_labels,
        validation_scores,
        optimization_metric=threshold_metric,
    )
    thresholds = {
        "evaluation_threshold": float(threshold_analysis["evaluation_threshold"]),
        "runtime_warn_threshold": float(threshold_analysis["runtime_warn_threshold"]),
        "runtime_block_threshold": float(threshold_analysis["runtime_block_threshold"]),
        "best_metric": threshold_analysis["best_metric"],
    }

    test_scores, test_prediction_time = _predict_scores(trainer, test_dataset)
    test_labels = test_df["label"].astype(int).tolist()
    test_metrics = evaluate_scores(test_labels, test_scores, thresholds["evaluation_threshold"])
    test_metrics["total_prediction_time_seconds"] = float(test_prediction_time)
    test_metrics["avg_prediction_time_seconds"] = float(test_prediction_time / max(len(test_labels), 1))

    DEFAULT_CONFUSION_DIR.mkdir(parents=True, exist_ok=True)
    confusion_path = DEFAULT_CONFUSION_DIR / f"{safe_name}_confusion_matrix.png"
    plot_confusion_matrix(test_metrics["confusion_matrix"], model_name, confusion_path)
    test_metrics["confusion_matrix_path"] = str(confusion_path)

    result = {
        "model_name": resolved_model_name,
        "model_alias": safe_name,
        "model_dir": str(model_dir),
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "dataset_source": dataset_source,
        "local_csv": str(local_csv) if local_csv else None,
        "processed_dataset_path": str(processed_path),
        "train_size": int(len(train_df)),
        "validation_size": int(len(validation_df)),
        "test_size": int(len(test_df)),
        "label_distribution": {
            str(label): int(count)
            for label, count in prepared_df["label"].value_counts().sort_index().items()
        },
        "training_config": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "gradient_accumulation_steps": int(gradient_accumulation_steps),
            "max_length": int(max_length),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "warmup_ratio": float(warmup_ratio),
            "fp16": bool(fp16_enabled and not load_model_in_half),
            "model_loaded_in_half_precision": bool(load_model_in_half),
            "device": "cuda" if use_cuda and torch.cuda.is_available() else "cpu",
            "gradient_checkpointing": bool(gradient_checkpointing),
            "freeze_base_model": bool(freeze_base_model),
            "freeze_encoder_layers": int(freeze_encoder_layers),
            "optim": optim,
            "early_stopping_patience": int(early_stopping_patience),
            "metric_for_best_model": metric_for_best_model,
        },
        "trainable_config": trainable_config,
        "training_time_seconds": float(training_time),
        "validation_prediction_time_seconds": float(validation_prediction_time),
        "threshold_analysis": threshold_analysis,
        "thresholds": thresholds,
        "test_metrics": test_metrics,
    }

    effective_dataset_name = (
        "geekyrakshit/prompt-injection-dataset"
        if local_csv and local_stem.endswith("_v3")
        else dataset_name
    )
    model_display_name = {
        "distilbert-base-uncased": "distilbert",
        "roberta-base": "roberta",
        "xlm-roberta-base": "xlm_roberta",
    }.get(resolved_model_name, safe_name)
    training_metadata = {
        "fine_tuned": True,
        "model_name": model_display_name if not safe_name.endswith("_v3") else safe_name,
        "model_alias": safe_name,
        "base_model": resolved_model_name,
        "dataset_name": effective_dataset_name,
        "dataset_config": dataset_config,
        "dataset_source": dataset_source,
        "dataset_path": str(local_csv) if local_csv else None,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "epochs": int(epochs),
        "label_mapping": {"0": "SAFE", "1": "INJECTION"},
        "training_mode": "partial_finetune" if freeze_base_model or freeze_encoder_layers else "full_finetune",
        "trainable_config": trainable_config,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "test_rows": int(len(test_df)),
        "train_size": int(len(train_df)),
        "validation_size": int(len(validation_df)),
        "test_size": int(len(test_df)),
        "metrics": test_metrics,
        "created_by": "src.train_transformers",
    }
    safe_write_text(
        model_dir / "training_metadata.json",
        json.dumps(training_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    safe_write_text(
        model_dir / "metrics.json",
        json.dumps(
            {
                "model_name": training_metadata["model_name"],
                "base_model": resolved_model_name,
                "dataset_name": effective_dataset_name,
                "dataset_path": str(local_csv) if local_csv else dataset_source,
                "trained_at": training_metadata["trained_at"],
                "epochs": int(epochs),
                "label_mapping": {"0": "SAFE", "1": "INJECTION"},
                "training_mode": training_metadata["training_mode"],
                "trainable_config": trainable_config,
                "train_rows": int(len(train_df)),
                "validation_rows": int(len(validation_df)),
                "test_rows": int(len(test_df)),
                "thresholds": thresholds,
                "metrics": test_metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    results_payload = update_json_object(DEFAULT_RESULTS_PATH, safe_name, result)
    update_json_object(DEFAULT_THRESHOLDS_PATH, safe_name, thresholds)
    _write_transformer_comparison(results_payload, DEFAULT_COMPARISON_PATH)
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Transformer prompt injection classifiers.")
    parser.add_argument("--model", required=True, choices=sorted(SUPPORTED_TRANSFORMER_MODELS))
    parser.add_argument("--dataset-config", default=DEFAULT_TRANSFORMER_DATASET_CONFIG, choices=["core", "full"])
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--output-dir", default=str(DEFAULT_RUNS_DIR))
    parser.add_argument("--use-cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-samples", type=int, default=0, help="Optional small subset for smoke tests.")
    parser.add_argument(
        "--local-csv",
        default="",
        help="Optional local CSV for offline/smoke tests. Default uses Hugging Face neuralchemy dataset.",
    )
    parser.add_argument(
        "--dataset",
        default="",
        help="Alias for --local-csv. Use datasets/unified/prompt_injection_transformer_ready.csv after unification.",
    )
    parser.add_argument(
        "--prefer-cached-arrow",
        action="store_true",
        help="Use local Hugging Face Arrow cache for neuralchemy dataset when Hub checks are unavailable.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default="",
        help=(
            "Optional model directory name under models/transformers. "
            "If omitted and --dataset ends with _v2/_v3.csv, saves to <model>_v2/<model>_v3."
        ),
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce VRAM usage for large Transformer models.",
    )
    parser.add_argument(
        "--freeze-base-model",
        action="store_true",
        help="Freeze the whole base Transformer and train only the classification head.",
    )
    parser.add_argument(
        "--freeze-encoder-layers",
        type=int,
        default=0,
        help="Freeze embeddings and the first N encoder layers. Useful for XLM-RoBERTa on 4GB VRAM.",
    )
    parser.add_argument(
        "--optim",
        default="adamw_torch",
        help="Trainer optimizer name, e.g. adamw_torch or adafactor.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument(
        "--metric-for-best-model",
        choices=["f1", "f2", "recall", "precision", "accuracy"],
        default="f1",
    )
    args = parser.parse_args()

    train_transformer_model(
        model_name=args.model,
        dataset_config=args.dataset_config,
        dataset_name=args.dataset_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        output_dir=args.output_dir,
        use_cuda=args.use_cuda,
        max_samples=args.max_samples or None,
        local_csv=args.dataset or args.local_csv or None,
        prefer_cached_arrow=args.prefer_cached_arrow,
        checkpoint_name=args.checkpoint_name or None,
        gradient_checkpointing=args.gradient_checkpointing,
        freeze_base_model=args.freeze_base_model,
        freeze_encoder_layers=args.freeze_encoder_layers,
        optim=args.optim,
        early_stopping_patience=args.early_stopping_patience,
        metric_for_best_model=args.metric_for_best_model,
    )


if __name__ == "__main__":
    main()

