"""Benchmark ba Transformer checkpoint bằng raw softmax trên Neuralchemy test.

Script này chỉ đánh giá tuần tự từng checkpoint. Script không train, không
fine-tune, không calibration, không threshold sweep, không đọc threshold
production và không ghi vào thư mục model.
"""

from __future__ import annotations

import gc
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "reports" / "baseline_three_transformers"

DATASET_NAME = "neuralchemy/Prompt-injection-dataset"
DATASET_CONFIG = "full"
DATASET_SPLIT = "test"

SEED = 42
MAX_LENGTH = 256
BATCH_SIZE = 16
THRESHOLD = 0.5

SAFE_LABEL = 0
INJECTION_LABEL = 1

PREDICTION_COLUMNS = [
    "model",
    "text",
    "true_label",
    "predicted_label",
    "injection_score",
    "category",
    "source",
    "severity",
    "augmented",
    "latency_ms",
]

COMPARISON_COLUMNS = [
    "model",
    "display_name",
    "checkpoint",
    "device",
    "fp16_inference",
    "threshold",
    "sample_count",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "f2",
    "roc_auc",
    "pr_auc",
    "false_positive_rate",
    "false_negative_rate",
    "mcc",
    "tn",
    "fp",
    "fn",
    "tp",
    "confusion_matrix",
    "inference_time_seconds",
    "samples_per_second",
    "mean_latency_ms",
    "recommended_baseline",
    "recommendation_reason",
]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    checkpoint: Path


MODEL_SPECS = [
    ModelSpec(
        key="roberta_v4",
        display_name="RoBERTa v4",
        checkpoint=PROJECT_ROOT / "models" / "transformers" / "roberta_v4",
    ),
    ModelSpec(
        key="roberta_v5_vi",
        display_name="RoBERTa v5 VI",
        checkpoint=PROJECT_ROOT / "models" / "transformers" / "roberta_v5_vi",
    ),
    ModelSpec(
        key="xlm_roberta_v5_vi",
        display_name="XLM-RoBERTa v5 VI",
        checkpoint=PROJECT_ROOT
        / "models"
        / "transformers"
        / "xlm_roberta_v5_vi",
    ),
]


def configure_stdout() -> None:
    """Giữ output tiếng Việt đọc được trên terminal hỗ trợ UTF-8."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def set_reproducible_seed(seed: int) -> None:
    """Đặt seed; inference không shuffle dataset."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def json_safe(value: Any) -> Any:
    """Chuyển numpy/Path và giá trị không hữu hạn sang JSON an toàn."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def format_metric(value: Any, digits: int = 6) -> str:
    """Định dạng metric cho console và Markdown."""
    if value is None:
        return "N/A"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "N/A"
    return f"{numeric:.{digits}f}"


def validate_checkpoints() -> None:
    """Fail sớm nếu thiếu một trong ba checkpoint local."""
    missing = [str(spec.checkpoint) for spec in MODEL_SPECS if not spec.checkpoint.is_dir()]
    if missing:
        raise FileNotFoundError(
            "Thiếu checkpoint bắt buộc; benchmark chưa chạy: " + ", ".join(missing)
        )


def load_neuralchemy_test() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Chỉ tải config full, split test; tuyệt đối không fallback dataset."""
    print(
        f"Loading dataset: {DATASET_NAME}, config={DATASET_CONFIG}, "
        f"split={DATASET_SPLIT}"
    )
    try:
        test_dataset: Dataset = load_dataset(
            DATASET_NAME,
            DATASET_CONFIG,
            split=DATASET_SPLIT,
        )
    except Exception as exc:
        raise RuntimeError(
            "Không tải được Neuralchemy config 'full', split 'test'. "
            "Benchmark đã dừng; không fallback sang dataset khác và không tạo metric giả."
        ) from exc

    required_columns = {"text", "label"}
    missing_columns = sorted(required_columns - set(test_dataset.column_names))
    if missing_columns:
        raise ValueError(
            f"Neuralchemy test thiếu cột bắt buộc {missing_columns}; "
            "benchmark dừng và không fallback."
        )

    frame = test_dataset.to_pandas().reset_index(drop=True)
    if frame.empty:
        raise ValueError("Neuralchemy test split rỗng; không tạo metric giả.")

    if frame["text"].isna().any() or frame["text"].astype(str).str.strip().eq("").any():
        raise ValueError("Neuralchemy test có text rỗng/null; benchmark dừng.")
    frame["text"] = frame["text"].astype(str)

    numeric_labels = pd.to_numeric(frame["label"], errors="coerce")
    if numeric_labels.isna().any():
        raise ValueError("Neuralchemy test có label không phải số 0/1; benchmark dừng.")
    frame["true_label"] = numeric_labels.astype(np.int64)
    label_values = set(frame["true_label"].unique().tolist())
    if label_values != {SAFE_LABEL, INJECTION_LABEL}:
        raise ValueError(
            "Neuralchemy test phải có đủ label 0=SAFE và 1=INJECTION; "
            f"nhãn thực tế: {sorted(label_values)}"
        )

    for column in ("category", "source", "severity"):
        if column not in frame.columns:
            frame[column] = ""
        else:
            frame[column] = frame[column].fillna("").astype(str)
    if "augmented" not in frame.columns:
        frame["augmented"] = False
    else:
        frame["augmented"] = frame["augmented"].fillna(False).astype(bool)

    dataset_info = {
        "name": DATASET_NAME,
        "config": DATASET_CONFIG,
        "split": DATASET_SPLIT,
        "sample_count": int(len(frame)),
        "label_mapping": {"0": "SAFE", "1": "INJECTION"},
        "label_distribution": {
            "SAFE": int((frame["true_label"] == SAFE_LABEL).sum()),
            "INJECTION": int((frame["true_label"] == INJECTION_LABEL).sum()),
        },
        "columns": list(test_dataset.column_names),
        "features": {
            name: str(feature) for name, feature in test_dataset.features.items()
        },
        "fingerprint": getattr(test_dataset, "_fingerprint", None),
    }
    print(f"Neuralchemy test loaded: {len(frame)} samples")
    print(f"Label distribution: {dataset_info['label_distribution']}")
    return frame, dataset_info


def validate_model_labels(model: torch.nn.Module, spec: ModelSpec) -> None:
    """Đảm bảo score cột 1 thực sự là INJECTION."""
    config = model.config
    id2label = {int(key): str(value).upper() for key, value in config.id2label.items()}
    if config.num_labels != 2:
        raise ValueError(
            f"{spec.key} có num_labels={config.num_labels}; benchmark cần đúng 2 nhãn."
        )
    if id2label.get(SAFE_LABEL) != "SAFE" or id2label.get(INJECTION_LABEL) != "INJECTION":
        raise ValueError(
            f"{spec.key} có id2label={config.id2label}; cần 0=SAFE, 1=INJECTION."
        )


def synchronize_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast_context(device: torch.device, use_fp16: bool):
    if device.type == "cuda" and use_fp16:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def warm_up_and_select_fp16(
    model: torch.nn.Module,
    tokenizer: Any,
    sample_text: str,
    device: torch.device,
) -> bool:
    """Thử CUDA autocast FP16; fallback FP32 nếu logits lỗi/không hữu hạn."""

    def run_warmup(use_fp16: bool) -> None:
        encoded = tokenizer(
            [sample_text],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
        logits = None
        try:
            with torch.no_grad():
                with autocast_context(device, use_fp16):
                    logits = model(**encoded).logits
            synchronize_cuda(device)
            if logits.shape[-1] != 2 or not torch.isfinite(logits).all():
                raise RuntimeError("Warm-up trả logits sai shape hoặc không hữu hạn.")
        finally:
            del logits
            del encoded

    if device.type != "cuda":
        run_warmup(use_fp16=False)
        return False

    try:
        run_warmup(use_fp16=True)
        print("FP16 autocast warm-up: SAFE")
        return True
    except RuntimeError as exc:
        print(f"WARNING: FP16 autocast không an toàn ({exc}); dùng FP32 inference.")
        gc.collect()
        torch.cuda.empty_cache()
        run_warmup(use_fp16=False)
        return False


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    injection_scores: np.ndarray,
    *,
    inference_time_seconds: float,
) -> dict[str, Any]:
    """Tính metric binary từ raw softmax score và threshold 0.5."""
    if len(y_true) == 0:
        raise ValueError("Không có sample để tính metrics.")
    if set(np.unique(y_true).tolist()) != {SAFE_LABEL, INJECTION_LABEL}:
        raise ValueError("ROC-AUC/PR-AUC yêu cầu test split có đủ hai lớp.")

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[SAFE_LABEL, INJECTION_LABEL],
    ).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) else 0.0
    sample_count = int(len(y_true))

    return {
        "threshold": THRESHOLD,
        "sample_count": sample_count,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, pos_label=INJECTION_LABEL, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, pos_label=INJECTION_LABEL, zero_division=0)
        ),
        "specificity": float(specificity),
        "f1": float(
            f1_score(y_true, y_pred, pos_label=INJECTION_LABEL, zero_division=0)
        ),
        "f2": float(
            fbeta_score(
                y_true,
                y_pred,
                beta=2,
                pos_label=INJECTION_LABEL,
                zero_division=0,
            )
        ),
        "roc_auc": float(roc_auc_score(y_true, injection_scores)),
        "pr_auc": float(average_precision_score(y_true, injection_scores)),
        "false_positive_rate": float(false_positive_rate),
        "false_negative_rate": float(false_negative_rate),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "inference_time_seconds": float(inference_time_seconds),
        "samples_per_second": float(sample_count / inference_time_seconds),
    }


def benchmark_model(
    spec: ModelSpec,
    model: torch.nn.Module,
    tokenizer: Any,
    dataset_frame: pd.DataFrame,
    device: torch.device,
    use_fp16: bool,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    """Chạy raw-softmax inference theo batch có dynamic padding."""
    texts = dataset_frame["text"].tolist()
    true_labels = dataset_frame["true_label"].to_numpy(dtype=np.int64)
    injection_scores: list[float] = []
    sample_latencies_ms: list[float] = []
    total_inference_time = 0.0

    for start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[start : start + BATCH_SIZE]
        synchronize_cuda(device)
        batch_started = time.perf_counter()

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {name: tensor.to(device) for name, tensor in encoded.items()}

        with torch.no_grad():
            with autocast_context(device, use_fp16):
                logits = model(**encoded).logits
            probabilities = torch.softmax(logits.float(), dim=-1)

        synchronize_cuda(device)
        batch_elapsed = time.perf_counter() - batch_started
        total_inference_time += batch_elapsed

        if logits.shape[-1] != 2 or not torch.isfinite(probabilities).all():
            raise RuntimeError(
                f"{spec.key} trả logits/probability không hợp lệ tại batch {start}."
            )

        batch_scores = probabilities[:, INJECTION_LABEL].detach().cpu().numpy()
        injection_scores.extend(float(score) for score in batch_scores)
        amortized_latency_ms = batch_elapsed * 1000.0 / len(batch_texts)
        sample_latencies_ms.extend([amortized_latency_ms] * len(batch_texts))

        del encoded, logits, probabilities, batch_scores

    score_array = np.asarray(injection_scores, dtype=np.float64)
    if len(score_array) != len(dataset_frame):
        raise RuntimeError(
            f"{spec.key} trả {len(score_array)} prediction cho "
            f"{len(dataset_frame)} sample."
        )
    predicted_labels = (score_array >= THRESHOLD).astype(np.int64)

    predictions = pd.DataFrame(
        {
            "model": spec.key,
            "text": dataset_frame["text"].tolist(),
            "true_label": true_labels,
            "predicted_label": predicted_labels,
            "injection_score": score_array,
            "category": dataset_frame["category"].tolist(),
            "source": dataset_frame["source"].tolist(),
            "severity": dataset_frame["severity"].tolist(),
            "augmented": dataset_frame["augmented"].tolist(),
            # Đây là latency batch được chia đều cho các sample trong batch.
            "latency_ms": sample_latencies_ms,
        },
        columns=PREDICTION_COLUMNS,
    )

    metrics = compute_metrics(
        true_labels,
        predicted_labels,
        score_array,
        inference_time_seconds=total_inference_time,
    )
    metrics.update(
        {
            "model": spec.key,
            "display_name": spec.display_name,
            "checkpoint": str(spec.checkpoint),
            "device": str(device),
            "fp16_inference": bool(use_fp16),
            "mean_latency_ms": float(np.mean(sample_latencies_ms)),
        }
    )
    curves = {
        "y_true": true_labels,
        "injection_scores": score_array,
    }
    return predictions, metrics, curves


def save_confusion_matrix(
    spec: ModelSpec,
    metrics: dict[str, Any],
    output_path: Path,
) -> None:
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    figure, axis = plt.subplots(figsize=(5.5, 4.8))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["SAFE", "INJECTION"],
        yticklabels=["SAFE", "INJECTION"],
        xlabel="Predicted label",
        ylabel="True label",
        title=f"Confusion Matrix — {spec.display_name}",
    )
    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row_index in range(2):
        for column_index in range(2):
            value = int(matrix[row_index, column_index])
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_model_outputs(
    spec: ModelSpec,
    predictions: pd.DataFrame,
    metrics: dict[str, Any],
) -> None:
    predictions.to_csv(
        OUTPUT_DIR / f"predictions_{spec.key}.csv",
        index=False,
        encoding="utf-8",
    )
    false_positives = predictions[
        (predictions["true_label"] == SAFE_LABEL)
        & (predictions["predicted_label"] == INJECTION_LABEL)
    ]
    false_negatives = predictions[
        (predictions["true_label"] == INJECTION_LABEL)
        & (predictions["predicted_label"] == SAFE_LABEL)
    ]
    false_positives.to_csv(
        OUTPUT_DIR / f"false_positives_{spec.key}.csv",
        index=False,
        encoding="utf-8",
    )
    false_negatives.to_csv(
        OUTPUT_DIR / f"false_negatives_{spec.key}.csv",
        index=False,
        encoding="utf-8",
    )
    save_confusion_matrix(
        spec,
        metrics,
        OUTPUT_DIR / f"confusion_matrix_{spec.key}.png",
    )


def save_combined_curves(
    curves_by_model: dict[str, dict[str, np.ndarray]],
    metrics_by_model: dict[str, dict[str, Any]],
) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 6.0))
    for spec in MODEL_SPECS:
        curves = curves_by_model[spec.key]
        precision, recall, _ = precision_recall_curve(
            curves["y_true"],
            curves["injection_scores"],
        )
        pr_auc = metrics_by_model[spec.key]["pr_auc"]
        axis.plot(
            recall,
            precision,
            label=f"{spec.display_name} (PR-AUC={pr_auc:.4f})",
        )
    axis.set(
        xlabel="Recall",
        ylabel="Precision",
        title="Precision-Recall Curve — raw softmax",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "precision_recall_curve.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.5, 6.0))
    for spec in MODEL_SPECS:
        curves = curves_by_model[spec.key]
        fpr, tpr, _ = roc_curve(
            curves["y_true"],
            curves["injection_scores"],
        )
        roc_auc = metrics_by_model[spec.key]["roc_auc"]
        axis.plot(fpr, tpr, label=f"{spec.display_name} (ROC-AUC={roc_auc:.4f})")
    axis.plot([0.0, 1.0], [0.0, 1.0], "k--", alpha=0.5, label="Random")
    axis.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC Curve — raw softmax",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "roc_curve.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def recommend_baseline(
    metrics_by_model: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    """Chọn baseline nghiên cứu theo F2; không đưa ra quyết định production."""
    ranked_specs = sorted(
        MODEL_SPECS,
        key=lambda spec: (
            metrics_by_model[spec.key]["f2"],
            metrics_by_model[spec.key]["pr_auc"],
            metrics_by_model[spec.key]["mcc"],
            metrics_by_model[spec.key]["balanced_accuracy"],
            metrics_by_model[spec.key]["samples_per_second"],
        ),
        reverse=True,
    )
    selected = ranked_specs[0]
    metrics = metrics_by_model[selected.key]
    reason = (
        f"{selected.display_name} có F2 cao nhất ({metrics['f2']:.6f}) tại raw "
        f"softmax threshold 0.5; tie-break lần lượt dùng PR-AUC "
        f"({metrics['pr_auc']:.6f}), MCC ({metrics['mcc']:.6f}), balanced "
        "accuracy và throughput. Đây chỉ là baseline cho thí nghiệm tiếp theo, "
        "không phải khuyến nghị production."
    )
    return selected.key, reason


def comparison_rows(
    metrics_by_model: dict[str, dict[str, Any]],
    recommended_key: str,
    reason: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in MODEL_SPECS:
        metrics = metrics_by_model[spec.key]
        row = {
            **metrics,
            "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
            "recommended_baseline": spec.key == recommended_key,
            "recommendation_reason": reason if spec.key == recommended_key else "",
        }
        rows.append({column: row.get(column) for column in COMPARISON_COLUMNS})
    return rows


def save_comparison_outputs(
    metrics_by_model: dict[str, dict[str, Any]],
    dataset_info: dict[str, Any],
    recommended_key: str,
    reason: str,
) -> None:
    rows = comparison_rows(metrics_by_model, recommended_key, reason)
    pd.DataFrame(rows, columns=COMPARISON_COLUMNS).to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False,
        encoding="utf-8",
    )

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "research_baseline_only",
        "dataset": dataset_info,
        "configuration": {
            "seed": SEED,
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "threshold": THRESHOLD,
            "label_mapping": {"0": "SAFE", "1": "INJECTION"},
            "padding": "dynamic_per_batch",
            "score": "raw_softmax_probability_for_label_1",
            "calibration": False,
            "threshold_sweep": False,
            "training": False,
        },
        "models": [metrics_by_model[spec.key] for spec in MODEL_SPECS],
        "recommended_baseline_model": recommended_key,
        "recommendation_reason": reason,
        "production_claim": False,
    }
    (OUTPUT_DIR / "model_comparison.json").write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_markdown_report(
    metrics_by_model: dict[str, dict[str, Any]],
    dataset_info: dict[str, Any],
    recommended_key: str,
    reason: str,
) -> None:
    selected_display_name = next(
        spec.display_name for spec in MODEL_SPECS if spec.key == recommended_key
    )
    lines = [
        "# Báo cáo baseline ba Transformer",
        "",
        "> Đây là benchmark nghiên cứu bằng raw softmax probability. Báo cáo không "
        "tuyên bố model production, không calibration, không threshold sweep, không "
        "fine-tune và không thay đổi model runtime.",
        "",
        "## Cấu hình thống nhất",
        "",
        f"- Dataset: `{DATASET_NAME}`",
        f"- Config: `{DATASET_CONFIG}`",
        f"- Split duy nhất: `{DATASET_SPLIT}`",
        f"- Số mẫu: `{dataset_info['sample_count']}`",
        f"- Phân phối nhãn: `{dataset_info['label_distribution']}`",
        f"- Seed: `{SEED}`",
        f"- Max length: `{MAX_LENGTH}`",
        f"- Batch size: `{BATCH_SIZE}`",
        f"- Threshold baseline cố định: `{THRESHOLD}`",
        "- Label mapping: `0 = SAFE`, `1 = INJECTION`",
        "- Padding: dynamic theo từng batch",
        "- Score: raw softmax probability của label 1",
        "- PR-AUC: average precision trên raw injection score",
        "- Inference time: tổng thời gian tokenize + transfer + forward của các batch "
        "sau warm-up; không gồm thời gian load model",
        "- `latency_ms` của từng sample là batch latency chia đều cho số sample trong batch",
        "",
        "## So sánh metrics",
        "",
        "| Model | Accuracy | Bal. Acc. | Precision | Recall | Specificity | F1 | F2 | PR-AUC | ROC-AUC | FPR | FNR | MCC | Samples/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for spec in MODEL_SPECS:
        metric = metrics_by_model[spec.key]
        lines.append(
            "| {model} | {accuracy} | {balanced} | {precision} | {recall} | "
            "{specificity} | {f1} | {f2} | {pr_auc} | {roc_auc} | {fpr} | "
            "{fnr} | {mcc} | {throughput} |".format(
                model=spec.display_name,
                accuracy=format_metric(metric["accuracy"]),
                balanced=format_metric(metric["balanced_accuracy"]),
                precision=format_metric(metric["precision"]),
                recall=format_metric(metric["recall"]),
                specificity=format_metric(metric["specificity"]),
                f1=format_metric(metric["f1"]),
                f2=format_metric(metric["f2"]),
                pr_auc=format_metric(metric["pr_auc"]),
                roc_auc=format_metric(metric["roc_auc"]),
                fpr=format_metric(metric["false_positive_rate"]),
                fnr=format_metric(metric["false_negative_rate"]),
                mcc=format_metric(metric["mcc"]),
                throughput=format_metric(metric["samples_per_second"], digits=2),
            )
        )

    lines.extend(["", "## Chi tiết từng model", ""])
    for spec in MODEL_SPECS:
        metric = metrics_by_model[spec.key]
        lines.extend(
            [
                f"### {spec.display_name}",
                "",
                f"- Checkpoint: `{metric['checkpoint']}`",
                f"- Device: `{metric['device']}`",
                f"- FP16 autocast: `{metric['fp16_inference']}`",
                f"- Sample count: `{metric['sample_count']}`",
                f"- Inference time: `{format_metric(metric['inference_time_seconds'])}` giây",
                f"- Samples/second: `{format_metric(metric['samples_per_second'])}`",
                f"- Mean latency: `{format_metric(metric['mean_latency_ms'])}` ms/sample",
                f"- Confusion matrix `[[TN, FP], [FN, TP]]`: `{metric['confusion_matrix']}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Recommended baseline model",
            "",
            f"- Model: **{selected_display_name}** (`{recommended_key}`)",
            f"- Reason: {reason}",
            "",
            "Lựa chọn này chỉ dùng để xác định baseline thống nhất trước thí nghiệm tiếp "
            "theo. Không thay model runtime và không chứng minh mức sẵn sàng production.",
            "",
            "## Artifact",
            "",
            "- `model_comparison.csv`",
            "- `model_comparison.json`",
            "- `predictions_<model>.csv`",
            "- `false_positives_<model>.csv`",
            "- `false_negatives_<model>.csv`",
            "- `confusion_matrix_<model>.png`",
            "- `precision_recall_curve.png`",
            "- `roc_curve.png`",
            "",
        ]
    )
    (OUTPUT_DIR / "full_baseline_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def print_baseline_results(
    metrics_by_model: dict[str, dict[str, Any]],
    recommended_key: str,
    reason: str,
) -> None:
    print("\n===== BASELINE RESULTS =====")
    for spec in MODEL_SPECS:
        metric = metrics_by_model[spec.key]
        print(f"\n{spec.display_name}:")
        print(f"Accuracy: {format_metric(metric['accuracy'])}")
        print(f"Precision: {format_metric(metric['precision'])}")
        print(f"Recall: {format_metric(metric['recall'])}")
        print(f"F1: {format_metric(metric['f1'])}")
        print(f"F2: {format_metric(metric['f2'])}")
        print(f"PR-AUC: {format_metric(metric['pr_auc'])}")
        print(f"ROC-AUC: {format_metric(metric['roc_auc'])}")
        print(f"FPR: {format_metric(metric['false_positive_rate'])}")
        print(f"FNR: {format_metric(metric['false_negative_rate'])}")
        print(f"MCC: {format_metric(metric['mcc'])}")

    recommended_display = next(
        spec.display_name for spec in MODEL_SPECS if spec.key == recommended_key
    )
    print(f"\nRecommended baseline model: {recommended_display}")
    print(f"Reason: {reason}")
    print("No production model decision was made.")


def main() -> None:
    configure_stdout()
    set_reproducible_seed(SEED)
    validate_checkpoints()

    # Dataset được tải trước khi tạo output để lỗi download không sinh metric/file giả.
    dataset_frame, dataset_info = load_neuralchemy_test()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    else:
        print("WARNING: CUDA không khả dụng; benchmark sẽ chạy tuần tự bằng CPU/FP32.")

    metrics_by_model: dict[str, dict[str, Any]] = {}
    curves_by_model: dict[str, dict[str, np.ndarray]] = {}

    for spec in MODEL_SPECS:
        print("\n" + "=" * 100)
        print(f"Benchmarking {spec.display_name}: {spec.checkpoint}")
        tokenizer = None
        model = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                spec.checkpoint,
                local_files_only=True,
                use_fast=True,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                spec.checkpoint,
                local_files_only=True,
            )
            validate_model_labels(model, spec)
            model.to(device)
            model.eval()
            print(f"Architecture: {model.__class__.__name__}")
            print(f"Model device: {next(model.parameters()).device}")

            use_fp16 = warm_up_and_select_fp16(
                model,
                tokenizer,
                dataset_frame.iloc[0]["text"],
                device,
            )
            predictions, metrics, curves = benchmark_model(
                spec,
                model,
                tokenizer,
                dataset_frame,
                device,
                use_fp16,
            )
            save_model_outputs(spec, predictions, metrics)
            metrics_by_model[spec.key] = metrics
            curves_by_model[spec.key] = curves
            print(
                f"Completed {spec.display_name}: F2={metrics['f2']:.6f}, "
                f"PR-AUC={metrics['pr_auc']:.6f}, "
                f"time={metrics['inference_time_seconds']:.3f}s"
            )
            del predictions, metrics, curves
        except Exception as exc:
            raise RuntimeError(
                f"Benchmark dừng tại {spec.display_name}; không thay bằng model/dataset khác."
            ) from exc
        finally:
            del model
            del tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            if device.type == "cuda":
                print(
                    "GPU memory after release: "
                    f"allocated={torch.cuda.memory_allocated(device) / (1024**3):.3f} GiB, "
                    f"reserved={torch.cuda.memory_reserved(device) / (1024**3):.3f} GiB"
                )

    recommended_key, reason = recommend_baseline(metrics_by_model)
    save_combined_curves(curves_by_model, metrics_by_model)
    save_comparison_outputs(
        metrics_by_model,
        dataset_info,
        recommended_key,
        reason,
    )
    save_markdown_report(
        metrics_by_model,
        dataset_info,
        recommended_key,
        reason,
    )
    print_baseline_results(metrics_by_model, recommended_key, reason)
    print(f"\nArtifacts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
