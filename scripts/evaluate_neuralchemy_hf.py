"""Evaluate existing Transformer prompt-injection models on Neuralchemy HF dataset.

No training, no fine-tuning, and no checkpoint overwrite are performed.

Outputs are written to ``evaluation_neuralchemy/`` by default.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration_runtime import (  # noqa: E402
    apply_probability_calibrator,
    canonical_model_key,
    get_calibrated_threshold_entry,
    load_runtime_calibrator,
)
from src.transformer_utils import (  # noqa: E402
    is_finetuned_transformer_checkpoint,
    resolve_transformer_model_dir,
)


MODEL_CONFIGS = {
    "roberta": {
        "display_name": "RoBERTa",
        "alias": "roberta",
    },
    "xlm_roberta": {
        "display_name": "XLM-RoBERTa",
        "alias": "xlm_roberta",
    },
}


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def _safe_json(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_json(v) for v in value]
    return value


def _normalise_label(value: Any, category: Any = None) -> int:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        text = str(category or "").strip().lower()
    else:
        text = str(value).strip().lower()
    if text in {"1", "true", "attack", "prompt_injection", "injection", "malicious", "jailbreak", "unsafe"}:
        return 1
    if text in {"0", "false", "benign", "safe", "clean", "normal"}:
        return 0
    try:
        return 1 if int(float(text)) != 0 else 0
    except (TypeError, ValueError):
        category_text = str(category or "").strip().lower()
        return 0 if category_text == "benign" else 1


def _find_text_column(columns: list[str]) -> str:
    for column in ["text", "prompt", "input", "instruction", "content", "query"]:
        if column in columns:
            return column
    raise ValueError(f"Không tìm thấy cột text trong dataset: {columns}")


def _find_label_column(columns: list[str]) -> str:
    for column in ["label", "labels", "category", "target", "is_malicious"]:
        if column in columns:
            return column
    raise ValueError(f"Không tìm thấy cột label trong dataset: {columns}")


def _load_neuralchemy_dataset(config_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    ds = load_dataset("neuralchemy/Prompt-injection-dataset", config_name)
    frames: list[pd.DataFrame] = []
    schema: dict[str, Any] = {"config": config_name, "splits": {}}
    for split_name, split_ds in ds.items():
        frame = split_ds.to_pandas()
        frame["split"] = split_name
        frames.append(frame)
        schema["splits"][split_name] = {
            "rows": len(frame),
            "columns": list(frame.columns),
            "features": {key: str(value) for key, value in split_ds.features.items()},
        }
    raw = pd.concat(frames, ignore_index=True)
    text_column = _find_text_column(list(raw.columns))
    label_column = _find_label_column(list(raw.columns))

    raw[text_column] = raw[text_column].fillna("").astype(str)
    before = len(raw)
    raw = raw[raw[text_column].str.strip().astype(bool)].copy()
    raw["true_label"] = [
        _normalise_label(label, category)
        for label, category in zip(raw[label_column], raw.get("category", pd.Series([None] * len(raw))))
    ]
    raw["row_id"] = np.arange(len(raw))
    raw["word_count"] = raw[text_column].str.split().str.len()
    raw["char_count"] = raw[text_column].str.len()
    schema["text_column"] = text_column
    schema["label_column"] = label_column
    schema["rows_before_cleaning"] = before
    schema["rows_after_cleaning"] = len(raw)
    schema["removed_empty_or_invalid_rows"] = before - len(raw)
    return raw.reset_index(drop=True), schema


def _dataset_summary(df: pd.DataFrame, schema: dict[str, Any], token_stats: dict[str, Any]) -> dict[str, Any]:
    total = len(df)
    label_counts = df["true_label"].value_counts().to_dict()
    split_counts = df["split"].value_counts().to_dict() if "split" in df.columns else {}
    summary: dict[str, Any] = {
        "dataset_name": "neuralchemy/Prompt-injection-dataset",
        "config": schema["config"],
        "source": "https://huggingface.co/datasets/neuralchemy/Prompt-injection-dataset",
        "text_column": schema["text_column"],
        "label_column": schema["label_column"],
        "rows_before_cleaning": schema["rows_before_cleaning"],
        "rows_after_cleaning": schema["rows_after_cleaning"],
        "removed_empty_or_invalid_rows": schema["removed_empty_or_invalid_rows"],
        "total_samples": total,
        "benign_count": int(label_counts.get(0, 0)),
        "attack_count": int(label_counts.get(1, 0)),
        "benign_ratio": float(label_counts.get(0, 0) / total) if total else 0.0,
        "attack_ratio": float(label_counts.get(1, 0) / total) if total else 0.0,
        "split_counts": {str(k): int(v) for k, v in split_counts.items()},
        "average_char_length": float(df["char_count"].mean()),
        "min_char_length": int(df["char_count"].min()),
        "max_char_length": int(df["char_count"].max()),
        "average_word_length": float(df["word_count"].mean()),
        "min_word_length": int(df["word_count"].min()),
        "max_word_length": int(df["word_count"].max()),
        "token_stats": token_stats,
    }
    for column in ["category", "source", "severity", "augmented"]:
        if column in df.columns:
            values = df[column].fillna("").astype(str).replace("", "(blank)").value_counts().to_dict()
            summary[f"{column}_counts"] = {str(k): int(v) for k, v in values.items()}
    if "tags" in df.columns:
        tag_counts: dict[str, int] = {}
        for item in df["tags"]:
            if isinstance(item, np.ndarray):
                tags = item.tolist()
            elif isinstance(item, list):
                tags = item
            elif pd.isna(item):
                tags = []
            else:
                tags = [str(item)]
            for tag in tags:
                tag_counts[str(tag)] = tag_counts.get(str(tag), 0) + 1
        summary["top_tags"] = dict(sorted(tag_counts.items(), key=lambda kv: kv[1], reverse=True)[:30])
    return summary


def _directory_size(path: Path) -> int:
    return int(sum(file.stat().st_size for file in path.rglob("*") if file.is_file()))


def _model_num_parameters(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _token_lengths(texts: list[str], model_alias: str, batch_size: int = 512, max_length: int = 512) -> dict[str, Any]:
    model_dir = resolve_transformer_model_dir(model_alias)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    lengths: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            truncation=True,
            padding=False,
            max_length=max_length,
            add_special_tokens=True,
        )
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    return {
        "tokenizer_model": model_alias,
        "min_tokens": int(np.min(lengths)) if lengths else 0,
        "max_tokens": int(np.max(lengths)) if lengths else 0,
        "avg_tokens": float(np.mean(lengths)) if lengths else 0.0,
        "p95_tokens": float(np.percentile(lengths, 95)) if lengths else 0.0,
        "truncated_at": max_length,
    }


def _load_threshold(model_key: str) -> dict[str, Any]:
    entry = get_calibrated_threshold_entry(model_key) or {}
    if entry:
        return {
            "threshold": float(entry.get("threshold_eval", 0.5)),
            "threshold_source": "models/calibrated_thresholds.json",
            "threshold_warn": float(entry.get("threshold_warn", 0.5)),
            "threshold_block": float(entry.get("threshold_block", 0.8)),
            "calibration_method": entry.get("calibration_method"),
            "score_type": entry.get("score_type"),
        }
    return {
        "threshold": 0.5,
        "threshold_source": "default_0.5",
        "threshold_warn": 0.5,
        "threshold_block": 0.8,
        "calibration_method": None,
        "score_type": "raw_softmax_probability",
    }


def _apply_calibration(model_key: str, raw_scores: np.ndarray) -> tuple[np.ndarray, str, str | None]:
    calibrator = load_runtime_calibrator(model_key)
    if calibrator is None:
        return raw_scores.astype(float), "raw_softmax_probability", None
    calibrated = np.array([apply_probability_calibrator(calibrator, float(score)) for score in raw_scores], dtype=float)
    return calibrated, "calibrated_probability", calibrator.__class__.__name__


def _predict_model(
    texts: list[str],
    model_key: str,
    batch_size: int,
    max_length: int,
    use_cuda: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    model_alias = MODEL_CONFIGS[model_key]["alias"]
    model_dir = resolve_transformer_model_dir(model_alias)
    if not is_finetuned_transformer_checkpoint(model_dir):
        raise FileNotFoundError(f"Checkpoint không hợp lệ hoặc chưa fine-tune: {model_dir}")
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    model.to(device)
    model.eval()

    raw_scores: list[float] = []
    predicted_batches = 0
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            probabilities = torch.softmax(outputs.logits, dim=-1)[:, 1]
            raw_scores.extend(float(value) for value in probabilities.detach().cpu().tolist())
            predicted_batches += 1
    inference_time = time.perf_counter() - started
    raw = np.array(raw_scores, dtype=float)
    probabilities, score_used, calibration_method = _apply_calibration(model_key, raw)
    threshold_info = _load_threshold(model_key)
    threshold = float(threshold_info["threshold"])
    predicted = (probabilities >= threshold).astype(int)
    result = pd.DataFrame(
        {
            "model": model_key,
            "raw_probability": raw,
            "calibrated_probability": probabilities if score_used == "calibrated_probability" else np.nan,
            "probability": probabilities,
            "score_used": score_used,
            "predicted_label": predicted,
            "threshold": threshold,
        }
    )
    metadata = {
        "model": model_key,
        "display_name": MODEL_CONFIGS[model_key]["display_name"],
        "model_path": str(model_dir),
        "model_size_bytes": _directory_size(model_dir),
        "num_parameters": _model_num_parameters(model),
        "runtime_device": str(device),
        "batch_size": batch_size,
        "max_length": max_length,
        "inference_time_seconds": float(inference_time),
        "avg_latency_ms": float(inference_time * 1000 / len(texts)) if texts else 0.0,
        "predicted_batches": predicted_batches,
        "score_used": score_used,
        "calibration_method": calibration_method or threshold_info.get("calibration_method"),
        **threshold_info,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, metadata


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, scores)
    try:
        roc_auc = roc_auc_score(y_true, scores)
    except ValueError:
        roc_auc = None
    try:
        pr_auc = auc(recall_curve, precision_curve)
    except ValueError:
        pr_auc = None
    try:
        avg_precision = average_precision_score(y_true, scores)
    except ValueError:
        avg_precision = None
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": None if roc_auc is None else float(roc_auc),
        "pr_auc": None if pr_auc is None else float(pr_auc),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "specificity": float(specificity),
        "sensitivity": float(sensitivity),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "true_positive_rate": float(sensitivity),
        "true_negative_rate": float(specificity),
        "average_precision": None if avg_precision is None else float(avg_precision),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "support": {
            "total": int(len(y_true)),
            "benign": int((y_true == 0).sum()),
            "attack": int((y_true == 1).sum()),
        },
    }


def _threshold_sweep(y_true: np.ndarray, scores: np.ndarray, model_key: str) -> pd.DataFrame:
    rows = []
    for threshold in np.arange(0.05, 1.0, 0.05):
        threshold = round(float(threshold), 2)
        y_pred = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        fnr = fn / (fn + tp) if (fn + tp) else 0.0
        rows.append(
            {
                "model": model_key,
                "threshold": threshold,
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall,
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "accuracy": accuracy_score(y_true, y_pred),
                "specificity": specificity,
                "fpr": fpr,
                "fnr": fnr,
                "youden_j": recall + specificity - 1.0,
                "balanced_accuracy": (recall + specificity) / 2.0,
            }
        )
    return pd.DataFrame(rows)


def _best_thresholds(sweep: pd.DataFrame, current_threshold: float) -> dict[str, Any]:
    def best_by(metric: str) -> dict[str, Any]:
        sorted_df = sweep.sort_values([metric, "threshold"], ascending=[False, False])
        row = sorted_df.iloc[0].to_dict()
        return {key: _safe_json(value) for key, value in row.items()}

    return {
        "current_threshold": float(current_threshold),
        "best_by_f1": best_by("f1"),
        "best_by_youden_j": best_by("youden_j"),
        "best_by_balanced_accuracy": best_by("balanced_accuracy"),
    }


def _write_plots(output_dir: Path, predictions: pd.DataFrame, metrics_by_model: dict[str, Any], sweep: pd.DataFrame) -> None:
    y_true = predictions.drop_duplicates("row_id").sort_values("row_id")["true_label"].to_numpy()

    plt.figure(figsize=(8, 6))
    for model_key in MODEL_CONFIGS:
        df = predictions[predictions["model"] == model_key].sort_values("row_id")
        fpr, tpr, _ = roc_curve(y_true, df["probability"].to_numpy())
        label = f"{MODEL_CONFIGS[model_key]['display_name']} (AUC={metrics_by_model[model_key]['roc_auc']:.3f})"
        plt.plot(fpr, tpr, label=label)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 6))
    for model_key in MODEL_CONFIGS:
        df = predictions[predictions["model"] == model_key].sort_values("row_id")
        precision, recall, _ = precision_recall_curve(y_true, df["probability"].to_numpy())
        label = f"{MODEL_CONFIGS[model_key]['display_name']} (AP={metrics_by_model[model_key]['average_precision']:.3f})"
        plt.plot(recall, precision, label=label)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curve.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, model_key in zip(axes, MODEL_CONFIGS):
        cm = metrics_by_model[model_key]["confusion_matrix"]
        matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        im = axis.imshow(matrix, cmap="Blues")
        axis.set_title(MODEL_CONFIGS[model_key]["display_name"])
        axis.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
        axis.set_yticks([0, 1], labels=["True 0", "True 1"])
        for (i, j), value in np.ndenumerate(matrix):
            axis.text(j, i, str(value), ha="center", va="center")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    fig.suptitle("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    for model_key in MODEL_CONFIGS:
        df = predictions[predictions["model"] == model_key]
        plt.hist(df[df["true_label"] == 0]["probability"], bins=30, alpha=0.35, label=f"{MODEL_CONFIGS[model_key]['display_name']} benign")
        plt.hist(df[df["true_label"] == 1]["probability"], bins=30, alpha=0.35, label=f"{MODEL_CONFIGS[model_key]['display_name']} attack")
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title("Probability Distribution")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "probability_distribution.png", dpi=180)
    plt.savefig(output_dir / "prediction_histogram.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for model_key in MODEL_CONFIGS:
        model_sweep = sweep[sweep["model"] == model_key]
        name = MODEL_CONFIGS[model_key]["display_name"]
        axes[0].plot(model_sweep["threshold"], model_sweep["f1"], marker="o", label=name)
        axes[1].plot(model_sweep["threshold"], model_sweep["precision"], marker="o", label=f"{name} Precision")
        axes[1].plot(model_sweep["threshold"], model_sweep["recall"], marker="x", label=f"{name} Recall")
    axes[0].set_title("Threshold vs F1")
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("F1")
    axes[1].set_title("Threshold vs Precision/Recall")
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Score")
    axes[0].legend()
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "threshold_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 6))
    for model_key in MODEL_CONFIGS:
        df = predictions[predictions["model"] == model_key].sort_values("row_id")
        frac_pos, mean_pred = calibration_curve(y_true, df["probability"].to_numpy(), n_bins=10, strategy="uniform")
        plt.plot(mean_pred, frac_pos, marker="o", label=MODEL_CONFIGS[model_key]["display_name"])
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "calibration_curve.png", dpi=180)
    plt.close()


def _format_metric(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _sample_errors(errors: pd.DataFrame, n: int = 5) -> str:
    if errors.empty:
        return "Không có mẫu."
    lines = []
    for _, row in errors.head(n).iterrows():
        text = str(row["text"]).replace("\n", " ")
        if len(text) > 220:
            text = text[:220] + "..."
        lines.append(f"- `{row['model']}` probability={row['probability']:.4f}, true={row['true_label']}, pred={row['predicted_label']}: {text}")
    return "\n".join(lines)


def _write_report(
    output_dir: Path,
    dataset_summary: dict[str, Any],
    metrics_by_model: dict[str, Any],
    threshold_summary: dict[str, Any],
    false_positive: pd.DataFrame,
    false_negative: pd.DataFrame,
) -> None:
    comparison_rows = []
    for model_key in MODEL_CONFIGS:
        metrics = metrics_by_model[model_key]
        comparison_rows.append(
            "| {model} | {accuracy} | {precision} | {recall} | {f1} | {roc_auc} | {pr_auc} | {balanced} | {mcc} | {specificity} | {sensitivity} | {time}s | {size_mb} MB |".format(
                model=MODEL_CONFIGS[model_key]["display_name"],
                accuracy=_format_metric(metrics["accuracy"]),
                precision=_format_metric(metrics["precision"]),
                recall=_format_metric(metrics["recall"]),
                f1=_format_metric(metrics["f1"]),
                roc_auc=_format_metric(metrics["roc_auc"]),
                pr_auc=_format_metric(metrics["pr_auc"]),
                balanced=_format_metric(metrics["balanced_accuracy"]),
                mcc=_format_metric(metrics["mcc"]),
                specificity=_format_metric(metrics["specificity"]),
                sensitivity=_format_metric(metrics["sensitivity"]),
                time=_format_metric(metrics["inference_time_seconds"]),
                size_mb=_format_metric(metrics["model_size_mb"]),
            )
        )

    model_sections = []
    for model_key in MODEL_CONFIGS:
        metrics = metrics_by_model[model_key]
        cm = metrics["confusion_matrix"]
        threshold = threshold_summary[model_key]
        model_sections.append(
            f"""### {MODEL_CONFIGS[model_key]['display_name']}

| Metric | Value |
| --- | ---: |
| Accuracy | {_format_metric(metrics['accuracy'])} |
| Precision | {_format_metric(metrics['precision'])} |
| Recall / Sensitivity / TPR | {_format_metric(metrics['recall'])} |
| F1-score | {_format_metric(metrics['f1'])} |
| ROC-AUC | {_format_metric(metrics['roc_auc'])} |
| PR-AUC | {_format_metric(metrics['pr_auc'])} |
| Average Precision | {_format_metric(metrics['average_precision'])} |
| Balanced Accuracy | {_format_metric(metrics['balanced_accuracy'])} |
| MCC | {_format_metric(metrics['mcc'])} |
| Specificity / TNR | {_format_metric(metrics['specificity'])} |
| FPR | {_format_metric(metrics['false_positive_rate'])} |
| FNR | {_format_metric(metrics['false_negative_rate'])} |
| Current threshold | {_format_metric(metrics['threshold'])} |
| Inference time | {_format_metric(metrics['inference_time_seconds'])}s |
| Avg latency | {_format_metric(metrics['avg_latency_ms'])} ms/sample |

Confusion Matrix: TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}.

Threshold tối ưu theo F1: {threshold['best_by_f1']['threshold']} với F1={threshold['best_by_f1']['f1']:.4f}.
Threshold tối ưu theo Youden's J: {threshold['best_by_youden_j']['threshold']} với J={threshold['best_by_youden_j']['youden_j']:.4f}.
Threshold tối ưu theo Balanced Accuracy: {threshold['best_by_balanced_accuracy']['threshold']} với Balanced Accuracy={threshold['best_by_balanced_accuracy']['balanced_accuracy']:.4f}.
"""
        )

    roberta = metrics_by_model["roberta"]
    xlm = metrics_by_model["xlm_roberta"]
    better_f1 = "RoBERTa" if roberta["f1"] >= xlm["f1"] else "XLM-RoBERTa"
    better_recall = "RoBERTa" if roberta["recall"] >= xlm["recall"] else "XLM-RoBERTa"
    production = "RoBERTa" if roberta["mcc"] >= xlm["mcc"] and roberta["precision"] >= xlm["precision"] else "XLM-RoBERTa"

    report = f"""# Neuralchemy Prompt Injection Evaluation Report

## Dataset Summary

- Dataset: `neuralchemy/Prompt-injection-dataset`
- Config: `{dataset_summary['config']}`
- Source: https://huggingface.co/datasets/neuralchemy/Prompt-injection-dataset
- Text column: `{dataset_summary['text_column']}`
- Label column: `{dataset_summary['label_column']}`
- Rows before cleaning: {dataset_summary['rows_before_cleaning']}
- Rows after cleaning: {dataset_summary['rows_after_cleaning']}
- Removed empty/invalid rows: {dataset_summary['removed_empty_or_invalid_rows']}
- Benign: {dataset_summary['benign_count']} ({dataset_summary['benign_ratio']:.2%})
- Attack: {dataset_summary['attack_count']} ({dataset_summary['attack_ratio']:.2%})
- Split counts: `{dataset_summary['split_counts']}`
- Avg char length: {dataset_summary['average_char_length']:.2f}
- Min/max char length: {dataset_summary['min_char_length']} / {dataset_summary['max_char_length']}
- Avg word length: {dataset_summary['average_word_length']:.2f}
- Min/max word length: {dataset_summary['min_word_length']} / {dataset_summary['max_word_length']}

Token statistics:

```json
{json.dumps(dataset_summary['token_stats'], ensure_ascii=False, indent=2)}
```

Class/category distribution:

```json
{json.dumps({k: v for k, v in dataset_summary.items() if k.endswith('_counts') or k == 'top_tags'}, ensure_ascii=False, indent=2)}
```

## Model Evaluation

{chr(10).join(model_sections)}

## Threshold Analysis

Không cập nhật threshold model. Báo cáo chỉ sweep threshold từ 0.05 đến 0.95.

Threshold hiện tại lấy từ runtime artifact `models/calibrated_thresholds.json` nếu có:

- RoBERTa: {metrics_by_model['roberta']['threshold']} ({metrics_by_model['roberta']['threshold_source']})
- XLM-RoBERTa: {metrics_by_model['xlm_roberta']['threshold']} ({metrics_by_model['xlm_roberta']['threshold_source']})

Threshold tối ưu có thể khác threshold hiện tại vì dataset Neuralchemy có phân bố khác dataset calibration cũ. Nếu threshold tối ưu cao/thấp hơn, đó là dấu hiệu model calibration/decision boundary chưa chuyển miền hoàn hảo sang dataset mới.

## Error Analysis

### False Positive điển hình

{_sample_errors(false_positive)}

Nguyên nhân thường gặp: dataset có hard negatives chứa từ khóa như `ignore`, `override`, `security`, hoặc ngữ cảnh security-adjacent nhưng label vẫn là benign.

### False Negative điển hình

{_sample_errors(false_negative)}

Nguyên nhân thường gặp: attack được viết gián tiếp, nhẹ, hoặc không chứa pattern quen thuộc; model có thể chưa đủ generalize sang cách viết mới của Neuralchemy.

## Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Balanced Accuracy | MCC | Specificity | Sensitivity | Inference time | Model size |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(comparison_rows)}

Nhận xét:

- Model có F1 cao hơn: {better_f1}.
- Model phát hiện prompt injection tốt hơn theo Recall: {better_recall}.
- Model phù hợp production hơn theo cân bằng Precision/MCC: {production}.

## Conclusion

- Dataset Neuralchemy core có {dataset_summary['total_samples']} mẫu, gồm {dataset_summary['benign_count']} benign và {dataset_summary['attack_count']} attack.
- Đây là dataset có nhiều hard negatives và security-adjacent benign prompts, nên phù hợp để kiểm tra generalization.
- Nếu hiệu năng giảm so với benchmark cũ, đó là dấu hiệu dataset mới khó hơn hoặc phân phối khác hơn.
- Có thể có dấu hiệu overfitting nếu model đạt rất cao trên dataset train/test nội bộ nhưng giảm mạnh trên Neuralchemy.
- Không nên fine-tune ngay nếu mục tiêu là đánh giá khách quan; sau khi báo cáo xong, có thể fine-tune thêm trên train split Neuralchemy nhưng cần giữ validation/test riêng để tránh leakage.

## Output Files

- `metrics_summary.csv`
- `metrics_summary.json`
- `prediction_full.csv`
- `false_positive.csv`
- `false_negative.csv`
- `threshold_sweep.csv`
- `threshold_summary.json`
- `evaluation_report.md`
- `roc_curve.png`
- `pr_curve.png`
- `confusion_matrix.png`
- `threshold_curve.png`
- `calibration_curve.png`
- `probability_distribution.png`
- `prediction_histogram.png`
"""
    (output_dir / "evaluation_report.md").write_text(report, encoding="utf-8")


def run_evaluation(args: argparse.Namespace) -> None:
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    df, schema = _load_neuralchemy_dataset(args.config)
    text_column = schema["text_column"]
    texts = df[text_column].astype(str).tolist()

    print(f"[INFO] Loaded Neuralchemy config={args.config}: {len(df)} rows")
    token_stats = {
        model_key: _token_lengths(texts, model_key, max_length=args.max_length)
        for model_key in MODEL_CONFIGS
    }
    dataset_summary = _dataset_summary(df, schema, token_stats)
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(_safe_json(dataset_summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metadata_by_model: dict[str, Any] = {}
    prediction_frames: list[pd.DataFrame] = []
    for model_key in MODEL_CONFIGS:
        print(f"[INFO] Evaluating {model_key}...")
        try:
            pred, metadata = _predict_model(
                texts,
                model_key=model_key,
                batch_size=args.batch_size,
                max_length=args.max_length,
                use_cuda=not args.cpu,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or args.cpu:
                raise
            print(f"[WARN] CUDA OOM for {model_key}; retrying on CPU with batch_size=8")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            pred, metadata = _predict_model(
                texts,
                model_key=model_key,
                batch_size=8,
                max_length=args.max_length,
                use_cuda=False,
            )
        metadata_by_model[model_key] = metadata
        prediction_frames.append(pred)

    base_columns = [
        "row_id",
        "split",
        text_column,
        "true_label",
        "category",
        "source",
        "severity",
        "group_id",
        "augmented",
        "tags",
        "word_count",
        "char_count",
    ]
    for column in base_columns:
        if column not in df.columns:
            df[column] = None
    base = df[base_columns].rename(columns={text_column: "text"})
    all_predictions = []
    metrics_rows = []
    metrics_by_model: dict[str, Any] = {}
    sweep_frames = []
    threshold_summary: dict[str, Any] = {}
    y_true = base["true_label"].to_numpy(dtype=int)
    for pred in prediction_frames:
        model_key = str(pred["model"].iloc[0])
        full = pd.concat([base.reset_index(drop=True), pred.reset_index(drop=True)], axis=1)
        scores = full["probability"].to_numpy(dtype=float)
        y_pred = full["predicted_label"].to_numpy(dtype=int)
        metrics = _metrics(y_true, y_pred, scores)
        metadata = metadata_by_model[model_key]
        metrics.update(
            {
                "model": model_key,
                "display_name": metadata["display_name"],
                "threshold": metadata["threshold"],
                "threshold_source": metadata["threshold_source"],
                "threshold_warn": metadata["threshold_warn"],
                "threshold_block": metadata["threshold_block"],
                "score_used": metadata["score_used"],
                "calibration_method": metadata["calibration_method"],
                "model_path": metadata["model_path"],
                "model_size_bytes": metadata["model_size_bytes"],
                "model_size_mb": metadata["model_size_bytes"] / (1024 * 1024),
                "num_parameters": metadata["num_parameters"],
                "runtime_device": metadata["runtime_device"],
                "batch_size": metadata["batch_size"],
                "max_length": metadata["max_length"],
                "inference_time_seconds": metadata["inference_time_seconds"],
                "avg_latency_ms": metadata["avg_latency_ms"],
            }
        )
        metrics_by_model[model_key] = metrics
        metrics_rows.append(metrics)
        sweep = _threshold_sweep(y_true, scores, model_key)
        sweep_frames.append(sweep)
        threshold_summary[model_key] = _best_thresholds(sweep, metadata["threshold"])
        all_predictions.append(full)

    predictions = pd.concat(all_predictions, ignore_index=True)
    threshold_sweep = pd.concat(sweep_frames, ignore_index=True)
    false_positive = predictions[(predictions["true_label"] == 0) & (predictions["predicted_label"] == 1)].copy()
    false_negative = predictions[(predictions["true_label"] == 1) & (predictions["predicted_label"] == 0)].copy()

    metrics_summary = pd.DataFrame(metrics_rows)
    metric_column_order = [
        "model",
        "display_name",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "mcc",
        "specificity",
        "sensitivity",
        "false_positive_rate",
        "false_negative_rate",
        "true_positive_rate",
        "true_negative_rate",
        "average_precision",
        "threshold",
        "score_used",
        "calibration_method",
        "inference_time_seconds",
        "avg_latency_ms",
        "model_size_mb",
        "num_parameters",
        "runtime_device",
    ]
    metrics_summary.to_csv(output_dir / "metrics_summary.csv", index=False, columns=[c for c in metric_column_order if c in metrics_summary.columns])
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(
            _safe_json(
                {
                    "dataset_summary": dataset_summary,
                    "models": metrics_by_model,
                    "model_metadata": metadata_by_model,
                }
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    prediction_columns = [
        "model",
        "split",
        "row_id",
        "text",
        "true_label",
        "predicted_label",
        "probability",
        "threshold",
        "raw_probability",
        "calibrated_probability",
        "score_used",
        "category",
        "source",
        "severity",
        "group_id",
        "augmented",
        "tags",
        "word_count",
        "char_count",
    ]
    predictions.to_csv(output_dir / "prediction_full.csv", index=False, columns=[c for c in prediction_columns if c in predictions.columns])
    false_positive.to_csv(output_dir / "false_positive.csv", index=False, columns=[c for c in prediction_columns if c in false_positive.columns])
    false_negative.to_csv(output_dir / "false_negative.csv", index=False, columns=[c for c in prediction_columns if c in false_negative.columns])
    threshold_sweep.to_csv(output_dir / "threshold_sweep.csv", index=False)
    (output_dir / "threshold_summary.json").write_text(
        json.dumps(_safe_json(threshold_summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_plots(output_dir, predictions, metrics_by_model, threshold_sweep)
    _write_report(output_dir, dataset_summary, metrics_by_model, threshold_summary, false_positive, false_negative)
    print(f"[OK] Evaluation complete: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="core", choices=["core", "full"])
    parser.add_argument("--output-dir", default="evaluation_neuralchemy")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    _configure_stdout()
    run_evaluation(parse_args())
