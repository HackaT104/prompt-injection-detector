"""Continue fine-tuning RoBERTa/XLM-RoBERTa v4 with Vietnamese replay data.

The parent v4 checkpoint is never overwritten. Both encoders must remain fully
trainable; the process stops instead of falling back to head-only training.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.file_utils import safe_write_text
from src.thresholding import choose_threshold
from src.train_transformers import TextClassificationDataset, _compute_metrics, _training_arguments
from src.transformer_utils import ID2LABEL, LABEL2ID, import_optional, softmax_positive_scores

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MODELS_DIR = PROJECT_ROOT / "models"
TRANSFORMER_MODELS_DIR = MODELS_DIR / "transformers"
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
RUNS_DIR = PROJECT_ROOT / "outputs" / "transformer_v5_runs"
THRESHOLDS_PATH = MODELS_DIR / "thresholds.json"

DEFAULT_VI_TRAIN = Path(os.environ.get("PROMPT_INJECTION_VI_TRAIN", r"F:\Tải về\vi_test\vi_train.csv"))
DEFAULT_VI_TEST = Path(os.environ.get("PROMPT_INJECTION_VI_TEST", r"F:\Tải về\vi_test\vi_test.csv"))
DEFAULT_REPLAY_TRAIN = PROCESSED_DIR / "hf_prompt_injection_train.csv"

VI_TRAIN_OUTPUT = PROCESSED_DIR / "vi_train_processed.csv"
VI_VALIDATION_OUTPUT = PROCESSED_DIR / "vi_validation_processed.csv"
VI_TEST_OUTPUT = PROCESSED_DIR / "vi_test_processed.csv"
REPLAY_OUTPUT = PROCESSED_DIR / "transformer_v5_replay_train.csv"
DATASET_REPORT_PATH = REPORTS_DIR / "transformer_v5_vi_dataset_summary.json"
RESULTS_STATE_PATH = REPORTS_DIR / "transformer_v5_vi_results.json"
EVALUATION_CSV_PATH = REPORTS_DIR / "transformer_v5_vi_evaluation.csv"
SUMMARY_PATH = REPORTS_DIR / "transformer_v5_vi_summary.md"
ERROR_CASES_PATH = REPORTS_DIR / "transformer_v5_vi_error_cases.csv"
SCORE_DISTRIBUTION_PATH = REPORTS_DIR / "transformer_v5_vi_score_distribution.csv"
THRESHOLD_SUMMARY_PATH = REPORTS_DIR / "threshold_summary_v5_vi.md"
ERROR_ANALYSIS_PATH = REPORTS_DIR / "error_analysis_v5.md"

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "roberta": {
        "output_name": "roberta_v5_vi",
        "expected_model_type": "roberta",
        "source_candidates": [
            TRANSFORMER_MODELS_DIR / "roberta_v4",
            TRANSFORMER_MODELS_DIR / "roberta_v4_colab",
            MODELS_DIR / "roberta_v4",
            MODELS_DIR / "roberta_v4_colab",
        ],
        "base_fallback": "roberta-base",
        "default_optim": "adamw_torch",
    },
    "xlm_roberta": {
        "output_name": "xlm_roberta_v5_vi",
        "expected_model_type": "xlm-roberta",
        "source_candidates": [
            TRANSFORMER_MODELS_DIR / "xlm_roberta_v4_colab",
            TRANSFORMER_MODELS_DIR / "xlm_roberta_v4",
            MODELS_DIR / "xlm_roberta_v4_colab",
            MODELS_DIR / "xlm_roberta_v4",
        ],
        "base_fallback": "xlm-roberta-base",
        "default_optim": "adafactor",
    },
}

LABEL_TEXT_MAPPING = {
    "0": 0, "benign": 0, "safe": 0, "normal": 0,
    "1": 1, "malicious": 1, "injection": 1, "attack": 1,
    "unsafe": 1, "prompt_injection": 1,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return " ".join(unicodedata.normalize("NFC", str(value)).strip().split())


def _dedup_key(value: Any) -> str:
    return _normalize_text(value).casefold()


def _normalize_label(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return int(value)
    if isinstance(value, (float, np.floating)) and not np.isnan(value) and int(value) in {0, 1}:
        return int(value)
    normalized = str(value).strip().lower()
    if normalized in LABEL_TEXT_MAPPING:
        return LABEL_TEXT_MAPPING[normalized]
    raise ValueError(f"Không thể map label {value!r}; chỉ chấp nhận SAFE/0 hoặc INJECTION/1.")


def _read_csv(path: str | Path, source_name: str) -> pd.DataFrame:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset: {target}")
    frame = pd.read_csv(target, encoding="utf-8-sig")
    missing = [column for column in ["text", "label"] if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset {target} thiếu cột bắt buộc: {missing}")
    frame = frame.copy()
    frame["text"] = frame["text"].map(_normalize_text)
    frame = frame[frame["text"].ne("")].copy()
    frame["label"] = frame["label"].map(_normalize_label).astype(int)
    if set(frame["label"].unique()) != {0, 1}:
        raise ValueError(f"Dataset {target} phải có đủ hai lớp 0 và 1.")
    frame["dedup_key"] = frame["text"].map(_dedup_key)
    frame["dataset_origin"] = source_name
    if "language" not in frame.columns:
        frame["language"] = "en" if source_name == "v4_replay" else "unknown"
    frame["language"] = frame["language"].fillna("unknown").astype(str).str.strip().str.lower()
    frame.loc[~frame["language"].isin(["en", "vi", "mixed"]), "language"] = "unknown"
    if "id" not in frame.columns:
        frame["id"] = [f"{source_name}_{index:07d}" for index in range(len(frame))]
    return frame


def _distribution(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "labels": {str(key): int(value) for key, value in frame["label"].value_counts().sort_index().items()},
        "languages": {str(key): int(value) for key, value in frame["language"].value_counts().items()},
        "origins": {str(key): int(value) for key, value in frame["dataset_origin"].value_counts().items()},
    }


def _stratified_sample(frame: pd.DataFrame, sample_size: int, random_state: int = 42) -> pd.DataFrame:
    if sample_size >= len(frame):
        return frame.sample(frac=1.0, random_state=random_state).copy()
    sampled, _ = train_test_split(
        frame, train_size=sample_size, random_state=random_state, stratify=frame["label"]
    )
    return sampled.copy()


def prepare_datasets(
    vi_train_path: str | Path = DEFAULT_VI_TRAIN,
    vi_test_path: str | Path = DEFAULT_VI_TEST,
    replay_train_path: str | Path = DEFAULT_REPLAY_TRAIN,
    replay_ratio: float = 0.80,
    random_state: int = 42,
) -> dict[str, Any]:
    """Create Vietnamese splits and a leakage-free 80/20 replay training set."""
    if not 0.0 < replay_ratio < 1.0:
        raise ValueError("replay_ratio phải nằm trong khoảng (0, 1).")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    vi_source = _read_csv(vi_train_path, "vi_new")
    vi_test = _read_csv(vi_test_path, "vi_new_test")
    source_rows, test_rows = len(vi_source), len(vi_test)
    vi_source = vi_source.drop_duplicates("dedup_key", keep="first").copy()
    vi_test = vi_test.drop_duplicates("dedup_key", keep="first").copy()

    test_keys = set(vi_test["dedup_key"])
    overlap_train_test = int(vi_source["dedup_key"].isin(test_keys).sum())
    vi_source = vi_source[~vi_source["dedup_key"].isin(test_keys)].copy()
    vi_train, vi_validation = train_test_split(
        vi_source, test_size=0.10, random_state=random_state, stratify=vi_source["label"]
    )
    vi_train, vi_validation = vi_train.copy(), vi_validation.copy()
    vi_train["split"], vi_validation["split"], vi_test["split"] = "train", "validation", "test"

    replay = _read_csv(replay_train_path, "v4_replay").drop_duplicates("dedup_key", keep="first")
    reserved = set(vi_train["dedup_key"]) | set(vi_validation["dedup_key"]) | set(vi_test["dedup_key"])
    replay_overlap = int(replay["dedup_key"].isin(reserved).sum())
    replay = replay[~replay["dedup_key"].isin(reserved)].copy()
    replay_count = int(round(len(vi_train) * replay_ratio / (1.0 - replay_ratio)))
    replay_sample = _stratified_sample(replay, replay_count, random_state)
    replay_sample["split"] = "train"
    mixed_train = pd.concat([replay_sample, vi_train], ignore_index=True, sort=False)
    mixed_train = mixed_train.drop_duplicates("dedup_key", keep="last")
    mixed_train = mixed_train.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    columns = [
        "id", "text", "label", "attack_type", "language", "severity",
        "injection_position", "obfuscation", "source", "dataset_origin", "split",
    ]
    for frame in [vi_train, vi_validation, vi_test, mixed_train]:
        for column in columns:
            if column not in frame.columns:
                frame[column] = ""
    vi_train[columns].to_csv(VI_TRAIN_OUTPUT, index=False, encoding="utf-8-sig")
    vi_validation[columns].to_csv(VI_VALIDATION_OUTPUT, index=False, encoding="utf-8-sig")
    vi_test[columns].to_csv(VI_TEST_OUTPUT, index=False, encoding="utf-8-sig")
    mixed_train[columns].to_csv(REPLAY_OUTPUT, index=False, encoding="utf-8-sig")

    summary = {
        "created_at": _utc_now(),
        "random_state": random_state,
        "requested_replay_ratio": replay_ratio,
        "actual_replay_ratio": float((mixed_train["dataset_origin"] == "v4_replay").mean()),
        "source_files": {
            "vi_train": str(Path(vi_train_path).resolve()),
            "vi_test": str(Path(vi_test_path).resolve()),
            "v4_replay": str(Path(replay_train_path).resolve()),
        },
        "cleaning": {
            "vi_train_original_rows": source_rows,
            "vi_test_original_rows": test_rows,
            "train_test_overlap_removed": overlap_train_test,
            "replay_overlap_removed": replay_overlap,
        },
        "splits": {
            "vi_train": _distribution(vi_train),
            "vi_validation": _distribution(vi_validation),
            "vi_test": _distribution(vi_test),
            "mixed_replay_train": _distribution(mixed_train),
        },
    }
    safe_write_text(DATASET_REPORT_PATH, json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary



def _is_checkpoint(path: Path) -> bool:
    return (
        path.exists()
        and (path / "config.json").exists()
        and any((path / filename).exists() for filename in ["model.safetensors", "pytorch_model.bin"])
    )


def resolve_parent_checkpoint(model_key: str, allow_base_fallback: bool = False) -> tuple[str, bool]:
    spec = MODEL_SPECS[model_key]
    for candidate in spec["source_candidates"]:
        if _is_checkpoint(candidate):
            return str(candidate.resolve()), True
    if allow_base_fallback:
        return str(spec["base_fallback"]), False
    checked = "\n- ".join(str(path) for path in spec["source_candidates"])
    raise FileNotFoundError(
        f"Không tìm thấy checkpoint v4 cho {model_key}. Đã kiểm tra:\n- {checked}\n"
        "Không fallback về base model khi chưa truyền --allow-base-fallback."
    )


def _validate_parent_config(transformers: Any, source: str, model_key: str, is_local_v4: bool) -> Any:
    config = transformers.AutoConfig.from_pretrained(source)
    expected_type = MODEL_SPECS[model_key]["expected_model_type"]
    if str(config.model_type) != expected_type:
        raise RuntimeError(f"Sai kiến trúc: cần {expected_type}, nhận {config.model_type} từ {source}.")
    if is_local_v4:
        id2label = {int(key): str(value).upper() for key, value in dict(config.id2label).items()}
        label2id = {str(key).upper(): int(value) for key, value in dict(config.label2id).items()}
        if id2label != ID2LABEL or label2id != LABEL2ID:
            raise RuntimeError(
                f"Label mapping v4 không hợp lệ: id2label={id2label}, label2id={label2id}."
            )
    return config


def _assert_full_encoder_trainable(model: Any) -> dict[str, Any]:
    # requires_grad is not persisted in a checkpoint. Explicitly unfreeze the old
    # XLM v4 head-only checkpoint, then verify every encoder parameter.
    for parameter in model.parameters():
        parameter.requires_grad = True
    base_model = getattr(model, "base_model", None)
    if base_model is None:
        raise RuntimeError("Không tìm thấy encoder/base_model để xác nhận full fine-tune.")
    frozen = [name for name, parameter in base_model.named_parameters() if not parameter.requires_grad]
    if frozen:
        raise RuntimeError(f"Encoder đang bị freeze: {', '.join(frozen[:10])}")
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    encoder_total = sum(parameter.numel() for parameter in base_model.parameters())
    encoder_trainable = sum(parameter.numel() for parameter in base_model.parameters() if parameter.requires_grad)
    if total != trainable or encoder_total != encoder_trainable:
        raise RuntimeError("Full fine-tune guard thất bại: vẫn còn parameter bị freeze.")
    return {
        "freeze_encoder": False,
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "trainable_ratio": float(trainable / max(total, 1)),
        "encoder_parameters": int(encoder_total),
        "encoder_trainable_parameters": int(encoder_trainable),
    }


def _load_processed(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame["text"] = frame["text"].map(_normalize_text)
    frame["label"] = frame["label"].map(_normalize_label).astype(int)
    if "language" not in frame:
        frame["language"] = "unknown"
    frame["language"] = frame["language"].fillna("unknown").astype(str).str.lower()
    return frame


def _to_dataset(frame: pd.DataFrame, tokenizer: Any, max_length: int) -> Any:
    """Pre-tokenize once; dynamic padding is applied later by the data collator."""
    torch = import_optional("torch")
    encoded = tokenizer(
        frame["text"].astype(str).tolist(),
        truncation=True,
        padding=False,
        max_length=max_length,
    )
    labels = frame["label"].astype(int).tolist()

    class EncodedDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(labels)

        def __getitem__(self, index: int) -> dict[str, Any]:
            item = {key: value[index] for key, value in encoded.items()}
            item["labels"] = labels[index]
            return item

    return EncodedDataset()


def _predict_scores(trainer: Any, dataset: Any) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    output = trainer.predict(dataset)
    return softmax_positive_scores(np.asarray(output.predictions)), time.perf_counter() - started


def _metrics(y_true: list[int], scores: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = [int(value) for value in confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()]
    result = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, pos_label=1, zero_division=0)),
        "f2": float(fbeta_score(y_true, predictions, beta=2, pos_label=1, zero_division=0)),
        "roc_auc": None,
        "average_precision": None,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "threshold": float(threshold),
        "rows": int(len(y_true)),
    }
    if len(set(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, scores))
        result["average_precision"] = float(average_precision_score(y_true, scores))
    return result


def _language_metrics(frame: pd.DataFrame, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for language in ["en", "vi", "mixed"]:
        mask = frame["language"].astype(str).str.lower().eq(language).to_numpy()
        if mask.any():
            rows[language] = _metrics(
                frame.loc[mask, "label"].astype(int).tolist(), scores[mask], threshold
            )
    return rows


def _latest_checkpoint(transformers: Any, run_dir: Path) -> str | None:
    if not run_dir.exists():
        return None
    try:
        return transformers.trainer_utils.get_last_checkpoint(str(run_dir))
    except Exception:
        return None


def _backup_existing_target(target: Path) -> Path | None:
    if not target.exists():
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = MODELS_DIR / "backups" / f"{target.name}_{timestamp}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(target, destination)
    return destination


def _error_rows(
    model_name: str,
    split_name: str,
    frame: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
) -> list[dict[str, Any]]:
    predictions = (scores >= threshold).astype(int)
    rows: list[dict[str, Any]] = []
    for position, (_, item) in enumerate(frame.reset_index(drop=True).iterrows()):
        truth, prediction = int(item["label"]), int(predictions[position])
        if truth == prediction:
            continue
        rows.append({
            "model": model_name,
            "split": split_name,
            "error_type": "false_positive" if truth == 0 else "false_negative",
            "id": item.get("id", position),
            "language": item.get("language", "unknown"),
            "ground_truth": truth,
            "predicted_label": prediction,
            "risk_score": float(scores[position]),
            "threshold": float(threshold),
            "attack_type": item.get("attack_type", ""),
            "source": item.get("source", item.get("dataset_origin", "")),
            "text": item["text"],
        })
    return rows


def _score_rows(
    model_name: str, split_name: str, frame: pd.DataFrame, scores: np.ndarray
) -> list[dict[str, Any]]:
    return [
        {
            "model": model_name,
            "split": split_name,
            "id": item.get("id", position),
            "language": item.get("language", "unknown"),
            "label": int(item["label"]),
            "risk_score": float(scores[position]),
        }
        for position, (_, item) in enumerate(frame.reset_index(drop=True).iterrows())
    ]


def _load_state() -> dict[str, Any]:
    if RESULTS_STATE_PATH.exists():
        return json.loads(RESULTS_STATE_PATH.read_text(encoding="utf-8-sig"))
    return {"created_at": _utc_now(), "models": {}}


def _save_state(state: dict[str, Any]) -> None:
    state["updated_at"] = _utc_now()
    safe_write_text(RESULTS_STATE_PATH, json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _update_thresholds(model_name: str, result: dict[str, Any]) -> None:
    payload = (
        json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8-sig"))
        if THRESHOLDS_PATH.exists() else {}
    )
    metrics = result["validation_metrics"]
    threshold_payload = {
        **result["thresholds"],
        "best_metric": "f2",
        **{key: metrics[key] for key in ["precision", "recall", "f1", "f2", "tn", "fp", "fn", "tp"]},
        "dataset": str(VI_VALIDATION_OUTPUT),
        "calibrated_at": _utc_now(),
        "model_path": result["output_model_path"],
        "parent_checkpoint": result["parent_checkpoint"],
        "warnings": [],
    }
    payload.setdefault("models", {})[model_name] = threshold_payload
    payload.setdefault("transformer_models", {})[model_name] = threshold_payload
    safe_write_text(THRESHOLDS_PATH, json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _fmt(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"



def _write_reports(state: dict[str, Any]) -> None:
    evaluation_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for model_name, result in state.get("models", {}).items():
        for split_name, metrics in result.get("evaluations", {}).items():
            evaluation_rows.append({"model": model_name, "split": split_name, **metrics})
        error_rows.extend(result.get("error_cases", []))
        score_rows.extend(result.get("score_distribution", []))

    pd.DataFrame(evaluation_rows).to_csv(EVALUATION_CSV_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(error_rows).to_csv(ERROR_CASES_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(score_rows).to_csv(SCORE_DISTRIBUTION_PATH, index=False, encoding="utf-8-sig")

    lines = [
        "# Tổng kết continue fine-tune Transformer v5 tiếng Việt",
        "",
        f"Cập nhật: {state.get('updated_at', _utc_now())}",
        "",
        "## Nguyên tắc migration",
        "",
        "- V5 warm-start từ checkpoint v4; không train lại từ base khi v4 tồn tại.",
        "- Training dùng replay 80% dữ liệu v4 và 20% dữ liệu mới.",
        "- Threshold chỉ được chọn trên validation tiếng Việt bằng F2.",
        "- Encoder phải trainable 100%; pipeline dừng nếu phát hiện freeze.",
        "",
        "## So sánh chính",
        "",
        "| Model | Split | Accuracy | Precision | Recall | F1 | F2 | ROC-AUC | PR-AUC | FP | FN |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in evaluation_rows:
        lines.append(
            f"| {row['model']} | {row['split']} | {_fmt(row.get('accuracy'))} | "
            f"{_fmt(row.get('precision'))} | {_fmt(row.get('recall'))} | {_fmt(row.get('f1'))} | "
            f"{_fmt(row.get('f2'))} | {_fmt(row.get('roc_auc'))} | "
            f"{_fmt(row.get('average_precision'))} | {row.get('fp', 'N/A')} | {row.get('fn', 'N/A')} |"
        )
    lines.extend(["", "## Theo ngôn ngữ trên Vietnamese test", ""])
    for model_name, result in state.get("models", {}).items():
        lines.extend([
            f"### {model_name}", "",
            "| Language | Rows | Precision | Recall | F1 | F2 | FP | FN |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for language, metrics in result.get("language_metrics", {}).items():
            lines.append(
                f"| {language} | {metrics['rows']} | {_fmt(metrics['precision'])} | "
                f"{_fmt(metrics['recall'])} | {_fmt(metrics['f1'])} | {_fmt(metrics['f2'])} | "
                f"{metrics['fp']} | {metrics['fn']} |"
            )
        lines.append("")
    lines.extend([
        "## DistilBERT", "",
        "DistilBERT không được train trong migration v5, được đánh dấu deprecated và "
        "không còn nằm trong runtime mặc định. Checkpoint cũ vẫn được backup để truy vết.",
        "", "## Bước tiếp theo", "",
        "Dùng v5 làm direct detector cho user_prompt. Với Context-Aware Detection, nên fine-tune "
        "một model riêng trên cặp USER_INTENT + EXTERNAL_CONTEXT và giữ nhãn indirect tách biệt.",
    ])
    safe_write_text(SUMMARY_PATH, "\n".join(lines) + "\n", encoding="utf-8")

    threshold_lines = [
        "# Threshold Transformer v5 tiếng Việt", "",
        "Quét 0.01 đến 0.99 trên validation, chọn theo F2. Test không tham gia calibration.",
        "",
        "| Model | Evaluation | Warn | Block | Precision | Recall | F1 | F2 | TN | FP | FN | TP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model_name, result in state.get("models", {}).items():
        thresholds, metrics = result["thresholds"], result["validation_metrics"]
        threshold_lines.append(
            f"| {model_name} | {thresholds['evaluation_threshold']:.2f} | "
            f"{thresholds['runtime_warn_threshold']:.2f} | {thresholds['runtime_block_threshold']:.2f} | "
            f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | "
            f"{metrics['f2']:.4f} | {metrics['tn']} | {metrics['fp']} | "
            f"{metrics['fn']} | {metrics['tp']} |"
        )
    threshold_lines.extend([
        "", "- evaluation_threshold: probability thành nhãn cho report.",
        "- warn_threshold: runtime bắt đầu cảnh báo.",
        "- block_threshold: luôn cao hơn warn và dùng để chặn.",
    ])
    safe_write_text(THRESHOLD_SUMMARY_PATH, "\n".join(threshold_lines) + "\n", encoding="utf-8")

    analysis = [
        "# Phân tích lỗi Transformer v5", "",
        "FP là prompt an toàn bị cảnh báo/chặn; FN là injection bị bỏ lọt.", "",
        "| Model | Language | FP | FN |",
        "| --- | --- | ---: | ---: |",
    ]
    if error_rows:
        errors = pd.DataFrame(error_rows)
        grouped = errors.groupby(["model", "language", "error_type"]).size().unstack(fill_value=0)
        for (model_name, language), counts in grouped.iterrows():
            analysis.append(
                f"| {model_name} | {language} | {int(counts.get('false_positive', 0))} | "
                f"{int(counts.get('false_negative', 0))} |"
            )
        for model_name in errors["model"].unique():
            analysis.extend(["", f"## {model_name}", ""])
            model_errors = errors[errors["model"] == model_name]
            for kind, title in [
                ("false_positive", "Top False Positives"),
                ("false_negative", "Top False Negatives"),
            ]:
                analysis.extend([f"### {title}", ""])
                selected = model_errors[model_errors["error_type"] == kind].sort_values(
                    "risk_score", ascending=(kind == "false_negative")
                ).head(10)
                if selected.empty:
                    analysis.append("Không có.")
                for _, item in selected.iterrows():
                    preview = str(item["text"]).replace("\n", " ")[:180]
                    analysis.append(
                        f"- {item['language']} score={float(item['risk_score']):.4f}: {preview}"
                    )
                analysis.append("")
    else:
        analysis.append("| N/A | N/A | 0 | 0 |")
    analysis.extend([
        "", "## Hướng cải thiện", "",
        "- FP tiếng Việt: thêm hard-negative cùng chủ đề và kiểu câu.",
        "- FN mixed/obfuscation: thêm code-switching và obfuscation, giữ test độc lập.",
        "- Chỉ recalibrate trên validation mới; không dùng test chọn threshold.",
    ])
    safe_write_text(ERROR_ANALYSIS_PATH, "\n".join(analysis) + "\n", encoding="utf-8")


def train_continue(
    model_key: str,
    epochs: float = 3,
    max_length: int = 128,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    use_cuda: bool = True,
    gradient_checkpointing: bool = True,
    resume_checkpoint: bool = True,
    allow_base_fallback: bool = False,
    optim: str | None = None,
    max_train_samples: int | None = None,
    output_name: str | None = None,
) -> dict[str, Any]:
    if model_key not in MODEL_SPECS:
        raise ValueError(f"Model không hỗ trợ: {model_key}")
    for path in [REPLAY_OUTPUT, VI_VALIDATION_OUTPUT, VI_TEST_OUTPUT]:
        if not path.exists():
            raise FileNotFoundError(f"Thiếu {path}; chạy --prepare-only trước.")

    torch = import_optional("torch")
    transformers = import_optional("transformers")
    source, is_local_v4 = resolve_parent_checkpoint(model_key, allow_base_fallback)
    _validate_parent_config(transformers, source, model_key, is_local_v4)
    spec = MODEL_SPECS[model_key]
    effective_name = output_name or spec["output_name"]
    model_dir = TRANSFORMER_MODELS_DIR / effective_name
    run_dir = RUNS_DIR / effective_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_df = _load_processed(REPLAY_OUTPUT)
    validation_df = _load_processed(VI_VALIDATION_OUTPUT)
    test_df = _load_processed(VI_TEST_OUTPUT)
    if max_train_samples and max_train_samples < len(train_df):
        train_df = _stratified_sample(train_df, max_train_samples).reset_index(drop=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(source, use_fast=True)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        source, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    )
    trainable_config = _assert_full_encoder_trainable(model)
    model.config.id2label = ID2LABEL
    model.config.label2id = LABEL2ID
    model.config.problem_type = "single_label_classification"
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    cuda_enabled = bool(use_cuda and torch.cuda.is_available())
    selected_optim = optim or spec["default_optim"]
    train_dataset = _to_dataset(train_df, tokenizer, max_length)
    validation_dataset = _to_dataset(validation_df, tokenizer, max_length)
    test_dataset = _to_dataset(test_df, tokenizer, max_length)
    args = _training_arguments(
        transformers,
        output_dir=str(run_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=max(1, batch_size),
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        disable_tqdm=True,
        load_best_model_at_end=True,
        metric_for_best_model="f2",
        greater_is_better=True,
        save_total_limit=2,
        fp16=cuda_enabled,
        gradient_checkpointing=gradient_checkpointing,
        optim=selected_optim,
        report_to=[],
        seed=42,
        dataloader_num_workers=0,
        remove_unused_columns=True,
    )
    trainer = transformers.Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=transformers.DataCollatorWithPadding(
            tokenizer=tokenizer, pad_to_multiple_of=8 if cuda_enabled else None
        ),
        compute_metrics=_compute_metrics,
        callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=2)],
    )
    resume_from = _latest_checkpoint(transformers, run_dir) if resume_checkpoint else None
    started = time.perf_counter()
    try:
        trainer.train(resume_from_checkpoint=resume_from)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except RuntimeError:
                    pass
            raise RuntimeError(
                "CUDA OOM trong full fine-tune. Pipeline đã dừng và KHÔNG chuyển sang head-only. "
                "Dùng --batch-size 1 --gradient-accumulation-steps 16 --optim adafactor "
                "hoặc chạy notebook Colab GPU."
            ) from exc
        raise
    training_seconds = time.perf_counter() - started



    validation_scores, validation_prediction_seconds = _predict_scores(trainer, validation_dataset)
    threshold_search = choose_threshold(
        validation_df["label"].astype(int).tolist(),
        validation_scores,
        optimization_metric="f2",
    )
    thresholds = {
        "evaluation_threshold": float(threshold_search["evaluation_threshold"]),
        "runtime_warn_threshold": float(threshold_search["runtime_warn_threshold"]),
        "runtime_block_threshold": float(threshold_search["runtime_block_threshold"]),
    }
    evaluation_threshold = thresholds["evaluation_threshold"]
    validation_metrics = _metrics(
        validation_df["label"].astype(int).tolist(), validation_scores, evaluation_threshold
    )
    test_scores, test_prediction_seconds = _predict_scores(trainer, test_dataset)
    test_metrics = _metrics(
        test_df["label"].astype(int).tolist(), test_scores, evaluation_threshold
    )
    test_metrics["prediction_time_seconds"] = float(test_prediction_seconds)
    test_metrics["avg_latency_ms"] = float(test_prediction_seconds / max(len(test_df), 1) * 1000)

    backup_path = _backup_existing_target(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    metadata = {
        "fine_tuned": True,
        "model_name": effective_name,
        "model_alias": effective_name,
        "base_model": spec["base_fallback"],
        "parent_checkpoint": source,
        "continued_from_v4": bool(is_local_v4),
        "resume_checkpoint": resume_from,
        "dataset_name": "v4_replay_80pct_plus_vi_20pct",
        "dataset_path": str(REPLAY_OUTPUT),
        "validation_path": str(VI_VALIDATION_OUTPUT),
        "test_path": str(VI_TEST_OUTPUT),
        "trained_at": _utc_now(),
        "epochs": float(epochs),
        "label_mapping": {"0": "SAFE", "1": "INJECTION"},
        "training_mode": "continued_full_finetune",
        "freeze_encoder": False,
        "trainable_config": trainable_config,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "test_rows": int(len(test_df)),
        "thresholds": thresholds,
        "metrics": test_metrics,
        "created_by": "src.train_transformers_continue",
    }
    safe_write_text(
        model_dir / "training_metadata.json",
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    safe_write_text(
        model_dir / "metrics.json",
        json.dumps(
            {"validation": validation_metrics, "test": test_metrics, "thresholds": thresholds},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    result = {
        "model": effective_name,
        "parent_checkpoint": source,
        "continued_from_v4": bool(is_local_v4),
        "output_model_path": str(model_dir.resolve()),
        "backup_of_previous_v5": None if backup_path is None else str(backup_path.resolve()),
        "training_seconds": float(training_seconds),
        "validation_prediction_seconds": float(validation_prediction_seconds),
        "training_config": {
            "epochs": float(epochs),
            "max_length": max_length,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "warmup_ratio": warmup_ratio,
            "weight_decay": weight_decay,
            "fp16": cuda_enabled,
            "gradient_checkpointing": gradient_checkpointing,
            "freeze_encoder": False,
            "optim": selected_optim,
            "resume_from_checkpoint": resume_from,
        },
        "trainable_config": trainable_config,
        "thresholds": thresholds,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "language_metrics": _language_metrics(test_df, test_scores, evaluation_threshold),
        "evaluations": {
            "v5_vi_validation": validation_metrics,
            "v5_vi_test": test_metrics,
        },
        "error_cases": _error_rows(
            effective_name, "v5_vi_test", test_df, test_scores, evaluation_threshold
        ),
        "score_distribution": [
            *_score_rows(effective_name, "v5_vi_validation", validation_df, validation_scores),
            *_score_rows(effective_name, "v5_vi_test", test_df, test_scores),
        ],
    }
    state = _load_state()
    state.setdefault("models", {})[effective_name] = result
    _save_state(state)
    _update_thresholds(effective_name, result)
    _write_reports(state)

    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continue fine-tune Transformer v4 với dữ liệu Việt Nam."
    )
    parser.add_argument("--model", choices=["roberta", "xlm_roberta", "all"], default="roberta")
    parser.add_argument("--vi-train", default=str(DEFAULT_VI_TRAIN))
    parser.add_argument("--vi-test", default=str(DEFAULT_VI_TEST))
    parser.add_argument("--replay-train", default=str(DEFAULT_REPLAY_TRAIN))
    parser.add_argument("--replay-ratio", type=float, default=0.80)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--optim", default="", help="Mặc định RoBERTa=adamw_torch, XLM-R=adafactor.")
    parser.add_argument("--use-cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--resume-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-base-fallback", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--output-name", default="", help="Chỉ dùng cho smoke test.")
    args = parser.parse_args()

    if not args.skip_prepare:
        summary = prepare_datasets(
            vi_train_path=args.vi_train,
            vi_test_path=args.vi_test,
            replay_train_path=args.replay_train,
            replay_ratio=args.replay_ratio,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.prepare_only:
        return

    model_keys = ["roberta", "xlm_roberta"] if args.model == "all" else [args.model]
    for model_key in model_keys:
        output_name = args.output_name or None
        if output_name and len(model_keys) > 1:
            output_name = f"{output_name}_{model_key}"
        print(f"\n=== Continue fine-tune {model_key} từ checkpoint v4 ===")
        result = train_continue(
            model_key=model_key,
            epochs=args.epochs,
            max_length=args.max_length,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            use_cuda=args.use_cuda,
            gradient_checkpointing=args.gradient_checkpointing,
            resume_checkpoint=args.resume_checkpoint,
            allow_base_fallback=args.allow_base_fallback,
            optim=args.optim or None,
            max_train_samples=args.max_train_samples or None,
            output_name=output_name,
        )
        print(json.dumps({
            "model": result["model"],
            "parent_checkpoint": result["parent_checkpoint"],
            "output_model_path": result["output_model_path"],
            "thresholds": result["thresholds"],
            "validation_metrics": result["validation_metrics"],
            "test_metrics": result["test_metrics"],
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

