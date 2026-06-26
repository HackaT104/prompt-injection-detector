"""Utilities for Transformer-based prompt injection classifiers."""

from __future__ import annotations

import gc
import importlib
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    fbeta_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.benign_intent import detect_benign_reference_intent
from src.file_utils import safe_write_text
from src.language_utils import detect_language
from src.preprocessing import clean_text, prepare_text_for_detection
from src.thresholding import choose_threshold, predict_with_threshold, confusion_rates


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRANSFORMER_MODELS_DIR = PROJECT_ROOT / "models" / "transformers"
MODELS_THRESHOLDS_PATH = PROJECT_ROOT / "models" / "thresholds.json"
MODEL_TRANSFORMER_THRESHOLDS_PATH = PROJECT_ROOT / "models" / "transformer_thresholds.json"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TRANSFORMER_THRESHOLDS_PATH = OUTPUTS_DIR / "transformer_thresholds.json"
DEFAULT_DATASET_NAME = "neuralchemy/Prompt-injection-dataset"
ID2LABEL = {0: "SAFE", 1: "INJECTION"}
LABEL2ID = {"SAFE": 0, "INJECTION": 1}
DEFAULT_TRANSFORMER_DATASET_CONFIG = "full"
DEFAULT_TRANSFORMER_WARN_THRESHOLD = 0.50
DEFAULT_TRANSFORMER_BLOCK_THRESHOLD = 0.80
CUDA_DISABLED_MODEL_DIRS: set[str] = set()
TRANSFORMER_MODEL_ALIASES = {
    "distilbert": "distilbert-base-uncased",
    "distilbert-base-uncased": "distilbert-base-uncased",
    "distilbert_v2": "distilbert-base-uncased",
    "distilbert-v2": "distilbert-base-uncased",
    "distilbert_v3": "distilbert-base-uncased",
    "distilbert-v3": "distilbert-base-uncased",
    "distilbert_v4": "distilbert-base-uncased",
    "distilbert-v4": "distilbert-base-uncased",
    "distilbert_legacy": "distilbert-base-uncased",
    "roberta": "roberta-base",
    "roberta-base": "roberta-base",
    "roberta_v2": "roberta-base",
    "roberta-v2": "roberta-base",
    "roberta_v3": "roberta-base",
    "roberta-v3": "roberta-base",
    "roberta_v4": "roberta-base",
    "roberta-v4": "roberta-base",
    "roberta_v5_vi": "roberta-base",
    "roberta-v5-vi": "roberta-base",
    "roberta_legacy": "roberta-base",
    "xlm_roberta": "xlm-roberta-base",
    "xlm-roberta": "xlm-roberta-base",
    "xlm-roberta-base": "xlm-roberta-base",
    "xlm_roberta_v3": "xlm-roberta-base",
    "xlm-roberta-v3": "xlm-roberta-base",
    "xlm_roberta_v4": "xlm-roberta-base",
    "xlm-roberta-v4": "xlm-roberta-base",
    "xlm_roberta_v5_vi": "xlm-roberta-base",
    "xlm-roberta-v5-vi": "xlm-roberta-base",
}
TRANSFORMER_MODEL_DIRS = {
    "distilbert-base-uncased": "distilbert",
    "roberta-base": "roberta",
    "xlm-roberta-base": "xlm_roberta",
}
TRANSFORMER_ALIAS_DIRS = {
    "distilbert_v2": "distilbert_v2",
    "distilbert-v2": "distilbert_v2",
    "distilbert_v3": "distilbert_v3",
    "distilbert-v3": "distilbert_v3",
    "distilbert_v4": "distilbert_v4",
    "distilbert-v4": "distilbert_v4",
    "distilbert_legacy": "distilbert",
    "roberta_v2": "roberta_v2",
    "roberta-v2": "roberta_v2",
    "roberta_v3": "roberta_v3",
    "roberta-v3": "roberta_v3",
    "roberta_v4": "roberta_v4",
    "roberta-v4": "roberta_v4",
    "roberta_v5_vi": "roberta_v5_vi",
    "roberta-v5-vi": "roberta_v5_vi",
    "roberta_legacy": "roberta",
    "xlm_roberta_v3": "xlm_roberta_v3",
    "xlm-roberta-v3": "xlm_roberta_v3",
    "xlm_roberta_v4": "xlm_roberta_v4",
    "xlm-roberta-v4": "xlm_roberta_v4",
    "xlm_roberta_v5_vi": "xlm_roberta_v5_vi",
    "xlm-roberta-v5-vi": "xlm_roberta_v5_vi",
}
SUPPORTED_TRANSFORMER_MODELS = set(TRANSFORMER_MODEL_ALIASES)

TEXT_COLUMN_CANDIDATES = [
    "ml_text",
    "model_text",
    "text",
    "canonical_text",
    "prompt",
    "user_prompt",
    "input",
    "content",
    "instruction",
    "query",
]
LABEL_COLUMN_CANDIDATES = ["label", "labels", "category", "type", "is_malicious", "target"]
LABEL_MAPPING = {
    "0": 0,
    "false": 0,
    "benign": 0,
    "safe": 0,
    "normal": 0,
    "clean": 0,
    "1": 1,
    "true": 1,
    "malicious": 1,
    "injection": 1,
    "prompt_injection": 1,
    "prompt injection": 1,
    "jailbreak": 1,
    "jailbreaking": 1,
    "unsafe": 1,
    "attack": 1,
}


def resolve_transformer_model_name(model_name: str) -> str:
    """Return the Hugging Face base model name for supported aliases."""
    normalized = str(model_name).strip().lower()
    if normalized not in TRANSFORMER_MODEL_ALIASES:
        raise ValueError(f"Transformer model không hợp lệ: {model_name}")
    return TRANSFORMER_MODEL_ALIASES[normalized]


def safe_model_dir_name(model_name: str) -> str:
    normalized = str(model_name).strip().lower()
    if normalized in TRANSFORMER_ALIAS_DIRS:
        return TRANSFORMER_ALIAS_DIRS[normalized]
    resolved_name = TRANSFORMER_MODEL_ALIASES.get(normalized, str(model_name))
    return TRANSFORMER_MODEL_DIRS.get(resolved_name, resolved_name.replace("/", "__"))


def resolve_transformer_model_dir(model_name: str) -> Path:
    normalized = str(model_name).strip().lower()
    explicit_dir = safe_model_dir_name(model_name)
    if normalized in TRANSFORMER_ALIAS_DIRS:
        return TRANSFORMER_MODELS_DIR / explicit_dir

    if normalized in {"distilbert", "distilbert-base-uncased"}:
        v4_dir = TRANSFORMER_MODELS_DIR / "distilbert_v4"
        if is_finetuned_transformer_checkpoint(v4_dir):
            return v4_dir
        v3_dir = TRANSFORMER_MODELS_DIR / "distilbert_v3"
        if is_finetuned_transformer_checkpoint(v3_dir):
            return v3_dir
        v2_dir = TRANSFORMER_MODELS_DIR / "distilbert_v2"
        if is_finetuned_transformer_checkpoint(v2_dir):
            return v2_dir
    if normalized in {"roberta", "roberta-base"}:
        v5_dir = TRANSFORMER_MODELS_DIR / "roberta_v5_vi"
        if is_finetuned_transformer_checkpoint(v5_dir):
            return v5_dir
        v4_dir = TRANSFORMER_MODELS_DIR / "roberta_v4"
        if is_finetuned_transformer_checkpoint(v4_dir):
            return v4_dir
        v3_dir = TRANSFORMER_MODELS_DIR / "roberta_v3"
        if is_finetuned_transformer_checkpoint(v3_dir):
            return v3_dir
        v2_dir = TRANSFORMER_MODELS_DIR / "roberta_v2"
        if is_finetuned_transformer_checkpoint(v2_dir):
            return v2_dir
    if normalized in {"xlm_roberta", "xlm-roberta", "xlm-roberta-base"}:
        v5_dir = TRANSFORMER_MODELS_DIR / "xlm_roberta_v5_vi"
        if is_finetuned_transformer_checkpoint(v5_dir):
            return v5_dir
        v4_dir = TRANSFORMER_MODELS_DIR / "xlm_roberta_v4"
        if is_finetuned_transformer_checkpoint(v4_dir):
            return v4_dir
        base_dir = TRANSFORMER_MODELS_DIR / "xlm_roberta"
        if is_finetuned_transformer_checkpoint(base_dir):
            return base_dir
        v3_dir = TRANSFORMER_MODELS_DIR / "xlm_roberta_v3"
        if is_finetuned_transformer_checkpoint(v3_dir):
            return v3_dir
    return TRANSFORMER_MODELS_DIR / explicit_dir


def is_finetuned_transformer_checkpoint(model_dir: str | Path) -> bool:
    """Accept only project fine-tuned checkpoints, not raw base model directories."""
    path = Path(model_dir)
    config_path = path / "config.json"
    metadata_path = path / "training_metadata.json"
    has_weights = any((path / name).exists() for name in ["model.safetensors", "pytorch_model.bin"])
    if not path.exists() or not config_path.exists() or not has_weights:
        return False

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    id2label = {int(key): value for key, value in config.get("id2label", {}).items()}
    label2id = config.get("label2id", {})
    labels_ok = id2label == ID2LABEL and label2id == LABEL2ID
    metadata_ok = metadata_path.exists()
    if metadata_ok:
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata_ok = metadata.get("fine_tuned") is True
        except (OSError, json.JSONDecodeError):
            metadata_ok = False
    return bool(labels_ok and metadata_ok)


def import_optional(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Thiếu dependency '{module_name}'. Hãy cài: "
            "pip install transformers datasets accelerate evaluate scikit-learn torch"
        ) from exc


def import_huggingface_load_dataset() -> Any:
    """Import HF datasets.load_dataset even though this repo has a local datasets/ folder."""
    original_sys_path = list(sys.path)
    local_datasets_module = sys.modules.get("datasets")
    project_root = PROJECT_ROOT.resolve()
    if local_datasets_module is not None and getattr(local_datasets_module, "__file__", None) is None:
        sys.modules.pop("datasets", None)

    try:
        sys.path = [path for path in sys.path if Path(path or ".").resolve() != project_root]
        datasets_module = importlib.import_module("datasets")
    finally:
        sys.path = original_sys_path

    load_dataset = getattr(datasets_module, "load_dataset", None)
    if load_dataset is None:
        raise ImportError("Không import được datasets.load_dataset từ Hugging Face.")
    return load_dataset


def normalize_transformer_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
        return int(value)
    if isinstance(value, float) and value in (0.0, 1.0):
        return int(value)

    normalized = str(value).strip().lower()
    if normalized in LABEL_MAPPING:
        return LABEL_MAPPING[normalized]
    raise ValueError(f"Nhãn không hỗ trợ cho Transformer dataset: {value!r}")


def _detect_column(columns: list[str], candidates: list[str], required_name: str) -> str:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    for column in columns:
        normalized = column.lower()
        if any(candidate in normalized for candidate in candidates):
            return column
    raise ValueError(f"Không tự nhận diện được cột {required_name}. Columns: {columns}")


def load_neuralchemy_dataframe(
    dataset_config: str = "core",
    dataset_name: str = DEFAULT_DATASET_NAME,
) -> pd.DataFrame:
    load_dataset = import_huggingface_load_dataset()
    dataset = load_dataset(dataset_name, dataset_config)
    frames: list[pd.DataFrame] = []
    for split_name, split_dataset in dataset.items():
        frame = split_dataset.to_pandas()
        frame["hf_split"] = split_name
        frames.append(frame)

    if not frames:
        raise ValueError(f"Dataset {dataset_name}/{dataset_config} không có split nào.")
    return pd.concat(frames, ignore_index=True)


def load_cached_neuralchemy_arrow_dataframe(dataset_config: str = "core") -> pd.DataFrame:
    """Load cached Hugging Face Arrow files when the Hub is unavailable."""
    datasets_module = import_optional("datasets")
    cache_root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "datasets"
        / "neuralchemy___prompt-injection-dataset"
        / dataset_config
        / "0.0.0"
    )
    arrow_files = sorted(cache_root.glob("*/prompt-injection-dataset-*.arrow"))
    if not arrow_files:
        raise FileNotFoundError(
            "Không tìm thấy Arrow cache cho neuralchemy/Prompt-injection-dataset. "
            "Hãy chạy khi có mạng bằng load_dataset trước, hoặc bỏ --prefer-cached-arrow."
        )

    frames: list[pd.DataFrame] = []
    for arrow_path in arrow_files:
        split_name = arrow_path.stem.replace("prompt-injection-dataset-", "")
        split_dataset = datasets_module.Dataset.from_file(str(arrow_path))
        frame = split_dataset.to_pandas()
        frame["hf_split"] = split_name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_neuralchemy_split_dataframes(
    dataset_config: str = DEFAULT_TRANSFORMER_DATASET_CONFIG,
    dataset_name: str = DEFAULT_DATASET_NAME,
    prefer_cached_arrow: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load neuralchemy splits as separate DataFrames to preserve official train/validation/test."""
    if prefer_cached_arrow:
        combined = load_cached_neuralchemy_arrow_dataframe(dataset_config)
    else:
        combined = load_neuralchemy_dataframe(dataset_config, dataset_name)
    split_column = "hf_split" if "hf_split" in combined.columns else "split"
    if split_column not in combined.columns:
        return {"all": combined}
    return {
        str(split_name): split_df.reset_index(drop=True)
        for split_name, split_df in combined.groupby(split_column)
    }


def prepare_transformer_dataframe(
    raw_df: pd.DataFrame,
    max_samples: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    text_column = _detect_column(list(raw_df.columns), TEXT_COLUMN_CANDIDATES, "text")
    label_column = _detect_column(list(raw_df.columns), LABEL_COLUMN_CANDIDATES, "label")
    working_df = raw_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    if max_samples is not None and max_samples > 0:
        working_df = working_df.head(max_samples).copy()

    rows: list[dict[str, Any]] = []
    for index, row in working_df.iterrows():
        raw_text = "" if pd.isna(row[text_column]) else str(row[text_column])
        if not raw_text.strip() or pd.isna(row[label_column]):
            continue
        label = normalize_transformer_label(row[label_column])
        cleaned_text = clean_text(raw_text)
        language_value = row.get("language", None)
        detected_language = (
            str(language_value).strip().lower()
            if language_value is not None and not pd.isna(language_value) and str(language_value).strip()
            else detect_language(raw_text)
        )
        rows.append(
            {
                "sample_id": f"transformer_{index}",
                "text": raw_text,
                "model_text": cleaned_text,
                "label": label,
                "detected_language": detected_language,
                "canonical_text": cleaned_text,
                "source_split": row.get("hf_split", row.get("split", "unknown")),
            }
        )

    df = pd.DataFrame(rows).drop_duplicates(subset=["model_text", "label"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("Dataset Transformer rỗng sau khi chuẩn hóa.")
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df


def _can_stratify(labels: pd.Series) -> bool:
    counts = labels.value_counts()
    return len(counts) == 2 and counts.min() >= 2


def split_transformer_dataframe(
    df: pd.DataFrame,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if set(df["label"].astype(int).unique()) != {0, 1}:
        raise ValueError("Dataset Transformer cần đủ 2 lớp: 0 benign và 1 malicious.")

    stratify = df["label"] if _can_stratify(df["label"]) else None
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=random_state,
        stratify=stratify,
    )
    temp_stratify = temp_df["label"] if _can_stratify(temp_df["label"]) else None
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_stratify,
    )
    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def split_transformer_dataframe_by_source(
    df: pd.DataFrame,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use official dataset splits when present; otherwise fall back to stratified split."""
    if "source_split" not in df.columns:
        return split_transformer_dataframe(df, random_state=random_state)

    split_values = set(df["source_split"].astype(str).str.lower().unique())
    required = {"train", "validation", "test"}
    if not required.issubset(split_values):
        return split_transformer_dataframe(df, random_state=random_state)

    train_df = df[df["source_split"].astype(str).str.lower() == "train"].copy()
    validation_df = df[df["source_split"].astype(str).str.lower() == "validation"].copy()
    test_df = df[df["source_split"].astype(str).str.lower() == "test"].copy()
    return (
        train_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def softmax_positive_scores(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probabilities[:, 1].astype(float)


def action_from_score(score: float, warn_threshold: float, block_threshold: float) -> str:
    if score >= block_threshold:
        return "block"
    if score >= warn_threshold:
        return "warn"
    return "allow"


@lru_cache(maxsize=4)
def _load_transformer_artifacts_cached(model_dir_text: str, use_cuda: bool) -> tuple[Any, Any, Any, Any]:
    torch = import_optional("torch")
    transformers = import_optional("transformers")
    model_dir = Path(model_dir_text)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return torch, tokenizer, model, device


def evaluate_scores(
    y_true: list[int],
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    y_pred = predict_with_threshold(scores, threshold)
    rates = confusion_rates(y_true, y_pred)
    result = {
        "evaluation_threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, pos_label=1, zero_division=0)),
        "false_positive_rate": float(rates["false_positive_rate"]),
        "false_negative_rate": float(rates["false_negative_rate"]),
        "confusion_matrix": rates["confusion_matrix"],
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["benign", "malicious"],
            output_dict=True,
            zero_division=0,
        ),
        "classification_report_text": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["benign", "malicious"],
            zero_division=0,
        ),
        "roc_auc": None,
        "average_precision": None,
    }
    if len(set(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, scores))
        result["average_precision"] = float(average_precision_score(y_true, scores))
    return result


def build_threshold_payload(threshold_analysis: dict[str, Any]) -> dict[str, float]:
    return {
        "evaluation_threshold": float(threshold_analysis["selected_threshold"]),
        "runtime_warn_threshold": float(
            threshold_analysis.get("runtime_warn_threshold", DEFAULT_TRANSFORMER_WARN_THRESHOLD)
        ),
        "runtime_block_threshold": float(
            threshold_analysis.get("runtime_block_threshold", DEFAULT_TRANSFORMER_BLOCK_THRESHOLD)
        ),
    }


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    safe_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def update_json_object(path: str | Path, key: str, value: dict[str, Any]) -> dict[str, Any]:
    target = Path(path)
    if target.exists():
        payload = json.loads(target.read_text(encoding="utf-8-sig"))
    else:
        payload = {"models": {}}
    payload.setdefault("models", {})[key] = value
    save_json(target, payload)
    return payload


@lru_cache(maxsize=8)
def _load_probability_calibrator(model_dir_text: str) -> Any | None:
    calibrator_path = Path(model_dir_text) / "probability_calibrator.joblib"
    if not calibrator_path.exists():
        return None
    return joblib.load(calibrator_path)


def predict_transformer(
    text: str,
    model_path: str | Path,
    model_name: str | None = None,
    max_length: int = 128,
    thresholds: dict[str, float] | None = None,
    use_cuda: bool = True,
) -> dict[str, Any]:
    """Run inference with a fine-tuned Transformer model directory."""
    model_dir = Path(model_path)
    if not is_finetuned_transformer_checkpoint(model_dir):
        raise FileNotFoundError(
            f"Transformer checkpoint not found or not fine-tuned: {model_dir}. "
            "Please fine-tune the model first."
        )

    runtime_warnings: list[str] = []
    model_dir_key = str(model_dir.resolve())
    effective_use_cuda = use_cuda and model_dir_key not in CUDA_DISABLED_MODEL_DIRS
    try:
        torch, tokenizer, model, device = _load_transformer_artifacts_cached(model_dir_key, effective_use_cuda)
    except RuntimeError as exc:
        if not effective_use_cuda or "out of memory" not in str(exc).lower():
            raise
        CUDA_DISABLED_MODEL_DIRS.add(model_dir_key)
        _load_transformer_artifacts_cached.cache_clear()
        try:
            torch = import_optional("torch")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        runtime_warnings.append("CUDA out of memory; Transformer inference fell back to CPU.")
        torch, tokenizer, model, device = _load_transformer_artifacts_cached(model_dir_key, False)

    prepared = prepare_text_for_detection(text)
    encoded = tokenizer(
        prepared["cleaned_text"],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded)
        probabilities_tensor = torch.softmax(outputs.logits, dim=-1)[0]
    logits = [float(value) for value in outputs.logits[0].detach().cpu().tolist()]
    probabilities_list = [float(value) for value in probabilities_tensor.detach().cpu().tolist()]
    raw_probabilities_list = list(probabilities_list)
    risk_score = float(probabilities_list[LABEL2ID["INJECTION"]])
    raw_risk_score = risk_score
    calibration_method = None
    calibrator = _load_probability_calibrator(str(model_dir.resolve()))
    if calibrator is not None:
        calibrated_score = calibrator.predict([raw_risk_score])[0]
        risk_score = float(min(1.0, max(0.0, calibrated_score)))
        probabilities_list = [1.0 - risk_score, risk_score]
        calibration_method = calibrator.__class__.__name__
    benign_guard = detect_benign_reference_intent(text)
    # Benign guard is kept as metadata only; the returned risk_score is the true model probability.
    confidence = float(max(probabilities_list))
    raw_predicted_label = int(np.argmax(raw_probabilities_list))
    predicted_label = int(np.argmax(probabilities_list))

    model_key = model_dir.name
    alias_model_key = safe_model_dir_name(model_name or model_dir.name)
    saved_thresholds: dict[str, float] | None = None
    if thresholds is None:
        for threshold_path in [MODEL_TRANSFORMER_THRESHOLDS_PATH, MODELS_THRESHOLDS_PATH, TRANSFORMER_THRESHOLDS_PATH]:
            if not threshold_path.exists():
                continue
            try:
                thresholds_payload = json.loads(threshold_path.read_text(encoding="utf-8-sig"))
                models_payload = thresholds_payload.get("models", thresholds_payload)
                saved_thresholds = (
                    models_payload.get(model_key)
                    or models_payload.get(alias_model_key)
                )
                if saved_thresholds:
                    break
            except (json.JSONDecodeError, OSError):
                saved_thresholds = None

    resolved_thresholds = thresholds or saved_thresholds or {
        "evaluation_threshold": DEFAULT_TRANSFORMER_WARN_THRESHOLD,
        "runtime_warn_threshold": DEFAULT_TRANSFORMER_WARN_THRESHOLD,
        "runtime_block_threshold": DEFAULT_TRANSFORMER_BLOCK_THRESHOLD,
    }
    resolved_thresholds = {
        "evaluation_threshold": float(
            resolved_thresholds.get(
                "evaluation_threshold",
                resolved_thresholds.get("warn_threshold", DEFAULT_TRANSFORMER_WARN_THRESHOLD),
            )
        ),
        "runtime_warn_threshold": float(
            resolved_thresholds.get(
                "runtime_warn_threshold",
                resolved_thresholds.get("warn_threshold", DEFAULT_TRANSFORMER_WARN_THRESHOLD),
            )
        ),
        "runtime_block_threshold": float(
            resolved_thresholds.get(
                "runtime_block_threshold",
                resolved_thresholds.get("block_threshold", DEFAULT_TRANSFORMER_BLOCK_THRESHOLD),
            )
        ),
    }
    warn_threshold = resolved_thresholds["runtime_warn_threshold"]
    block_threshold = max(resolved_thresholds["runtime_block_threshold"], warn_threshold)
    evaluation_label = 1 if risk_score >= resolved_thresholds["evaluation_threshold"] else 0
    runtime_action = action_from_score(
        risk_score,
        warn_threshold=warn_threshold,
        block_threshold=block_threshold,
    )
    return {
        "text": text,
        "model": model_name or model_dir.name,
        "label": evaluation_label,
        "risk_score": round(risk_score, 8),
        "raw_risk_score": round(raw_risk_score, 8),
        "confidence": round(confidence, 8),
        "probabilities": {
            "safe": round(float(probabilities_list[LABEL2ID["SAFE"]]), 8),
            "injection": round(float(probabilities_list[LABEL2ID["INJECTION"]]), 8),
        },
        "raw_probabilities": {
            "safe": round(float(raw_probabilities_list[LABEL2ID["SAFE"]]), 8),
            "injection": round(float(raw_probabilities_list[LABEL2ID["INJECTION"]]), 8),
        },
        "calibration_method": calibration_method,
        "benign_guard": benign_guard,
        "logits": [round(value, 6) for value in logits],
        "predicted_label": predicted_label,
        "raw_predicted_label": raw_predicted_label,
        "evaluation_label": evaluation_label,
        "runtime_action": runtime_action,
        "thresholds": resolved_thresholds,
        "runtime_device": str(device),
        "warnings": runtime_warnings,
        "detected_language": prepared["detected_language"],
        "canonical_text": prepared["cleaned_text"],
    }


def diagnose_transformer(
    text: str,
    model_name: str,
    dataset_config: str = DEFAULT_TRANSFORMER_DATASET_CONFIG,
    use_cuda: bool = True,
) -> dict[str, Any]:
    resolved_name = resolve_transformer_model_name(model_name)
    model_dir = resolve_transformer_model_dir(resolved_name)
    checkpoint_exists = is_finetuned_transformer_checkpoint(model_dir)
    base = {
        "dataset_name": DEFAULT_DATASET_NAME,
        "dataset_config": dataset_config,
        "model": safe_model_dir_name(resolved_name),
        "model_path": str(model_dir),
        "checkpoint_exists": checkpoint_exists,
        "id2label": {str(key): value for key, value in ID2LABEL.items()},
        "label2id": LABEL2ID,
        "warn_threshold": DEFAULT_TRANSFORMER_WARN_THRESHOLD,
        "block_threshold": DEFAULT_TRANSFORMER_BLOCK_THRESHOLD,
        "thresholds": {
            "evaluation_threshold": DEFAULT_TRANSFORMER_WARN_THRESHOLD,
            "runtime_warn_threshold": DEFAULT_TRANSFORMER_WARN_THRESHOLD,
            "runtime_block_threshold": DEFAULT_TRANSFORMER_BLOCK_THRESHOLD,
        },
    }
    if not checkpoint_exists:
        return {
            **base,
            "available": False,
            "action": "model_not_ready",
            "risk_score": None,
            "confidence": None,
            "probabilities": None,
            "logits": None,
            "predicted_label": None,
            "error": "Transformer checkpoint not found. Please fine-tune the model first.",
        }

    prediction = predict_transformer(
        text=text,
        model_path=model_dir,
        model_name=resolved_name,
        thresholds={
            "evaluation_threshold": DEFAULT_TRANSFORMER_WARN_THRESHOLD,
            "runtime_warn_threshold": DEFAULT_TRANSFORMER_WARN_THRESHOLD,
            "runtime_block_threshold": DEFAULT_TRANSFORMER_BLOCK_THRESHOLD,
        },
        use_cuda=use_cuda,
    )
    return {
        **base,
        "available": True,
        "logits": prediction["logits"],
        "probabilities": prediction["probabilities"],
        "predicted_label": prediction["predicted_label"],
        "predicted_class": prediction["predicted_label"],
        "risk_score": prediction["risk_score"],
        "confidence": prediction["confidence"],
        "action": prediction["runtime_action"],
        "thresholds": prediction["thresholds"],
        "input": {
            "original_text": prediction["text"],
            "detected_language": prediction["detected_language"],
            "canonical_text": prediction["canonical_text"],
        },
    }


def clear_transformer_runtime_cache() -> None:
    """Release cached Transformer objects between multi-model comparisons."""
    _load_transformer_artifacts_cached.cache_clear()
    _load_probability_calibrator.cache_clear()
    gc.collect()
    try:
        torch = import_optional("torch")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
