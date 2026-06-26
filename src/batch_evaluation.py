"""Batch dataset evaluation for prompt injection detector models."""

from __future__ import annotations

import csv
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.advanced_detection import detect_prompt_advanced
from src.transformer_utils import is_finetuned_transformer_checkpoint, resolve_transformer_model_dir


SUPPORTED_BATCH_MODELS = [
    "logistic_regression",
    "linear_svm",
    "random_forest",
    "distilbert",
    "roberta",
    "xlm_roberta",
    "hybrid",
]
SUPPORTED_BATCH_MODEL_SET = set(SUPPORTED_BATCH_MODELS) | {
    "distilbert_v2",
    "roberta_v2",
    "distilbert_v3",
    "roberta_v3",
    "xlm_roberta_v3",
    "distilbert_v4",
    "roberta_v4",
    "xlm_roberta_v4",
}
MODEL_ALIASES = {
    "logistic": "logistic_regression",
    "logistic_regression": "logistic_regression",
    "linear_svm": "linear_svm",
    "svm": "linear_svm",
    "rf": "random_forest",
    "random_forest": "random_forest",
    "distilbert": "distilbert",
    "distilbert-base-uncased": "distilbert",
    "distilbert_v2": "distilbert_v2",
    "distilbert-v2": "distilbert_v2",
    "distilbert_v3": "distilbert_v3",
    "distilbert-v3": "distilbert_v3",
    "distilbert_v4": "distilbert_v4",
    "distilbert-v4": "distilbert_v4",
    "roberta": "roberta",
    "roberta-base": "roberta",
    "roberta_v2": "roberta_v2",
    "roberta-v2": "roberta_v2",
    "roberta_v3": "roberta_v3",
    "roberta-v3": "roberta_v3",
    "roberta_v4": "roberta_v4",
    "roberta-v4": "roberta_v4",
    "xlm_roberta": "xlm_roberta",
    "xlm-roberta": "xlm_roberta",
    "xlm-roberta-base": "xlm_roberta",
    "xlm_roberta_v3": "xlm_roberta_v3",
    "xlm-roberta-v3": "xlm_roberta_v3",
    "xlm_roberta_v4": "xlm_roberta_v4",
    "xlm-roberta-v4": "xlm_roberta_v4",
    "hybrid": "hybrid",
}
CSV_PREFIXES = {
    "logistic_regression": "logistic",
    "linear_svm": "svm",
    "random_forest": "rf",
    "distilbert": "distilbert",
    "roberta": "roberta",
    "distilbert_v2": "distilbert_v2",
    "roberta_v2": "roberta_v2",
    "distilbert_v3": "distilbert_v3",
    "roberta_v3": "roberta_v3",
    "distilbert_v4": "distilbert_v4",
    "roberta_v4": "roberta_v4",
    "xlm_roberta": "xlm_roberta",
    "xlm_roberta_v3": "xlm_roberta_v3",
    "xlm_roberta_v4": "xlm_roberta_v4",
    "hybrid": "hybrid",
}
TEXT_COLUMN_CANDIDATES = ["text", "prompt", "input", "instruction", "user_prompt", "content"]
LABEL_COLUMN_CANDIDATES = ["label", "ground_truth", "ground_truth_label", "target", "class", "is_malicious"]
CATEGORY_COLUMN_CANDIDATES = ["category", "attack_type", "type"]
METADATA_COLUMNS = ["category", "source", "language", "attack_type", "context", "response"]
LABEL_TEXT_MAPPING = {
    "benign": 0,
    "safe": 0,
    "normal": 0,
    "clean": 0,
    "legitimate": 0,
    "malicious": 1,
    "injection": 1,
    "prompt_injection": 1,
    "prompt injection": 1,
    "attack": 1,
    "unsafe": 1,
    "jailbreak": 1,
    "jailbreaking": 1,
}
DEFAULT_MAX_BATCH_ITEMS = 1000


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return not str(value).strip()


def _python_scalar(value: Any) -> Any:
    if _is_blank(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    return value


def _lower_key_map(item: dict[str, Any]) -> dict[str, str]:
    return {str(key).strip().lower(): str(key) for key in item.keys()}


def _get_value_by_column(item: dict[str, Any], column: str | None, default: Any = None) -> Any:
    if column is None:
        return default
    if column in item:
        return item.get(column, default)
    key_map = _lower_key_map(item)
    original = key_map.get(str(column).strip().lower())
    return item.get(original, default) if original is not None else default


def _detect_column_from_items(items: list[dict[str, Any]], candidates: list[str]) -> str | None:
    observed: dict[str, str] = {}
    for item in items:
        for lower_key, original_key in _lower_key_map(item).items():
            observed.setdefault(lower_key, original_key)
    for candidate in candidates:
        if candidate in observed:
            return observed[candidate]
    return None


def _dataset_format_from_suffix(filename: str) -> str:
    suffix = Path(filename or "").suffix.lower().lstrip(".")
    return f"uploaded_{suffix}" if suffix else "uploaded_dataset"


def _source_from_dataset_name(dataset_name: str | None) -> str:
    return _dataset_format_from_suffix(dataset_name or "")


def _attach_parse_metadata(rows: list[dict[str, Any]], filename: str) -> list[dict[str, Any]]:
    dataset_format = _dataset_format_from_suffix(filename)
    return [
        {
            **{str(key): _python_scalar(value) for key, value in row.items()},
            "_dataset_format": dataset_format,
            "_source_filename": filename,
        }
        for row in rows
    ]


def normalize_model_name(model_name: str) -> str:
    normalized = str(model_name or "").strip().lower().replace("-", "_")
    normalized = normalized.replace("distilbert_base_uncased", "distilbert-base-uncased")
    normalized = normalized.replace("roberta_base", "roberta-base")
    normalized = MODEL_ALIASES.get(normalized, normalized)
    if normalized not in SUPPORTED_BATCH_MODEL_SET:
        try:
            if is_finetuned_transformer_checkpoint(resolve_transformer_model_dir(normalized)):
                return normalized
        except Exception:
            pass
        raise ValueError(f"Unsupported model for batch evaluation: {model_name}")
    return normalized


def normalize_selected_models(models: list[str] | None) -> list[str]:
    if not models:
        return list(SUPPORTED_BATCH_MODELS)
    selected: list[str] = []
    for model_name in models:
        normalized = normalize_model_name(model_name)
        if normalized not in selected:
            selected.append(normalized)
    return selected


def normalize_hybrid_config(config: dict[str, Any] | None) -> dict[str, Any]:
    config = dict(config or {})
    strategy = str(
        config.get("decision_strategy")
        or config.get("strategy")
        or "weighted_voting"
    ).strip().lower()
    strategy_map = {
        "max_risk": "maximum_risk",
        "max": "maximum_risk",
        "maximum_risk": "maximum_risk",
        "majority": "majority_vote",
        "majority_vote": "majority_vote",
        "weighted": "weighted_voting",
        "weighted_vote": "weighted_voting",
        "weighted_voting": "weighted_voting",
    }
    traditional = str(config.get("traditional_model") or "random_forest").strip().lower()
    if traditional in {"svm", "linear-svm"}:
        traditional = "linear_svm"
    if traditional in {"logistic", "lr"}:
        traditional = "logistic_regression"
    transformer = str(config.get("transformer_model") or "xlm_roberta_v3").strip().lower()
    if transformer == "distilbert-base-uncased":
        transformer = "distilbert"
    if transformer == "roberta-base":
        transformer = "roberta"
    if transformer in {"distilbert-v2", "distilbert_v2"}:
        transformer = "distilbert_v2"
    if transformer in {"roberta-v2", "roberta_v2"}:
        transformer = "roberta_v2"
    if transformer in {"distilbert-v3", "distilbert_v3"}:
        transformer = "distilbert_v3"
    if transformer in {"roberta-v3", "roberta_v3"}:
        transformer = "roberta_v3"
    if transformer in {"distilbert-v4", "distilbert_v4"}:
        transformer = "distilbert_v4"
    if transformer in {"roberta-v4", "roberta_v4"}:
        transformer = "roberta_v4"
    if transformer in {"xlm-roberta", "xlm-roberta-base"}:
        transformer = "xlm_roberta"
    if transformer in {"xlm-roberta-v3", "xlm_roberta_v3"}:
        transformer = "xlm_roberta_v3"
    if transformer in {"xlm-roberta-v4", "xlm_roberta_v4"}:
        transformer = "xlm_roberta_v4"
    return {
        "traditional_model": traditional,
        "transformer_model": transformer,
        "use_rule_based": bool(config.get("use_rule_based", True)),
        "decision_strategy": strategy_map.get(strategy, "weighted_voting"),
    }


def _find_text_column(columns: list[str]) -> str | None:
    lower_to_original = {column.strip().lower(): column for column in columns}
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in lower_to_original:
            return lower_to_original[candidate]
    return None


def parse_dataset_content(filename: str, content: str) -> list[dict[str, Any]]:
    """Parse CSV, JSON, JSONL or TXT content into batch item dictionaries."""
    suffix = Path(filename or "dataset.csv").suffix.lower()
    if suffix == ".csv" or not suffix:
        frame = pd.read_csv(io.StringIO(content), keep_default_na=False)
        return records_from_dataframe(frame, filename=filename)
    if suffix == ".json":
        payload = json.loads(content)
        if isinstance(payload, dict):
            if isinstance(payload.get("items"), list):
                return _attach_parse_metadata([dict(item) for item in payload["items"]], filename)
            if isinstance(payload.get("data"), list):
                return _attach_parse_metadata([dict(item) for item in payload["data"]], filename)
            return _attach_parse_metadata([dict(payload)], filename)
        if isinstance(payload, list):
            return _attach_parse_metadata([dict(item) for item in payload], filename)
        raise ValueError("JSON dataset must be an object, an item list or contain an items/data list.")
    if suffix == ".jsonl":
        rows = []
        for line_no, line in enumerate(content.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
                if not isinstance(parsed, dict):
                    raise ValueError("each JSONL line must be a JSON object")
                rows.append(dict(parsed))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
        return _attach_parse_metadata(rows, filename)
    if suffix == ".txt":
        return _attach_parse_metadata(
            [
                {"id": str(index), "text": line.strip()}
                for index, line in enumerate(content.splitlines(), start=1)
                if line.strip()
            ],
            filename,
        )
    raise ValueError("Unsupported file type. Supported formats: .csv, .json, .jsonl, .txt")


def records_from_dataframe(frame: pd.DataFrame, filename: str = "uploaded.csv") -> list[dict[str, Any]]:
    # Preserve every original column. Validation performs schema detection and normalization later.
    rows = [
        {str(key): _python_scalar(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]
    return _attach_parse_metadata(rows, filename)


def _normalize_label(value: Any) -> int | None:
    if _is_blank(value):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and float(value).is_integer():
        int_value = int(value)
        return int_value if int_value in {0, 1} else None
    text = str(value).strip().lower()
    if text in {"0", "1"}:
        return int(text)
    return LABEL_TEXT_MAPPING.get(text)


def _public_columns(items: list[dict[str, Any]]) -> list[str]:
    columns = sorted({str(key) for item in items for key in item.keys() if not str(key).startswith("_")})
    return columns


def _label_mapping_for_values(values: list[Any]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for value in values:
        if _is_blank(value):
            continue
        normalized = _normalize_label(value)
        if normalized is not None:
            mapping[str(value).strip()] = normalized
    return mapping


def validate_batch_items(
    items: list[dict[str, Any]],
    max_items: int = DEFAULT_MAX_BATCH_ITEMS,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    normalized_items: list[dict[str, Any]] = []

    if not items:
        errors.append("Dataset is empty.")
        return {
            "valid": False,
            "errors": errors,
            "warnings": warnings,
            "rows": 0,
            "total_rows": 0,
            "has_ground_truth": False,
            "text_column_detected": None,
            "label_column_detected": None,
            "category_column_detected": None,
            "detected_columns": [],
            "label_mapping": {},
            "items": [],
        }

    detected_columns = _public_columns(items)
    text_column = _detect_column_from_items(items, TEXT_COLUMN_CANDIDATES)
    label_column = _detect_column_from_items(items, LABEL_COLUMN_CANDIDATES)
    category_column = _detect_column_from_items(items, CATEGORY_COLUMN_CANDIDATES)
    if text_column is None:
        errors.append(
            "Missing required column: text. Supported text columns: "
            + ", ".join(TEXT_COLUMN_CANDIDATES)
            + f". Detected columns: {', '.join(detected_columns) or 'none'}"
        )

    total_rows = len(items)
    if len(items) > max_items:
        warnings.append(f"Dataset has {len(items)} rows; only the first max_items={max_items} rows will be evaluated.")
        items = items[:max_items]

    has_any_label = bool(
        label_column
        and any(not _is_blank(_get_value_by_column(item, label_column)) for item in items)
    )
    raw_label_values = [
        _get_value_by_column(item, label_column)
        for item in items
        if label_column is not None
    ]
    label_mapping = _label_mapping_for_values(raw_label_values)
    seen_ids: set[str] = set()
    default_source = _source_from_dataset_name(dataset_name) if dataset_name else None

    for row_number, raw_item in enumerate(items, start=1):
        text_value = _get_value_by_column(raw_item, text_column) if text_column else None
        if _is_blank(text_value):
            errors.append(f"Empty text at row {row_number}; text_column_detected={text_column!r}")
            continue

        raw_id = _get_value_by_column(raw_item, "id")
        item_id = str(raw_id).strip() if not _is_blank(raw_id) else str(row_number)
        if item_id in seen_ids:
            errors.append(f"Duplicate id at row {row_number}: {item_id}")
        seen_ids.add(item_id)

        raw_label = _get_value_by_column(raw_item, label_column) if label_column else None
        label: int | None = None
        if has_any_label:
            label = _normalize_label(raw_label)
            if label is None:
                errors.append(
                    f"Invalid label at row {row_number}: {raw_label!r}. "
                    "Supported numeric labels are 0/1; supported text labels are: "
                    + ", ".join(sorted(LABEL_TEXT_MAPPING))
                )

        attack_type = _get_value_by_column(raw_item, "attack_type")
        category = _get_value_by_column(raw_item, category_column) if category_column else None
        if _is_blank(category) and not _is_blank(attack_type):
            category = attack_type

        source = _get_value_by_column(raw_item, "source")
        if _is_blank(source):
            source = _get_value_by_column(raw_item, "_dataset_format") or default_source or "uploaded_dataset"

        original_prompt = _get_value_by_column(raw_item, "prompt")
        if _is_blank(original_prompt):
            original_prompt = text_value

        original_record = {
            str(key): _python_scalar(value)
            for key, value in raw_item.items()
            if not str(key).startswith("_")
        }
        normalized = {
            "id": item_id,
            "text": str(text_value),
            "normalized_text": str(text_value),
            "label": label,
            "ground_truth_label": label,
            "original_prompt": str(original_prompt),
            "original_label": None if _is_blank(raw_label) else str(raw_label),
            "category": None if _is_blank(category) else str(category),
            "attack_type": None if _is_blank(attack_type) else str(attack_type),
            "context": None if _is_blank(_get_value_by_column(raw_item, "context")) else str(_get_value_by_column(raw_item, "context")),
            "response": None if _is_blank(_get_value_by_column(raw_item, "response")) else str(_get_value_by_column(raw_item, "response")),
            "source": None if _is_blank(source) else str(source),
            "language": None if _is_blank(_get_value_by_column(raw_item, "language")) else str(_get_value_by_column(raw_item, "language")),
            "text_column_detected": text_column,
            "label_column_detected": label_column,
            "category_column_detected": category_column,
            "original_record": original_record,
        }
        normalized_items.append(normalized)

    if not has_any_label:
        warnings.append("Ground-truth labels not provided. Metrics are not available.")

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "rows": len(items),
        "total_rows": total_rows,
        "max_items": max_items,
        "has_ground_truth": has_any_label and not errors,
        "text_column_detected": text_column,
        "label_column_detected": label_column,
        "category_column_detected": category_column,
        "detected_columns": detected_columns,
        "label_mapping": label_mapping,
        "items": [] if errors else normalized_items,
    }

def _prediction_not_ready(model_name: str, message: str, latency_ms: float = 0.0) -> dict[str, Any]:
    return {
        "requested_model": model_name,
        "loaded_model": None,
        "predicted_label": None,
        "risk_score": None,
        "confidence": None,
        "action": "model_not_ready",
        "latency_ms": round(float(latency_ms), 2),
        "available": False,
        "message": message or "Checkpoint not found or inference failed.",
        "error": message or "Checkpoint not found or inference failed.",
    }


def run_single_prediction(
    text: str,
    model_name: str,
    hybrid_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    try:
        result = detect_prompt_advanced(
            text=text,
            input_type="text",
            model=model_name,
            hybrid_config=hybrid_config if model_name == "hybrid" else None,
        )
    except Exception as exc:  # one model must not fail the whole batch
        return _prediction_not_ready(model_name, str(exc), (time.perf_counter() - started_at) * 1000)

    decision = result.get("decision", {}) or {}
    action = decision.get("action") or "model_not_ready"
    risk_score = decision.get("risk_score")
    available = action not in {"model_not_ready", "unavailable"} and risk_score is not None
    if not available:
        message = "; ".join(str(item) for item in result.get("warnings", []) or [])
        if not message:
            message = result.get("explanation") or "Checkpoint not found or inference failed."
        prediction = _prediction_not_ready(model_name, message, decision.get("processing_time_ms", 0.0))
        prediction["loaded_model"] = decision.get("model")
        return prediction

    return {
        "requested_model": model_name,
        "loaded_model": decision.get("model"),
        "predicted_label": decision.get("label"),
        "risk_score": risk_score,
        "confidence": decision.get("confidence"),
        "action": action,
        "latency_ms": decision.get("processing_time_ms", round((time.perf_counter() - started_at) * 1000, 2)),
        "available": True,
        "message": None,
        "error": None,
    }


def clear_model_cache(model_name: str) -> None:
    """Release cached Transformer artifacts between batch model passes."""
    if model_name not in {
        "distilbert",
        "roberta",
        "xlm_roberta",
        "distilbert_v2",
        "roberta_v2",
        "distilbert_v3",
        "roberta_v3",
        "xlm_roberta_v3",
        "distilbert_v4",
        "roberta_v4",
        "xlm_roberta_v4",
        "hybrid",
    }:
        return
    try:
        from src.transformer_utils import _load_transformer_artifacts_cached

        _load_transformer_artifacts_cached.cache_clear()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _round_metric(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def calculate_model_metrics(results: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    pairs: list[tuple[int, int, dict[str, Any], dict[str, Any]]] = []
    risk_scores: list[float] = []
    latencies: list[float] = []
    unavailable_count = 0

    for row in results:
        prediction = row["predictions"].get(model_name, {})
        if not prediction.get("available"):
            unavailable_count += 1
            continue
        label = row.get("ground_truth_label")
        predicted = prediction.get("predicted_label")
        if label is None or predicted is None:
            continue
        pairs.append((int(label), int(predicted), row, prediction))
        if prediction.get("risk_score") is not None:
            risk_scores.append(float(prediction["risk_score"]))
        if prediction.get("latency_ms") is not None:
            latencies.append(float(prediction["latency_ms"]))

    if not pairs:
        return {
            "available_prompts": 0,
            "unavailable_prompts": unavailable_count,
            "message": "No available predictions for metrics.",
        }

    tp = sum(1 for truth, pred, _, _ in pairs if truth == 1 and pred == 1)
    fp = sum(1 for truth, pred, _, _ in pairs if truth == 0 and pred == 1)
    tn = sum(1 for truth, pred, _, _ in pairs if truth == 0 and pred == 0)
    fn = sum(1 for truth, pred, _, _ in pairs if truth == 1 and pred == 0)
    total = len(pairs)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    false_positives = [
        _mistake_row(row, prediction)
        for truth, pred, row, prediction in pairs
        if truth == 0 and pred == 1
    ]
    false_negatives = [
        _mistake_row(row, prediction)
        for truth, pred, row, prediction in pairs
        if truth == 1 and pred == 0
    ]
    return {
        "available_prompts": total,
        "unavailable_prompts": unavailable_count,
        "accuracy": _round_metric((tp + tn) / total),
        "precision": _round_metric(precision),
        "recall": _round_metric(recall),
        "f1": _round_metric(f1),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "false_positive_rate": _round_metric(fp / (fp + tn) if (fp + tn) else 0.0),
        "false_negative_rate": _round_metric(fn / (fn + tp) if (fn + tp) else 0.0),
        "avg_latency_ms": _round_metric(sum(latencies) / len(latencies) if latencies else 0.0),
        "avg_risk_score": _round_metric(sum(risk_scores) / len(risk_scores) if risk_scores else 0.0),
        "false_positives": false_positives[:20],
        "false_negatives": false_negatives[:20],
    }


def _mistake_row(row: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "text": row.get("text"),
        "ground_truth_label": row.get("ground_truth_label"),
        "predicted_label": prediction.get("predicted_label"),
        "risk_score": prediction.get("risk_score"),
        "action": prediction.get("action"),
    }


def find_disagreement_cases(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for row in results:
        labels = {
            model_name: prediction.get("predicted_label")
            for model_name, prediction in row.get("predictions", {}).items()
            if prediction.get("available") and prediction.get("predicted_label") is not None
        }
        if len(set(labels.values())) > 1:
            cases.append(
                {
                    "id": row.get("id"),
                    "text": row.get("text"),
                    "ground_truth_label": row.get("ground_truth_label"),
                    "model_labels": labels,
                }
            )
    return cases[:50]


def evaluate_batch_items(
    items: list[dict[str, Any]],
    models: list[str] | None = None,
    hybrid_config: dict[str, Any] | None = None,
    dataset_name: str | None = None,
    max_items: int = DEFAULT_MAX_BATCH_ITEMS,
) -> dict[str, Any]:
    selected_models = normalize_selected_models(models)
    normalized_hybrid_config = normalize_hybrid_config(hybrid_config)
    validation = validate_batch_items(items, max_items=max_items, dataset_name=dataset_name)
    metadata = {
        "dataset_name": dataset_name or "uploaded_dataset",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selected_models": selected_models,
        "hybrid_config": normalized_hybrid_config,
    }

    if not validation["valid"]:
        return {
            "status": "failed",
            "metadata": metadata,
            "validation": {key: value for key, value in validation.items() if key != "items"},
            "summary": {
                "total_prompts": validation["total_rows"],
                "has_ground_truth": False,
                "message": "Dataset validation failed.",
            },
            "results": [],
            "exports": {},
        }

    rows: list[dict[str, Any]] = []
    for item in validation["items"]:
        rows.append(
            {
                "id": item["id"],
                "text": item["text"],
                "original_prompt": item.get("original_prompt"),
                "normalized_text": item.get("normalized_text", item["text"]),
                "original_label": item.get("original_label"),
                "ground_truth_label": item.get("ground_truth_label", item.get("label")),
                "category": item.get("category"),
                "attack_type": item.get("attack_type"),
                "context": item.get("context"),
                "response": item.get("response"),
                "source": item.get("source"),
                "language": item.get("language"),
                "original_record": item.get("original_record", {}),
                "predictions": {},
            }
        )

    for model_name in selected_models:
        for row in rows:
            row["predictions"][model_name] = run_single_prediction(
                row["text"],
                model_name,
                hybrid_config=normalized_hybrid_config,
            )
        clear_model_cache(model_name)

    has_ground_truth = bool(validation["has_ground_truth"])
    summary: dict[str, Any] = {
        "total_prompts": len(rows),
        "has_ground_truth": has_ground_truth,
        "selected_models": selected_models,
    }
    if has_ground_truth:
        summary["models"] = {
            model_name: calculate_model_metrics(rows, model_name)
            for model_name in selected_models
        }
        summary["model_disagreement_cases"] = find_disagreement_cases(rows)
    else:
        summary["message"] = "Ground-truth labels not provided. Metrics are not available."
        summary["models"] = {}
        summary["model_disagreement_cases"] = find_disagreement_cases(rows)

    response_without_exports = {
        "status": "completed",
        "metadata": metadata,
        "validation": {key: value for key, value in validation.items() if key != "items"},
        "summary": summary,
        "results": rows,
    }
    exports = build_exports(response_without_exports, selected_models)
    return {**response_without_exports, "exports": exports}


def build_exports(payload: dict[str, Any], selected_models: list[str]) -> dict[str, Any]:
    return {
        "csv_filename": "batch_predictions_report.csv",
        "csv_content": build_csv_report(payload["results"], selected_models),
        "json_filename": "batch_predictions_report.json",
        "json_content": json.dumps(payload, ensure_ascii=False, indent=2),
        "markdown_filename": "batch_evaluation_report.md",
        "markdown_content": build_markdown_report(payload, selected_models),
    }


def build_csv_report(results: list[dict[str, Any]], selected_models: list[str]) -> str:
    output = io.StringIO()
    base_fields = ["id", "original_prompt", "normalized_text", "original_label", "ground_truth_label", "attack_type", "category", "context", "response", "source", "language"]
    fields = list(base_fields)
    for model_name in selected_models:
        prefix = CSV_PREFIXES.get(model_name, model_name)
        fields.extend(
            [
                f"{prefix}_pred",
                f"{prefix}_risk",
                f"{prefix}_confidence",
                f"{prefix}_action",
                f"{prefix}_latency_ms",
                f"{prefix}_available",
                f"{prefix}_message",
            ]
        )

    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for result in results:
        row = {field: result.get(field) for field in base_fields}
        for model_name in selected_models:
            prefix = CSV_PREFIXES.get(model_name, model_name)
            prediction = result["predictions"].get(model_name, {})
            row[f"{prefix}_pred"] = prediction.get("predicted_label")
            row[f"{prefix}_risk"] = prediction.get("risk_score")
            row[f"{prefix}_confidence"] = prediction.get("confidence")
            row[f"{prefix}_action"] = prediction.get("action")
            row[f"{prefix}_latency_ms"] = prediction.get("latency_ms")
            row[f"{prefix}_available"] = prediction.get("available")
            row[f"{prefix}_message"] = prediction.get("message") or prediction.get("error")
        writer.writerow(row)
    return output.getvalue()


def _escape_markdown_cell(value: Any, limit: int | None = None) -> str:
    text = "" if value is None else str(value)
    text = text.replace("|", "\\|").replace("\n", " ")
    if limit and len(text) > limit:
        text = text[: limit - 3] + "..."
    return text


def build_markdown_report(payload: dict[str, Any], selected_models: list[str]) -> str:
    metadata = payload["metadata"]
    summary = payload["summary"]
    lines = [
        "# Batch Evaluation Report",
        "",
        "## 1. Dataset uploaded",
        "",
        f"- Dataset: `{metadata.get('dataset_name')}`",
        f"- Generated at: `{metadata.get('generated_at')}`",
        f"- Number of prompts: `{summary.get('total_prompts')}`",
        f"- Ground truth available: `{summary.get('has_ground_truth')}`",
        f"- Text column detected: `{payload.get('validation', {}).get('text_column_detected')}`",
        f"- Label column detected: `{payload.get('validation', {}).get('label_column_detected')}`",
        f"- Category column detected: `{payload.get('validation', {}).get('category_column_detected')}`",
        "",
        "## 2. Selected models",
        "",
        *[f"- `{model_name}`" for model_name in selected_models],
        "",
        "## 3. Metrics per model",
        "",
    ]

    if not summary.get("has_ground_truth"):
        lines.append("Ground-truth labels not provided. Metrics are not available.")
    else:
        lines.extend([
            "| Model | Accuracy | Precision | Recall | F1 | TP | FP | TN | FN | Avg latency ms | Avg risk |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for model_name, metrics in summary.get("models", {}).items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        _escape_markdown_cell(model_name),
                        _escape_markdown_cell(metrics.get("accuracy")),
                        _escape_markdown_cell(metrics.get("precision")),
                        _escape_markdown_cell(metrics.get("recall")),
                        _escape_markdown_cell(metrics.get("f1")),
                        _escape_markdown_cell(metrics.get("tp")),
                        _escape_markdown_cell(metrics.get("fp")),
                        _escape_markdown_cell(metrics.get("tn")),
                        _escape_markdown_cell(metrics.get("fn")),
                        _escape_markdown_cell(metrics.get("avg_latency_ms")),
                        _escape_markdown_cell(metrics.get("avg_risk_score")),
                    ]
                )
                + " |"
            )

    lines.extend(["", "## 4. Top false positives", ""])
    lines.extend(_mistakes_markdown(summary, "false_positives"))
    lines.extend(["", "## 5. Top false negatives", ""])
    lines.extend(_mistakes_markdown(summary, "false_negatives"))
    lines.extend(["", "## 6. Model disagreement cases", ""])
    disagreements = summary.get("model_disagreement_cases", [])
    if not disagreements:
        lines.append("No disagreement cases found.")
    else:
        lines.extend(["| ID | Prompt | Ground truth | Model labels |", "|---|---|---:|---|"])
        for case in disagreements[:20]:
            lines.append(
                f"| {_escape_markdown_cell(case.get('id'))} | "
                f"{_escape_markdown_cell(case.get('text'), 100)} | "
                f"{_escape_markdown_cell(case.get('ground_truth_label'))} | "
                f"{_escape_markdown_cell(json.dumps(case.get('model_labels'), ensure_ascii=False))} |"
            )

    lines.extend(["", "## 7. Full prediction table", ""])
    header = ["ID", "Original prompt", "Normalized text", "Original label", "Ground truth", "Category"]
    for model_name in selected_models:
        header.extend([f"{model_name} label", f"{model_name} risk", f"{model_name} action"])
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for row in payload["results"]:
        cells = [
            _escape_markdown_cell(row.get("id")),
            _escape_markdown_cell(row.get("original_prompt", row.get("text")), 100),
            _escape_markdown_cell(row.get("normalized_text", row.get("text")), 100),
            _escape_markdown_cell(row.get("original_label")),
            _escape_markdown_cell(row.get("ground_truth_label")),
            _escape_markdown_cell(row.get("attack_type") or row.get("category")),
        ]
        for model_name in selected_models:
            prediction = row["predictions"].get(model_name, {})
            cells.extend(
                [
                    _escape_markdown_cell(prediction.get("predicted_label")),
                    _escape_markdown_cell(prediction.get("risk_score")),
                    _escape_markdown_cell(prediction.get("action")),
                ]
            )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _mistakes_markdown(summary: dict[str, Any], key: str) -> list[str]:
    if not summary.get("has_ground_truth"):
        return ["Ground-truth labels not provided."]
    lines: list[str] = []
    found = False
    for model_name, metrics in summary.get("models", {}).items():
        mistakes = metrics.get(key, [])
        if not mistakes:
            continue
        found = True
        lines.append(f"### {model_name}")
        lines.append("| ID | Prompt | Ground truth | Predicted | Risk | Action |")
        lines.append("|---|---|---:|---:|---:|---|")
        for item in mistakes[:10]:
            lines.append(
                f"| {_escape_markdown_cell(item.get('id'))} | "
                f"{_escape_markdown_cell(item.get('text'), 100)} | "
                f"{_escape_markdown_cell(item.get('ground_truth_label'))} | "
                f"{_escape_markdown_cell(item.get('predicted_label'))} | "
                f"{_escape_markdown_cell(item.get('risk_score'))} | "
                f"{_escape_markdown_cell(item.get('action'))} |"
            )
        lines.append("")
    return lines if found else ["No cases found."]
