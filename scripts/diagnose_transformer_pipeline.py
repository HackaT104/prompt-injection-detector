"""Diagnose Transformer inference, labels, tokenizers, splits and score distributions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.batch_evaluation import parse_dataset_content, validate_batch_items  # noqa: E402
from src.thresholding import predict_with_threshold  # noqa: E402
from src.transformer_utils import (  # noqa: E402
    ID2LABEL,
    LABEL2ID,
    _load_transformer_artifacts_cached,
    import_optional,
    is_finetuned_transformer_checkpoint,
    resolve_transformer_model_dir,
    softmax_positive_scores,
    normalize_transformer_label,
    split_transformer_dataframe_by_source,
)
from src.preprocessing import clean_text  # noqa: E402


DEFAULT_TEST_DATASET = PROJECT_ROOT / "datasets" / "test" / "Prompt_INJECTION_And_Benign_DATASET.jsonl"
DEFAULT_TRAIN_DATASET = PROJECT_ROOT / "datasets" / "unified" / "prompt_injection_transformer_ready_v3.csv"
DEFAULT_MODELS = ["distilbert_v3", "roberta_v3", "xlm_roberta_v3"]
REPORTS_DIR = PROJECT_ROOT / "reports"
THRESHOLDS_PATH = PROJECT_ROOT / "models" / "transformer_thresholds.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose Transformer prompt injection models.")
    parser.add_argument("--test-dataset", default=str(DEFAULT_TEST_DATASET))
    parser.add_argument("--train-dataset", default=str(DEFAULT_TRAIN_DATASET))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--use-cuda", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def read_text_dataset(path: Path) -> list[dict[str, Any]]:
    content = path.read_text(encoding="utf-8-sig", errors="replace")
    return parse_dataset_content(path.name, content)


def load_labeled_test(path: Path) -> tuple[list[dict[str, Any]], list[str], list[int], dict[str, Any]]:
    items = read_text_dataset(path)
    validation = validate_batch_items(items, max_items=1_000_000, dataset_name=path.name)
    if not validation["valid"]:
        raise ValueError("Dataset validation failed: " + "; ".join(validation["errors"]))
    if not validation["has_ground_truth"]:
        raise ValueError("Diagnostics requires a labeled test dataset.")
    rows = validation["items"]
    texts = [str(row["text"]) for row in rows]
    labels = [int(row["ground_truth_label"]) for row in rows]
    dataset_info = {
        "path": str(path),
        "rows": len(rows),
        "text_column": validation["text_column_detected"],
        "label_column": validation["label_column_detected"],
        "label_distribution": {
            "safe_0": int(sum(label == 0 for label in labels)),
            "injection_1": int(sum(label == 1 for label in labels)),
        },
    }
    return rows, texts, labels, dataset_info


def load_thresholds() -> dict[str, dict[str, float]]:
    if not THRESHOLDS_PATH.exists():
        return {}
    payload = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8-sig"))
    return payload.get("models", payload)


def model_config_status(model_dir: Path) -> dict[str, Any]:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    tokenizer_config_path = model_dir / "tokenizer_config.json"
    tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8")) if tokenizer_config_path.exists() else {}
    id2label = {int(key): value for key, value in config.get("id2label", {}).items()}
    label2id = config.get("label2id", {})
    return {
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures", []),
        "id2label": id2label,
        "label2id": label2id,
        "label_mapping_ok": id2label == ID2LABEL and label2id == LABEL2ID,
        "tokenizer_class_config": tokenizer_config.get("tokenizer_class"),
        "model_config": config,
    }


def expected_tokenizer_ok(model_name: str, tokenizer_class: str, model_type: str) -> bool:
    lowered = tokenizer_class.lower()
    if model_name.startswith("distilbert"):
        return "distilbert" in lowered and model_type == "distilbert"
    if model_name.startswith("roberta"):
        return "roberta" in lowered and "xlm" not in lowered and model_type == "roberta"
    if model_name.startswith("xlm_roberta"):
        return "xlm" in lowered and "roberta" in lowered and model_type == "xlm-roberta"
    return True


def predict_scores(
    model_name: str,
    texts: list[str],
    batch_size: int,
    max_length: int,
    use_cuda: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    model_dir = resolve_transformer_model_dir(model_name)
    if not is_finetuned_transformer_checkpoint(model_dir):
        raise FileNotFoundError(f"Checkpoint is not fine-tuned or missing: {model_dir}")

    torch, tokenizer, model, device = _load_transformer_artifacts_cached(str(model_dir.resolve()), use_cuda)
    scores: list[float] = []
    first_logits: list[float] | None = None
    first_probs: list[float] | None = None
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
        with torch.no_grad():
            outputs = model(**encoded)
        logits = outputs.logits.detach().cpu().numpy()
        probabilities = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        if first_logits is None:
            first_logits = [float(value) for value in logits[0].tolist()]
            first_probs = [float(value) for value in probabilities[0].tolist()]
        scores.extend(float(score) for score in softmax_positive_scores(logits))

    runtime = {
        "model_dir": str(model_dir),
        "tokenizer_class_runtime": tokenizer.__class__.__name__,
        "device": str(device),
        "first_logits": first_logits,
        "first_probabilities": first_probs,
        "softmax_sum_first": None if first_probs is None else float(sum(first_probs)),
        "risk_score_source": "softmax(logits)[:, 1]",
    }
    _load_transformer_artifacts_cached.cache_clear()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return np.asarray(scores, dtype=float), runtime


def metrics_for_scores(y_true: list[int], scores: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = predict_with_threshold(scores, threshold)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in cm.ravel()]
    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(set(y_true)) == 2 else None,
        "average_precision": float(average_precision_score(y_true, scores)) if len(set(y_true)) == 2 else None,
        "confusion_matrix": cm.tolist(),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
    return metrics


def distribution_rows(model_name: str, y_true: list[int], scores: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bins = [round(i / 10, 1) for i in range(0, 11)]
    for label in [0, 1]:
        label_scores = np.asarray([score for truth, score in zip(y_true, scores) if truth == label], dtype=float)
        for left, right in zip(bins[:-1], bins[1:]):
            if right == 1.0:
                count = int(((label_scores >= left) & (label_scores <= right)).sum())
            else:
                count = int(((label_scores >= left) & (label_scores < right)).sum())
            rows.append(
                {
                    "model": model_name,
                    "label": label,
                    "label_name": "SAFE" if label == 0 else "INJECTION",
                    "risk_bin": f"{left:.1f}-{right:.1f}",
                    "count": count,
                }
            )
    return rows


def error_rows(
    model_name: str,
    dataset_rows: list[dict[str, Any]],
    y_true: list[int],
    scores: np.ndarray,
    threshold: float,
) -> list[dict[str, Any]]:
    y_pred = predict_with_threshold(scores, threshold)
    rows: list[dict[str, Any]] = []
    for row, truth, pred, score in zip(dataset_rows, y_true, y_pred, scores):
        if int(truth) == int(pred):
            continue
        rows.append(
            {
                "model": model_name,
                "id": row.get("id"),
                "error_type": "FP" if int(truth) == 0 and int(pred) == 1 else "FN",
                "ground_truth": int(truth),
                "predicted": int(pred),
                "risk_score": float(score),
                "threshold": float(threshold),
                "text": row.get("text"),
                "category": row.get("category"),
                "language": row.get("language"),
            }
        )
    return sorted(rows, key=lambda item: item["risk_score"], reverse=True)


def analyze_training_dataset(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    raw_df = pd.read_csv(path, encoding="utf-8-sig")
    rows: list[dict[str, Any]] = []
    for index, row in raw_df.iterrows():
        raw_text = "" if pd.isna(row.get("text")) else str(row.get("text"))
        if not raw_text.strip() or pd.isna(row.get("label")):
            continue
        rows.append(
            {
                "sample_id": f"diagnostic_{index}",
                "text": raw_text,
                "model_text": clean_text(raw_text),
                "label": normalize_transformer_label(row.get("label")),
                "detected_language": str(row.get("language", "unknown")),
                "source_split": str(row.get("split", "unknown")),
            }
        )
    prepared = pd.DataFrame(rows).drop_duplicates(subset=["model_text", "label"]).reset_index(drop=True)
    train_df, validation_df, test_df = split_transformer_dataframe_by_source(prepared)
    split_map = {"train": train_df, "validation": validation_df, "test": test_df}
    text_sets = {name: set(df["model_text"].astype(str)) for name, df in split_map.items()}

    def distribution(df: pd.DataFrame, column: str) -> dict[str, int]:
        if column not in df.columns:
            return {}
        return {str(key): int(value) for key, value in df[column].value_counts(dropna=False).to_dict().items()}

    return {
        "path": str(path),
        "exists": True,
        "raw_rows": int(len(raw_df)),
        "prepared_rows_after_dedup": int(len(prepared)),
        "duplicates_removed_by_prepare": int(len(raw_df) - len(prepared)),
        "raw_duplicate_text_count": int(raw_df.duplicated(subset=["text"]).sum()) if "text" in raw_df.columns else None,
        "raw_duplicate_text_label_count": int(raw_df.duplicated(subset=["text", "label"]).sum())
        if {"text", "label"}.issubset(raw_df.columns)
        else None,
        "split_sizes": {name: int(len(df)) for name, df in split_map.items()},
        "split_label_distribution": {name: distribution(df, "label") for name, df in split_map.items()},
        "split_language_distribution": {name: distribution(df, "detected_language") for name, df in split_map.items()},
        "overlap_train_validation": int(len(text_sets["train"] & text_sets["validation"])),
        "overlap_train_test": int(len(text_sets["train"] & text_sets["test"])),
        "overlap_validation_test": int(len(text_sets["validation"] & text_sets["test"])),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return "" if value is None else str(value)


def pass_criteria(metrics: dict[str, Any]) -> bool:
    threshold = float(metrics["threshold"])
    return (
        float(metrics["f1"]) >= 0.97
        and float(metrics["recall"]) >= 0.95
        and float(metrics["precision"]) >= 0.95
        and (metrics["roc_auc"] is not None and float(metrics["roc_auc"]) >= 0.98)
        and 0.2 <= threshold <= 0.8
    )


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Transformer Diagnostics",
        "",
        "Report kiểm tra inference, label mapping, tokenizer, dataset split, score distribution và lỗi FP/FN.",
        "",
        "## Test Dataset",
        "",
        f"- Path: `{payload['test_dataset']['path']}`",
        f"- Rows: `{payload['test_dataset']['rows']}`",
        f"- Text column: `{payload['test_dataset']['text_column']}`",
        f"- Label column: `{payload['test_dataset']['label_column']}`",
        f"- SAFE/0: `{payload['test_dataset']['label_distribution']['safe_0']}`",
        f"- INJECTION/1: `{payload['test_dataset']['label_distribution']['injection_1']}`",
        "",
        "## Training Dataset / Split Check",
        "",
    ]
    split = payload["training_dataset"]
    if split.get("exists"):
        lines.extend(
            [
                f"- Path: `{split['path']}`",
                f"- Raw rows: `{split['raw_rows']}`",
                f"- Prepared rows after dedup: `{split['prepared_rows_after_dedup']}`",
                f"- Duplicates removed by prepare: `{split['duplicates_removed_by_prepare']}`",
                f"- Raw duplicate text count: `{split['raw_duplicate_text_count']}`",
                f"- Raw duplicate text+label count: `{split['raw_duplicate_text_label_count']}`",
                f"- Split sizes: `{split['split_sizes']}`",
                f"- Overlap train/validation: `{split['overlap_train_validation']}`",
                f"- Overlap train/test: `{split['overlap_train_test']}`",
                f"- Overlap validation/test: `{split['overlap_validation_test']}`",
            ]
        )
    else:
        lines.append(f"- Training dataset not found: `{split['path']}`")

    lines.extend(
        [
            "",
            "## Model Metrics",
            "",
            "| Model | Mapping OK | Tokenizer OK | Threshold | AUC-ROC | PR-AUC | Accuracy | Precision | Recall | F1 | F2 | TN | FP | FN | TP | Decision |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in payload["models"]:
        metrics = row["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["model"],
                    str(row["label_mapping_ok"]),
                    str(row["tokenizer_ok"]),
                    fmt(metrics["threshold"]),
                    fmt(metrics["roc_auc"]),
                    fmt(metrics["average_precision"]),
                    fmt(metrics["accuracy"]),
                    fmt(metrics["precision"]),
                    fmt(metrics["recall"]),
                    fmt(metrics["f1"]),
                    fmt(metrics["f2"]),
                    str(metrics["tn"]),
                    str(metrics["fp"]),
                    str(metrics["fn"]),
                    str(metrics["tp"]),
                    row["decision"],
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Inference Checks",
            "",
        ]
    )
    for row in payload["models"]:
        runtime = row["runtime"]
        lines.extend(
            [
                f"### {row['model']}",
                "",
                f"- Model dir: `{runtime['model_dir']}`",
                f"- Runtime tokenizer: `{runtime['tokenizer_class_runtime']}`",
                f"- Config tokenizer: `{row['tokenizer_class_config']}`",
                f"- Model type: `{row['model_type']}`",
                f"- Risk score source: `{runtime['risk_score_source']}`",
                f"- First logits: `{runtime['first_logits']}`",
                f"- First probabilities: `{runtime['first_probabilities']}`",
                f"- First probability sum: `{runtime['softmax_sum_first']}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Output Files",
            "",
            "- Score distribution: `reports/transformer_score_distribution.csv`",
            "- Error cases: `reports/transformer_error_cases.csv`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    test_dataset = Path(args.test_dataset)
    if not test_dataset.is_absolute():
        test_dataset = PROJECT_ROOT / test_dataset
    train_dataset = Path(args.train_dataset)
    if not train_dataset.is_absolute():
        train_dataset = PROJECT_ROOT / train_dataset

    dataset_rows, texts, y_true, test_info = load_labeled_test(test_dataset.resolve())
    thresholds = load_thresholds()
    training_info = analyze_training_dataset(train_dataset.resolve())

    model_rows: list[dict[str, Any]] = []
    distribution_output: list[dict[str, Any]] = []
    error_output: list[dict[str, Any]] = []

    import_optional("torch")
    for model_name in args.models:
        print(f"[diagnostics] {model_name}")
        model_dir = resolve_transformer_model_dir(model_name)
        config_status = model_config_status(model_dir)
        scores, runtime = predict_scores(
            model_name,
            texts,
            batch_size=args.batch_size,
            max_length=args.max_length,
            use_cuda=args.use_cuda,
        )
        threshold_payload = thresholds.get(model_name) or thresholds.get(model_dir.name) or {}
        threshold = float(threshold_payload.get("evaluation_threshold", 0.5))
        metrics = metrics_for_scores(y_true, scores, threshold)
        tokenizer_ok = expected_tokenizer_ok(
            model_name,
            runtime["tokenizer_class_runtime"],
            str(config_status["model_type"]),
        )
        model_pass = bool(config_status["label_mapping_ok"] and tokenizer_ok and pass_criteria(metrics))
        decision = "keep" if model_pass else "needs_retrain_or_recalibration"
        model_rows.append(
            {
                "model": model_name,
                **config_status,
                "tokenizer_ok": tokenizer_ok,
                "runtime": runtime,
                "threshold_payload": threshold_payload,
                "metrics": metrics,
                "decision": decision,
            }
        )
        distribution_output.extend(distribution_rows(model_name, y_true, scores))
        error_output.extend(error_rows(model_name, dataset_rows, y_true, scores, threshold))

    payload = {
        "test_dataset": test_info,
        "training_dataset": training_info,
        "models": model_rows,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(
        REPORTS_DIR / "transformer_score_distribution.csv",
        distribution_output,
        ["model", "label", "label_name", "risk_bin", "count"],
    )
    write_csv(
        REPORTS_DIR / "transformer_error_cases.csv",
        error_output,
        ["model", "id", "error_type", "ground_truth", "predicted", "risk_score", "threshold", "text", "category", "language"],
    )
    (REPORTS_DIR / "transformer_diagnostics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    write_markdown(REPORTS_DIR / "transformer_diagnostics.md", payload)
    print(json.dumps({"report": str(REPORTS_DIR / "transformer_diagnostics.md")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
