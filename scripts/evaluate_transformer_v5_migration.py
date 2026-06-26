"""Compare Transformer v4/v5 on Vietnamese and English holdout datasets."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.transformer_utils import ID2LABEL, LABEL2ID, import_optional

MODELS_DIR = PROJECT_ROOT / "models"
TRANSFORMER_DIR = MODELS_DIR / "transformers"
REPORTS_DIR = PROJECT_ROOT / "reports"
VI_TEST_PATH = PROJECT_ROOT / "datasets" / "processed" / "vi_test_processed.csv"
EN_TEST_PATH = PROJECT_ROOT / "datasets" / "processed" / "hf_prompt_injection_test.csv"
THRESHOLDS_PATH = MODELS_DIR / "thresholds.json"
EVALUATION_PATH = REPORTS_DIR / "transformer_v5_vi_evaluation.csv"
SUMMARY_PATH = REPORTS_DIR / "transformer_v5_vi_summary.md"
ERROR_CASES_PATH = REPORTS_DIR / "transformer_v5_vi_error_cases.csv"
SCORES_PATH = REPORTS_DIR / "transformer_v5_vi_score_distribution.csv"
ERROR_ANALYSIS_PATH = REPORTS_DIR / "error_analysis_v5.md"

MODEL_PAIRS = {
    "roberta": ("roberta_v4", "roberta_v5_vi"),
    "xlm_roberta": ("xlm_roberta_v4", "xlm_roberta_v5_vi"),
}


def _load_frame(path: Path, default_language: str) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame = frame.dropna(subset=["text", "label"]).copy()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame["label"] = frame["label"].astype(int)
    if "language" not in frame:
        frame["language"] = default_language
    frame["language"] = frame["language"].fillna(default_language).astype(str).str.lower()
    if "id" not in frame:
        frame["id"] = [f"{path.stem}_{index:07d}" for index in range(len(frame))]
    return frame


def _is_checkpoint(path: Path) -> bool:
    return (
        (path / "config.json").exists()
        and any((path / filename).exists() for filename in ["pytorch_model.bin", "model.safetensors"])
    )


def _thresholds() -> dict[str, Any]:
    payload = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8-sig"))
    return payload.get("models", payload)


def _predict(model_path: Path, texts: list[str], batch_size: int, use_cuda: bool) -> tuple[np.ndarray, float]:
    torch = import_optional("torch")
    transformers = import_optional("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path)
    config_id2label = {int(key): str(value).upper() for key, value in model.config.id2label.items()}
    if config_id2label != ID2LABEL or int(model.config.label2id.get("INJECTION", -1)) != LABEL2ID["INJECTION"]:
        raise RuntimeError(f"Label mapping sai trong {model_path}: {model.config.id2label}")
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    scores: list[float] = []
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start:start + batch_size],
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(**encoded).logits
            probabilities = torch.softmax(logits.float(), dim=-1)[:, LABEL2ID["INJECTION"]]
            scores.extend(probabilities.detach().cpu().numpy().astype(float).tolist())
    elapsed = time.perf_counter() - started
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.asarray(scores, dtype=float), elapsed


def _metrics(labels: list[int], scores: np.ndarray, threshold: float, elapsed: float) -> dict[str, Any]:
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = [int(value) for value in confusion_matrix(labels, predicted, labels=[0, 1]).ravel()]
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, pos_label=1, zero_division=0)),
        "recall": float(recall_score(labels, predicted, pos_label=1, zero_division=0)),
        "f1": float(f1_score(labels, predicted, pos_label=1, zero_division=0)),
        "f2": float(fbeta_score(labels, predicted, beta=2, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "threshold": float(threshold),
        "rows": len(labels),
        "prediction_time_seconds": elapsed,
        "avg_latency_ms": elapsed / max(len(labels), 1) * 1000,
    }


def _fmt(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"


def evaluate(batch_size: int = 16, use_cuda: bool = True) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    vi_test = _load_frame(VI_TEST_PATH, "unknown")
    en_test = _load_frame(EN_TEST_PATH, "en")
    thresholds = _thresholds()
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    distributions: list[dict[str, Any]] = []

    if EVALUATION_PATH.exists():
        existing = pd.read_csv(EVALUATION_PATH, encoding="utf-8-sig")
        rows.extend(existing[existing["split"].astype(str).str.contains("validation")].to_dict("records"))

    for family, versions in MODEL_PAIRS.items():
        for version in versions:
            model_path = TRANSFORMER_DIR / version
            if not _is_checkpoint(model_path):
                print(f"Skip {version}: checkpoint not found")
                continue
            threshold = float(thresholds.get(version, {}).get("evaluation_threshold", 0.5))
            for split_name, frame in [("vi_test", vi_test), ("english_v4_test", en_test)]:
                print(f"Evaluate {version} on {split_name} ({len(frame)} rows)")
                scores, elapsed = _predict(
                    model_path, frame["text"].astype(str).tolist(), batch_size, use_cuda
                )
                metric = _metrics(frame["label"].astype(int).tolist(), scores, threshold, elapsed)
                rows.append({
                    "family": family,
                    "model": version,
                    "generation": "v5" if "_v5_" in version else "v4",
                    "split": split_name,
                    **metric,
                })
                predictions = (scores >= threshold).astype(int)
                for index, item in frame.reset_index(drop=True).iterrows():
                    truth, prediction = int(item["label"]), int(predictions[index])
                    if "_v5_" in version:
                        distributions.append({
                            "model": version,
                            "split": split_name,
                            "id": item["id"],
                            "language": item["language"],
                            "label": truth,
                            "risk_score": float(scores[index]),
                        })
                        if truth != prediction:
                            errors.append({
                                "model": version,
                                "split": split_name,
                                "error_type": "false_positive" if truth == 0 else "false_negative",
                                "id": item["id"],
                                "language": item["language"],
                                "ground_truth": truth,
                                "predicted_label": prediction,
                                "risk_score": float(scores[index]),
                                "threshold": threshold,
                                "attack_type": item.get("attack_type", ""),
                                "source": item.get("source", ""),
                                "text": item["text"],
                            })

    result_df = pd.DataFrame(rows)
    result_df["family"] = result_df["family"].fillna(
        result_df["model"].map(
            lambda value: "xlm_roberta" if str(value).startswith("xlm_") else "roberta"
        )
    )
    result_df.to_csv(EVALUATION_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(errors).to_csv(ERROR_CASES_PATH, index=False, encoding="utf-8-sig")
    pd.DataFrame(distributions).to_csv(SCORES_PATH, index=False, encoding="utf-8-sig")

    lines = [
        "# Tổng kết migration Transformer v4 sang v5 tiếng Việt", "",
        "V5 tiếp tục từ checkpoint v4 với replay 80/20; threshold v5 được chọn trên validation.",
        "", "## Full comparison", "",
        "| Family | Model | Split | Accuracy | Precision | Recall | F1 | F2 | ROC-AUC | PR-AUC | FP | FN |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in result_df.iterrows():
        lines.append(
            f"| {row.get('family', '')} | {row['model']} | {row['split']} | "
            f"{_fmt(row.get('accuracy'))} | {_fmt(row.get('precision'))} | {_fmt(row.get('recall'))} | "
            f"{_fmt(row.get('f1'))} | {_fmt(row.get('f2'))} | {_fmt(row.get('roc_auc'))} | "
            f"{_fmt(row.get('average_precision'))} | {row.get('fp', 'N/A')} | {row.get('fn', 'N/A')} |"
        )
    lines.extend(["", "## Delta v5 so với v4", "",
                  "| Family | Split | Delta F1 | Delta ROC-AUC | Delta FP | Delta FN |",
                  "| --- | --- | ---: | ---: | ---: | ---: |"])
    for family, (v4, v5) in MODEL_PAIRS.items():
        for split in ["vi_test", "english_v4_test"]:
            old = result_df[(result_df["model"] == v4) & (result_df["split"] == split)]
            new = result_df[(result_df["model"] == v5) & (result_df["split"] == split)]
            if old.empty or new.empty:
                continue
            old_row, new_row = old.iloc[0], new.iloc[0]
            lines.append(
                f"| {family} | {split} | {new_row['f1'] - old_row['f1']:+.4f} | "
                f"{new_row['roc_auc'] - old_row['roc_auc']:+.4f} | "
                f"{int(new_row['fp'] - old_row['fp']):+d} | {int(new_row['fn'] - old_row['fn']):+d} |"
            )
    lines.extend([
        "", "## DistilBERT", "",
        "DistilBERT đã được backup và deprecated; không còn trong runtime mặc định.",
        "", "## Context-Aware Detection", "",
        "Bước tiếp theo là đánh giá riêng user prompt và external context, sau đó hợp nhất "
        "direct score, indirect score và rule signals bằng policy có calibration.",
    ])
    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    analysis = [
        "# Phân tích lỗi Transformer v5", "",
        "| Model | Split | Language | FP | FN |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    if errors:
        error_df = pd.DataFrame(errors)
        grouped = error_df.groupby(["model", "split", "language", "error_type"]).size().unstack(fill_value=0)
        for (model, split, language), counts in grouped.iterrows():
            analysis.append(
                f"| {model} | {split} | {language} | "
                f"{int(counts.get('false_positive', 0))} | {int(counts.get('false_negative', 0))} |"
            )
        for model in error_df["model"].unique():
            analysis.extend(["", f"## {model}", ""])
            for kind, title in [("false_positive", "Top FP"), ("false_negative", "Top FN")]:
                analysis.extend([f"### {title}", ""])
                selected = error_df[
                    (error_df["model"] == model) & (error_df["error_type"] == kind)
                ].sort_values("risk_score", ascending=(kind == "false_negative")).head(10)
                for _, item in selected.iterrows():
                    preview = str(item["text"]).replace("\n", " ")[:180]
                    analysis.append(
                        f"- {item['split']}/{item['language']} score={item['risk_score']:.4f}: {preview}"
                    )
    ERROR_ANALYSIS_PATH.write_text("\n".join(analysis) + "\n", encoding="utf-8")
    print(f"Saved: {EVALUATION_PATH}")
    print(f"Saved: {SUMMARY_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Transformer v4/v5 migration.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--use-cuda", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    evaluate(args.batch_size, args.use_cuda)


if __name__ == "__main__":
    main()

