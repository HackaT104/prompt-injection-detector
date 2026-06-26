"""Post-hoc probability calibration for DistilBERT v3.

The script fits an isotonic calibrator on the Transformer validation split, saves it
beside the checkpoint, and refreshes threshold files using calibrated scores.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.thresholding import choose_threshold, metrics_at_threshold  # noqa: E402
from src.transformer_utils import (  # noqa: E402
    _load_transformer_artifacts_cached,
    import_optional,
    normalize_transformer_label,
    resolve_transformer_model_dir,
    softmax_positive_scores,
    split_transformer_dataframe_by_source,
)
from src.preprocessing import clean_text  # noqa: E402


DATASET_PATH = PROJECT_ROOT / "datasets" / "unified" / "prompt_injection_transformer_ready_v3.csv"
MODEL_NAME = "distilbert_v3"
REPORT_PATH = PROJECT_ROOT / "reports" / "distilbert_probability_calibration.md"
JSON_REPORT_PATH = PROJECT_ROOT / "reports" / "distilbert_probability_calibration.json"
THRESHOLDS_PATHS = [
    PROJECT_ROOT / "models" / "transformer_thresholds.json",
    PROJECT_ROOT / "models" / "thresholds.json",
]


def predict_scores(texts: list[str], batch_size: int = 32, max_length: int = 128, use_cuda: bool = False) -> np.ndarray:
    model_dir = resolve_transformer_model_dir(MODEL_NAME)
    torch, tokenizer, model, device = _load_transformer_artifacts_cached(str(model_dir.resolve()), use_cuda)
    scores: list[float] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
        scores.extend(float(score) for score in softmax_positive_scores(outputs.logits.detach().cpu().numpy()))
    _load_transformer_artifacts_cached.cache_clear()
    return np.asarray(scores, dtype=float)


def load_validation_data() -> tuple[list[str], list[int]]:
    raw_df = pd.read_csv(DATASET_PATH, encoding="utf-8-sig")
    rows: list[dict[str, Any]] = []
    for index, row in raw_df.iterrows():
        raw_text = "" if pd.isna(row.get("text")) else str(row.get("text"))
        if not raw_text.strip() or pd.isna(row.get("label")):
            continue
        rows.append(
            {
                "sample_id": f"calibration_{index}",
                "text": raw_text,
                "model_text": clean_text(raw_text),
                "label": normalize_transformer_label(row.get("label")),
                "source_split": str(row.get("split", "unknown")),
            }
        )
    prepared = pd.DataFrame(rows).drop_duplicates(subset=["model_text", "label"]).reset_index(drop=True)
    _train_df, validation_df, _test_df = split_transformer_dataframe_by_source(prepared)
    return validation_df["model_text"].tolist(), validation_df["label"].astype(int).tolist()


def update_threshold_files(payload: dict[str, Any]) -> None:
    for path in THRESHOLDS_PATHS:
        if path.exists():
            current = json.loads(path.read_text(encoding="utf-8-sig"))
        else:
            current = {"models": {}}
        current.setdefault("models", {})
        current["models"][MODEL_NAME] = payload
        current["models"]["distilbert_v3"] = payload
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    import_optional("torch")
    texts, labels = load_validation_data()
    raw_scores = predict_scores(texts, use_cuda=True)
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(raw_scores, labels)
    calibrated_scores = np.asarray(calibrator.predict(raw_scores), dtype=float)

    raw_threshold = choose_threshold(labels, raw_scores, optimization_metric="f1")
    calibrated_threshold = choose_threshold(labels, calibrated_scores, optimization_metric="f1")
    selected = calibrated_threshold["selected_metrics"]
    model_dir = resolve_transformer_model_dir(MODEL_NAME)
    calibrator_path = model_dir / "probability_calibrator.joblib"
    joblib.dump(calibrator, calibrator_path)

    threshold_payload = {
        "evaluation_threshold": float(calibrated_threshold["evaluation_threshold"]),
        "runtime_warn_threshold": float(calibrated_threshold["runtime_warn_threshold"]),
        "runtime_block_threshold": float(calibrated_threshold["runtime_block_threshold"]),
        "best_metric": calibrated_threshold["best_metric"],
        "precision": float(selected["precision"]),
        "recall": float(selected["recall"]),
        "f1": float(selected["f1"]),
        "f2": float(selected["f2"]),
        "tn": int(selected["true_negative"]),
        "fp": int(selected["false_positive"]),
        "fn": int(selected["false_negative"]),
        "tp": int(selected["true_positive"]),
        "dataset": str(DATASET_PATH),
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_dir),
        "probability_calibrator": str(calibrator_path),
        "calibration_method": "IsotonicRegression",
        "warnings": [],
    }
    update_threshold_files(threshold_payload)

    raw_metrics = metrics_at_threshold(labels, raw_scores, raw_threshold["evaluation_threshold"])
    calibrated_metrics = metrics_at_threshold(labels, calibrated_scores, calibrated_threshold["evaluation_threshold"])
    report = {
        "model": MODEL_NAME,
        "dataset": str(DATASET_PATH),
        "rows": len(labels),
        "calibrator_path": str(calibrator_path),
        "raw_brier": float(brier_score_loss(labels, raw_scores)),
        "calibrated_brier": float(brier_score_loss(labels, calibrated_scores)),
        "raw_log_loss": float(log_loss(labels, np.clip(raw_scores, 1e-7, 1 - 1e-7))),
        "calibrated_log_loss": float(log_loss(labels, np.clip(calibrated_scores, 1e-7, 1 - 1e-7))),
        "raw_roc_auc": float(roc_auc_score(labels, raw_scores)),
        "calibrated_roc_auc": float(roc_auc_score(labels, calibrated_scores)),
        "raw_average_precision": float(average_precision_score(labels, raw_scores)),
        "calibrated_average_precision": float(average_precision_score(labels, calibrated_scores)),
        "raw_threshold": raw_threshold,
        "calibrated_threshold": calibrated_threshold,
        "raw_metrics": raw_metrics,
        "calibrated_metrics": calibrated_metrics,
    }
    JSON_REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# DistilBERT Probability Calibration",
        "",
        f"- Model: `{MODEL_NAME}`",
        f"- Dataset: `{DATASET_PATH}`",
        f"- Validation rows: `{len(labels)}`",
        f"- Calibrator: `IsotonicRegression`",
        f"- Calibrator path: `{calibrator_path}`",
        "",
        "| Metric | Raw | Calibrated |",
        "| --- | ---: | ---: |",
        f"| Brier score | {report['raw_brier']:.6f} | {report['calibrated_brier']:.6f} |",
        f"| Log loss | {report['raw_log_loss']:.6f} | {report['calibrated_log_loss']:.6f} |",
        f"| ROC-AUC | {report['raw_roc_auc']:.6f} | {report['calibrated_roc_auc']:.6f} |",
        f"| Average precision | {report['raw_average_precision']:.6f} | {report['calibrated_average_precision']:.6f} |",
        f"| Evaluation threshold | {raw_threshold['evaluation_threshold']:.4f} | {calibrated_threshold['evaluation_threshold']:.4f} |",
        f"| Warn threshold | {raw_threshold['runtime_warn_threshold']:.4f} | {calibrated_threshold['runtime_warn_threshold']:.4f} |",
        f"| Block threshold | {raw_threshold['runtime_block_threshold']:.4f} | {calibrated_threshold['runtime_block_threshold']:.4f} |",
        "",
        "Runtime now uses calibrated probability for `distilbert_v3` when `probability_calibrator.joblib` is present.",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_PATH), "calibrator": str(calibrator_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
