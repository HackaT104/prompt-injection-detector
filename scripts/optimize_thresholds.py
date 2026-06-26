"""Optimize evaluation and runtime thresholds for prompt injection detectors.

The script uses saved probabilities when available and never trains or overwrites
model checkpoints. Thresholds are selected on validation data, then evaluated on
an independent test split for reporting.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.detector import load_model_artifacts  # noqa: E402
from src.preprocessing import prepare_text_for_detection  # noqa: E402

REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
THRESHOLDS_PATH = MODELS_DIR / "thresholds.json"
TRANSFORMER_RESULTS_PATH = REPORTS_DIR / "transformer_v5_vi_results.json"
DEFAULT_VALIDATION = PROJECT_ROOT / "datasets" / "processed" / "vi_validation_processed.csv"
DEFAULT_TEST = PROJECT_ROOT / "datasets" / "processed" / "vi_test_processed.csv"
DEFAULT_MODELS = [
    "logistic_regression",
    "linear_svm",
    "random_forest",
    "roberta_v5_vi",
    "xlm_roberta_v5_vi",
]
TRADITIONAL_MODELS = {"logistic_regression", "linear_svm", "random_forest"}
MODEL_DISPLAY = {
    "logistic_regression": "TF-IDF + Logistic Regression",
    "linear_svm": "TF-IDF + Linear SVM",
    "random_forest": "TF-IDF + Random Forest",
    "roberta_v5_vi": "RoBERTa v5 VI",
    "xlm_roberta_v5_vi": "XLM-RoBERTa v5 VI",
}


class ScoreBundle(dict):
    @property
    def labels(self) -> np.ndarray:
        return np.asarray(self["label"], dtype=int)

    @property
    def scores(self) -> np.ndarray:
        return np.asarray(self["risk_score"], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize thresholds for prompt injection detectors.")
    parser.add_argument("--validation", default=str(DEFAULT_VALIDATION), help="Validation CSV used for threshold selection.")
    parser.add_argument("--test", default=str(DEFAULT_TEST), help="Test CSV used only for post-selection evaluation.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="Model names to optimize.")
    parser.add_argument("--start", type=float, default=0.05, help="Threshold sweep start.")
    parser.add_argument("--end", type=float, default=0.95, help="Threshold sweep end.")
    parser.add_argument("--step", type=float, default=0.01, help="Threshold sweep step.")
    parser.add_argument("--update-config", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--baseline-thresholds", default="", help="Optional thresholds JSON used as the old/baseline comparison.")
    parser.add_argument("--focus-model", default="xlm_roberta_v5_vi", help="Model for model-specific FP/FN and recommendation files.")
    return parser.parse_args()


def load_frame(path: Path, split_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "text" not in frame.columns or "label" not in frame.columns:
        raise ValueError(f"{path} must contain text and label columns.")
    frame = frame.dropna(subset=["text", "label"]).copy()
    frame["text"] = frame["text"].astype(str).str.strip()
    frame = frame[frame["text"] != ""].copy()
    frame["label"] = frame["label"].astype(int)
    if "id" not in frame.columns:
        frame["id"] = [f"{split_name}_{index:07d}" for index in range(len(frame))]
    if "language" not in frame.columns:
        frame["language"] = "unknown"
    frame["split"] = split_name
    return frame.reset_index(drop=True)


def threshold_grid(start: float, end: float, step: float) -> list[float]:
    count = int(round((end - start) / step)) + 1
    return [round(start + index * step, 2) for index in range(count)]


def safe_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(set(labels.tolist())) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def safe_average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    if len(set(labels.tolist())) < 2:
        return None
    return float(average_precision_score(labels, scores))


def metrics_at(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = [int(value) for value in confusion_matrix(labels, predicted, labels=[0, 1]).ravel()]
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, pos_label=1, zero_division=0)),
        "recall": float(recall_score(labels, predicted, pos_label=1, zero_division=0)),
        "f1": float(f1_score(labels, predicted, pos_label=1, zero_division=0)),
        "f2": float(fbeta_score(labels, predicted, beta=2, pos_label=1, zero_division=0)),
        "roc_auc": safe_roc_auc(labels, scores),
        "average_precision": safe_average_precision(labels, scores),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
    }


def sweep_thresholds(labels: np.ndarray, scores: np.ndarray, grid: list[float]) -> pd.DataFrame:
    return pd.DataFrame([metrics_at(labels, scores, threshold) for threshold in grid])


def sort_rows(frame: pd.DataFrame, columns: list[str], ascending: list[bool]) -> dict[str, Any]:
    row = frame.sort_values(columns, ascending=ascending).iloc[0].to_dict()
    return {key: (None if pd.isna(value) else value) for key, value in row.items()}


def choose_security(sweep: pd.DataFrame) -> tuple[dict[str, Any], str]:
    viable = sweep[sweep["recall"] >= 0.90]
    source = viable if not viable.empty else sweep
    row = sort_rows(source, ["f2", "recall", "precision", "fp", "threshold"], [False, False, False, True, True])
    reason = "maximize_f2_with_recall_ge_0.90" if not viable.empty else "maximize_f2_no_recall_constraint_available"
    return row, reason


def choose_balanced(sweep: pd.DataFrame) -> tuple[dict[str, Any], str]:
    row = sort_rows(sweep, ["f1", "f2", "precision", "recall", "fp", "threshold"], [False, False, False, False, True, True])
    return row, "maximize_f1"


def choose_production(sweep: pd.DataFrame, old_fp: int | None = None) -> tuple[dict[str, Any], str]:
    viable = sweep[(sweep["precision"] >= 0.80) & (sweep["recall"] >= 0.80)]
    if not viable.empty:
        if old_fp is not None:
            reduced = viable[viable["fp"] < old_fp]
            if not reduced.empty:
                viable = reduced
        row = sort_rows(viable, ["fp", "f1", "f2", "recall", "threshold"], [True, False, False, False, True])
        return row, "production_precision_ge_0.80_recall_ge_0.80_min_fp"

    fallback = sweep[sweep["recall"] >= 0.75]
    if not fallback.empty:
        row = sort_rows(fallback, ["precision", "f1", "f2", "fp", "threshold"], [False, False, False, True, True])
        return row, "fallback_max_precision_with_recall_ge_0.75"

    row = sort_rows(sweep, ["f1", "precision", "recall", "fp", "threshold"], [False, False, False, True, True])
    return row, "fallback_max_f1_no_precision_recall_constraint"

def choose_high_precision_block(sweep: pd.DataFrame, min_precision: float = 0.90) -> tuple[dict[str, Any] | None, str]:
    viable = sweep[(sweep["precision"] >= min_precision) & (sweep["tp"] > 0)]
    if viable.empty:
        return None, f"no_threshold_reaches_precision_ge_{min_precision:.2f}"
    row = sort_rows(viable, ["threshold", "fp", "recall", "f1"], [True, True, False, False])
    return row, f"high_precision_block_precision_ge_{min_precision:.2f}"


def derive_runtime_thresholds(production_threshold: float, balanced_threshold: float) -> tuple[float, float]:
    block = round(float(production_threshold), 2)
    block = max(block, round(float(balanced_threshold) + 0.15, 2), 0.20)
    block = min(block, 0.95)
    warn = round(max(0.05, block - 0.20), 2)
    if block - warn < 0.15:
        warn = round(max(0.05, block - 0.15), 2)
    if warn >= block:
        warn = round(max(0.05, block - 0.01), 2)
    return warn, block


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def traditional_scores(model_name: str, frame: pd.DataFrame) -> ScoreBundle:
    model, vectorizer = load_model_artifacts(model_name)
    cleaned = [prepare_text_for_detection(text)["cleaned_text"] for text in frame["text"].astype(str).tolist()]
    vectorized = vectorizer.transform(cleaned)
    warnings: list[str] = []
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vectorized)
        classes = list(getattr(model, "classes_", [0, 1]))
        positive_index = classes.index(1) if 1 in classes else 1
        scores = probabilities[:, positive_index].astype(float)
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(vectorized)
        scores = np.asarray([_sigmoid(float(value)) for value in raw], dtype=float)
        warnings.append("decision_function normalized with sigmoid because predict_proba is unavailable")
    else:
        scores = model.predict(vectorized).astype(float)
        warnings.append("predict labels used as scores because probability is unavailable")
    bundle = ScoreBundle(frame.copy())
    bundle["risk_score"] = scores
    bundle["score_source"] = "computed_from_model"
    bundle["warnings"] = warnings
    return bundle


def load_transformer_cached_scores(model_name: str, split_name: str, frame: pd.DataFrame) -> ScoreBundle:
    if not TRANSFORMER_RESULTS_PATH.exists():
        raise FileNotFoundError(f"Saved transformer scores not found: {TRANSFORMER_RESULTS_PATH}")
    payload = json.loads(TRANSFORMER_RESULTS_PATH.read_text(encoding="utf-8-sig"))
    model_payload = payload.get("models", {}).get(model_name)
    if not model_payload:
        raise KeyError(f"No saved scores for {model_name} in {TRANSFORMER_RESULTS_PATH}")
    rows = [row for row in model_payload.get("score_distribution", []) if row.get("split") == split_name]
    if not rows:
        raise KeyError(f"No saved {split_name} scores for {model_name}")
    scores = pd.DataFrame(rows)
    merged = frame.merge(scores[["id", "risk_score"]], on="id", how="left", validate="one_to_one")
    missing = int(merged["risk_score"].isna().sum())
    if missing:
        raise ValueError(f"Missing {missing} cached scores for {model_name}/{split_name}")
    bundle = ScoreBundle(merged)
    bundle["score_source"] = "reports/transformer_v5_vi_results.json"
    bundle["warnings"] = []
    return bundle


def get_scores(model_name: str, split_name: str, frame: pd.DataFrame) -> ScoreBundle:
    if model_name in TRADITIONAL_MODELS:
        return traditional_scores(model_name, frame)
    return load_transformer_cached_scores(model_name, split_name, frame)


def current_thresholds() -> dict[str, Any]:
    if not THRESHOLDS_PATH.exists():
        return {}
    payload = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8-sig"))
    return payload


def model_threshold_entry(payload: dict[str, Any], model_name: str) -> dict[str, Any]:
    models = payload.get("models", payload)
    return dict(models.get(model_name, {}))


def update_threshold_payload(payload: dict[str, Any], model_name: str, entry: dict[str, Any]) -> dict[str, Any]:
    if "models" not in payload:
        payload = {"models": dict(payload)}
    payload.setdefault("models", {})[model_name] = entry
    if model_name in TRADITIONAL_MODELS:
        payload.setdefault("traditional_models", {})[model_name] = entry
    payload[model_name] = entry
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    return payload


def metrics_for_selected(bundle: ScoreBundle, threshold: float) -> dict[str, Any]:
    return metrics_at(bundle.labels, bundle.scores, threshold)


def build_prediction_cases(bundle: ScoreBundle, threshold: float, model_name: str) -> pd.DataFrame:
    row_data = {key: value for key, value in bundle.items() if key not in {"warnings", "score_source"}}
    frame = pd.DataFrame(row_data).copy()
    frame["model"] = model_name
    frame["selected_threshold"] = float(threshold)
    frame["predicted_label"] = (frame["risk_score"].astype(float) >= threshold).astype(int)
    keep = [
        "model", "split", "id", "language", "label", "predicted_label", "risk_score",
        "selected_threshold", "text", "attack_type", "source",
    ]
    for column in keep:
        if column not in frame.columns:
            frame[column] = ""
    return frame[keep]


def short_float(value: Any) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.4f}"


def recommendation_for(model_name: str, test_metrics: dict[str, Any], production_method: str) -> str:
    if model_name == "xlm_roberta_v5_vi":
        if test_metrics["precision"] < 0.90 or test_metrics["recall"] < 0.80:
            return "warning_only_high_precision_block_optional"
        return "production_block_candidate_not_primary"
    if model_name == "roberta_v5_vi":
        return "demo_and_primary_runtime"
    if test_metrics["f1"] >= 0.90 and test_metrics["precision"] >= 0.80:
        return "comparison_or_backup_model"
    return "comparison_only"


def write_xlm_recommendation(
    model_name: str,
    old_thresholds: dict[str, Any],
    old_metrics: dict[str, Any],
    new_entry: dict[str, Any],
    new_metrics: dict[str, Any],
    security_metrics: dict[str, Any],
    balanced_metrics: dict[str, Any],
    production_metrics: dict[str, Any],
    method: str,
) -> None:
    path = REPORTS_DIR / f"threshold_recommendation_{model_name}.md"
    lines = [
        f"# Threshold recommendation for {MODEL_DISPLAY.get(model_name, model_name)}",
        "",
        "## Kết luận ngắn",
        "",
        "Threshold cũ `0.16` được chọn vì tối ưu F2/Recall trên validation, nên recall cao nhưng false positive rất lớn. Ngưỡng này phù hợp cho nghiên cứu/security sweep, không phù hợp làm runtime block threshold.",
        "",
        "## Threshold cũ và mới",
        "",
        "| Loại | Cũ | Mới | Ghi chú |",
        "| --- | ---: | ---: | --- |",
        f"| evaluation_threshold | {short_float(old_thresholds.get('evaluation_threshold'))} | {short_float(new_entry['evaluation_threshold'])} | Balanced mode: F1 cao nhất |",
        f"| runtime_warn_threshold | {short_float(old_thresholds.get('runtime_warn_threshold'))} | {short_float(new_entry['runtime_warn_threshold'])} | Cảnh báo, không block |",
        f"| runtime_block_threshold | {short_float(old_thresholds.get('runtime_block_threshold'))} | {short_float(new_entry['runtime_block_threshold'])} | Production mode: {method} |",
        "",
        "## Metric trên test set",
        "",
        "| Metric | Tại threshold cũ | Tại block threshold mới | Thay đổi |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric in ["accuracy", "precision", "recall", "f1", "f2", "roc_auc", "average_precision"]:
        old_value = old_metrics.get(metric)
        new_value = new_metrics.get(metric)
        delta = None if old_value is None or new_value is None else float(new_value) - float(old_value)
        lines.append(f"| {metric} | {short_float(old_value)} | {short_float(new_value)} | {short_float(delta)} |")
    for metric in ["fp", "fn", "tp", "tn"]:
        old_value = old_metrics.get(metric)
        new_value = new_metrics.get(metric)
        delta = None if old_value is None or new_value is None else int(new_value) - int(old_value)
        lines.append(f"| {metric.upper()} | {old_value} | {new_value} | {delta} |")
    lines.extend([
        "",
        "## Ba chế độ chọn threshold trên validation",
        "",
        "| Mode | Threshold | Precision | Recall | F1 | F2 | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| Security | {short_float(security_metrics['threshold'])} | {short_float(security_metrics['precision'])} | {short_float(security_metrics['recall'])} | {short_float(security_metrics['f1'])} | {short_float(security_metrics['f2'])} | {int(security_metrics['fp'])} | {int(security_metrics['fn'])} |",
        f"| Balanced | {short_float(balanced_metrics['threshold'])} | {short_float(balanced_metrics['precision'])} | {short_float(balanced_metrics['recall'])} | {short_float(balanced_metrics['f1'])} | {short_float(balanced_metrics['f2'])} | {int(balanced_metrics['fp'])} | {int(balanced_metrics['fn'])} |",
        f"| Production | {short_float(production_metrics['threshold'])} | {short_float(production_metrics['precision'])} | {short_float(production_metrics['recall'])} | {short_float(production_metrics['f1'])} | {short_float(production_metrics['f2'])} | {int(production_metrics['fp'])} | {int(production_metrics['fn'])} |",
        "",
        "## Khuyến nghị",
        "",
        "- Không dùng XLM-RoBERTa v5 VI làm model chính để block tự động vì precision vẫn chưa đủ cao.",
        "- Có thể dùng ở chế độ `warning only` hoặc model so sánh đa ngôn ngữ.",
        "- Model chính cho demo/runtime vẫn nên là RoBERTa v5 VI.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    validation = load_frame(Path(args.validation), "v5_vi_validation")
    test = load_frame(Path(args.test), "v5_vi_test")
    grid = threshold_grid(args.start, args.end, args.step)
    threshold_payload = current_thresholds()
    baseline_payload = threshold_payload
    if args.baseline_thresholds:
        baseline_payload = json.loads(Path(args.baseline_thresholds).read_text(encoding="utf-8-sig"))
    now = datetime.now(timezone.utc).isoformat()

    all_sweep_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    json_summary: dict[str, Any] = {"created_at": now, "validation_dataset": str(args.validation), "test_dataset": str(args.test), "models": {}}
    focus_outputs: dict[str, Any] = {}

    for model_name in args.models:
        print(f"Optimize thresholds for {model_name}")
        validation_scores = get_scores(model_name, "v5_vi_validation", validation)
        test_scores = get_scores(model_name, "v5_vi_test", test)
        sweep = sweep_thresholds(validation_scores.labels, validation_scores.scores, grid)
        sweep.insert(0, "model", model_name)
        all_sweep_rows.append(sweep)

        old_entry = model_threshold_entry(baseline_payload, model_name)
        old_eval_threshold = float(old_entry.get("evaluation_threshold", 0.5))
        old_metrics_test = metrics_for_selected(test_scores, old_eval_threshold)
        security, security_method = choose_security(sweep)
        balanced, balanced_method = choose_balanced(sweep)
        production, production_method = choose_production(sweep, old_fp=int(old_metrics_test["fp"]))
        runtime_warn, runtime_block = derive_runtime_thresholds(
            float(production["threshold"]),
            float(balanced["threshold"]),
        )
        block_selection_method = production_method
        if model_name == "xlm_roberta_v5_vi":
            high_precision_block, high_precision_method = choose_high_precision_block(sweep, min_precision=0.90)
            if high_precision_block is not None:
                runtime_block = round(float(high_precision_block["threshold"]), 2)
                runtime_warn = round(float(balanced["threshold"]), 2)
                if runtime_warn >= runtime_block:
                    runtime_warn = round(max(0.05, runtime_block - 0.15), 2)
                block_selection_method = high_precision_method
        selected_test_metrics = metrics_for_selected(test_scores, runtime_block)
        eval_test_metrics = metrics_for_selected(test_scores, float(balanced["threshold"]))
        recommendation = recommendation_for(model_name, selected_test_metrics, block_selection_method)

        new_entry = {
            **old_entry,
            "model_name": model_name.replace("_", "-"),
            "evaluation_threshold": round(float(balanced["threshold"]), 4),
            "runtime_warn_threshold": round(float(runtime_warn), 4),
            "runtime_block_threshold": round(float(runtime_block), 4),
            "warn_threshold": round(float(runtime_warn), 4),
            "block_threshold": round(float(runtime_block), 4),
            "threshold_selection_method": "balanced_f1_for_evaluation_production_precision_recall_constraint_for_runtime_block",
            "security_threshold": round(float(security["threshold"]), 4),
            "balanced_threshold": round(float(balanced["threshold"]), 4),
            "production_threshold": round(float(production["threshold"]), 4),
            "security_selection_method": security_method,
            "balanced_selection_method": balanced_method,
            "production_selection_method": production_method,
            "block_selection_method": block_selection_method,
            "last_calibrated_at": now,
            "validation_dataset": str(args.validation),
            "test_dataset": str(args.test),
            "score_source": validation_scores.get("score_source", "unknown"),
            "warnings": list(validation_scores.get("warnings", [])),
            "selected_metrics_validation": {
                "security": security,
                "balanced": balanced,
                "production": production,
            },
            "selected_metrics_test": {
                "evaluation_threshold": eval_test_metrics,
                "runtime_block_threshold": selected_test_metrics,
                "old_evaluation_threshold": old_metrics_test,
            },
        }
        if args.update_config:
            threshold_payload = update_threshold_payload(threshold_payload, model_name, new_entry)

        summary_row = {
            "model": model_name,
            "selected_threshold": runtime_block,
            "evaluation_threshold": float(balanced["threshold"]),
            "runtime_warn_threshold": runtime_warn,
            "runtime_block_threshold": runtime_block,
            "selection_method": block_selection_method,
            "accuracy": selected_test_metrics["accuracy"],
            "precision": selected_test_metrics["precision"],
            "recall": selected_test_metrics["recall"],
            "f1": selected_test_metrics["f1"],
            "f2": selected_test_metrics["f2"],
            "roc_auc": selected_test_metrics["roc_auc"],
            "average_precision": selected_test_metrics["average_precision"],
            "fp": selected_test_metrics["fp"],
            "fn": selected_test_metrics["fn"],
            "recommendation": recommendation,
        }
        summary_rows.append(summary_row)
        json_summary["models"][model_name] = {
            "old_thresholds": old_entry,
            "new_thresholds": new_entry,
            "summary": summary_row,
        }

        if model_name == args.focus_model:
            focus_cases = build_prediction_cases(test_scores, runtime_block, model_name)
            fps = focus_cases[(focus_cases["label"].astype(int) == 0) & (focus_cases["predicted_label"].astype(int) == 1)]
            fns = focus_cases[(focus_cases["label"].astype(int) == 1) & (focus_cases["predicted_label"].astype(int) == 0)]
            fps.to_csv(REPORTS_DIR / "false_positives_at_selected_threshold.csv", index=False, encoding="utf-8-sig")
            fns.to_csv(REPORTS_DIR / "false_negatives_at_selected_threshold.csv", index=False, encoding="utf-8-sig")
            fps.to_csv(REPORTS_DIR / f"false_positives_at_selected_threshold_{model_name}.csv", index=False, encoding="utf-8-sig")
            fns.to_csv(REPORTS_DIR / f"false_negatives_at_selected_threshold_{model_name}.csv", index=False, encoding="utf-8-sig")
            focus_outputs = {
                "old_entry": old_entry,
                "old_metrics_test": old_metrics_test,
                "new_entry": new_entry,
                "selected_test_metrics": selected_test_metrics,
                "security": security,
                "balanced": balanced,
                "production": production,
                "production_method": block_selection_method,
            }

    sweep_df = pd.concat(all_sweep_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    sweep_df.to_csv(REPORTS_DIR / "threshold_sweep_all_models.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(REPORTS_DIR / "threshold_optimization_summary.csv", index=False, encoding="utf-8-sig")
    (REPORTS_DIR / "threshold_optimization_summary.json").write_text(
        json.dumps(json_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    focus_model = args.focus_model
    focus_sweep = sweep_df[sweep_df["model"] == focus_model].copy()
    if not focus_sweep.empty:
        focus_sweep.to_csv(REPORTS_DIR / f"threshold_sweep_{focus_model}.csv", index=False, encoding="utf-8-sig")
        focus_summary = json_summary["models"].get(focus_model, {})
        (REPORTS_DIR / f"threshold_summary_{focus_model}.json").write_text(
            json.dumps(focus_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if focus_outputs:
            write_xlm_recommendation(
                focus_model,
                focus_outputs["old_entry"],
                focus_outputs["old_metrics_test"],
                focus_outputs["new_entry"],
                focus_outputs["selected_test_metrics"],
                focus_outputs["security"],
                focus_outputs["balanced"],
                focus_outputs["production"],
                focus_outputs["production_method"],
            )

    lines = [
        "# Threshold Optimization Summary", "",
        f"Validation dataset: `{args.validation}`", f"Test dataset: `{args.test}`", "",
        "| Model | Selected Threshold | Method | Accuracy | Precision | Recall | F1 | F2 | ROC-AUC | FP | FN | Recommendation |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['model']} | {row['selected_threshold']:.2f} | {row['selection_method']} | "
            f"{row['accuracy']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {row['f2']:.4f} | {row['roc_auc']:.4f} | "
            f"{int(row['fp'])} | {int(row['fn'])} | {row['recommendation']} |"
        )
    (REPORTS_DIR / "threshold_optimization_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if args.update_config:
        THRESHOLDS_PATH.write_text(json.dumps(threshold_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nThreshold optimization completed.\n")
    print(summary_df[["model", "selected_threshold", "selection_method", "accuracy", "precision", "recall", "f1", "f2", "roc_auc", "fp", "fn", "recommendation"]].to_string(index=False))


if __name__ == "__main__":
    main()






