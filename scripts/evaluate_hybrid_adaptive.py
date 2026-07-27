"""Evaluate Adaptive Risk Fusion + Decision Policy Engine on external benchmarks.

The evaluator reuses already-computed Transformer predictions whenever possible:

- calibrated predictions from direct external calibration reports
- raw Transformer predictions plus runtime probability calibrators as fallback
- BIPIA ablation model scores plus runtime probability calibrators

It does not use Logistic Regression, Linear SVM or Random Forest in the runtime
hybrid configuration. Those models are intentionally left as research baselines.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
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
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration_runtime import apply_probability_calibrator, load_runtime_calibrator
from src.detection.adaptive_risk_fusion import AdaptiveRiskFusion
from src.detection.policy_engine import DecisionPolicyEngine
from src.rule_based import detect_by_rules


REPORT_ROOT = PROJECT_ROOT / "reports" / "hybrid_adaptive_evaluation"

DATASETS: dict[str, dict[str, Any]] = {
    "direct_all": {
        "path": PROJECT_ROOT / "data" / "external_benchmark" / "direct" / "direct_all_normalized.csv",
        "calibration_key": "all",
        "text_column": "text",
        "source_type": "user_prompt",
        "description": "Direct Prompt Injection External Benchmark - all normalized datasets.",
    },
    "deepset": {
        "path": PROJECT_ROOT / "data" / "external_benchmark" / "direct" / "deepset_normalized.csv",
        "calibration_key": "deepset",
        "text_column": "text",
        "source_type": "user_prompt",
        "description": "Deepset prompt-injection benchmark subset.",
    },
    "bipia": {
        "path": PROJECT_ROOT / "data" / "external_benchmark" / "bipia" / "bipia_normalized.csv",
        "calibration_key": "bipia",
        "text_column": "external_content",
        "task_column": "user_task",
        "source_type": "external_content",
        "description": "BIPIA external-content benchmark.",
    },
}

MODEL_KEYS = {
    "roberta": "roberta",
    "xlm_roberta": "xlm_roberta",
}

ABLATION_CONFIGS = [
    ("rule_only", "Rule-only", "No"),
    ("roberta_only", "RoBERTa-only", "No"),
    ("xlm_only", "XLM-R-only", "No"),
    ("roberta_xlm_weighted", "RoBERTa + XLM-R weighted fusion", "No"),
    ("adaptive_fusion", "Rule + RoBERTa + XLM-R adaptive fusion", "Yes"),
    ("adaptive_fusion_policy", "Adaptive fusion + policy engine", "Yes"),
]


def _as_int_label(value: Any) -> int:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "attack", "unsafe", "injection", "malicious"}:
            return 1
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return default
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _threshold_decision(score: float) -> tuple[str, str]:
    if score >= 0.70:
        return "BLOCK", "high"
    if score >= 0.30:
        return "WARN", "medium"
    return "SAFE", "safe"


def _compute_metrics(y_true: list[int], y_pred: list[int], scores: list[float]) -> dict[str, Any]:
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    balanced_accuracy = (recall + specificity) / 2.0

    try:
        roc_auc = roc_auc_score(y_true, scores) if len(set(y_true)) > 1 else None
    except ValueError:
        roc_auc = None
    try:
        pr_auc = average_precision_score(y_true, scores) if len(set(y_true)) > 1 else None
    except ValueError:
        pr_auc = None

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "f2": round(float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)), 6),
        "roc_auc": None if roc_auc is None else round(float(roc_auc), 6),
        "pr_auc": None if pr_auc is None else round(float(pr_auc), 6),
        "specificity_tnr": round(float(specificity), 6),
        "fpr": round(float(fpr), 6),
        "fnr": round(float(fnr), 6),
        "balanced_accuracy": round(float(balanced_accuracy), 6),
        "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 6) if len(set(y_pred)) > 1 else 0.0,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "false_positive_count": int(fp),
        "false_negative_count": int(fn),
        "support": {
            "total": len(y_true),
            "positive": int(sum(y_true)),
            "negative": int(len(y_true) - sum(y_true)),
        },
    }


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _apply_calibrator_if_available(model_key: str, raw_score: float) -> tuple[float, str]:
    calibrator = load_runtime_calibrator(model_key)
    if calibrator is None:
        return raw_score, "raw_probability_no_calibrator"
    return float(apply_probability_calibrator(calibrator, raw_score)), "calibrated_from_raw_prediction"


def _score_column(df: pd.DataFrame) -> str:
    for column in ["calibrated_score", "model_score", "score", "final_score", "risk_score"]:
        if column in df.columns:
            return column
    raise ValueError(f"Không tìm thấy cột score trong dataframe: {list(df.columns)}")


def _load_direct_scores(dataset_key: str, model_key: str) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    calibration_key = DATASETS[dataset_key]["calibration_key"]
    calibrated_path = (
        PROJECT_ROOT
        / "reports"
        / "direct_external_evaluation"
        / "calibration"
        / calibration_key
        / model_key
        / "calibrated_predictions.csv"
    )
    raw_path = (
        PROJECT_ROOT
        / "reports"
        / "direct_external_evaluation"
        / model_key
        / ("deepset" if dataset_key == "deepset" else "")
        / "predictions.csv"
    )
    if dataset_key != "deepset":
        raw_path = PROJECT_ROOT / "reports" / "direct_external_evaluation" / model_key / "predictions.csv"

    frames: list[pd.DataFrame] = []
    calibrated = _read_csv_if_exists(calibrated_path)
    if calibrated is not None:
        frames.append(
            pd.DataFrame(
                {
                    "id": calibrated["id"].astype(str),
                    f"{model_key}_score": calibrated["calibrated_score"].astype(float),
                    f"{model_key}_raw_score": calibrated.get("raw_score", calibrated["calibrated_score"]).astype(float),
                    f"{model_key}_score_used": "calibrated_probability",
                }
            )
        )
    else:
        warnings.append(f"Thiếu calibrated_predictions.csv cho {dataset_key}/{model_key}: {calibrated_path}")

    raw = _read_csv_if_exists(raw_path)
    if raw is not None:
        raw_col = _score_column(raw)
        raw_rows = []
        for _, row in raw.iterrows():
            raw_score = _safe_float(row[raw_col])
            score, score_used = _apply_calibrator_if_available(model_key, raw_score)
            raw_rows.append(
                {
                    "id": str(row["id"]),
                    f"{model_key}_score": score,
                    f"{model_key}_raw_score": raw_score,
                    f"{model_key}_score_used": score_used,
                }
            )
        frames.append(pd.DataFrame(raw_rows))
    else:
        warnings.append(f"Thiếu raw predictions.csv cho {dataset_key}/{model_key}: {raw_path}")

    if not frames:
        raise FileNotFoundError(f"Không có score Transformer cho {dataset_key}/{model_key}.")

    merged = pd.concat(frames, ignore_index=True)
    # Keep calibrated rows first when both calibrated and calibrated-from-raw exist.
    merged = merged.drop_duplicates(subset=["id"], keep="first")
    return merged, warnings


def _load_bipia_scores(model_key: str) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    model_dir = "roberta" if model_key == "roberta" else "xlm_roberta"
    path = PROJECT_ROOT / "reports" / "bipia_evaluation" / model_dir / "predictions.csv"
    df = _read_csv_if_exists(path)
    if df is None:
        raise FileNotFoundError(f"Không có BIPIA predictions cho {model_key}: {path}")
    score_col = _score_column(df)
    rows = []
    for _, row in df.iterrows():
        raw_score = _safe_float(row[score_col])
        score, score_used = _apply_calibrator_if_available(model_key, raw_score)
        rows.append(
            {
                "id": str(row["id"]),
                f"{model_key}_score": score,
                f"{model_key}_raw_score": raw_score,
                f"{model_key}_score_used": score_used,
            }
        )
    if not rows:
        warnings.append(f"BIPIA predictions rỗng cho {model_key}: {path}")
    return pd.DataFrame(rows).drop_duplicates(subset=["id"], keep="first"), warnings


def _load_model_scores(dataset_key: str) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    score_frames = []
    for model_key in MODEL_KEYS:
        if dataset_key == "bipia":
            frame, model_warnings = _load_bipia_scores(model_key)
        else:
            frame, model_warnings = _load_direct_scores(dataset_key, model_key)
        warnings.extend(model_warnings)
        score_frames.append(frame)

    scores = score_frames[0]
    for frame in score_frames[1:]:
        scores = scores.merge(frame, on="id", how="outer")
    return scores, warnings


def _row_text(row: pd.Series, config: dict[str, Any]) -> str:
    text_column = config["text_column"]
    return "" if pd.isna(row.get(text_column)) else str(row.get(text_column, ""))


def _row_language(row: pd.Series) -> str:
    value = row.get("language", "unknown")
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "unknown"
    normalized = str(value).strip().lower()
    if normalized in {"en", "vi", "mixed"}:
        return normalized
    return "unknown"


def _decide_ablation(
    *,
    ablation_key: str,
    language: str,
    source_type: str,
    rule_score: float,
    roberta_score: float,
    xlm_score: float,
    rule_result: dict[str, Any],
    fusion: dict[str, Any],
    policy: dict[str, Any],
) -> tuple[int, float, str, str]:
    if ablation_key == "rule_only":
        risk = rule_score
        decision, risk_level = _threshold_decision(risk)
    elif ablation_key == "roberta_only":
        risk = roberta_score
        decision, risk_level = _threshold_decision(risk)
    elif ablation_key == "xlm_only":
        risk = xlm_score
        decision, risk_level = _threshold_decision(risk)
    elif ablation_key == "roberta_xlm_weighted":
        model_only = AdaptiveRiskFusion().fuse(
            language=language,
            source_type=source_type,
            rule_score=0.0,
            rule_matches=[],
            roberta_score=roberta_score,
            xlm_score=xlm_score,
            scores_are_calibrated=True,
            highest_severity="none",
            has_high_severity_rule=False,
            has_critical_rule=False,
        )
        risk = float(model_only["model_risk"])
        decision, risk_level = _threshold_decision(risk)
    elif ablation_key == "adaptive_fusion":
        risk = float(fusion["final_risk"])
        decision, risk_level = _threshold_decision(risk)
    elif ablation_key == "adaptive_fusion_policy":
        risk = float(fusion["final_risk"])
        decision = str(policy["decision"])
        risk_level = str(policy["risk_level"])
    else:  # pragma: no cover - protected by fixed config list
        raise ValueError(f"Unknown ablation config: {ablation_key}")

    return (0 if decision == "SAFE" else 1), round(float(risk), 6), decision, risk_level


def evaluate_dataset(dataset_key: str, limit: int | None = None) -> dict[str, Any]:
    if dataset_key not in DATASETS:
        raise ValueError(f"Dataset không hợp lệ: {dataset_key}. Hỗ trợ: {sorted(DATASETS)}")
    config = DATASETS[dataset_key]
    dataset_path = Path(config["path"])
    if not dataset_path.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset: {dataset_path}")

    dataset = pd.read_csv(dataset_path)
    if limit is not None:
        dataset = dataset.head(limit).copy()

    scores, score_warnings = _load_model_scores(dataset_key)
    dataset["id"] = dataset["id"].astype(str)
    scores["id"] = scores["id"].astype(str)
    merged = dataset.merge(scores, on="id", how="left")
    missing_roberta = int(merged["roberta_score"].isna().sum())
    missing_xlm = int(merged["xlm_roberta_score"].isna().sum())
    if missing_roberta:
        score_warnings.append(f"Thiếu RoBERTa score cho {missing_roberta} dòng; gán score=0 và đánh dấu missing.")
    if missing_xlm:
        score_warnings.append(f"Thiếu XLM-R score cho {missing_xlm} dòng; gán score=0 và đánh dấu missing.")

    output_dir = REPORT_ROOT / dataset_key
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_rows: list[dict[str, Any]] = []
    ablation_records: dict[str, dict[str, list[Any]]] = {
        key: {"y_true": [], "y_pred": [], "scores": []}
        for key, _, _ in ABLATION_CONFIGS
    }

    fusion_engine = AdaptiveRiskFusion()
    policy_engine = DecisionPolicyEngine()
    source_type = str(config["source_type"])

    for _, row in merged.iterrows():
        text = _row_text(row, config)
        language = _row_language(row)
        label = _as_int_label(row.get("label", 0))
        roberta_score = _safe_float(row.get("roberta_score"))
        xlm_score = _safe_float(row.get("xlm_roberta_score"))

        rule_result = detect_by_rules(text, source_type=source_type)
        rule_score = _safe_float(rule_result.get("rule_score", rule_result.get("risk_score", 0.0)))
        scores_are_calibrated = (
            str(row.get("roberta_score_used", "")).startswith("calibrated")
            and str(row.get("xlm_roberta_score_used", "")).startswith("calibrated")
        )

        fusion = fusion_engine.fuse(
            text=text,
            language=language,
            source_type=source_type,
            rule_score=rule_score,
            rule_matches=rule_result.get("matched_rules", []),  # type: ignore[arg-type]
            roberta_score=roberta_score,
            xlm_score=xlm_score,
            scores_are_calibrated=scores_are_calibrated,
            highest_severity=str(rule_result.get("highest_severity", "none")),
            has_high_severity_rule=bool(rule_result.get("has_high_severity_rule", False)),
            has_critical_rule=bool(rule_result.get("has_critical_rule", False)),
        )
        policy = policy_engine.decide(
            final_risk=float(fusion["final_risk"]),
            model_risk=float(fusion["model_risk"]),
            rule_score=rule_score,
            roberta_score=roberta_score,
            xlm_score=xlm_score,
            highest_severity=str(rule_result.get("highest_severity", "none")),
            has_high_severity_rule=bool(rule_result.get("has_high_severity_rule", False)),
            has_critical_rule=bool(rule_result.get("has_critical_rule", False)),
            source_type=source_type,
            language=language,
            rule_matches=rule_result.get("matched_rules", []),  # type: ignore[arg-type]
            weights=fusion["weights"],
            fusion_method=str(fusion["fusion_method"]),
            scores_are_calibrated=scores_are_calibrated,
            benign_reference_intent=bool(
                isinstance(rule_result.get("benign_guard"), dict)
                and rule_result.get("benign_guard", {}).get("triggered")
            ),
        )

        main_pred = 0 if policy["decision"] == "SAFE" else 1
        for ablation_key, _, _ in ABLATION_CONFIGS:
            pred, risk, _, _ = _decide_ablation(
                ablation_key=ablation_key,
                language=language,
                source_type=source_type,
                rule_score=rule_score,
                roberta_score=roberta_score,
                xlm_score=xlm_score,
                rule_result=rule_result,
                fusion=fusion,
                policy=policy,
            )
            ablation_records[ablation_key]["y_true"].append(label)
            ablation_records[ablation_key]["y_pred"].append(pred)
            ablation_records[ablation_key]["scores"].append(risk)

        prediction_rows.append(
            {
                "id": row["id"],
                "text": text,
                "label": label,
                "predicted_label": main_pred,
                "risk_level": policy["risk_level"],
                "decision": policy["decision"],
                "final_risk": fusion["final_risk"],
                "model_risk": fusion["model_risk"],
                "rule_score": rule_score,
                "roberta_score": roberta_score,
                "xlm_score": xlm_score,
                "roberta_raw_score": row.get("roberta_raw_score"),
                "xlm_raw_score": row.get("xlm_roberta_raw_score"),
                "roberta_score_used": row.get("roberta_score_used"),
                "xlm_score_used": row.get("xlm_roberta_score_used"),
                "language": language,
                "source_type": source_type,
                "weights": json.dumps(fusion["weights"], ensure_ascii=False),
                "highest_rule_severity": rule_result.get("highest_severity", "none"),
                "decision_policy": policy["decision_policy"],
                "reasons": json.dumps(policy["reasons"], ensure_ascii=False),
                "fusion_reasons": json.dumps(fusion["reasons"], ensure_ascii=False),
                "matched_rules": json.dumps(rule_result.get("matched_rules", []), ensure_ascii=False),
                "attack_type": row.get("attack_type", ""),
                "source_task": row.get("source_task", row.get("dataset_name", "")),
                "user_task": row.get(config.get("task_column", ""), ""),
            }
        )

    predictions = pd.DataFrame(prediction_rows)
    y_true = predictions["label"].astype(int).tolist()
    y_pred = predictions["predicted_label"].astype(int).tolist()
    y_score = predictions["final_risk"].astype(float).tolist()
    metrics = _compute_metrics(y_true, y_pred, y_score)

    ablation_metrics = {}
    for ablation_key, label_name, context_aware in ABLATION_CONFIGS:
        record = ablation_records[ablation_key]
        ablation_metrics[ablation_key] = {
            "model": label_name,
            "context_aware": context_aware,
            **_compute_metrics(
                [int(v) for v in record["y_true"]],
                [int(v) for v in record["y_pred"]],
                [float(v) for v in record["scores"]],
            ),
        }

    predictions_path = output_dir / "predictions.csv"
    metrics_path = output_dir / "metrics.json"
    fp_path = output_dir / "false_positives.csv"
    fn_path = output_dir / "false_negatives.csv"
    report_path = output_dir / "evaluation_report_vi.md"

    predictions.to_csv(predictions_path, index=False)
    predictions[(predictions["label"] == 0) & (predictions["predicted_label"] == 1)].to_csv(fp_path, index=False)
    predictions[(predictions["label"] == 1) & (predictions["predicted_label"] == 0)].to_csv(fn_path, index=False)

    score_source_summary = {
        "roberta": predictions["roberta_score_used"].fillna("missing").value_counts().to_dict(),
        "xlm_roberta": predictions["xlm_score_used"].fillna("missing").value_counts().to_dict(),
    }
    payload = {
        "dataset": dataset_key,
        "dataset_path": str(dataset_path),
        "description": config["description"],
        "limit": limit,
        "source_type": source_type,
        "runtime_signals": ["rule_based", "roberta", "xlm_roberta"],
        "metrics": metrics,
        "ablation_metrics": ablation_metrics,
        "score_source_summary": score_source_summary,
        "warnings": score_warnings,
        "output_files": {
            "predictions": str(predictions_path),
            "false_positives": str(fp_path),
            "false_negatives": str(fn_path),
            "report": str(report_path),
        },
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_build_dataset_report(payload), encoding="utf-8")

    _write_global_reports()
    return payload


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _build_dataset_report(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    cm = metrics["confusion_matrix"]
    warning_lines = "\n".join(f"- {warning}" for warning in payload["warnings"]) or "- Không có warning."
    ablation_lines = []
    for key, label_name, context_aware in ABLATION_CONFIGS:
        item = payload["ablation_metrics"][key]
        ablation_lines.append(
            "| {model} | {context} | {acc} | {prec} | {rec} | {f1} | {f2} | {roc} | {pr} | {fp} | {fn} |".format(
                model=label_name,
                context=context_aware,
                acc=_fmt(item["accuracy"]),
                prec=_fmt(item["precision"]),
                rec=_fmt(item["recall"]),
                f1=_fmt(item["f1"]),
                f2=_fmt(item["f2"]),
                roc=_fmt(item["roc_auc"]),
                pr=_fmt(item["pr_auc"]),
                fp=item["false_positive_count"],
                fn=item["false_negative_count"],
            )
        )

    return f"""# Báo cáo Hybrid Adaptive Evaluation - {payload['dataset']}

## Cấu hình

- Dataset: `{payload['dataset_path']}`
- Mô tả: {payload['description']}
- Limit: {payload['limit'] if payload['limit'] is not None else 'Không giới hạn'}
- Source type runtime: `{payload['source_type']}`
- Runtime signals: Rule-based + RoBERTa + XLM-RoBERTa
- Không dùng Logistic Regression / Random Forest trong runtime hybrid.

## Metrics chính - Adaptive Fusion + Policy Engine

| Metric | Giá trị |
| --- | ---: |
| Accuracy | {_fmt(metrics['accuracy'])} |
| Precision | {_fmt(metrics['precision'])} |
| Recall | {_fmt(metrics['recall'])} |
| F1-score | {_fmt(metrics['f1'])} |
| F2-score | {_fmt(metrics['f2'])} |
| ROC-AUC | {_fmt(metrics['roc_auc'])} |
| PR-AUC | {_fmt(metrics['pr_auc'])} |
| Specificity/TNR | {_fmt(metrics['specificity_tnr'])} |
| FPR | {_fmt(metrics['fpr'])} |
| FNR | {_fmt(metrics['fnr'])} |
| Balanced Accuracy | {_fmt(metrics['balanced_accuracy'])} |
| MCC | {_fmt(metrics['mcc'])} |
| False Positive | {metrics['false_positive_count']} |
| False Negative | {metrics['false_negative_count']} |

Confusion Matrix: TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']}.

## Bảng ablation

| Model | Context-aware/Policy | Accuracy | Precision | Recall | F1 | F2 | ROC-AUC | PR-AUC | FP | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(ablation_lines)}

## Nguồn score

```json
{json.dumps(payload['score_source_summary'], ensure_ascii=False, indent=2)}
```

## Warning / lưu ý vận hành

{warning_lines}
"""


def _load_completed_payloads() -> list[dict[str, Any]]:
    payloads = []
    for metrics_path in sorted(REPORT_ROOT.glob("*/metrics.json")):
        try:
            payloads.append(json.loads(metrics_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return payloads


def _write_global_reports() -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    payloads = _load_completed_payloads()
    if not payloads:
        return

    ablation_lines = []
    for payload in payloads:
        for key, label_name, context_aware in ABLATION_CONFIGS:
            item = payload["ablation_metrics"][key]
            ablation_lines.append(
                "| {dataset} | {model} | {context} | {acc} | {prec} | {rec} | {f1} | {f2} | {roc} | {pr} | {fp} | {fn} |".format(
                    dataset=payload["dataset"],
                    model=label_name,
                    context=context_aware,
                    acc=_fmt(item["accuracy"]),
                    prec=_fmt(item["precision"]),
                    rec=_fmt(item["recall"]),
                    f1=_fmt(item["f1"]),
                    f2=_fmt(item["f2"]),
                    roc=_fmt(item["roc_auc"]),
                    pr=_fmt(item["pr_auc"]),
                    fp=item["false_positive_count"],
                    fn=item["false_negative_count"],
                )
            )

    summary = f"""# Hybrid Adaptive Ablation Summary

| Dataset | Model | Context-aware/Policy | Accuracy | Precision | Recall | F1 | F2 | ROC-AUC | PR-AUC | FP | FN |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(ablation_lines)}
"""
    (REPORT_ROOT / "hybrid_ablation_summary_vi.md").write_text(summary, encoding="utf-8")

    main_lines = []
    for payload in payloads:
        metrics = payload["metrics"]
        main_lines.append(
            f"| {payload['dataset']} | {_fmt(metrics['accuracy'])} | {_fmt(metrics['precision'])} | {_fmt(metrics['recall'])} | {_fmt(metrics['f1'])} | {_fmt(metrics['f2'])} | {_fmt(metrics['roc_auc'])} | {_fmt(metrics['pr_auc'])} | {metrics['false_positive_count']} | {metrics['false_negative_count']} |"
        )

    analysis = _build_final_analysis(payloads)
    report = f"""# Báo cáo Adaptive Risk Fusion + Decision Policy Engine

## Tóm tắt cấu hình runtime

Runtime hybrid mới chỉ sử dụng 3 tín hiệu: Rule-based, RoBERTa calibrated score, XLM-RoBERTa calibrated score.
Logistic Regression, Linear SVM và Random Forest không tham gia runtime hybrid; chúng chỉ còn là baseline nghiên cứu/đánh giá.

## Bảng tổng hợp chính

| Dataset | Accuracy | Precision | Recall | F1 | F2 | ROC-AUC | PR-AUC | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(main_lines)}

## Phân tích cuối

{analysis}
"""
    (REPORT_ROOT / "adaptive_risk_fusion_policy_engine_report_vi.md").write_text(report, encoding="utf-8")


def _build_final_analysis(payloads: list[dict[str, Any]]) -> str:
    by_dataset = {payload["dataset"]: payload for payload in payloads}
    lines = []
    if "bipia" in by_dataset:
        bipia = by_dataset["bipia"]
        roberta = bipia["ablation_metrics"]["roberta_only"]
        xlm = bipia["ablation_metrics"]["xlm_only"]
        policy = bipia["metrics"]
        lines.append(
            f"1. Trên BIPIA, RoBERTa-only đạt F1={_fmt(roberta['f1'])}, Recall={_fmt(roberta['recall'])}. "
            "Đây là chỉ báo generalization trực tiếp của RoBERTa khi không có rule/policy hỗ trợ."
        )
        better = "XLM-RoBERTa" if xlm["f1"] >= roberta["f1"] else "RoBERTa"
        lines.append(
            f"2. So sánh BIPIA: XLM-R F1={_fmt(xlm['f1'])}, RoBERTa F1={_fmt(roberta['f1'])}; "
            f"model đang nhỉnh hơn theo F1 là {better}."
        )
        adaptive = bipia["ablation_metrics"]["adaptive_fusion"]
        lines.append(
            f"3. Context/rule adaptive fusion trước policy có FP={adaptive['false_positive_count']}, FN={adaptive['false_negative_count']}; "
            f"sau Policy Engine có FP={policy['false_positive_count']}, FN={policy['false_negative_count']}. "
            "Chênh lệch này cho biết policy đang giảm/làm tăng loại lỗi nào."
        )
    else:
        lines.append("1. Chưa có kết quả BIPIA; chạy `--dataset bipia --limit 1000` để phân tích generalization.")

    lines.append(
        "4. Random Forest không nằm trong runtime hybrid mới. Khi cần so sánh truyền thống-vs-Transformer, dùng các báo cáo baseline/direct hoặc BIPIA ablation cũ."
    )
    lines.append(
        "5. Các mẫu cả Rule/RoBERTa/XLM-R đều thất bại có thể lọc từ `predictions.csv`: label=1, rule_score<0.30, roberta_score<0.30, xlm_score<0.30."
    )
    lines.append(
        "6. Hướng cải thiện tiếp theo: calibration riêng cho BIPIA/external-content, thêm language-aware threshold, và hard-negative mining cho các false positive có keyword benign."
    )
    return "\n\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = evaluate_dataset(args.dataset, limit=args.limit)
    metrics = payload["metrics"]
    print(
        f"[OK] {args.dataset}: accuracy={metrics['accuracy']}, precision={metrics['precision']}, "
        f"recall={metrics['recall']}, f1={metrics['f1']}, fp={metrics['false_positive_count']}, fn={metrics['false_negative_count']}"
    )
    print(f"[OK] Reports: {REPORT_ROOT / args.dataset}")


if __name__ == "__main__":
    main()
