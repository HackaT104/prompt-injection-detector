"""Evaluate the external-content indirect prompt injection pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.indirect_pipeline import detect_indirect_content


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate indirect prompt injection detection.")
    parser.add_argument("--dataset", default="datasets/test/indirect_test_cases.csv")
    parser.add_argument("--model", default="roberta")
    parser.add_argument("--output-dir", default="reports/indirect_evaluation")
    parser.add_argument("--use-cuda", action="store_true")
    return parser.parse_args()


def evaluate(dataset_path: Path, model_name: str, output_dir: Path, use_cuda: bool) -> dict:
    data = pd.read_csv(dataset_path)
    required = {"id", "user_task", "external_content", "label"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions: list[dict] = []
    for row in data.to_dict(orient="records"):
        started = time.perf_counter()
        result = detect_indirect_content(
            user_task=str(row["user_task"]),
            external_content=str(row["external_content"]),
            source_type=str(row.get("source_type") or "raw_text"),
            source_name=str(row.get("source_name") or row["id"]),
            model_name=model_name,
            safe_context_policy="exclude",
            use_cuda=use_cuda,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        predicted = int(result["recommended_action"] != "allow")
        predictions.append(
            {
                **row,
                "predicted_label": predicted,
                "final_score": result["final_score"],
                "rule_score": result["rule_score"],
                "model_score": result["model_score"],
                "context_score": result["context_score"],
                "recommended_action": result["recommended_action"],
                "risk_level": result["risk_level"],
                "matched_rule_count": len(result["matched_rules"]),
                "latency_ms": round(latency_ms, 3),
                "explanation": result["explanation"],
            }
        )

    predictions_df = pd.DataFrame(predictions)
    labels = predictions_df["label"].astype(int).to_numpy()
    predicted = predictions_df["predicted_label"].astype(int).to_numpy()
    scores = predictions_df["final_score"].astype(float).to_numpy()
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    metrics = {
        "dataset": str(dataset_path.resolve()),
        "model": model_name,
        "rows": int(len(predictions_df)),
        "positive_label": 1,
        "decision_threshold": 0.50,
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, pos_label=1, zero_division=0)),
        "recall": float(recall_score(labels, predicted, pos_label=1, zero_division=0)),
        "f1": float(f1_score(labels, predicted, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "average_latency_ms": float(predictions_df["latency_ms"].mean()),
    }

    predictions_df.to_csv(output_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    predictions_df[(predictions_df["label"] == 0) & (predictions_df["predicted_label"] == 1)].to_csv(
        output_dir / "false_positives.csv", index=False, encoding="utf-8-sig"
    )
    predictions_df[(predictions_df["label"] == 1) & (predictions_df["predicted_label"] == 0)].to_csv(
        output_dir / "false_negatives.csv", index=False, encoding="utf-8-sig"
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = "\n".join(
        [
            "# Indirect Prompt Injection Evaluation",
            "",
            f"- Dataset: `{metrics['dataset']}`",
            f"- Model: `{model_name}`",
            f"- Rows: `{metrics['rows']}`",
            "- Positive class: `1 = indirect injection`",
            "",
            "| Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN | FP | FN | TP | Avg latency |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            (
                f"| {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
                f"{metrics['f1']:.4f} | {metrics['roc_auc']:.4f} | {metrics['pr_auc']:.4f} | "
                f"{tn} | {fp} | {fn} | {tp} | {metrics['average_latency_ms']:.2f} ms |"
            ),
            "",
            "Full predictions are in `predictions.csv`; error cases are in `false_positives.csv` and `false_negatives.csv`.",
        ]
    )
    (output_dir / "metrics.md").write_text(markdown + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    args = parse_args()
    metrics = evaluate(
        dataset_path=(PROJECT_ROOT / args.dataset).resolve() if not Path(args.dataset).is_absolute() else Path(args.dataset),
        model_name=args.model,
        output_dir=(PROJECT_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir),
        use_cuda=args.use_cuda,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
