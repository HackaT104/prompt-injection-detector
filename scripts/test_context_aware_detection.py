"""Run the external benchmark through the explainable context-aware pipeline."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.pipeline import run_hybrid_detection  # noqa: E402


DEFAULT_BENCHMARK = PROJECT_ROOT / "data" / "external_benchmark" / "external_prompt_injection_benchmark.csv"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test hybrid context-aware prompt injection detection.")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--use-ml", action="store_true", help="Include local ML model score if model artifacts exist.")
    parser.add_argument("--use-transformer", action="store_true", help="Include Transformer score. Slower.")
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    tn = fp = fn = tp = 0
    for true, pred in zip(y_true, y_pred):
        if true == 0 and pred == 0:
            tn += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 1 and pred == 0:
            fn += 1
        elif true == 1 and pred == 1:
            tp += 1
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "accuracy": _safe_div(tp + tn, len(y_true)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def _load_rows(path: Path, max_rows: int | None = None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if max_rows is not None:
        return rows[:max_rows]
    return rows


def main() -> None:
    args = _parse_args()
    rows = _load_rows(args.benchmark, args.max_rows)
    y_true: list[int] = []
    y_pred: list[int] = []

    print("Context-aware benchmark")
    print(f"Benchmark: {args.benchmark}")
    print(f"Rows: {len(rows)}")
    print(f"Use ML: {args.use_ml}")
    print(f"Use Transformer: {args.use_transformer}")
    print("")

    for row in rows:
        label = int(row["label"])
        result = run_hybrid_detection(
            user_prompt=row["user_task"],
            user_task=row["user_task"],
            external_content=row.get("external_content", ""),
            use_ml=args.use_ml,
            use_transformer=args.use_transformer,
            use_cuda=False,
        )
        predicted = int(result["label"])
        y_true.append(label)
        y_pred.append(predicted)
        reasons = " | ".join(str(reason) for reason in result["reasons"][:3])
        print(
            f"{row['id']}\t"
            f"pred={predicted}\t"
            f"true={label}\t"
            f"risk={result['risk_level']}\t"
            f"score={result['final_score']:.4f}\t"
            f"reasons={reasons}"
        )

    metric = _metrics(y_true, y_pred)
    print("")
    print("Summary")
    print(f"accuracy={metric['accuracy']:.4f}")
    print(f"precision={metric['precision']:.4f}")
    print(f"recall={metric['recall']:.4f}")
    print(f"f1={metric['f1']:.4f}")
    print(f"confusion_matrix={metric['confusion_matrix']}")


if __name__ == "__main__":
    main()
