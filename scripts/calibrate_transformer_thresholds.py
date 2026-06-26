"""Calibrate runtime thresholds for fine-tuned Transformer checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.thresholding import _candidate_thresholds, metrics_at_threshold  # noqa: E402
from src.transformer_utils import (  # noqa: E402
    DEFAULT_TRANSFORMER_BLOCK_THRESHOLD,
    DEFAULT_TRANSFORMER_WARN_THRESHOLD,
    OUTPUTS_DIR,
    TRANSFORMER_THRESHOLDS_PATH,
    _load_transformer_artifacts_cached,
    import_optional,
    is_finetuned_transformer_checkpoint,
    prepare_transformer_dataframe,
    resolve_transformer_model_dir,
    safe_model_dir_name,
    softmax_positive_scores,
    split_transformer_dataframe_by_source,
)

DEFAULT_DATASET = PROJECT_ROOT / "datasets" / "unified" / "prompt_injection_transformer_ready_v2.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "transformer_threshold_calibration.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate Transformer thresholds on a validation split.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--models", nargs="+", default=["distilbert", "roberta"])
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--use-cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-runtime-warn", type=float, default=DEFAULT_TRANSFORMER_WARN_THRESHOLD)
    parser.add_argument("--min-runtime-block", type=float, default=DEFAULT_TRANSFORMER_BLOCK_THRESHOLD)
    parser.add_argument("--recall-target", type=float, default=0.98)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def predict_scores_batch(
    texts: list[str],
    model_dir: Path,
    max_length: int,
    batch_size: int,
    use_cuda: bool,
) -> np.ndarray:
    torch, tokenizer, model, device = _load_transformer_artifacts_cached(str(model_dir.resolve()), use_cuda)
    scores: list[float] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
        batch_scores = softmax_positive_scores(outputs.logits.detach().cpu().numpy())
        scores.extend(float(score) for score in batch_scores)
    return np.asarray(scores, dtype=float)


def select_thresholds(y_true: list[int], scores: np.ndarray, recall_target: float) -> dict[str, Any]:
    candidates = [metrics_at_threshold(y_true, scores, threshold) for threshold in _candidate_thresholds(scores)]
    best_f1 = sorted(
        candidates,
        key=lambda row: (float(row["f1"]), float(row["recall"]), float(row["precision"]), -float(row["false_positive_rate"])),
        reverse=True,
    )[0]
    recall_candidates = [row for row in candidates if float(row["recall"]) >= recall_target]
    if recall_candidates:
        recall_priority = sorted(
            recall_candidates,
            key=lambda row: (float(row["f1"]), float(row["precision"]), -float(row["false_positive_rate"])),
            reverse=True,
        )[0]
    else:
        recall_priority = sorted(
            candidates,
            key=lambda row: (float(row["recall"]), float(row["f1"]), float(row["precision"])),
            reverse=True,
        )[0]

    selected = best_f1
    evaluation_threshold = float(selected["threshold"])
    runtime_warn_threshold = float(max(DEFAULT_TRANSFORMER_WARN_THRESHOLD, evaluation_threshold))
    runtime_block_threshold = float(max(DEFAULT_TRANSFORMER_BLOCK_THRESHOLD, runtime_warn_threshold))
    return {
        "evaluation_threshold": evaluation_threshold,
        "runtime_warn_threshold": runtime_warn_threshold,
        "runtime_block_threshold": runtime_block_threshold,
        "best_f1_threshold": best_f1,
        "recall_priority_threshold": recall_priority,
        "recall_target": float(recall_target),
        "candidate_count": len(candidates),
    }


def update_thresholds_file(model_key: str, thresholds: dict[str, float]) -> None:
    if TRANSFORMER_THRESHOLDS_PATH.exists():
        payload = json.loads(TRANSFORMER_THRESHOLDS_PATH.read_text(encoding="utf-8"))
    else:
        payload = {"models": {}}
    payload.setdefault("models", {})[model_key] = thresholds
    TRANSFORMER_THRESHOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRANSFORMER_THRESHOLDS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    torch = import_optional("torch")
    if args.use_cuda and not torch.cuda.is_available():
        print("CUDA not available; calibrating on CPU.")

    raw_df = pd.read_csv(dataset_path, encoding="utf-8-sig")
    prepared_df = prepare_transformer_dataframe(raw_df)
    _train_df, validation_df, _test_df = split_transformer_dataframe_by_source(prepared_df)
    y_true = validation_df["label"].astype(int).tolist()
    texts = validation_df["model_text"].tolist()

    report: dict[str, Any] = {
        "dataset": str(dataset_path),
        "validation_size": len(validation_df),
        "models": {},
    }
    for model_name in args.models:
        model_dir = resolve_transformer_model_dir(model_name)
        model_key = model_dir.name
        if not is_finetuned_transformer_checkpoint(model_dir):
            report["models"][model_name] = {
                "available": False,
                "model_dir": str(model_dir),
                "message": "Checkpoint not found or not fine-tuned.",
            }
            continue

        scores = predict_scores_batch(
            texts=texts,
            model_dir=model_dir,
            max_length=args.max_length,
            batch_size=args.batch_size,
            use_cuda=args.use_cuda,
        )
        calibration = select_thresholds(y_true, scores, recall_target=args.recall_target)
        saved_thresholds = {
            "evaluation_threshold": float(calibration["evaluation_threshold"]),
            "runtime_warn_threshold": float(calibration["runtime_warn_threshold"]),
            "runtime_block_threshold": float(calibration["runtime_block_threshold"]),
        }
        update_thresholds_file(model_key, saved_thresholds)
        alias_key = safe_model_dir_name(model_name)
        if alias_key != model_key:
            update_thresholds_file(alias_key, saved_thresholds)

        report["models"][model_name] = {
            "available": True,
            "model_dir": str(model_dir),
            "model_key": model_key,
            "saved_thresholds": saved_thresholds,
            "calibration": calibration,
        }

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
