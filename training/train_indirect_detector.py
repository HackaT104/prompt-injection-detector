"""Train an indirect prompt injection detector without overwriting direct models."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import prepare_text_for_detection

DEFAULT_DATA_PATH = PROJECT_ROOT / "datasets" / "processed" / "bipia_indirect.csv"
INDIRECT_MODELS_DIR = PROJECT_ROOT / "models" / "indirect"
INDIRECT_REPORTS_DIR = PROJECT_ROOT / "reports" / "indirect"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def _canonical_combined_text(user_intent: str, context: str) -> str:
    prepared_user = prepare_text_for_detection(user_intent)
    prepared_context = prepare_text_for_detection(context)
    return (
        f"USER_INTENT: {prepared_user['cleaned_text']}\n"
        f"CONTEXT: {prepared_context['cleaned_text']}"
    )


def _load_training_frame(path: str | Path) -> pd.DataFrame:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Cannot find processed BIPIA CSV: {data_path}. "
            "Run: python training/load_bipia_dataset.py"
        )
    df = pd.read_csv(data_path)
    required_columns = {"user_intent", "context", "label"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Processed BIPIA CSV missing columns: {sorted(missing)}")
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    df["canonical_combined_text"] = [
        _canonical_combined_text(user_intent, context)
        for user_intent, context in zip(df["user_intent"].fillna(""), df["context"].fillna(""))
    ]
    return df


def _make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=20000,
        min_df=1,
        lowercase=False,
        token_pattern=r"(?u)\b[\w][\w./\\:-]*\b",
    )


def _make_calibrated_svm(y_train: list[int]) -> CalibratedClassifierCV:
    min_class_count = min(pd.Series(y_train).value_counts().to_dict().values())
    cv = max(2, min(3, min_class_count))
    base = LinearSVC(class_weight="balanced", random_state=RANDOM_STATE, max_iter=5000)
    try:
        return CalibratedClassifierCV(estimator=base, cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base, cv=cv)


def _evaluate(model: Any, vectorizer: TfidfVectorizer, X_test: list[str], y_test: list[int]) -> dict[str, Any]:
    start = time.perf_counter()
    X_vec = vectorizer.transform(X_test)
    scores = model.predict_proba(X_vec)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_vec)
    y_pred = (scores >= 0.5).astype(int)
    elapsed = time.perf_counter() - start
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y_test, y_pred, labels=[0, 1], zero_division=0),
        "avg_prediction_time_seconds": float(elapsed / max(len(y_test), 1)),
    }


def train_indirect_detector(data_path: str | Path = DEFAULT_DATA_PATH, train_svm: bool = True) -> dict[str, Any]:
    df = _load_training_frame(data_path)
    X = df["canonical_combined_text"].tolist()
    y = df["label"].astype(int).tolist()
    if set(y) != {0, 1}:
        raise ValueError("Indirect detector training requires both labels: 0 benign and 1 malicious.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    INDIRECT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    INDIRECT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    vectorizer = _make_vectorizer()
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    start = time.perf_counter()
    X_train_vec = vectorizer.fit_transform(X_train)
    model.fit(X_train_vec, y_train)
    training_time = time.perf_counter() - start

    joblib.dump(model, INDIRECT_MODELS_DIR / "logistic_regression_model.joblib")
    joblib.dump(vectorizer, INDIRECT_MODELS_DIR / "logistic_regression_vectorizer.joblib")

    metrics: dict[str, Any] = {
        "dataset_path": str(data_path),
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "label_distribution": {
            str(label): int(count)
            for label, count in df["label"].value_counts().sort_index().items()
        },
        "models": {
            "logistic_regression": {
                "training_time_seconds": float(training_time),
                **_evaluate(model, vectorizer, X_test, y_test),
            }
        },
    }

    if train_svm:
        svm_vectorizer = _make_vectorizer()
        svm_model = _make_calibrated_svm(y_train)
        start = time.perf_counter()
        X_train_svm = svm_vectorizer.fit_transform(X_train)
        svm_model.fit(X_train_svm, y_train)
        svm_training_time = time.perf_counter() - start
        joblib.dump(svm_model, INDIRECT_MODELS_DIR / "linear_svm_model.joblib")
        joblib.dump(svm_vectorizer, INDIRECT_MODELS_DIR / "linear_svm_vectorizer.joblib")
        metrics["models"]["linear_svm"] = {
            "training_time_seconds": float(svm_training_time),
            **_evaluate(svm_model, svm_vectorizer, X_test, y_test),
        }

    (INDIRECT_REPORTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train indirect prompt injection detector.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Path to processed BIPIA CSV.")
    parser.add_argument("--skip-svm", action="store_true", help="Skip optional calibrated Linear SVM.")
    args = parser.parse_args()

    train_indirect_detector(args.data, train_svm=not args.skip_svm)


if __name__ == "__main__":
    main()
