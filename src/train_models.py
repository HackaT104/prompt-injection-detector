"""Train TF-IDF + Logistic Regression and TF-IDF + Linear SVM models."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.compare_models import compare_models, save_model_comparison_reports
from src.data_augmentation import (
    BENIGN_CANONICAL_AUGMENTATIONS,
    SECURITY_CANONICAL_AUGMENTATIONS,
    create_augmented_dataset,
    generate_vietnamese_variants,
)
from src.data_loader import (
    auto_detect_label_column,
    auto_detect_text_column,
    build_dataset_summary,
    load_jsonl_dataset,
    normalize_labels,
    print_dataset_summary,
    save_dataset_summary,
)
from src.evaluate import evaluate_model, save_metrics_json, save_metrics_markdown
from src.file_utils import safe_write_text
from src.preprocessing import clean_text, prepare_text_for_detection
from src.thresholding import choose_threshold, get_positive_class_scores


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Prompt_INJECTION_And_Benign_DATASET.jsonl"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DIRECT_MODEL_ARTIFACT_NAMES = [
    "logistic_regression_model.joblib",
    "logistic_regression_vectorizer.joblib",
    "linear_svm_model.joblib",
    "linear_svm_vectorizer.joblib",
    "random_forest_model.joblib",
    "random_forest_vectorizer.joblib",
    "model_thresholds.json",
]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
TFIDF_TOKEN_PATTERN = r"(?u)\b[\w][\w./\\:-]*\b"
CUSTOM_STOPWORDS = {
    "this",
    "that",
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "is",
    "you",
    "your",
    "yours",
    "i",
    "me",
    "my",
    "we",
    "us",
    "our",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "am",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "having",
    "for",
    "with",
    "about",
    "as",
    "at",
    "by",
    "from",
    "into",
    "it",
    "its",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "where",
    "when",
    "why",
    "how",
    "can",
    "could",
    "would",
    "should",
    "will",
    "shall",
    "if",
    "then",
    "than",
}
SECURITY_TERMS_TO_KEEP = {
    "ignore",
    "previous",
    "instruction",
    "instructions",
    "system",
    "prompt",
    "reveal",
    "bypass",
    "jailbreak",
    "dan",
    "token",
    "api",
    "key",
    "password",
    "credential",
    "credentials",
    "os.system",
    "whoami",
    "eval",
    "exec",
}
if CUSTOM_STOPWORDS.intersection(SECURITY_TERMS_TO_KEEP):
    raise ValueError("Custom stopwords must not contain security-critical terms.")

TFIDF_CONFIG = {
    "ngram_range": (1, 2),
    "max_features": 10000,
    "min_df": 2,
    "max_df": 0.9,
    "lowercase": False,
    "token_pattern": TFIDF_TOKEN_PATTERN,
    "stop_words": sorted(CUSTOM_STOPWORDS),
}


def _min_df_for_sample_size(sample_size: int) -> int:
    return 2 if sample_size >= 100 else 1


def _tfidf_config_for_sample_size(sample_size: int | None = None) -> dict[str, Any]:
    config = TFIDF_CONFIG.copy()
    if sample_size is not None:
        config["min_df"] = _min_df_for_sample_size(sample_size)
    return config


def make_tfidf_vectorizer(sample_size: int | None = None) -> TfidfVectorizer:
    return TfidfVectorizer(**_tfidf_config_for_sample_size(sample_size))


def _distribution(values: list[int]) -> dict[str, int]:
    counter = Counter(int(value) for value in values)
    return {str(label): int(counter.get(label, 0)) for label in [0, 1]}


def _load_training_dataframe(data_path: str | Path) -> tuple[pd.DataFrame, str, str, str]:
    """Load either the original JSONL dataset or a merged direct CSV dataset."""
    dataset_path = Path(data_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset: {dataset_path}")

    suffix = dataset_path.suffix.lower()
    if suffix == ".jsonl":
        df = load_jsonl_dataset(dataset_path)
        dataset_kind = "jsonl"
    elif suffix == ".csv":
        df = pd.read_csv(dataset_path, encoding="utf-8-sig")
        dataset_kind = "csv"
    else:
        raise ValueError("Training dataset phải là file .jsonl hoặc .csv.")

    if df.empty:
        raise ValueError("Dataset rỗng, không thể train model.")

    text_column = auto_detect_text_column(df)
    label_column = auto_detect_label_column(df)
    df = normalize_labels(df, label_column=label_column)
    df[text_column] = df[text_column].fillna("").astype(str)
    df = df[df[text_column].str.strip() != ""].copy()
    if df.empty:
        raise ValueError("Dataset không có prompt/text hợp lệ sau khi lọc dòng rỗng.")

    return df, text_column, label_column, dataset_kind


def backup_existing_direct_artifacts(models_dir: str | Path = MODELS_DIR) -> dict[str, Any]:
    """Backup current direct detector artifacts before overwriting them."""
    base_dir = Path(models_dir)
    existing_files = [base_dir / name for name in DIRECT_MODEL_ARTIFACT_NAMES if (base_dir / name).exists()]
    if not existing_files:
        return {"created": False, "backup_dir": None, "files": []}

    backup_dir = base_dir / "backups" / f"direct_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    copied_files: list[str] = []
    for artifact_path in existing_files:
        destination = backup_dir / artifact_path.name
        shutil.copy2(artifact_path, destination)
        copied_files.append(str(destination))

    return {
        "created": True,
        "backup_dir": str(backup_dir),
        "files": copied_files,
    }


def _canonicalize_texts(texts: list[str]) -> list[str]:
    return [prepare_text_for_detection(text)["cleaned_text"] for text in texts]


def _augment_training_split(X_train_raw: list[str], y_train: list[int]) -> tuple[list[str], list[int], int]:
    augmented_texts: list[str] = []
    augmented_labels: list[int] = []
    seen_augmented: set[tuple[str, int]] = set()
    for prompt, label in zip(X_train_raw, y_train):
        for variant in generate_vietnamese_variants(prompt):
            key = (variant, int(label))
            if key not in seen_augmented:
                seen_augmented.add(key)
                augmented_texts.append(variant)
                augmented_labels.append(int(label))

    for _, vietnamese_variant, label, _ in SECURITY_CANONICAL_AUGMENTATIONS:
        key = (vietnamese_variant, int(label))
        if key not in seen_augmented:
            seen_augmented.add(key)
            augmented_texts.append(vietnamese_variant)
            augmented_labels.append(int(label))

    for _, benign_variant, label, _ in BENIGN_CANONICAL_AUGMENTATIONS:
        key = (benign_variant, int(label))
        if key not in seen_augmented:
            seen_augmented.add(key)
            augmented_texts.append(benign_variant)
            augmented_labels.append(int(label))

    return X_train_raw + augmented_texts, y_train + augmented_labels, len(augmented_texts)


def _make_calibrated_linear_svm(y_train: list[int]) -> CalibratedClassifierCV:
    min_class_count = min(Counter(y_train).values())
    cv = max(2, min(3, min_class_count))
    base_model = LinearSVC(class_weight="balanced", random_state=RANDOM_STATE, max_iter=5000)
    try:
        return CalibratedClassifierCV(estimator=base_model, cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=base_model, cv=cv)


def train_logistic_regression(X_train: list[str], y_train: list[int]) -> tuple[Any, Any, float]:
    vectorizer = make_tfidf_vectorizer(len(X_train))
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)

    start = time.perf_counter()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    model.fit(X_train_vectorized, y_train)
    training_time = time.perf_counter() - start
    return model, vectorizer, training_time


def train_linear_svm(X_train: list[str], y_train: list[int]) -> tuple[Any, Any, float]:
    vectorizer = make_tfidf_vectorizer(len(X_train))
    model = _make_calibrated_linear_svm(y_train)

    start = time.perf_counter()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    model.fit(X_train_vectorized, y_train)
    training_time = time.perf_counter() - start
    return model, vectorizer, training_time


def train_random_forest(X_train: list[str], y_train: list[int]) -> tuple[Any, Any, float]:
    vectorizer = make_tfidf_vectorizer(len(X_train))
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    start = time.perf_counter()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    model.fit(X_train_vectorized, y_train)
    training_time = time.perf_counter() - start
    return model, vectorizer, training_time


def run_cross_validation(X: list[str], y: list[int]) -> dict[str, Any]:
    counts = Counter(y)
    min_class_count = min(counts.values())
    if min_class_count < 2:
        return {
            "skipped": True,
            "reason": "Mỗi lớp cần ít nhất 2 mẫu để chạy stratified cross-validation.",
        }

    n_splits = min(5, min_class_count)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, pos_label=1, zero_division=0),
        "recall": make_scorer(recall_score, pos_label=1, zero_division=0),
        "f1": make_scorer(f1_score, pos_label=1, zero_division=0),
    }

    pipelines = {
        "logistic_regression": Pipeline(
            steps=[
                (
                    "tfidf",
                    make_tfidf_vectorizer(len(X)),
                ),
                (
                    "classifier",
                    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
                ),
            ]
        ),
        "linear_svm": Pipeline(
            steps=[
                (
                    "tfidf",
                    make_tfidf_vectorizer(len(X)),
                ),
                (
                    "classifier",
                    LinearSVC(class_weight="balanced", random_state=RANDOM_STATE, max_iter=5000),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                (
                    "tfidf",
                    make_tfidf_vectorizer(len(X)),
                ),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=None,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    result: dict[str, Any] = {
        "skipped": False,
        "n_splits": n_splits,
        "models": {},
    }
    for model_name, pipeline in pipelines.items():
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=None)
        result["models"][model_name] = {
            metric_name.replace("test_", ""): {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "scores": [float(value) for value in values],
            }
            for metric_name, values in scores.items()
            if metric_name.startswith("test_")
        }
    return result


def _save_cross_validation(cross_validation: dict[str, Any]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_write_text(
        REPORTS_DIR / "cross_validation.json",
        json.dumps(cross_validation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _save_threshold_analysis(threshold_analysis: dict[str, Any]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_write_text(
        REPORTS_DIR / "threshold_analysis.json",
        json.dumps(threshold_analysis, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Threshold analysis",
        "",
        "Threshold được chọn trên validation set để giảm false positive nhưng vẫn giữ recall cao cho lớp malicious.",
        "",
    ]
    for model_key, analysis in threshold_analysis.get("models", {}).items():
        selected = analysis.get("selected_metrics", {})
        lines.extend(
            [
                f"## {model_key}",
                "",
                f"- Evaluation threshold: {analysis.get('selected_threshold', 0):.4f}",
                f"- Runtime warn threshold: {analysis.get('runtime_warn_threshold', 0):.4f}",
                f"- Runtime block threshold: {analysis.get('runtime_block_threshold', 0):.4f}",
                f"- Validation precision: {selected.get('precision', 0):.4f}",
                f"- Validation recall: {selected.get('recall', 0):.4f}",
                f"- Validation F1-score: {selected.get('f1', 0):.4f}",
                f"- Validation FPR: {selected.get('false_positive_rate', 0):.4f}",
                f"- Validation FNR: {selected.get('false_negative_rate', 0):.4f}",
                f"- Lý do chọn: {analysis.get('selection_reason')}",
                "",
            ]
        )
    safe_write_text(REPORTS_DIR / "threshold_analysis.md", "\n".join(lines), encoding="utf-8")


def _save_runtime_thresholds(threshold_analysis: dict[str, Any]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    thresholds = {
        model_key: {
            "evaluation_threshold": float(analysis["selected_threshold"]),
            "runtime_warn_threshold": float(analysis["runtime_warn_threshold"]),
            "runtime_block_threshold": float(analysis["runtime_block_threshold"]),
        }
        for model_key, analysis in threshold_analysis.get("models", {}).items()
    }
    safe_write_text(
        MODELS_DIR / "model_thresholds.json",
        json.dumps(thresholds, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def select_model_thresholds(
    X_train: list[str],
    y_train: list[int],
    X_validation: list[str],
    y_validation: list[int],
) -> dict[str, Any]:
    """Train temporary models on train-core and select thresholds on validation."""
    logistic_model, logistic_vectorizer, _ = train_logistic_regression(X_train, y_train)
    logistic_scores = get_positive_class_scores(logistic_model, logistic_vectorizer, X_validation)
    logistic_analysis = choose_threshold(y_validation, logistic_scores)

    svm_model, svm_vectorizer, _ = train_linear_svm(X_train, y_train)
    svm_scores = get_positive_class_scores(svm_model, svm_vectorizer, X_validation)
    svm_analysis = choose_threshold(y_validation, svm_scores)

    random_forest_model, random_forest_vectorizer, _ = train_random_forest(X_train, y_train)
    random_forest_scores = get_positive_class_scores(random_forest_model, random_forest_vectorizer, X_validation)
    random_forest_analysis = choose_threshold(y_validation, random_forest_scores)

    return {
        "validation_size": int(len(y_validation)),
        "validation_label_distribution": _distribution(y_validation),
        "selection_policy": {
            "min_recall": 0.95,
            "min_precision": 0.90,
            "runtime_warn_threshold_fixed": 0.50,
            "runtime_block_threshold_fixed": 0.80,
        },
        "models": {
            "logistic_regression": logistic_analysis,
            "linear_svm": svm_analysis,
            "random_forest": random_forest_analysis,
        },
    }


def _extract_linear_coefficients(model: Any) -> Any:
    if hasattr(model, "coef_"):
        return model.coef_[0]

    calibrated_classifiers = getattr(model, "calibrated_classifiers_", None)
    if not calibrated_classifiers:
        return None

    coefficients = []
    for calibrated_classifier in calibrated_classifiers:
        estimator = getattr(calibrated_classifier, "estimator", None)
        if estimator is not None and hasattr(estimator, "coef_"):
            coefficients.append(estimator.coef_[0])
    if not coefficients:
        return None

    import numpy as np

    return np.mean(coefficients, axis=0)


def _feature_stopword_hits(feature: str) -> list[str]:
    tokens = feature.lower().replace("/", " ").replace("\\", " ").replace(":", " ").split()
    return sorted({token for token in tokens if token in CUSTOM_STOPWORDS})


def extract_feature_analysis(model: Any, vectorizer: Any, top_n: int = 20) -> dict[str, Any]:
    """Extract top positive and negative TF-IDF features when the model exposes coefficients."""
    feature_names = vectorizer.get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        weighted_features = [
            (feature, float(importance))
            for feature, importance in zip(feature_names, model.feature_importances_)
            if float(importance) > 0 and any(character.isalpha() for character in feature)
        ]
        top_importances = sorted(weighted_features, key=lambda item: item[1], reverse=True)[:top_n]
        suspected_biased_features: list[dict[str, Any]] = []
        for feature, importance in top_importances:
            stopword_hits = _feature_stopword_hits(feature)
            if stopword_hits:
                suspected_biased_features.append(
                    {
                        "feature": feature,
                        "direction": "importance",
                        "weight": importance,
                        "stopword_hits": stopword_hits,
                    }
                )
        return {
            "available": True,
            "analysis_type": "feature_importance",
            "top_feature_importances": [
                {"feature": feature, "importance": importance}
                for feature, importance in top_importances
            ],
            "suspected_biased_features": suspected_biased_features,
            "bias_warning": bool(suspected_biased_features),
            "note": (
                "Random Forest feature importance chỉ cho biết feature quan trọng, "
                "không cho biết feature nghiêng về malicious hay benign."
            ),
        }

    coefficients = _extract_linear_coefficients(model)
    if coefficients is None:
        return {
            "available": False,
            "reason": "Model không expose hệ số tuyến tính để phân tích feature.",
        }

    feature_names = vectorizer.get_feature_names_out()
    weighted_features = [
        (feature, float(weight))
        for feature, weight in zip(feature_names, coefficients)
        if any(character.isalpha() for character in feature) and len(feature.strip()) > 1
    ]
    top_malicious = sorted(weighted_features, key=lambda item: item[1], reverse=True)[:top_n]
    top_benign = sorted(weighted_features, key=lambda item: item[1])[:top_n]
    suspected_biased_features: list[dict[str, Any]] = []
    for direction, features in [
        ("malicious", top_malicious),
        ("benign", top_benign),
    ]:
        for feature, weight in features:
            stopword_hits = _feature_stopword_hits(feature)
            if stopword_hits:
                suspected_biased_features.append(
                    {
                        "feature": feature,
                        "direction": direction,
                        "weight": weight,
                        "stopword_hits": stopword_hits,
                    }
                )

    return {
        "available": True,
        "analysis_type": "linear_weights",
        "top_malicious_features": [
            {"feature": feature, "weight": weight}
            for feature, weight in top_malicious
        ],
        "top_benign_features": [
            {"feature": feature, "weight": weight}
            for feature, weight in top_benign
        ],
        "suspected_biased_features": suspected_biased_features,
        "bias_warning": bool(suspected_biased_features),
    }


def save_feature_analysis_markdown(metrics: dict[str, Any], output_path: str | Path) -> None:
    """Save a readable feature-bias report for linear TF-IDF models."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Feature analysis",
        "",
        "Báo cáo này kiểm tra các feature TF-IDF có trọng số lớn trong Logistic Regression và Linear SVM.",
        "Mục tiêu là phát hiện spurious correlation: các từ quá chung bị học nhầm thành tín hiệu malicious hoặc benign.",
        "",
        "## Custom stopwords",
        "",
        ", ".join(f"`{word}`" for word in sorted(CUSTOM_STOPWORDS)),
        "",
        "Các token security quan trọng như `ignore`, `previous`, `instructions`, `system`, `prompt`, `reveal`, `bypass`, `api`, `key`, `whoami`, `eval`, `exec` không nằm trong stopwords.",
        "",
    ]

    for model_name, model_metrics in metrics.get("models", {}).items():
        feature_analysis = model_metrics.get("feature_analysis", {})
        lines.extend([f"## {model_name}", ""])
        if not feature_analysis.get("available"):
            lines.extend([f"- Không có feature analysis: {feature_analysis.get('reason')}", ""])
            continue

        if feature_analysis.get("analysis_type") == "feature_importance":
            lines.extend(
                [
                    "### Top feature importances",
                    "",
                    "Random Forest feature importance chỉ cho biết feature quan trọng, không nói trực tiếp feature nghiêng về malicious hay benign.",
                    "",
                ]
            )
            for item in feature_analysis.get("top_feature_importances", [])[:20]:
                lines.append(f"- `{item['feature']}`: {item['importance']:.6f}")
        else:
            lines.extend(
                [
                    "### Top malicious indicators",
                    "",
                    "Với Logistic Regression/Linear SVM, trọng số dương nghiêng về malicious.",
                    "",
                ]
            )
            for item in feature_analysis.get("top_malicious_features", [])[:20]:
                lines.append(f"- `{item['feature']}`: {item['weight']:.6f}")

            lines.extend(
                [
                    "",
                    "### Top benign indicators",
                    "",
                    "Với Logistic Regression/Linear SVM, trọng số âm nghiêng về benign.",
                    "",
                ]
            )
            for item in feature_analysis.get("top_benign_features", [])[:20]:
                lines.append(f"- `{item['feature']}`: {item['weight']:.6f}")

        suspected = feature_analysis.get("suspected_biased_features", [])
        lines.extend(["", "### Suspected biased features", ""])
        if suspected:
            for item in suspected:
                hits = ", ".join(f"`{word}`" for word in item.get("stopword_hits", []))
                lines.append(
                    f"- `{item['feature']}` ({item['direction']}, weight={item['weight']:.6f}) chứa stopword: {hits}"
                )
        else:
            lines.append("- Không phát hiện stopword trong top malicious/top benign features.")

        lines.append("")

    lines.extend(
        [
            "## Giải thích ngắn",
            "",
            "Spurious correlation xảy ra khi model học nhầm một từ phổ biến thành dấu hiệu tấn công chỉ vì từ đó xuất hiện lệch trong dataset train.",
            "Custom stopwords giúp loại các từ quá chung khỏi vocabulary TF-IDF, còn feature analysis giúp kiểm tra thủ công xem model còn đang dựa vào tín hiệu nhiễu hay không.",
            "",
        ]
    )
    safe_write_text(path, "\n".join(lines), encoding="utf-8")


def _vectorizer_config_for_report(vectorizer: TfidfVectorizer) -> dict[str, Any]:
    params = vectorizer.get_params()
    return {
        "ngram_range": list(params["ngram_range"]),
        "max_features": params["max_features"],
        "min_df": params["min_df"],
        "max_df": params["max_df"],
        "lowercase": params["lowercase"],
        "token_pattern": params["token_pattern"],
        "stop_words": list(params["stop_words"]) if params["stop_words"] is not None else None,
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
    }


def train_all(data_path: str | Path = DEFAULT_DATA_PATH, run_cv: bool = True) -> dict[str, Any]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    data_path = Path(data_path)
    df, text_column, label_column, dataset_kind = _load_training_dataframe(data_path)

    print_dataset_summary(df)
    dataset_summary = save_dataset_summary(df, REPORTS_DIR / "dataset_summary.md")

    if dataset_kind == "jsonl":
        augmented_dataset = create_augmented_dataset(data_path)
        saved_augmentation_path: str | None = str(PROJECT_ROOT / "data" / "processed" / "augmented_multilingual_dataset.csv")
        saved_augmentation_rows = int(len(augmented_dataset))
    else:
        saved_augmentation_path = None
        saved_augmentation_rows = 0

    X_raw = df[text_column].fillna("").astype(str).tolist()
    y = df["label_normalized"].astype(int).tolist()

    counts = Counter(y)
    if set(counts.keys()) != {0, 1}:
        raise ValueError("Dataset cần có đủ 2 lớp: 0 = benign và 1 = malicious.")
    if min(counts.values()) < 2:
        raise ValueError("Mỗi lớp cần ít nhất 2 mẫu để chia train/test có stratify.")

    X_train_raw, X_test_raw, y_train_base, y_test = train_test_split(
        X_raw,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train_for_model_raw, y_train, train_augmentation_count = _augment_training_split(
        X_train_raw,
        y_train_base,
    )
    X_train = _canonicalize_texts(X_train_for_model_raw)
    X_test = _canonicalize_texts(X_test_raw)
    X_original_canonical = _canonicalize_texts(X_raw)

    X_train_core, X_validation, y_train_core, y_validation = train_test_split(
        X_train,
        y_train,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    threshold_analysis = select_model_thresholds(
        X_train_core,
        y_train_core,
        X_validation,
        y_validation,
    )
    model_backup = backup_existing_direct_artifacts()
    _save_threshold_analysis(threshold_analysis)
    _save_runtime_thresholds(threshold_analysis)

    logistic_model, logistic_vectorizer, logistic_time = train_logistic_regression(X_train, y_train)
    joblib.dump(logistic_model, MODELS_DIR / "logistic_regression_model.joblib")
    joblib.dump(logistic_vectorizer, MODELS_DIR / "logistic_regression_vectorizer.joblib")

    svm_model, svm_vectorizer, svm_time = train_linear_svm(X_train, y_train)
    joblib.dump(svm_model, MODELS_DIR / "linear_svm_model.joblib")
    joblib.dump(svm_vectorizer, MODELS_DIR / "linear_svm_vectorizer.joblib")

    random_forest_model, random_forest_vectorizer, random_forest_time = train_random_forest(X_train, y_train)
    joblib.dump(random_forest_model, MODELS_DIR / "random_forest_model.joblib")
    joblib.dump(random_forest_vectorizer, MODELS_DIR / "random_forest_vectorizer.joblib")

    logistic_metrics = evaluate_model(
        logistic_model,
        logistic_vectorizer,
        X_test,
        y_test,
        "logistic_regression",
        train_size=len(y_train),
        reports_dir=REPORTS_DIR,
        threshold=float(threshold_analysis["models"]["logistic_regression"]["selected_threshold"]),
        runtime_warn_threshold=float(threshold_analysis["models"]["logistic_regression"]["runtime_warn_threshold"]),
        runtime_block_threshold=float(threshold_analysis["models"]["logistic_regression"]["runtime_block_threshold"]),
    )
    logistic_metrics["training_time_seconds"] = float(logistic_time)
    logistic_metrics["feature_analysis"] = extract_feature_analysis(logistic_model, logistic_vectorizer)
    logistic_metrics["tfidf_config"] = _vectorizer_config_for_report(logistic_vectorizer)

    svm_metrics = evaluate_model(
        svm_model,
        svm_vectorizer,
        X_test,
        y_test,
        "linear_svm",
        train_size=len(y_train),
        reports_dir=REPORTS_DIR,
        threshold=float(threshold_analysis["models"]["linear_svm"]["selected_threshold"]),
        runtime_warn_threshold=float(threshold_analysis["models"]["linear_svm"]["runtime_warn_threshold"]),
        runtime_block_threshold=float(threshold_analysis["models"]["linear_svm"]["runtime_block_threshold"]),
    )
    svm_metrics["training_time_seconds"] = float(svm_time)
    svm_metrics["feature_analysis"] = extract_feature_analysis(svm_model, svm_vectorizer)
    svm_metrics["tfidf_config"] = _vectorizer_config_for_report(svm_vectorizer)

    random_forest_metrics = evaluate_model(
        random_forest_model,
        random_forest_vectorizer,
        X_test,
        y_test,
        "random_forest",
        train_size=len(y_train),
        reports_dir=REPORTS_DIR,
        threshold=float(threshold_analysis["models"]["random_forest"]["selected_threshold"]),
        runtime_warn_threshold=float(threshold_analysis["models"]["random_forest"]["runtime_warn_threshold"]),
        runtime_block_threshold=float(threshold_analysis["models"]["random_forest"]["runtime_block_threshold"]),
    )
    random_forest_metrics["training_time_seconds"] = float(random_forest_time)
    random_forest_metrics["feature_analysis"] = extract_feature_analysis(random_forest_model, random_forest_vectorizer)
    random_forest_metrics["tfidf_config"] = _vectorizer_config_for_report(random_forest_vectorizer)

    metrics: dict[str, Any] = {
        "dataset": dataset_summary,
        "training_dataset_source": {
            "path": str(data_path),
            "kind": dataset_kind,
            "text_column": text_column,
            "label_column": label_column,
        },
        "model_backup": model_backup,
        "split": {
            "train_size": int(len(y_train_base)),
            "test_size": int(len(y_test)),
            "model_train_size_after_augmentation": int(len(y_train)),
            "train_augmentation_count": int(train_augmentation_count),
            "test_size_ratio": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratify": True,
            "train_label_distribution": _distribution(y_train_base),
            "model_train_label_distribution_after_augmentation": _distribution(y_train),
            "test_label_distribution": _distribution(y_test),
        },
        "augmentation": {
            "enabled": True,
            "saved_path": saved_augmentation_path,
            "total_augmented_rows_saved": saved_augmentation_rows,
            "train_augmented_rows_used": int(train_augmentation_count),
            "note": "Augmented Vietnamese prompts are generated from the training split only for model fitting; the test split remains the original dataset. The saved augmented CSV is generated only when training from the original JSONL source.",
        },
        "validation": {
            "train_core_size": int(len(y_train_core)),
            "validation_size": int(len(y_validation)),
            "validation_size_ratio_within_train": VALIDATION_SIZE,
            "train_core_label_distribution": _distribution(y_train_core),
            "validation_label_distribution": _distribution(y_validation),
            "threshold_analysis_path": str(REPORTS_DIR / "threshold_analysis.json"),
        },
        "threshold_analysis": threshold_analysis,
        "feature_analysis_report_path": str(REPORTS_DIR / "feature_analysis.md"),
        "models": {
            "logistic_regression": logistic_metrics,
            "linear_svm": svm_metrics,
            "random_forest": random_forest_metrics,
        },
    }

    save_feature_analysis_markdown(metrics, REPORTS_DIR / "feature_analysis.md")
    save_metrics_json(metrics, REPORTS_DIR / "metrics.json")
    comparison = compare_models(REPORTS_DIR / "metrics.json")
    metrics["model_comparison"] = comparison
    metrics["best_model"] = comparison["best_model"]
    metrics["model_comparison_report_path"] = str(REPORTS_DIR / "model_comparison.md")
    save_metrics_json(metrics, REPORTS_DIR / "metrics.json")
    save_metrics_markdown(metrics, REPORTS_DIR / "metrics.md")
    save_feature_analysis_markdown(metrics, REPORTS_DIR / "feature_analysis.md")
    save_model_comparison_reports(
        metrics,
        comparison,
        json_output=REPORTS_DIR / "model_comparison.json",
        md_output=REPORTS_DIR / "model_comparison.md",
    )

    if run_cv:
        cross_validation = run_cross_validation(X_original_canonical, y)
        _save_cross_validation(cross_validation)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train prompt injection detection models.")
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA_PATH),
        help="Path to direct training dataset (.jsonl or merged .csv).",
    )
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Skip cross-validation.",
    )
    args = parser.parse_args()

    metrics = train_all(args.data, run_cv=not args.skip_cv)
    print("\n=== Training completed ===")
    print(f"Metrics JSON: {REPORTS_DIR / 'metrics.json'}")
    print(f"Metrics Markdown: {REPORTS_DIR / 'metrics.md'}")
    print(f"Threshold analysis: {REPORTS_DIR / 'threshold_analysis.md'}")
    print(f"Feature analysis: {REPORTS_DIR / 'feature_analysis.md'}")
    print(f"Model comparison: {REPORTS_DIR / 'model_comparison.md'}")
    print(f"Best model: {metrics.get('best_model', {}).get('name')}")

    logistic_metrics = metrics.get("models", {}).get("logistic_regression", {})
    print("\n=== Logistic Regression threshold ===")
    print(f"Evaluation threshold: {logistic_metrics.get('evaluation_threshold')}")
    print(f"Runtime warn threshold: {logistic_metrics.get('runtime_warn_threshold')}")
    print(f"Runtime block threshold: {logistic_metrics.get('runtime_block_threshold')}")

    feature_analysis = logistic_metrics.get("feature_analysis", {})
    if feature_analysis.get("available"):
        print("\nTop malicious indicators:")
        for item in feature_analysis.get("top_malicious_features", [])[:10]:
            print(f"- {item['feature']}: {item['weight']:.6f}")
        print("\nTop benign indicators:")
        for item in feature_analysis.get("top_benign_features", [])[:10]:
            print(f"- {item['feature']}: {item['weight']:.6f}")
        suspected = feature_analysis.get("suspected_biased_features", [])
        if suspected:
            print("\nWARNING: suspected biased features found in top features:")
            for item in suspected[:10]:
                print(f"- {item['feature']} ({item['direction']}): {item['stopword_hits']}")


if __name__ == "__main__":
    main()
