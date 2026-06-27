"""Prepare direct prompt-injection external benchmarks.

The normalized files are benchmark-only artifacts. They are not written to any
training folder and should not be used for model training.

Default command:
    python scripts/prepare_direct_benchmarks.py

Outputs:
    data/external_benchmark/direct/deepset_normalized.csv
    data/external_benchmark/direct/neuralchemy_normalized.csv
    data/external_benchmark/direct/rogue_security_normalized.csv
    data/external_benchmark/direct/cyberec_normalized.csv
    data/external_benchmark/direct/direct_all_normalized.csv
    data/external_benchmark/direct/direct_benchmark_metadata.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import clean_text  # noqa: E402
from src.transformer_utils import import_huggingface_load_dataset  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "external_benchmark" / "direct"
OUTPUT_COLUMNS = [
    "id",
    "dataset_name",
    "text",
    "label",
    "attack_type",
    "language",
    "source_label",
]

DATASET_SPECS = {
    "deepset": {
        "hf_name": "deepset/prompt-injections",
        "hf_config": None,
        "output_file": "deepset_normalized.csv",
    },
    "neuralchemy": {
        "hf_name": "neuralchemy/Prompt-injection-dataset",
        "hf_config": "full",
        "fallback_configs": ["core", None],
        "output_file": "neuralchemy_normalized.csv",
    },
    "rogue_security": {
        "hf_name": "rogue-security/prompt-injections-benchmark",
        "hf_config": None,
        "output_file": "rogue_security_normalized.csv",
    },
    "cyberec": {
        "hf_name": "cyberec/llm-prompt-injection-attacks",
        "hf_config": None,
        "output_file": "cyberec_normalized.csv",
    },
}

SAFE_LABELS = {
    "0",
    "false",
    "safe",
    "benign",
    "clean",
    "normal",
    "not_injection",
    "not injection",
    "non-injection",
    "non injection",
    "negative",
    "legitimate",
}
INJECTION_LABELS = {
    "1",
    "true",
    "injection",
    "prompt_injection",
    "prompt injection",
    "malicious",
    "attack",
    "jailbreak",
    "jail_break",
    "instruction_override",
    "instruction override",
    "role_hijack",
    "role hijack",
    "data_exfiltration",
    "data exfiltration",
    "positive",
}
INJECTION_LABEL_HINTS = [
    "inject",
    "jailbreak",
    "override",
    "hijack",
    "exfil",
    "malicious",
    "attack",
    "bypass",
]

TEXT_COLUMN_CANDIDATES = [
    "text",
    "prompt",
    "input",
    "instruction",
    "content",
    "query",
    "user_prompt",
    "attack",
    "jailbreak",
]
LABEL_COLUMN_CANDIDATES = [
    "label",
    "is_prompt_injection",
    "is_injection",
    "is_malicious",
    "class",
    "category",
    "type",
    "source_label",
    "target",
]
ATTACK_TYPE_CANDIDATES = [
    "attack_type",
    "category",
    "type",
    "class",
    "risk_category",
    "prompt_type",
]
LANGUAGE_CANDIDATES = ["language", "lang", "locale"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize external direct prompt-injection benchmarks.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_SPECS),
        default=list(DATASET_SPECS),
        help="Datasets to prepare.",
    )
    parser.add_argument(
        "--max-rows-per-dataset",
        type=int,
        default=None,
        help="Optional deterministic cap per dataset for quick experiments.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def _empty_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _normalize_column_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _find_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {_normalize_column_name(column): column for column in columns}
    for candidate in candidates:
        key = _normalize_column_name(candidate)
        if key in normalized:
            return normalized[key]
    for column in columns:
        column_key = _normalize_column_name(column)
        if any(_normalize_column_name(candidate) in column_key for candidate in candidates):
            return column
    return None


def _stable_hash(text: str, length: int = 10) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:length]


def _source_label(row: dict[str, Any], label_column: str | None, attack_column: str | None) -> str:
    if label_column and row.get(label_column) is not None:
        return str(row.get(label_column)).strip()
    if attack_column and row.get(attack_column) is not None:
        return str(row.get(attack_column)).strip()
    return ""


def _normalize_label(value: Any, row: dict[str, Any] | None = None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if float(value) in {0.0, 1.0}:
            return int(value)
    normalized = re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()
    compact = normalized.replace(" ", "_")
    if normalized in SAFE_LABELS or compact in SAFE_LABELS:
        return 0
    if normalized in INJECTION_LABELS or compact in INJECTION_LABELS:
        return 1
    if any(hint in normalized for hint in INJECTION_LABEL_HINTS):
        return 1
    if normalized in {"benign prompt", "benign user request"}:
        return 0

    if row:
        joined = " ".join(str(v).lower() for v in row.values() if v is not None)
        if any(hint in joined for hint in INJECTION_LABEL_HINTS):
            return 1
    return None


def _infer_label(row: dict[str, Any], label_column: str | None, attack_column: str | None) -> int | None:
    if label_column:
        label = _normalize_label(row.get(label_column), row)
        if label is not None:
            return label
    if attack_column:
        label = _normalize_label(row.get(attack_column), row)
        if label is not None:
            return label
    for column in LABEL_COLUMN_CANDIDATES + ATTACK_TYPE_CANDIDATES:
        if column in row:
            label = _normalize_label(row.get(column), row)
            if label is not None:
                return label
    return None


def _attack_type(row: dict[str, Any], label: int, attack_column: str | None, source_label: str) -> str:
    if label == 0:
        return "safe"
    if attack_column and row.get(attack_column) is not None:
        value = str(row.get(attack_column)).strip()
        if value and _normalize_label(value) != 0:
            return _normalize_column_name(value).upper()
    if source_label and _normalize_label(source_label) != 0:
        return _normalize_column_name(source_label).upper()
    return "PROMPT_INJECTION"


def _language(row: dict[str, Any], language_column: str | None) -> str:
    if language_column and row.get(language_column) is not None:
        value = str(row.get(language_column)).strip().lower()
        return value or "unknown"
    return "en"


def _dataset_to_rows(dataset: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if hasattr(dataset, "items"):
        for split_name, split_dataset in dataset.items():
            for raw in split_dataset:
                row = dict(raw)
                row["_split"] = split_name
                rows.append(row)
    else:
        for raw in dataset:
            row = dict(raw)
            row["_split"] = "default"
            rows.append(row)
    return rows


def _load_hf_dataset(spec: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    load_dataset = import_huggingface_load_dataset()
    configs = [spec.get("hf_config"), *spec.get("fallback_configs", [])]
    last_error: Exception | None = None
    attempted: list[str] = []
    for config in configs:
        config_label = "" if config is None else str(config)
        attempted.append(config_label or "<default>")
        try:
            if config is None:
                dataset = load_dataset(spec["hf_name"])
            else:
                dataset = load_dataset(spec["hf_name"], config)
            rows = _dataset_to_rows(dataset)
            return rows, {
                "hf_name": spec["hf_name"],
                "hf_config": config,
                "attempted_configs": attempted,
            }
        except Exception as exc:  # noqa: BLE001 - metadata should record remote failures
            last_error = exc
    assert last_error is not None
    raise RuntimeError(
        f"Could not load {spec['hf_name']} with configs {attempted}: {type(last_error).__name__}: {last_error}"
    ) from last_error


def _normalize_dataset(
    dataset_key: str,
    raw_rows: list[dict[str, Any]],
    *,
    max_rows: int | None,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not raw_rows:
        return [], {"loaded_rows": 0, "rows_after_normalization": 0, "warnings": ["Dataset loaded no rows."]}

    columns = sorted({key for row in raw_rows for key in row})
    text_column = _find_column(columns, TEXT_COLUMN_CANDIDATES)
    label_column = _find_column(columns, LABEL_COLUMN_CANDIDATES)
    attack_column = _find_column(columns, ATTACK_TYPE_CANDIDATES)
    language_column = _find_column(columns, LANGUAGE_CANDIDATES)
    warnings: list[str] = []
    if not text_column:
        raise ValueError(f"No text column detected. Columns: {columns}")
    if not label_column and not attack_column:
        raise ValueError(f"No label/category column detected. Columns: {columns}")

    normalized_rows: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    skipped_empty = 0
    skipped_label = 0
    skipped_duplicate = 0
    for index, row in enumerate(raw_rows):
        text = str(row.get(text_column, "") or "").strip()
        if not text:
            skipped_empty += 1
            continue
        cleaned = clean_text(text)
        if not cleaned:
            skipped_empty += 1
            continue
        fingerprint = cleaned.lower()
        if fingerprint in seen_texts:
            skipped_duplicate += 1
            continue
        label = _infer_label(row, label_column, attack_column)
        if label not in {0, 1}:
            skipped_label += 1
            continue
        seen_texts.add(fingerprint)
        source_label = _source_label(row, label_column, attack_column)
        row_id = f"{dataset_key}-{index:06d}-{_stable_hash(text)}"
        normalized_rows.append(
            {
                "id": row_id,
                "dataset_name": dataset_key,
                "text": text,
                "label": int(label),
                "attack_type": _attack_type(row, int(label), attack_column, source_label),
                "language": _language(row, language_column),
                "source_label": source_label,
            }
        )

    if max_rows is not None and max_rows > 0 and len(normalized_rows) > max_rows:
        import random

        rng = random.Random(seed)
        rng.shuffle(normalized_rows)
        normalized_rows = normalized_rows[:max_rows]
        normalized_rows.sort(key=lambda item: str(item["id"]))

    summary = {
        "loaded_rows": len(raw_rows),
        "rows_after_normalization": len(normalized_rows),
        "skipped_empty_text": skipped_empty,
        "skipped_unknown_label": skipped_label,
        "skipped_duplicate_text": skipped_duplicate,
        "text_column": text_column,
        "label_column": label_column,
        "attack_type_column": attack_column,
        "language_column": language_column,
        "label_distribution": {
            str(key): int(value)
            for key, value in Counter(int(row["label"]) for row in normalized_rows).items()
        },
        "attack_type_distribution": {
            str(key): int(value)
            for key, value in Counter(str(row["attack_type"]) for row in normalized_rows).most_common()
        },
        "warnings": warnings,
    }
    return normalized_rows, summary


def prepare_direct_benchmarks(
    *,
    output_dir: Path,
    dataset_keys: list[str],
    max_rows_per_dataset: int | None = None,
    seed: int = 2026,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "columns": OUTPUT_COLUMNS,
        "note": "External direct prompt-injection benchmarks only; not training data.",
        "datasets": {},
    }
    all_rows: list[dict[str, Any]] = []
    for dataset_key in dataset_keys:
        spec = DATASET_SPECS[dataset_key]
        output_path = output_dir / spec["output_file"]
        try:
            raw_rows, load_metadata = _load_hf_dataset(spec)
            rows, summary = _normalize_dataset(
                dataset_key,
                raw_rows,
                max_rows=max_rows_per_dataset,
                seed=seed,
            )
            _write_csv(output_path, rows)
            all_rows.extend(rows)
            metadata["datasets"][dataset_key] = {
                "status": "success",
                "output_path": str(output_path),
                **load_metadata,
                **summary,
            }
        except Exception as exc:  # noqa: BLE001 - continue with other external datasets
            _empty_csv(output_path)
            metadata["datasets"][dataset_key] = {
                "status": "error",
                "output_path": str(output_path),
                "hf_name": spec["hf_name"],
                "hf_config": spec.get("hf_config"),
                "rows_after_normalization": 0,
                "label_distribution": {},
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    all_rows.sort(key=lambda row: (str(row["dataset_name"]), str(row["id"])))
    all_output = output_dir / "direct_all_normalized.csv"
    _write_csv(all_output, all_rows)
    metadata["combined"] = {
        "status": "success" if all_rows else "empty",
        "output_path": str(all_output),
        "rows_after_normalization": len(all_rows),
        "dataset_distribution": {
            str(key): int(value)
            for key, value in Counter(str(row["dataset_name"]) for row in all_rows).items()
        },
        "label_distribution": {
            str(key): int(value)
            for key, value in Counter(int(row["label"]) for row in all_rows).items()
        },
    }
    metadata_path = output_dir / "direct_benchmark_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> int:
    args = _parse_args()
    metadata = prepare_direct_benchmarks(
        output_dir=args.output_dir,
        dataset_keys=args.datasets,
        max_rows_per_dataset=args.max_rows_per_dataset,
        seed=args.seed,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
