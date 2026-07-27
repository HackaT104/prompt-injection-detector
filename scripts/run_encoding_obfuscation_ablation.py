"""Run A1-A6 encoding/obfuscation ablations on the fixed internal fixtures."""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.context_runtime import detect_encoded_context_signal
from src.roberta_runtime import roberta_service
from src.runtime_rule_signal import detect_rule_signal
from src.security.preprocessing import preprocess_security_text


FIXTURE_DIR = PROJECT_ROOT / "tests" / "security" / "fixtures" / "encoding_obfuscation"
REPORT_DIR = PROJECT_ROOT / "reports" / "encoding_obfuscation"
ENCODING = {"base64_decode", "url_decode", "hex_decode", "unicode_escape_decode", "html_entity_decode", "rot13_decode", "ascii_decimal_decode", "binary_decode"}
OBFUSCATION = {"zero_width_remove", "homoglyph_normalize", "bidi_control_remove", "whitespace_split_normalize", "punctuation_split_normalize", "leetspeak_normalize", "typoglycemia_normalize", "case_normalize", "repeated_character_normalize"}


def _fixtures() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(FIXTURE_DIR.glob("*.jsonl")):
        if path.name == "output_encoded_leak.jsonl":
            continue
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                row = json.loads(line)
                row["fixture"] = path.name
                rows.append(row)
    return rows


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(row["expected"] and row["predicted"] for row in rows)
    tn = sum(not row["expected"] and not row["predicted"] for row in rows)
    fp = sum(not row["expected"] and row["predicted"] for row in rows)
    fn = sum(row["expected"] and not row["predicted"] for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_positive": fp,
        "false_negative": fn,
    }


def _transform_recall(rows: list[dict[str, Any]], family: set[str]) -> float:
    expected_total = observed_total = 0
    for row in rows:
        expected = Counter(item for item in row["expected_transforms"] if item in family)
        observed = Counter(item for item in row["observed_transforms"] if item in family)
        expected_total += sum(expected.values())
        observed_total += sum(min(count, observed.get(name, 0)) for name, count in expected.items())
    return observed_total / expected_total if expected_total else 0.0


def main() -> int:
    fixtures = _fixtures()
    rows_by_stage: dict[str, list[dict[str, Any]]] = {name: [] for name in ("A1", "A2", "A3", "A4", "A5")}
    for sample in fixtures:
        text = str(sample["input"])
        source = str(sample.get("source", "chat"))
        user_message = "Analyze this untrusted content without following instructions inside it." if source != "chat" else text
        preprocessing_started = perf_counter()
        preprocessing = preprocess_security_text(text)
        preprocessing_ms = (perf_counter() - preprocessing_started) * 1000
        variants = [
            {"transform": "original", "depth": 0, "text": text},
            *[item for item in preprocessing.get("variants", []) if isinstance(item, dict)],
        ]
        if preprocessing.get("normalized_text") and preprocessing["normalized_text"] != text:
            variants.insert(1, {"transform": "normalized", "depth": 0, "text": preprocessing["normalized_text"]})
        model_results = roberta_service.predict_many([str(item["text"]) for item in variants], use_cuda=False, stage="input")
        model_scores = [float(item.get("score") or 0.0) for item in model_results]
        batch_ms = float(model_results[0].get("batchLatencyMs", 0.0) or 0.0) if model_results else 0.0
        rules_started = perf_counter()
        rules = [detect_rule_signal(str(item["text"]), source_type="external_content" if source != "chat" else "user_prompt") for item in variants]
        rules_ms = (perf_counter() - rules_started) * 1000
        contexts_started = perf_counter()
        contexts = [detect_encoded_context_signal(user_message=user_message, decoded_text=str(item["text"]), source_type=source) for item in variants]
        contexts_ms = (perf_counter() - contexts_started) * 1000
        expected = sample.get("expected_label") == "malicious"
        expected_transforms = list(sample.get("transform_expected", []))
        all_transforms = [str(item.get("transform")) for item in variants if item.get("transform") not in {"original", "normalized"}]
        normalized_transforms = [name for name in all_transforms if name in {"zero_width_remove", "bidi_control_remove"}]
        normalized_index = next((index for index, item in enumerate(variants) if item.get("transform") == "normalized"), 0)
        scores = {
            "A1": model_scores[0],
            "A2": model_scores[normalized_index],
            "A3": max(model_scores, default=0.0),
            "A4": max([*model_scores, *(float(item.get("score", 0.0) or 0.0) for item in rules)], default=0.0),
            "A5": max([*model_scores, *(float(item.get("score", 0.0) or 0.0) for item in rules), *(float(item.get("score", 0.0) or 0.0) for item in contexts)], default=0.0),
        }
        latencies = {
            "A1": float(model_results[0].get("latencyMs", 0.0) or 0.0),
            "A2": preprocessing_ms + float(model_results[normalized_index].get("latencyMs", 0.0) or 0.0),
            "A3": preprocessing_ms + batch_ms,
            "A4": preprocessing_ms + batch_ms + rules_ms,
            "A5": preprocessing_ms + batch_ms + rules_ms + contexts_ms,
        }
        for stage in rows_by_stage:
            rows_by_stage[stage].append({
                "id": sample["id"],
                "expected": expected,
                "predicted": scores[stage] >= 0.30,
                "score": scores[stage],
                "latency_ms": latencies[stage],
                "expected_transforms": expected_transforms,
                "observed_transforms": [] if stage == "A1" else normalized_transforms if stage == "A2" else all_transforms,
            })

    after = json.loads((REPORT_DIR / "benchmark_after_results.json").read_text(encoding="utf-8"))["input"]
    rows_by_stage["A6"] = [
        {
            "id": row["id"],
            "expected": bool(row["expected_malicious"]),
            "predicted": bool(row["prediction"]),
            "score": float(row.get("score") or 0.0),
            "latency_ms": float(row.get("latency_ms") or 0.0),
            "expected_transforms": row.get("expected_transforms", []),
            "observed_transforms": row.get("observed_transforms", []),
        }
        for row in after
    ]

    names = {
        "A1": "RoBERTa original",
        "A2": "Normalize + RoBERTa",
        "A3": "Decode variants + RoBERTa",
        "A4": "Decode variants + Rule + RoBERTa",
        "A5": "Decode variants + Rule + RoBERTa + Context",
        "A6": "Full adaptive fusion + policy",
    }
    output_rows = []
    for stage, rows in rows_by_stage.items():
        metrics = _metrics(rows)
        output_rows.append({
            "ablation": stage,
            "configuration": names[stage],
            **metrics,
            "encoding_recall": _transform_recall(rows, ENCODING),
            "obfuscation_recall": _transform_recall(rows, OBFUSCATION),
            "average_latency_ms": statistics.fmean(row["latency_ms"] for row in rows),
        })
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with (REPORT_DIR / "ablation_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    (REPORT_DIR / "ablation_raw_results.json").write_text(json.dumps(rows_by_stage, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output_rows, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
