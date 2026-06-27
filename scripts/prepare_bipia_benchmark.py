"""Prepare BIPIA as an external/OOD benchmark for the detector.

This script does not create training data. It reads the BIPIA benchmark files,
creates safe rows from clean contexts, creates indirect-injection rows by
inserting BIPIA attack strings into those contexts, and writes the normalized
CSV used by ``scripts/evaluate_bipia.py``.

Default command:
    python scripts/prepare_bipia_benchmark.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BIPIA_URL = "https://github.com/microsoft/BIPIA.git"
DEFAULT_BIPIA_ROOT = PROJECT_ROOT / "data" / "external_benchmark" / "bipia"
DEFAULT_RAW_REPO = DEFAULT_BIPIA_ROOT / "raw" / "BIPIA"
DEFAULT_OUTPUT = DEFAULT_BIPIA_ROOT / "bipia_normalized.csv"
DEFAULT_METADATA = DEFAULT_BIPIA_ROOT / "bipia_normalized.metadata.json"
DEFAULT_TASKS = ["email", "table", "qa", "abstract"]
DEFAULT_POSITIONS = ["end", "start", "middle"]

OUTPUT_COLUMNS = [
    "id",
    "user_task",
    "external_content",
    "label",
    "attack_type",
    "source_task",
    "language",
    "expected_behavior",
    "difficulty",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize BIPIA into the project benchmark schema.")
    parser.add_argument("--bipia-root", type=Path, default=DEFAULT_RAW_REPO, help="Local microsoft/BIPIA repo path.")
    parser.add_argument("--repo-url", default=BIPIA_URL, help="BIPIA Git repository URL.")
    parser.add_argument("--download", action="store_true", help="Clone BIPIA if --bipia-root is missing.")
    parser.add_argument("--no-download", action="store_true", help="Fail instead of cloning when BIPIA is missing.")
    parser.add_argument("--split", default="test", choices=["test", "train"], help="BIPIA split to read.")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS), help="Comma-separated source tasks.")
    parser.add_argument("--max-samples", type=int, default=1000, help="Maximum normalized rows to write.")
    parser.add_argument("--seed", type=int, default=2026, help="Deterministic sampling seed.")
    parser.add_argument("--positions", default=",".join(DEFAULT_POSITIONS), help="Attack insertion positions.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Normalized CSV output path.")
    return parser.parse_args()


def _clone_if_needed(repo_path: Path, repo_url: str, should_download: bool) -> None:
    if repo_path.exists():
        return
    if not should_download:
        raise FileNotFoundError(
            f"BIPIA repo not found at {repo_path}. Run with --download or clone {repo_url} there."
        )
    repo_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(repo_path)],
        check=True,
    )


def _repo_commit(repo_path: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_attacks(path: Path) -> list[dict[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    attacks: list[dict[str, str]] = []
    for group_name, prompts in raw.items():
        for index, prompt in enumerate(prompts):
            attacks.append(
                {
                    "attack_group": str(group_name),
                    "attack_index": str(index),
                    "attack_name": f"{group_name}-{index}",
                    "attack_text": str(prompt),
                }
            )
    return attacks


def _slug(value: str, max_len: int = 36) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return normalized[:max_len] or "item"


def _stable_hash(text: str, length: int = 10) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:length]


def _default_user_task(source_task: str, sample: dict[str, Any]) -> str:
    question = str(sample.get("question", "")).strip()
    if question:
        return question
    defaults = {
        "email": "Answer the user's question based on this email.",
        "table": "Answer the user's question based on this table.",
        "qa": "Answer the user's question based on this web document.",
        "abstract": "Summarize this document.",
        "code": "Fix the bug using the external answer.",
    }
    return defaults.get(source_task, f"Process this {source_task} content.")


def _extract_context(sample: dict[str, Any]) -> str:
    for key in ["context", "document", "content", "text", "email", "webpage", "table"]:
        value = sample.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return json.dumps(sample, ensure_ascii=False)


def _insert_attack(context: str, attack: str, position: str, seed: int) -> str:
    position = position.lower()
    if position == "start":
        return f"{attack}\n\n{context}"
    if position == "middle":
        rng = random.Random(seed)
        separators = [match.start() + 1 for match in re.finditer(r"[.!?]\s+", context)]
        if separators:
            index = rng.choice(separators)
        else:
            index = max(0, len(context) // 2)
        return f"{context[:index].rstrip()}\n\n{attack}\n\n{context[index:].lstrip()}"
    return f"{context}\n\n{attack}"


def _difficulty_for_attack(attack_text: str) -> str:
    normalized = attack_text.lower()
    easy_terms = [
        "ignore previous",
        "system prompt",
        "developer message",
        "api key",
        "execute",
        "command",
        "os.system",
        "subprocess",
    ]
    medium_terms = [
        "your response",
        "your answer",
        "your reply",
        "reply using",
        "encode",
        "base64",
        "add a sentence",
        "modify your",
        "include in your",
        "integrate",
    ]
    if any(term in normalized for term in easy_terms):
        return "easy"
    if any(term in normalized for term in medium_terms):
        return "medium"
    return "hard"


def _read_contexts(benchmark_dir: Path, tasks: list[str], split: str) -> tuple[list[dict[str, Any]], list[str]]:
    contexts: list[dict[str, Any]] = []
    warnings: list[str] = []
    for task in tasks:
        path = benchmark_dir / task / f"{split}.jsonl"
        if not path.exists():
            warnings.append(
                f"Skipped {task}: missing {path}. BIPIA requires users to generate qa/abstract contexts separately."
            )
            continue
        for index, sample in enumerate(_load_jsonl(path)):
            contexts.append({"source_task": task, "source_index": index, "sample": sample})
    return contexts, warnings


def _safe_rows(contexts: list[dict[str, Any]], max_rows: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in contexts[:max_rows]:
        source_task = str(item["source_task"])
        index = int(item["source_index"])
        sample = dict(item["sample"])
        user_task = _default_user_task(source_task, sample)
        context = _extract_context(sample)
        rows.append(
            {
                "id": f"bipia-{source_task}-{index:05d}-safe",
                "user_task": user_task,
                "external_content": context,
                "label": "0",
                "attack_type": "safe",
                "source_task": source_task,
                "language": "en",
                "expected_behavior": "safe",
                "difficulty": "easy",
            }
        )
    return rows


def _injection_rows(
    contexts: list[dict[str, Any]],
    attacks: list[dict[str, str]],
    *,
    max_rows: int,
    positions: list[str],
    seed: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not contexts or not attacks or max_rows <= 0:
        return rows
    candidates = [
        (position, item, attack)
        for position in positions
        for item in contexts
        for attack in attacks
    ]
    rng = random.Random(seed)
    rng.shuffle(candidates)
    for position, item, attack in candidates[:max_rows]:
        source_task = str(item["source_task"])
        index = int(item["source_index"])
        sample = dict(item["sample"])
        user_task = _default_user_task(source_task, sample)
        context = _extract_context(sample)
        attack_text = attack["attack_text"]
        attack_name = attack["attack_name"]
        injected = _insert_attack(
            context,
            attack_text,
            position,
            seed=seed + index + len(rows),
        )
        rows.append(
            {
                "id": (
                    f"bipia-{source_task}-{index:05d}-"
                    f"{_slug(attack_name)}-{position}-{_stable_hash(attack_text)}"
                ),
                "user_task": user_task,
                "external_content": injected,
                "label": "1",
                "attack_type": "indirect_injection",
                "source_task": source_task,
                "language": "en",
                "expected_behavior": "block",
                "difficulty": _difficulty_for_attack(attack_text),
            }
        )
    return rows


def prepare_bipia(
    *,
    repo_path: Path,
    output_path: Path,
    split: str = "test",
    tasks: list[str] | None = None,
    max_samples: int = 1000,
    positions: list[str] | None = None,
    seed: int = 2026,
    download: bool = False,
    repo_url: str = BIPIA_URL,
) -> dict[str, Any]:
    _clone_if_needed(repo_path, repo_url, should_download=download)
    benchmark_dir = repo_path / "benchmark"
    if not benchmark_dir.exists():
        raise FileNotFoundError(f"BIPIA benchmark directory not found: {benchmark_dir}")

    selected_tasks = tasks or DEFAULT_TASKS
    selected_positions = positions or DEFAULT_POSITIONS
    contexts, warnings = _read_contexts(benchmark_dir, selected_tasks, split)
    if not contexts:
        raise RuntimeError(f"No BIPIA context rows found for tasks={selected_tasks} split={split}.")
    attack_file = benchmark_dir / f"text_attack_{split}.json"
    if not attack_file.exists():
        raise FileNotFoundError(attack_file)
    attacks = _load_attacks(attack_file)

    safe_target = min(len(contexts), max(1, max_samples // 3))
    safe = _safe_rows(contexts, safe_target)
    injection_target = max(0, max_samples - len(safe))
    injection = _injection_rows(
        contexts,
        attacks,
        max_rows=injection_target,
        positions=selected_positions,
        seed=seed,
    )
    rows = safe + injection

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    label_counts = Counter(row["label"] for row in rows)
    source_counts = Counter(row["source_task"] for row in rows)
    difficulty_counts = Counter(row["difficulty"] for row in rows)
    metadata = {
        "source": "microsoft/BIPIA",
        "source_url": repo_url,
        "source_commit": _repo_commit(repo_path),
        "split": split,
        "selected_tasks": selected_tasks,
        "selected_positions": selected_positions,
        "output_path": str(output_path),
        "rows": len(rows),
        "safe_rows": int(label_counts.get("0", 0)),
        "injection_rows": int(label_counts.get("1", 0)),
        "source_task_counts": dict(sorted(source_counts.items())),
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "attack_prompt_count": len(attacks),
        "context_rows_available": len(contexts),
        "warnings": warnings,
        "data_leakage_note": (
            "BIPIA is prepared only under data/external_benchmark/bipia and is not copied "
            "into any training dataset by this script."
        ),
    }
    DEFAULT_METADATA.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    args = _parse_args()
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    positions = [position.strip() for position in args.positions.split(",") if position.strip()]
    should_download = args.download or not args.no_download
    metadata = prepare_bipia(
        repo_path=args.bipia_root,
        output_path=args.output,
        split=args.split,
        tasks=tasks,
        max_samples=args.max_samples,
        positions=positions,
        seed=args.seed,
        download=should_download,
        repo_url=args.repo_url,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
