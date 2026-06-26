"""Back up and deprecate DistilBERT without deleting its checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_SOURCE = MODELS_DIR / "transformers" / "distilbert_v4"


def deprecate_distilbert(source: str | Path = DEFAULT_SOURCE) -> dict[str, str]:
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Không tìm thấy DistilBERT checkpoint: {source_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    deprecated_root = MODELS_DIR / "deprecated"
    backup_path = deprecated_root / f"distilbert_{timestamp}"
    deprecated_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_path, backup_path)

    payload = {
        "model": source_path.name,
        "status": "deprecated",
        "deprecated_at": datetime.now(timezone.utc).isoformat(),
        "reason": "Replaced in the default runtime by RoBERTa v5 VI and XLM-RoBERTa v5 VI.",
        "original_checkpoint": str(source_path.resolve()),
        "backup_checkpoint": str(backup_path.resolve()),
        "runtime_policy": "Not selected by default; explicit legacy usage remains backward compatible.",
    }
    (backup_path / "DEPRECATION.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (source_path / "DEPRECATED.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    index_path = deprecated_root / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8-sig")) if index_path.exists() else {"models": []}
    index.setdefault("models", []).append(payload)
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Backup and deprecate DistilBERT.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    args = parser.parse_args()
    print(json.dumps(deprecate_distilbert(args.source), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

