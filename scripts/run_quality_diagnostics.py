from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quality_reports import run_quality_reports


if __name__ == "__main__":
    summary = run_quality_reports()
    print(json.dumps(summary, ensure_ascii=True, indent=2))
