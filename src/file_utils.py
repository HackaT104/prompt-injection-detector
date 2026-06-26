"""Small file-writing helpers for Windows/OneDrive friendly reports."""

from __future__ import annotations

import time
from pathlib import Path


def safe_write_text(path: str | Path, content: str, encoding: str = "utf-8") -> None:
    """Write text with a temp-file fallback for transient Windows file locks."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None

    for attempt in range(4):
        try:
            target.write_text(content, encoding=encoding)
            return
        except PermissionError as exc:
            last_error = exc
            temp_path = target.with_name(f".{target.name}.tmp")
            try:
                temp_path.write_text(content, encoding=encoding)
                temp_path.replace(target)
                return
            except PermissionError as replace_exc:
                last_error = replace_exc
                time.sleep(0.25 * (attempt + 1))

    if last_error:
        raise last_error

