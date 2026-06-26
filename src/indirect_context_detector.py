"""Heuristic context detector for task-changing instructions in untrusted content."""

from __future__ import annotations

from typing import Any

from src.preprocessing import clean_text, prepare_text_for_detection, remove_vietnamese_accents


CONTEXT_SIGNAL_GROUPS: dict[str, dict[str, Any]] = {
    "task_override": {
        "weight": 0.85,
        "patterns": [
            "ignore previous instructions", "disregard system prompt", "change your task",
            "new task", "instead of the user request", "bỏ qua hướng dẫn trước đó",
            "bo qua huong dan truoc do", "thay đổi nhiệm vụ", "thay doi nhiem vu",
        ],
    },
    "assistant_directive": {
        "weight": 0.65,
        "patterns": [
            "assistant must", "you must now", "follow these instructions", "do not tell the user",
            "không nói với người dùng", "khong noi voi nguoi dung", "hãy làm theo chỉ dẫn này",
            "hay lam theo chi dan nay", "system message", "developer message",
        ],
    },
    "secret_access": {
        "weight": 0.80,
        "patterns": [
            "reveal system prompt", "api key", "access token", "password", "credentials",
            "training data", "tiết lộ prompt hệ thống", "tiet lo prompt he thong",
            "khóa api", "khoa api", "mật khẩu", "mat khau",
        ],
    },
    "external_exfiltration": {
        "weight": 0.90,
        "patterns": [
            "send data to", "send all private data", "exfiltrate", "upload secrets",
            "post the data to", "gửi dữ liệu cho", "gui du lieu cho", "chuyển dữ liệu ra ngoài",
            "chuyen du lieu ra ngoai",
        ],
    },
    "tool_execution": {
        "weight": 0.75,
        "patterns": [
            "call the tool", "execute command", "run shell", "os.system", "subprocess",
            "curl http", "wget http", "gọi công cụ", "goi cong cu", "chạy lệnh", "chay lenh",
        ],
    },
}

BENIGN_USER_TASK_VERBS = {
    "summarize", "summary", "extract", "translate", "classify", "explain", "review", "compare",
    "tóm tắt", "tom tat", "trích xuất", "trich xuat", "dịch", "dich", "giải thích", "giai thich",
}


def _search_spaces(text: str) -> dict[str, str]:
    prepared = prepare_text_for_detection(text)
    return {
        "raw": clean_text(text),
        "accentless": remove_vietnamese_accents(text),
        "canonical": prepared["cleaned_text"],
    }


def detect_context_manipulation(user_task: str, external_text: str) -> dict[str, Any]:
    """Estimate whether untrusted text is an instruction to the assistant, not task data."""
    spaces = _search_spaces(external_text)
    user_spaces = _search_spaces(user_task)
    matched_signals: list[dict[str, Any]] = []
    matched_groups: set[str] = set()

    for group_name, config in CONTEXT_SIGNAL_GROUPS.items():
        for pattern in config["patterns"]:
            normalized = clean_text(pattern)
            accentless = remove_vietnamese_accents(pattern)
            for space_name, value in spaces.items():
                if normalized in value or accentless in value:
                    matched_signals.append(
                        {
                            "group": group_name,
                            "pattern": pattern,
                            "matched_in": space_name,
                            "weight": float(config["weight"]),
                        }
                    )
                    matched_groups.add(group_name)
                    break

    user_haystack = " ".join(user_spaces.values())
    user_task_is_content_operation = any(verb in user_haystack for verb in BENIGN_USER_TASK_VERBS)
    instruction_conflict = user_task_is_content_operation and bool(
        matched_groups & {"task_override", "assistant_directive", "tool_execution", "external_exfiltration"}
    )

    if not matched_signals:
        score = 0.0
    else:
        strongest = max(signal["weight"] for signal in matched_signals)
        multi_signal_bonus = 0.07 * max(0, len(matched_groups) - 1)
        conflict_bonus = 0.10 if instruction_conflict else 0.0
        score = min(1.0, strongest + multi_signal_bonus + conflict_bonus)

    if matched_groups:
        explanation = "External content contains assistant-directed or task-changing signals: " + ", ".join(sorted(matched_groups)) + "."
    else:
        explanation = "External content appears to be task data; no assistant-directed instruction signal matched."

    return {
        "context_score": round(float(score), 4),
        "matched_signals": matched_signals,
        "instruction_conflict": instruction_conflict,
        "user_task_is_content_operation": user_task_is_content_operation,
        "explanation": explanation,
    }
