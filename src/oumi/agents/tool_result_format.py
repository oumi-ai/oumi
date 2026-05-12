# Copyright 2025 - Oumi
# Licensed under the Apache License, Version 2.0
"""Compact tool outputs to fit in small-model context windows."""

from __future__ import annotations

import json
from typing import Any

_TRUNCATION_HINT_KEY = "__truncated"
_TRUNCATION_NOTE_KEY = "__truncation_note"


def compact_tool_output(
    output: Any,
    *,
    max_chars: int = 8192,
    keep_list_items: int = 5,
    keep_string_chars: int = 1024,
) -> Any:
    """Best-effort shrink of a tool output to fit within ``max_chars``."""
    try:
        raw = json.dumps(output, default=str)
    except (TypeError, ValueError):
        return {
            _TRUNCATION_HINT_KEY: True,
            _TRUNCATION_NOTE_KEY: "Tool output was not JSON-serializable.",
            "repr": repr(output)[:keep_string_chars],
        }
    if len(raw) <= max_chars:
        return output

    clipped = _clip(output, keep_list_items, keep_string_chars)
    if isinstance(clipped, dict):
        clipped = {
            **clipped,
            _TRUNCATION_HINT_KEY: True,
            _TRUNCATION_NOTE_KEY: (
                f"Output exceeded {max_chars} chars; lists clipped to "
                f"{keep_list_items} items, strings to {keep_string_chars} chars."
            ),
        }
    return clipped


def _clip(value: Any, keep_list_items: int, keep_string_chars: int) -> Any:
    if isinstance(value, str):
        if len(value) <= keep_string_chars:
            return value
        return (
            value[:keep_string_chars]
            + f"… [{len(value) - keep_string_chars} more chars]"
        )
    if isinstance(value, list):
        head = [
            _clip(v, keep_list_items, keep_string_chars)
            for v in value[:keep_list_items]
        ]
        if len(value) > keep_list_items:
            head.append(f"… [{len(value) - keep_list_items} more items]")
        return head
    if isinstance(value, dict):
        return {
            k: _clip(v, keep_list_items, keep_string_chars) for k, v in value.items()
        }
    return value
