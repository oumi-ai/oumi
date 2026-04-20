# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared utilities for environment runtimes."""

from __future__ import annotations

import json
import re
from typing import Any

from oumi.core.configs.params.grounding_params import GroundingFact
from oumi.utils.str_utils import repair_json_braces

TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def strip_tool_call_blocks(text: str) -> str:
    """Remove every ``<tool_call>...</tool_call>`` block from ``text``."""
    return TOOL_CALL_RE.sub("", text)


def close_dangling_tool_call(text: str) -> str:
    """Re-append ``</tool_call>`` when stop-sequence inference stripped it."""
    opens = text.count("<tool_call>")
    closes = text.count("</tool_call>")
    if opens > closes:
        return text + "</tool_call>"
    return text


def truncate_after_last_tool_call(text: str) -> str:
    """Return the text prefix up to and including the LAST ``</tool_call>``."""
    last_close = text.rfind("</tool_call>")
    if last_close == -1:
        return text
    return text[: last_close + len("</tool_call>")]


def canonicalize_tool_call_bodies(text: str) -> str:
    """Replace each ``<tool_call>`` body with a canonical JSON re-serialization.

    Models occasionally emit malformed JSON inside tool calls — most commonly
    an extra trailing ``}`` (``}}}``) or a missing close when the stop
    sequence fires early. Without this pass, the malformation is persisted
    verbatim into the output dataset (it would only be repaired transiently
    during tool execution). Downstream consumers of the dataset would then
    need to re-implement the repair.

    For each ``<tool_call>...</tool_call>`` block, this function attempts to
    brace-repair the body and, on success, substitutes it with the compact
    ``json.dumps`` form. Bodies that cannot be repaired are left untouched;
    the tool-executor surfaces a structured error message for those.
    """

    def _replace(match: re.Match[str]) -> str:
        body = match.group(1).strip()
        repaired = repair_json_braces(body)
        if repaired is None:
            return match.group(0)
        try:
            parsed = json.loads(repaired)
        except json.JSONDecodeError:
            return match.group(0)
        return f"<tool_call>{json.dumps(parsed)}</tool_call>"

    return TOOL_CALL_RE.sub(_replace, text)


def _format_grounding_value(value: Any) -> str:
    """Render a fact value as a quoted string or bare literal."""
    if isinstance(value, str):
        return f'"{value}"'
    return str(value)


def describe_grounding_default(facts: list[GroundingFact]) -> str:
    """Render grounding facts as a bulleted markdown block.

    Each fact's ``data`` dict is rendered as a single ``key=value,
    key=value`` line. Returns "" for an empty fact list.
    """
    if not facts:
        return ""
    lines: list[str] = []
    for fact in facts:
        parts = [
            f"{key}={_format_grounding_value(value)}"
            for key, value in fact.data.items()
        ]
        lines.append(f"- {', '.join(parts)}")
    return "\n".join(lines)
