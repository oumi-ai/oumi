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
from typing import Any

from oumi.core.configs.params.grounding_params import GroundingFact


def _format_grounding_value(value: Any) -> str:
    """Render a fact value as a quoted string or bare literal."""
    if isinstance(value, str):
        return json.dumps(value)
    return str(value)


def describe_grounding_default(facts: list[GroundingFact]) -> str:
    """Render grounding facts as a bulleted markdown block.

    Each fact's ``data`` dict is rendered as a single ``key=value,
    key=value`` line. Facts with empty ``data`` are skipped.
    """
    lines: list[str] = []
    for fact in facts:
        if not fact.data:
            continue
        parts = [
            f"{key}={_format_grounding_value(value)}"
            for key, value in fact.data.items()
        ]
        lines.append(f"- {', '.join(parts)}")
    return "\n".join(lines)
