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

import dataclasses
import importlib
import json
from collections.abc import Callable
from typing import Any, TypeVar

import jsonschema

from oumi.core.configs.params.base_params import BaseParams
from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.configs.params.grounding_params import GroundingFact
from oumi.core.configs.params.tool_params import ToolError, ToolParams
from oumi.core.types.tool_call import ToolResult

_KwargsT = TypeVar("_KwargsT", bound=BaseParams)


def parse_env_kwargs(
    kwargs_cls: type[_KwargsT], params: EnvironmentParams, *, env_label: str
) -> _KwargsT:
    """Build a validated env-kwargs dataclass from ``params.env_kwargs``.

    Rejects unrecognized keys (naming them) before constructing and
    finalize-validating the dataclass. Shared by concrete environments.
    """
    raw_kwargs = params.env_kwargs or {}
    known = {field.name for field in dataclasses.fields(kwargs_cls)}
    unknown = set(raw_kwargs) - known
    if unknown:
        raise ValueError(
            f"{env_label} got unknown env_kwargs: {sorted(unknown)}. "
            f"Known: {sorted(known)}"
        )
    kwargs = kwargs_cls(**raw_kwargs)
    kwargs.finalize_and_validate()
    return kwargs


def import_executor(dotted: str, tool_id: str) -> Callable[..., Any]:
    """Resolve a dotted import path to a callable. Raises ValueError on failure."""
    module_path, _, attr = dotted.rpartition(".")
    if not module_path or not attr:
        raise ValueError(
            f"Tool '{tool_id}': executor '{dotted}' must be a dotted import "
            f"path (e.g. 'pkg.module.fn')."
        )
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ValueError(
            f"Tool '{tool_id}': cannot import executor module '{module_path}': {e}"
        ) from e
    executor = getattr(module, attr, None)
    if executor is None:
        raise ValueError(
            f"Tool '{tool_id}': module '{module_path}' has no attribute '{attr}'."
        )
    if not callable(executor):
        raise ValueError(
            f"Tool '{tool_id}': executor '{dotted}' resolved to a non-callable."
        )
    return executor


def validate_executor_result(tool: ToolParams, result: Any) -> ToolResult:
    """Check an executor return is a ToolResult conforming to ``output_schema``."""
    if not isinstance(result, ToolResult):
        raise ToolError(
            f"Tool '{tool.id}' executor must return ToolResult, got "
            f"{type(result).__name__}."
        )
    if tool.output_schema is not None:
        try:
            jsonschema.validate(result.output, tool.output_schema)
        except jsonschema.ValidationError as e:
            raise ToolError(
                f"Tool '{tool.id}' executor output failed schema validation: {e}"
            ) from e
    return result


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
