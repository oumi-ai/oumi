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

"""Synthetic environment backed by LLM-simulated tool execution."""

from __future__ import annotations

import copy
import dataclasses
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jsonschema

from oumi.core.configs.inference_config import InferenceConfig
from oumi.core.configs.params.base_params import BaseParams
from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.configs.params.guided_decoding_params import GuidedDecodingParams
from oumi.core.configs.params.tool_params import ToolError, ToolParams
from oumi.core.registry import register_environment
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.core.types.tool_call import ToolResult
from oumi.environments.base_environment import BaseEnvironment
from oumi.utils.str_utils import extract_json

if TYPE_CHECKING:
    from oumi.core.inference.base_inference_engine import BaseInferenceEngine


@dataclass
class SyntheticStateParams(BaseParams):
    """Optional state configuration for a synthetic environment."""

    state_schema: dict[str, Any] | None = None
    initial_state: dict[str, Any] | None = None

    def __post_init__(self):
        """Validate state config consistency."""
        if self.state_schema is not None and self.initial_state is not None:
            jsonschema.validate(self.initial_state, self.state_schema)


@dataclass
class SyntheticEnvironmentKwargs(BaseParams):
    """Type-specific kwargs for SyntheticEnvironment."""

    system_prompt: str = ""
    state_params: SyntheticStateParams | None = None
    cache_by_input: bool = True

    def __post_init__(self) -> None:
        """Coerce state_params dict into SyntheticStateParams if needed."""
        if isinstance(self.state_params, dict):
            self.state_params = SyntheticStateParams(**self.state_params)

    def __finalize_and_validate__(self) -> None:
        """Finalize and validate the kwargs."""
        if not self.system_prompt:
            raise ValueError(
                "SyntheticEnvironmentKwargs.system_prompt cannot be empty."
            )
        if self.state_params is not None and self.cache_by_input:
            raise ValueError(
                "SyntheticEnvironmentKwargs.cache_by_input must be False when "
                "state_params is provided."
            )


@register_environment("synthetic")
class SyntheticEnvironment(BaseEnvironment):
    """LLM-simulated environment with optional mutable state."""

    def __init__(
        self,
        params: EnvironmentParams,
        kwargs: SyntheticEnvironmentKwargs,
    ) -> None:
        """Initialize a SyntheticEnvironment with the given params and kwargs."""
        self._params = params
        self._kwargs = kwargs
        self._cache: dict[str, ToolResult] = {}
        self._state: dict[str, Any] | None = (
            copy.deepcopy(kwargs.state_params.initial_state)
            if kwargs.state_params is not None
            and kwargs.state_params.initial_state is not None
            else None
        )
        self._engine: BaseInferenceEngine | None = None
        self._base_inference_config: InferenceConfig | None = None

    def attach_inference(
        self,
        engine: BaseInferenceEngine,
        base_config: InferenceConfig,
    ) -> None:
        """Inject the orchestrator's inference engine + base config."""
        self._engine = engine
        self._base_inference_config = base_config

    @classmethod
    def from_params(cls, params: EnvironmentParams) -> SyntheticEnvironment:
        """Build a SyntheticEnvironment from its params object."""
        kwargs = SyntheticEnvironmentKwargs(**(params.env_kwargs or {}))
        kwargs.finalize_and_validate()
        return cls(params, kwargs)

    @property
    def current_state(self) -> dict[str, Any] | None:
        """Return the current in-memory state snapshot."""
        if self._state is None:
            return None
        return copy.deepcopy(self._state)

    @staticmethod
    def _cache_key(tool_id: str, arguments: dict[str, Any]) -> str:
        """Build a stable cache key from tool id and arguments."""
        return f"{tool_id}::{json.dumps(arguments, sort_keys=True)}"

    def _resolve_cached(
        self, tool_id: str, arguments: dict[str, Any]
    ) -> ToolResult | None:
        """Look up a cached result for the given tool call."""
        if not self._kwargs.cache_by_input:
            return None
        result = self._cache.get(self._cache_key(tool_id, arguments))
        if result is None:
            return None
        return ToolResult(
            output=copy.deepcopy(result.output),
            updated_state=copy.deepcopy(result.updated_state),
        )

    def _cache_result(
        self, tool_id: str, arguments: dict[str, Any], result: ToolResult
    ) -> None:
        """Store a generated result in the cache."""
        if not self._kwargs.cache_by_input:
            return
        self._cache[self._cache_key(tool_id, arguments)] = ToolResult(
            output=copy.deepcopy(result.output),
            updated_state=copy.deepcopy(result.updated_state),
        )

    def _lookup_tool(self, tool_id: str) -> ToolParams:
        for tool in self._params.tools:
            if tool.id == tool_id:
                return tool
        raise ValueError(
            f"Tool '{tool_id}' not found in environment '{self._params.id}'. "
            f"Available tools: {[tool.id for tool in self._params.tools]}"
        )

    def step(self, calls: list[tuple[str, dict[str, Any]]]) -> list[ToolResult]:
        """Execute synthetic tool calls. Cache-misses batched per tool_id.

        Raises:
            RuntimeError: If ``attach_inference`` was not called.
            ValueError: If any tool id is unknown.
            ToolError: On simulator parse failure or output_schema mismatch.
        """
        if not calls:
            return []
        for tool_id, _ in calls:
            self._lookup_tool(tool_id)
        if self._engine is None or self._base_inference_config is None:
            raise RuntimeError(
                "SyntheticEnvironment.step called before attach_inference(). "
                "Wire the synthesizer's engine via attach_inference(engine, "
                "base_config) before invoking step()."
            )

        results: list[ToolResult | None] = [None] * len(calls)
        misses: list[tuple[int, str, dict[str, Any]]] = []
        for i, (tool_id, args) in enumerate(calls):
            cached = self._resolve_cached(tool_id, args)
            if cached is not None:
                results[i] = cached
            else:
                misses.append((i, tool_id, args))

        groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for i, tool_id, args in misses:
            groups.setdefault(tool_id, []).append((i, args))

        for tool_id, group in groups.items():
            tool = self._lookup_tool(tool_id)
            convs = [self._build_call_conv(tool, args) for _, args in group]
            inferred = self._engine.infer(convs, self._simulator_inference_config(tool))
            if len(inferred) != len(group):
                raise RuntimeError(
                    f"Simulator returned {len(inferred)} responses for "
                    f"{len(group)} calls to '{tool_id}'."
                )
            for (idx, args), conv in zip(group, inferred):
                raw = self._extract_text(conv)
                result = self._parse_and_validate(raw, tool)
                self._cache_result(tool_id, args, result)
                results[idx] = result

        assert all(r is not None for r in results), (
            "every call must produce a ToolResult"
        )
        return results  # type: ignore[return-value]

    def _build_simulator_system_prompt(self, tool: ToolParams) -> str:
        """Compose the simulator system prompt: env persona + tool schema."""
        return (
            f"{self._kwargs.system_prompt}\n\n"
            f"You are simulating the `{tool.id}` tool. Respond ONLY with a "
            f"JSON object matching the tool's output schema. Do NOT include "
            f"explanations, markdown, or surrounding prose.\n\n"
            f"Tool schema:\n{json.dumps(tool.to_llm_schema(), indent=2)}"
        )

    def _build_call_conv(
        self, tool: ToolParams, arguments: dict[str, Any]
    ) -> Conversation:
        """Build the simulator conversation for one tool call."""
        user_payload = json.dumps(
            {"tool": tool.id, "arguments": arguments}, sort_keys=True
        )
        return Conversation(
            messages=[
                Message(
                    role=Role.SYSTEM,
                    content=self._build_simulator_system_prompt(tool),
                ),
                Message(role=Role.USER, content=user_payload),
            ]
        )

    def _simulator_inference_config(self, tool: ToolParams) -> InferenceConfig:
        """Overlay guided decoding for the tool's output_schema onto base_config.

        Tools without ``output_schema`` get the permissive ``{"type": "object"}``
        constraint. Mirrors ``ConversationSynthesizer._planner_inference_config``.
        """
        assert self._base_inference_config is not None
        schema = tool.output_schema or {"type": "object"}
        sim_gen = dataclasses.replace(
            self._base_inference_config.generation,
            guided_decoding=GuidedDecodingParams(json=schema),
        )
        return dataclasses.replace(self._base_inference_config, generation=sim_gen)

    @staticmethod
    def _extract_text(conv: Conversation) -> str:
        """Pull the simulator's text response from an inferred conversation.

        Returns ``""`` (which forces the ``ToolError`` path in
        ``_parse_and_validate``) when the last message is not an assistant
        turn — guards against a passthrough/partial-failure path where the
        engine returns ``convs`` unchanged and ``messages[-1]`` is still the
        user payload (itself valid JSON of the form
        ``{"tool": ..., "arguments": ...}``).
        """
        if not conv.messages:
            return ""
        last = conv.messages[-1]
        if last.role != Role.ASSISTANT:
            return ""
        content = last.content
        return content.strip() if isinstance(content, str) else ""

    @staticmethod
    def _parse_and_validate(raw: str, tool: ToolParams) -> ToolResult:
        """Parse simulator output and validate against ``tool.output_schema``."""
        if not raw:
            raise ToolError(f"Simulator returned empty response for '{tool.id}'.")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            extracted = extract_json(raw, expected_type=dict)
            if extracted is None:
                raise ToolError(
                    f"Simulator output for '{tool.id}' is not valid JSON: {raw[:200]!r}"
                ) from None
            parsed = extracted
        if not isinstance(parsed, dict):
            raise ToolError(
                f"Simulator output for '{tool.id}' must be a JSON object, "
                f"got {type(parsed).__name__}."
            )
        if tool.output_schema is not None:
            try:
                jsonschema.validate(parsed, tool.output_schema)
            except jsonschema.ValidationError as e:
                raise ToolError(
                    f"Simulator output for '{tool.id}' failed schema validation: {e}"
                ) from e
        return ToolResult(output=parsed)
