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

"""Synthetic environment backed by LLM-simulated or Python-executed tools.

Stateless mode (``state_params=None``) batches LLM-simulated tool outputs
per tool id, cached by ``(tool_id, args)``. Individual tools may still
opt into Python execution by setting ``executor`` -- LLM simulation is
the fallback for tools without one. Stateful mode (``state_params`` is
set) requires every tool to define ``executor``; the env runs them
sequentially so state mutations thread through the batch.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jsonschema

from oumi.core.configs.inference_config import InferenceConfig
from oumi.core.configs.params.base_params import BaseParams
from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.configs.params.grounding_params import (
    GroundingFact,
    StateGroundingConfig,
)
from oumi.core.configs.params.guided_decoding_params import GuidedDecodingParams
from oumi.core.configs.params.tool_params import ToolError, ToolParams
from oumi.core.registry import register_environment
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.core.types.tool_call import ToolResult
from oumi.environments.base_environment import BaseEnvironment
from oumi.environments.utils import import_executor, validate_executor_result
from oumi.utils.str_utils import extract_json

if TYPE_CHECKING:
    from oumi.core.inference.base_inference_engine import BaseInferenceEngine


@dataclass
class SyntheticStateParams(BaseParams):
    """Optional state configuration for a synthetic environment.

    State grounding for these pools is declared at the env level
    via ``EnvironmentParams.grounding.state`` — each entry's
    ``state_path`` must resolve to a ``list[dict]`` in ``initial_state``.
    """

    state_schema: dict[str, Any] | None = None
    initial_state: dict[str, Any] | None = None

    def __post_init__(self):
        """Validate state config consistency."""
        if self.state_schema is not None and self.initial_state is not None:
            jsonschema.validate(self.initial_state, self.state_schema)


@dataclass
class SyntheticEnvironmentKwargs(BaseParams):
    """Type-specific kwargs for SyntheticEnvironment."""

    tool_persona: str = ""
    state_params: SyntheticStateParams | None = None
    cache_by_input: bool = True

    def __post_init__(self) -> None:
        """Coerce state_params dict into SyntheticStateParams if needed."""
        if isinstance(self.state_params, dict):
            self.state_params = SyntheticStateParams(**self.state_params)

    def __finalize_and_validate__(self) -> None:
        """Finalize and validate the kwargs."""
        if not self.tool_persona:
            raise ValueError("SyntheticEnvironmentKwargs.tool_persona cannot be empty.")
        if self.state_params is not None and self.cache_by_input:
            raise ValueError(
                "SyntheticEnvironmentKwargs.cache_by_input must be False when "
                "state_params is provided."
            )


@register_environment("synthetic")
class SyntheticEnvironment(BaseEnvironment):
    """LLM-simulated environment with optional mutable state.

    See the module docstring for the stateless vs stateful contract.
    """

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
        self._state_schema: dict[str, Any] | None = (
            kwargs.state_params.state_schema
            if kwargs.state_params is not None
            else None
        )
        self._state_grounding: list[StateGroundingConfig] = (
            list(params.grounding.state) if params.grounding is not None else []
        )
        if self._state is None and self._state_grounding:
            raise ValueError(
                f"SyntheticEnvironment '{params.id}': grounding.state is "
                f"configured but the env has no state (state_params with "
                f"initial_state is required)."
            )
        self._executors: dict[str, Callable[..., Any]] = {
            tool.id: import_executor(tool.executor, tool.id)
            for tool in params.tools
            if tool.executor
        }
        if self._state is not None:
            missing = [t.id for t in params.tools if not t.executor]
            if missing:
                raise ValueError(
                    "SyntheticEnvironment in stateful mode (state_params with "
                    "initial_state set) requires every tool to define an executor; "
                    "LLM-simulated tools cannot mutate state. Missing executor: "
                    f"{missing}"
                )
        if self._state is not None:
            self._validate_state_grounding()
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

    def requires_isolation(self) -> bool:
        """Stateful synth envs need per-sample isolation; stateless do not."""
        return self._state is not None

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
        """Execute tool calls. See module docstring for routing rules.

        Raises:
            ValueError: If any tool id is unknown.
            RuntimeError: If an LLM-simulated tool is invoked before
                ``attach_inference`` was called.
            ToolError: On simulator parse failure or schema mismatch.
        """
        if not calls:
            return []
        for tool_id, _ in calls:
            self._lookup_tool(tool_id)

        stateful = self._state is not None
        results: list[ToolResult | None] = [None] * len(calls)
        sim_misses: list[tuple[int, str, dict[str, Any]]] = []
        for i, (tool_id, args) in enumerate(calls):
            if tool_id in self._executors:
                if stateful:
                    results[i] = self._step_stateful_one(tool_id, args)
                else:
                    results[i] = self._step_executor_one(tool_id, args)
                continue
            cached = self._resolve_cached(tool_id, args)
            if cached is not None:
                results[i] = cached
            else:
                sim_misses.append((i, tool_id, args))

        if sim_misses:
            if self._engine is None or self._base_inference_config is None:
                raise RuntimeError(
                    "SyntheticEnvironment.step called before "
                    "attach_inference(). Wire the synthesizer's engine via "
                    "attach_inference(engine, base_config) before invoking "
                    "step()."
                )
            groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
            for i, tool_id, args in sim_misses:
                groups.setdefault(tool_id, []).append((i, args))

            for tool_id, group in groups.items():
                tool = self._lookup_tool(tool_id)
                convs = [self._build_call_conv(tool, args) for _, args in group]
                inferred = self._engine.infer(
                    convs, self._simulator_inference_config(tool)
                )
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

    def build_call_conversation(
        self, tool_id: str, arguments: dict[str, Any]
    ) -> Conversation:
        """Build the simulator conversation for one tool call."""
        return self._build_call_conv(self._lookup_tool(tool_id), arguments)

    def parse_tool_response(self, tool_id: str, response: Conversation) -> ToolResult:
        """Extract a ToolResult from a simulator response conversation."""
        tool = self._lookup_tool(tool_id)
        return self._parse_and_validate(self._extract_text(response), tool)

    def _step_executor_one(self, tool_id: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a stateless tool via its executor callable."""
        tool = self._lookup_tool(tool_id)
        tool.validate_arguments(arguments)
        result = self._executors[tool_id](arguments=arguments)
        validate_executor_result(tool, result)
        if result.updated_state is not None:
            raise ToolError(
                f"Tool '{tool.id}' executor returned updated_state but the "
                f"environment is stateless."
            )
        return result

    def _step_stateful_one(self, tool_id: str, arguments: dict[str, Any]) -> ToolResult:
        """Dispatch a stateful tool and commit ``updated_state`` after validation.

        ``state_in`` is a deep copy so the executor's reference to ``state``
        can't reach back into ``self._state``: executors that mutate ``state``
        in place (or hand the same dict back as ``updated_state``) end up
        touching the copy, and ``self._state`` is only reassigned via the
        explicit deepcopy of ``result.updated_state`` below.
        """
        assert self._state is not None
        tool = self._lookup_tool(tool_id)
        tool.validate_arguments(arguments)
        state_in = copy.deepcopy(self._state)
        result = self._executors[tool_id](arguments=arguments, state=state_in)
        validate_executor_result(tool, result)
        if result.updated_state is not None:
            if tool.read_only:
                raise ToolError(
                    f"Tool '{tool_id}' is read_only but executor returned "
                    f"updated_state. Read-only tools must not mutate state."
                )
            if self._state_schema is not None:
                try:
                    jsonschema.validate(result.updated_state, self._state_schema)
                except jsonschema.ValidationError as e:
                    raise ToolError(
                        f"Tool '{tool_id}' updated_state failed state_schema "
                        f"validation: {e}"
                    ) from e
            self._state = copy.deepcopy(result.updated_state)
        return result

    def sample_grounding(
        self,
        n: int,
        *,
        rng: random.Random,
        tool_ids: set[str] | None = None,
    ) -> list[GroundingFact]:
        """Project grounding facts from ``grounding.state`` pools.

        No-op for stateless envs or envs without ``grounding.state`` entries.
        ``tool_ids`` is accepted for ``BaseEnvironment`` signature compatibility
        but ignored — state grounding is pool-scoped, not tool-scoped.

        ``_validate_state_grounding`` at init guarantees each ``state_path``
        resolves to a list in ``self._state``, and ``state_schema`` validation
        on every commit keeps it that way, so the projection loop trusts the
        shape.
        """
        del tool_ids
        if self._state is None or not self._state_grounding:
            return []
        pool: list[GroundingFact] = []
        for cfg in self._state_grounding:
            whitelist = set(cfg.fields)
            for row in self._state[cfg.state_path]:
                projected = {k: v for k, v in row.items() if k in whitelist}
                pool.append(GroundingFact(data=projected))
        return rng.sample(pool, min(n, len(pool)))

    def _validate_state_grounding(self) -> None:
        """Validate each ``grounding.state`` entry against current state."""
        assert self._state is not None
        for cfg in self._state_grounding:
            if cfg.state_path not in self._state:
                raise ValueError(
                    f"SyntheticEnvironment '{self._params.id}': grounding "
                    f"state_path '{cfg.state_path}' is not present in "
                    f"initial_state. Top-level keys: "
                    f"{sorted(self._state.keys())}."
                )
            rows = self._state[cfg.state_path]
            if not isinstance(rows, list):
                raise ValueError(
                    f"SyntheticEnvironment '{self._params.id}': grounding "
                    f"state_path '{cfg.state_path}' must resolve to a list, "
                    f"got {type(rows).__name__}."
                )

    def _build_simulator_system_prompt(self, tool: ToolParams) -> str:
        """Compose the simulator system prompt: tool persona + tool schema."""
        return (
            f"{self._kwargs.tool_persona}\n\n"
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
        """Return the assistant's text response, or ``""`` to trigger a ToolError.

        Engines that passthrough on partial failure can leave the user payload
        (itself valid JSON) as ``messages[-1]``; the role guard forces the
        ToolError path so we don't echo arguments back as a tool result.
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
