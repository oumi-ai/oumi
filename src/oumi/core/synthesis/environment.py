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

"""Stateful environment and registry for agentic tool synthesis."""

import copy
import json
import re
from collections.abc import Callable
from typing import Any

import jsonschema

from oumi.core.configs.params.tool_params import (
    ToolAttribute,
    ToolEnvironmentAttribute,
    ToolOutputStrategy,
)
from oumi.core.synthesis.tool_executor import clean_json_output, is_valid_json
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.utils.json_patch import (
    JsonPatchError,
    JsonPatchValidationError,
    apply_json_patch,
    parse_patch_response,
)
from oumi.utils.logging import logger
from oumi.utils.str_utils import extract_json

_MAX_STATE_UPDATE_RETRIES = (
    2  # Max retries for state updates that fail schema validation.
)

_MAX_RESULT_RETRIES = 2


def _parse_top_level_json(text: str) -> Any:
    """Parse JSON from a response, stripping markdown fences if present."""
    stripped = text.strip()
    if not stripped:
        return None

    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", stripped)
    if fence_match:
        stripped = fence_match.group(1).strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


class GeneratedToolEnvironment:
    """Stateful environment for tool synthesis.

    Maintains a JSON state document that evolves as tools read/write to it.
    Builds prompts and applies responses — does not call inference itself.
    """

    def __init__(self, config: ToolEnvironmentAttribute):
        """Initialize the environment with config."""
        self._config = config
        self._state: dict[str, Any] = copy.deepcopy(config.initial_state or {})
        self._state_schema: dict[str, Any] | None = (
            copy.deepcopy(config.state_schema) if config.state_schema else None
        )

    @property
    def state(self) -> dict[str, Any]:
        """Current state of the environment."""
        return self._state

    def summarize_for_planner(self) -> dict[str, Any]:
        """Return a compact state view for planner grounding."""
        return {
            "name": self._config.name,
            "state": self._state,
        }

    def set_state(self, state: dict[str, Any], validate: bool = True) -> bool:
        """Set state, optionally skipping schema validation."""
        if validate and not self._validate_state(state):
            return False
        self._state = copy.deepcopy(state)
        return True

    def set_schema(self, schema: dict[str, Any]) -> None:
        """Set the state schema."""
        self._state_schema = copy.deepcopy(schema)

    def build_result_prompt(
        self,
        tool: ToolAttribute,
        arguments: dict[str, Any],
        retry: bool = False,
    ) -> Conversation:
        """Build the prompt for generating a tool result."""
        system_parts = [
            self._config.system_prompt,
            f"\nCurrent state:\n{json.dumps(self._state, indent=2)}",
        ]

        user_parts = [
            f"Tool '{tool.name}' called with arguments:\n"
            f"{json.dumps(arguments, indent=2)}\n\n"
            "Produce the tool's JSON output based on the current state. "
            "No markdown fences. Start with { or [.",
        ]
        if retry:
            user_parts.append(
                "\nYour previous output was not valid JSON. "
                "Output only a complete JSON object or array."
            )
        if tool.output_schema:
            user_parts.insert(
                0,
                f"Expected output schema:\n"
                f"{json.dumps(tool.output_schema, indent=2)}\n\n",
            )

        messages = [
            Message(role=Role.SYSTEM, content="\n".join(system_parts)),
            Message(role=Role.USER, content="\n".join(user_parts)),
        ]
        return Conversation(messages=messages)

    def build_write_state_update_prompt(
        self,
        tool: ToolAttribute,
        arguments: dict[str, Any],
        retry: bool = False,
        retry_error: str | None = None,
    ) -> Conversation:
        """Build prompt for state-first write: decide patch before generating result."""
        system_parts = [self._config.system_prompt]
        if self._state_schema:
            system_parts.append(
                f"\nState schema:\n{json.dumps(self._state_schema, indent=2)}"
            )

        example_success_user = (
            'Current state:\n{"files": {"a.txt": "hello"}}\n\n'
            "Tool 'WriteFile' was called with:\n"
            '{"path": "a.txt", "content": "world"}\n\n'
            "Decide the state update that should happen before the tool result "
            "is generated. Output ONLY a JSON object with keys: "
            '"patch" (RFC 6902 array) and "error" (null or string).'
        )
        example_success_assistant = (
            '{"patch": [{"op": "replace", "path": "/files/a.txt", '
            '"value": "world"}], "error": null}'
        )

        example_failure_user = (
            'Current state:\n{"files": {"a.txt": "hello"}}\n\n'
            "Tool 'DeleteFile' was called with:\n"
            '{"path": "missing.txt"}\n\n'
            "Decide the state update that should happen before the tool result "
            "is generated. Output ONLY a JSON object with keys: "
            '"patch" (RFC 6902 array) and "error" (null or string).'
        )
        example_failure_assistant = (
            '{"patch": [], "error": "File \'missing.txt\' does not exist."}'
        )

        example_path = self._build_example_path()
        user_parts = [
            f"Current state:\n{json.dumps(self._state, indent=2)}\n\n"
            f"Tool '{tool.name}' was called with:\n"
            f"{json.dumps(arguments, indent=2)}\n\n"
            "Decide the state update that should happen before the tool result "
            "is generated.\n"
            "Output ONLY a JSON object with:\n"
            '- "patch": RFC 6902 JSON Patch array\n'
            '- "error": null when the update can be applied, otherwise a short '
            "error string explaining why the write should be rejected.\n"
            "If the write should succeed with no changes, return an empty patch "
            'array and "error": null.\n'
            "Paths must use JSON Pointer syntax, e.g. "
            f"{example_path}."
        ]
        if retry:
            user_parts.append(
                "\nIMPORTANT: Your previous output could not be applied. "
                "Return ONLY a valid JSON object with keys 'patch' and 'error'."
            )
        if retry_error:
            user_parts.append(f"\nPrevious error: {retry_error}")

        messages = [
            Message(role=Role.SYSTEM, content="\n".join(system_parts)),
            Message(role=Role.USER, content=example_success_user),
            Message(role=Role.ASSISTANT, content=example_success_assistant),
            Message(role=Role.USER, content=example_failure_user),
            Message(role=Role.ASSISTANT, content=example_failure_assistant),
            Message(role=Role.USER, content="\n".join(user_parts)),
        ]
        return Conversation(messages=messages)

    def build_write_result_prompt(
        self,
        tool: ToolAttribute,
        arguments: dict[str, Any],
        patch_ops: list[dict[str, Any]],
        patch_succeeded: bool,
        pre_patch_state: dict[str, Any] | None = None,
        retry: bool = False,
        retry_error: str | None = None,
    ) -> Conversation:
        """Build the prompt for generating a write tool result after state update."""
        system_parts = [self._config.system_prompt]

        user_parts = []
        if tool.output_schema:
            user_parts.append(
                f"Expected output schema:\n{json.dumps(tool.output_schema, indent=2)}\n"
            )

        user_parts.append(
            f"Tool '{tool.name}' was called with arguments:\n"
            f"{json.dumps(arguments, indent=2)}\n\n"
        )
        if patch_succeeded:
            if pre_patch_state is not None:
                user_parts.append(
                    f"State BEFORE the tool call:\n"
                    f"{json.dumps(pre_patch_state, indent=2)}\n\n"
                    f"State AFTER the tool call:\n"
                    f"{json.dumps(self._state, indent=2)}\n\n"
                    "The patch was applied successfully. Generate a SUCCESS "
                    "result that reflects the change from the before state to "
                    "the after state. Do not re-check preconditions — the "
                    "operation already succeeded."
                )
            else:
                user_parts.append(
                    f"Current state (after update):\n"
                    f"{json.dumps(self._state, indent=2)}\n\n"
                    f"Applied state update:\n"
                    f"{json.dumps(patch_ops, indent=2)}\n\n"
                    "The patch was applied successfully. Generate a SUCCESS "
                    "result. Do not re-check preconditions — the operation "
                    "already succeeded."
                )
        else:
            user_parts.append(
                f"Current state:\n{json.dumps(self._state, indent=2)}\n\n"
                "The state update failed. Return a JSON error response "
                "describing the failure."
            )

        user_parts.append("No markdown fences. Start with { or [.")
        if retry:
            user_parts.append(
                "\nIMPORTANT: Your previous output was not valid JSON. "
                "Output only a complete JSON object or array."
            )
        if retry_error:
            user_parts.append(f"\nPrevious error: {retry_error}")

        return Conversation(
            messages=[
                Message(role=Role.SYSTEM, content="\n".join(system_parts)),
                Message(role=Role.USER, content="\n".join(user_parts)),
            ]
        )

    def _build_example_path(self) -> str:
        """Build a dynamic JSON Pointer example from current state keys.

        Walks the first key at each level up to depth 2 to produce something
        like "/users/1/name". Falls back to "/key/value" if state is empty.
        """
        parts: list[str] = []
        current: Any = self._state
        for _ in range(3):
            if isinstance(current, dict) and current:
                key = next(iter(current))
                parts.append(str(key))
                current = current[key]
            else:
                break
        if not parts:
            return "/key/value"
        return "/" + "/".join(parts)

    def apply_result(self, response: Conversation) -> str:
        """Extract the tool result text from an inference response."""
        return self._extract_text(response)

    def apply_state_update_returning_patch(
        self, response: Conversation
    ) -> tuple[bool, list[dict[str, Any]], str | None]:
        """Parse and apply a write-state update. Returns (succeeded, ops, error)."""
        text = self._extract_text(response)
        explicit_error: str | None = None

        parsed = _parse_top_level_json(text)
        if isinstance(parsed, dict):
            parsed_obj = parsed
            raw_patch = parsed_obj.get("patch", [])
            if not isinstance(raw_patch, list):
                return False, [], "Write-state response 'patch' must be a list."

            error_value = parsed_obj.get("error")
            if error_value is None:
                explicit_error = None
            elif isinstance(error_value, str):
                explicit_error = error_value.strip() or None
            else:
                explicit_error = str(error_value)

            if explicit_error:
                return False, [], explicit_error
            patch = raw_patch
        elif isinstance(parsed, list):
            patch = parsed
        else:
            patch = parse_patch_response(text)

        if patch is None:
            error = f"Failed to parse JSON Patch: {text[:200]}"
            logger.warning(error)
            return False, [], error

        try:
            self._state = apply_json_patch(
                self._state, patch, schema=self._state_schema
            )
            return True, patch, None
        except JsonPatchError as e:
            error = f"JSON Patch application failed: {e}"
            logger.warning(error)
            return False, patch, str(e)
        except JsonPatchValidationError as e:
            error = f"Patched state failed schema validation: {e}"
            logger.warning(error)
            return False, patch, str(e)

    def apply_state_update(self, response: Conversation) -> bool:
        """Convenience wrapper: apply patch and return success bool."""
        succeeded, _, _ = self.apply_state_update_returning_patch(response)
        return succeeded

    def _validate_state(self, state: dict[str, Any]) -> bool:
        if not self._state_schema:
            return True
        try:
            jsonschema.validate(instance=state, schema=self._state_schema)
            return True
        except jsonschema.ValidationError as e:
            logger.warning(f"State validation failed: {e.message}")
            return False

    @staticmethod
    def _extract_text(conversation: Conversation) -> str:
        """Extract text content from the last message of a conversation."""
        last_msg = conversation.messages[-1]
        content = last_msg.content
        if isinstance(content, str):
            return content
        return ""


_ID_SUFFIX_PATTERN = re.compile(r"^(.+)_id$")


def _pluralize(word: str) -> str:
    """Naive English pluralization for collection name inference."""
    if word.endswith("s"):
        if word.endswith("ss") or word.endswith("us"):
            return word + "es"
        return word
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


def _format_tool_definitions(tools: list[ToolAttribute]) -> str:
    """Format tool definitions for inclusion in a state generation prompt."""
    lines: list[str] = []
    for tool in tools:
        parts = [f"- {tool.name}: {tool.description}"]
        if tool.parameters:
            parts.append(f"  Parameters: {json.dumps(tool.parameters, indent=2)}")
        if tool.output_schema:
            parts.append(f"  Output: {json.dumps(tool.output_schema, indent=2)}")
        if tool.read_only:
            parts.append("  Read-only: true")
        lines.append("\n".join(parts))
    return "\n\n".join(lines)


def build_dependency_graph(
    schema: dict[str, Any],
) -> dict[str, set[str]]:
    """Build FK-based dependency graph from a state schema."""
    properties = schema.get("properties", {})
    collection_names = set(properties.keys())
    graph: dict[str, set[str]] = {name: set() for name in collection_names}

    for collection_name, collection_schema in properties.items():
        additional = collection_schema.get("additionalProperties", {})
        record_props = additional.get("properties", {})

        for field_name in record_props:
            match = _ID_SUFFIX_PATTERN.match(field_name)
            if not match:
                continue
            entity = match.group(1)
            target_collection = _pluralize(entity)
            if target_collection == collection_name:
                continue
            if target_collection in collection_names:
                graph[collection_name].add(target_collection)

    return graph


def sort_into_waves(
    graph: dict[str, set[str]],
) -> list[list[str]]:
    """Topological sort into parallel waves. Breaks cycles by inbound count."""
    if not graph:
        return []

    remaining = {name: set(deps) for name, deps in graph.items()}
    waves: list[list[str]] = []

    while remaining:
        ready = sorted(name for name, deps in remaining.items() if not deps)

        if not ready:
            inbound_counts = {name: 0 for name in remaining}
            for deps in remaining.values():
                for dep in deps:
                    if dep in inbound_counts:
                        inbound_counts[dep] += 1
            ready = [min(inbound_counts, key=lambda n: inbound_counts[n])]
            logger.warning(
                f"Cycle detected in dependency graph. Breaking by "
                f"forcing '{ready[0]}' into current wave."
            )

        waves.append(ready)

        resolved = set(ready)
        for name in ready:
            del remaining[name]
        for deps in remaining.values():
            deps -= resolved

    return waves


_DEFAULT_RECORD_COUNT = "3-8"


def build_collection_prompt(
    config: ToolEnvironmentAttribute,
    collection_name: str,
    sub_schema: dict[str, Any],
    existing_state: dict[str, Any],
    scenario_context: str | None = None,
    record_count: str = _DEFAULT_RECORD_COUNT,
    retry_error: str | None = None,
) -> Conversation:
    """Build a prompt for generating one collection's records."""
    system_msg = (
        f"You are populating data for an environment.\n\n"
        f"Environment: {config.name}\n"
        f"Description: {config.description}\n\n"
        f"{config.system_prompt}"
    )

    user_parts: list[str] = []

    if scenario_context:
        user_parts.append(f"Scenario: {scenario_context}\n")

    if existing_state:
        state_lines = []
        for coll_name, coll_data in existing_state.items():
            state_lines.append(f"{coll_name}: {json.dumps(coll_data, indent=2)}")
        user_parts.append("Existing state:\n" + "\n\n".join(state_lines) + "\n")

    schema_str = json.dumps(sub_schema, indent=2)
    user_parts.append(
        f"Generate {record_count} records for the "
        f"'{collection_name}' collection.\n\n"
        f"Each record must match this schema:\n{schema_str}\n"
    )

    fk_fields = [
        f for f in sub_schema.get("properties", {}) if _ID_SUFFIX_PATTERN.match(f)
    ]
    if fk_fields and existing_state:
        refs = []
        for fk in fk_fields:
            match = _ID_SUFFIX_PATTERN.match(fk)
            if match:
                target = _pluralize(match.group(1))
                if target in existing_state:
                    refs.append(f"- {fk} must reference an existing ID from '{target}'")
        if refs:
            user_parts.append("Referential integrity:\n" + "\n".join(refs) + "\n")

    user_parts.append(
        "Output a JSON object keyed by string IDs (e.g., "
        '"1", "2" or domain-appropriate IDs). '
        "No markdown fences. Start with {."
    )

    if retry_error:
        user_parts.append(
            f"\nIMPORTANT: Your previous output failed "
            f"validation: {retry_error}\n"
            "Fix the issue and output only valid JSON."
        )

    messages = [
        Message(role=Role.SYSTEM, content=system_msg),
        Message(role=Role.USER, content="\n".join(user_parts)),
    ]
    return Conversation(messages=messages)


def build_state_generation_prompt(
    config: ToolEnvironmentAttribute,
    tools: list[ToolAttribute],
    scenario_context: str | None = None,
    retry_error: str | None = None,
) -> Conversation:
    """Build prompt for generating full environment state (no schema path)."""
    system_msg = (
        f"You are populating initial data for an environment.\n\n"
        f"Environment: {config.name}\n"
        f"Description: {config.description}\n\n"
        f"{config.system_prompt}"
    )

    user_parts: list[str] = []

    if scenario_context:
        user_parts.append(f"Scenario: {scenario_context}\n")

    if tools:
        user_parts.append(
            "Tools that operate on this environment:\n"
            + _format_tool_definitions(tools)
            + "\n"
        )

    user_parts.append(
        "Generate the initial state for this environment as a JSON object. "
        "The state should be realistic and internally consistent. "
        "It must contain enough data that the tools above can operate on it "
        "meaningfully (e.g., lookups return results, filters have matches).\n\n"
        "No markdown fences. Start with {."
    )

    if retry_error:
        user_parts.append(
            f"\nIMPORTANT: Your previous output failed: {retry_error}\n"
            "Fix the issue and output only valid JSON."
        )

    messages = [
        Message(role=Role.SYSTEM, content=system_msg),
        Message(role=Role.USER, content="\n".join(user_parts)),
    ]
    return Conversation(messages=messages)


def validate_collection(
    collection_name: str,
    data: Any,
    sub_schema: dict[str, Any],
    existing_state: dict[str, Any],
) -> tuple[bool, str | None]:
    """Validate collection data: schema conformance and FK integrity."""
    if not isinstance(data, dict):
        return (
            False,
            f"Expected a dict keyed by ID, got {type(data).__name__}.",
        )

    for record_id, record in data.items():
        try:
            jsonschema.validate(instance=record, schema=sub_schema)
        except jsonschema.ValidationError as e:
            return (
                False,
                f"Record '{record_id}' failed schema validation: {e.message}",
            )

    for record_id, record in data.items():
        if not isinstance(record, dict):
            continue
        for field_name, value in record.items():
            match = _ID_SUFFIX_PATTERN.match(field_name)
            if not match or not isinstance(value, str):
                continue
            entity = match.group(1)
            target_collection = _pluralize(entity)
            if target_collection == collection_name:
                continue
            if target_collection not in existing_state:
                continue
            if value not in existing_state[target_collection]:
                return (
                    False,
                    f"Record '{record_id}' references "
                    f"{field_name}='{value}' but '{value}' "
                    f"does not exist in '{target_collection}'.",
                )

    return True, None


_MAX_COLLECTION_RETRIES = 2


def _infer_exact(
    inference_engine: Any,
    prompts: list[Conversation],
    inference_config: Any,
    context: str,
) -> list[Conversation]:
    """Run batched inference and enforce one response per prompt."""
    responses = inference_engine.infer(prompts, inference_config=inference_config)
    if len(responses) != len(prompts):
        raise RuntimeError(
            f"{context}: inference returned {len(responses)} responses "
            f"for {len(prompts)} prompts."
        )
    return responses


def _extract_response_text(response: Conversation) -> str:
    """Extract text from the last message of an inference response."""
    if not response.messages:
        return ""
    content = response.messages[-1].content
    return content if isinstance(content, str) else ""


class EnvironmentRegistry:
    """Builds environments once, then copies N times for parallel samples."""

    def __init__(self):
        """Initialize environment registery."""
        self._built: dict[str, GeneratedToolEnvironment] = {}

    def register_static(self, config: ToolEnvironmentAttribute) -> None:
        """Register an environment that needs no LLM generation."""
        env = GeneratedToolEnvironment(config=config)
        self._built[config.id] = env

    def build(
        self,
        config: ToolEnvironmentAttribute,
        tools: list[ToolAttribute],
        inference_engine: Any,
        inference_config: Any,
        scenario_context: str | None = None,
    ) -> None:
        """Build a fully populated environment via schema-based."""
        if config.state_schema:
            schema = copy.deepcopy(config.state_schema)
            state = self._build_from_schema(
                config,
                schema,
                inference_engine,
                inference_config,
                scenario_context,
            )
            env = GeneratedToolEnvironment(config=config)
            env.set_schema(schema)
            env.set_state(state, validate=False)
        else:
            state = self._build_from_description(
                config,
                tools,
                inference_engine,
                inference_config,
                scenario_context,
            )
            env = GeneratedToolEnvironment(config=config)
            env.set_state(state, validate=False)

        self._built[config.id] = env

    def create_copies(self, env_id: str, n: int) -> list[GeneratedToolEnvironment]:
        """Return n independent deepcopies of a built environment."""
        if env_id not in self._built:
            raise KeyError(
                f"Environment '{env_id}' not found. "
                f"Call build() or register_static() first."
            )
        source = self._built[env_id]
        return [copy.deepcopy(source) for _ in range(n)]

    def _build_from_schema(
        self,
        config: ToolEnvironmentAttribute,
        schema: dict[str, Any],
        inference_engine: Any,
        inference_config: Any,
        scenario_context: str | None,
    ) -> dict[str, Any]:
        """Generate state using wave-based per-collection prompts."""
        graph = build_dependency_graph(schema)
        waves = sort_into_waves(graph)

        state: dict[str, Any] = {}
        properties = schema.get("properties", {})

        for wave in waves:
            prompts = []
            wave_collections = []

            for collection_name in wave:
                collection_schema = properties.get(collection_name, {})
                sub_schema = collection_schema.get("additionalProperties", {})
                prompt = build_collection_prompt(
                    config=config,
                    collection_name=collection_name,
                    sub_schema=sub_schema,
                    existing_state=state,
                    scenario_context=scenario_context,
                )
                prompts.append(prompt)
                wave_collections.append((collection_name, sub_schema))

            if not prompts:
                continue

            responses = _infer_exact(
                inference_engine=inference_engine,
                prompts=prompts,
                inference_config=inference_config,
                context=f"Environment '{config.id}' collection generation",
            )

            for (collection_name, sub_schema), response in zip(
                wave_collections, responses
            ):
                success = self._process_collection_response(
                    collection_name=collection_name,
                    sub_schema=sub_schema,
                    response=response,
                    state=state,
                    config=config,
                    scenario_context=scenario_context,
                    inference_engine=inference_engine,
                    inference_config=inference_config,
                )
                if not success:
                    logger.warning(
                        f"Collection '{collection_name}' failed after "
                        f"{_MAX_COLLECTION_RETRIES} retries. Skipping."
                    )

        return state

    def _build_from_description(
        self,
        config: ToolEnvironmentAttribute,
        tools: list[ToolAttribute],
        inference_engine: Any,
        inference_config: Any,
        scenario_context: str | None,
    ) -> dict[str, Any]:
        """Generate state from env description + tool defs (no schema)."""
        prompt = build_state_generation_prompt(
            config=config,
            tools=tools,
            scenario_context=scenario_context,
        )
        responses = _infer_exact(
            inference_engine=inference_engine,
            prompts=[prompt],
            inference_config=inference_config,
            context=f"Environment '{config.id}' state generation",
        )
        text = _extract_response_text(responses[0])
        parsed = extract_json(text, expected_type=dict)

        if isinstance(parsed, dict) and parsed:
            return parsed

        retry_error = (
            f"Could not parse JSON from response: {text[:200]}"
            if not isinstance(parsed, dict)
            else "Generated state was empty."
        )
        for _ in range(_MAX_COLLECTION_RETRIES):
            retry_prompt = build_state_generation_prompt(
                config=config,
                tools=tools,
                scenario_context=scenario_context,
                retry_error=retry_error,
            )
            retry_responses = _infer_exact(
                inference_engine=inference_engine,
                prompts=[retry_prompt],
                inference_config=inference_config,
                context=f"Environment '{config.id}' state generation retry",
            )
            text = _extract_response_text(retry_responses[0])
            parsed = extract_json(text, expected_type=dict)

            if isinstance(parsed, dict) and parsed:
                return parsed

            retry_error = (
                f"Could not parse JSON from response: {text[:200]}"
                if not isinstance(parsed, dict)
                else "Generated state was empty."
            )

        logger.warning(
            f"State generation for '{config.id}' failed after "
            f"{_MAX_COLLECTION_RETRIES} retries. Using empty state."
        )
        return {}

    def _process_collection_response(
        self,
        collection_name: str,
        sub_schema: dict[str, Any],
        response: Any,
        state: dict[str, Any],
        config: ToolEnvironmentAttribute,
        scenario_context: str | None,
        inference_engine: Any,
        inference_config: Any,
    ) -> bool:
        """Process a collection generation response with retry on failure.

        Returns True if the collection was successfully populated.
        """
        text = _extract_response_text(response)
        parsed = extract_json(text, expected_type=dict)

        if isinstance(parsed, dict):
            ok, error = validate_collection(collection_name, parsed, sub_schema, state)
            if ok:
                state[collection_name] = parsed
                return True
            retry_error = error
        else:
            retry_error = f"Could not parse JSON from response: {text[:200]}"

        for _ in range(_MAX_COLLECTION_RETRIES):
            retry_prompt = build_collection_prompt(
                config=config,
                collection_name=collection_name,
                sub_schema=sub_schema,
                existing_state=state,
                scenario_context=scenario_context,
                retry_error=retry_error,
            )
            retry_responses = _infer_exact(
                inference_engine=inference_engine,
                prompts=[retry_prompt],
                inference_config=inference_config,
                context=(
                    f"Environment '{config.id}' collection '{collection_name}' retry"
                ),
            )
            text = _extract_response_text(retry_responses[0])
            parsed = extract_json(text, expected_type=dict)

            if isinstance(parsed, dict):
                ok, error = validate_collection(
                    collection_name,
                    parsed,
                    sub_schema,
                    state,
                )
                if ok:
                    state[collection_name] = parsed
                    return True
                retry_error = error
            else:
                retry_error = f"Could not parse JSON from response: {text[:200]}"

        return False


def resolve_env_tool(
    tool_executor: Any,
    tool_call: dict,
    idx_envs: dict[str, GeneratedToolEnvironment] | None,
) -> tuple[GeneratedToolEnvironment | None, ToolAttribute | None]:
    """Look up (env, tool) for an environment-bound tool call, or (None, None)."""
    if idx_envs is None:
        return None, None
    tool_obj = tool_executor.get_tool_by_name(tool_call["name"])
    if tool_obj and tool_obj.environment:
        env = idx_envs.get(tool_obj.environment)
        if env:
            return env, tool_obj
    return None, None


def is_env_tool_missing_env(tool_executor: Any, tool_call: dict) -> bool:
    """Return True if tool_call targets an ENVIRONMENT tool whose env is missing."""
    tool = tool_executor.get_tool_by_name(tool_call["name"])
    if tool and tool.output_strategy == ToolOutputStrategy.ENVIRONMENT:
        logger.warning(
            f"Environment not found for tool '{tool_call['name']}', skipping."
        )
        return True
    return False


def serialize_env_states(envs: dict[str, GeneratedToolEnvironment]) -> str:
    """Serialize planner-oriented environment summaries as stable JSON."""
    summaries = {
        env_id: env.summarize_for_planner()
        for env_id, env in sorted(envs.items(), key=lambda item: item[0])
    }
    return json.dumps(summaries, indent=2, sort_keys=True, ensure_ascii=True)


def format_environment_config(
    formatter: Any,
    sample: dict,
    config: ToolEnvironmentAttribute,
) -> ToolEnvironmentAttribute:
    """Format environment prompt fields with sample-specific attributes."""
    return ToolEnvironmentAttribute(
        id=config.id,
        name=formatter.format(
            sample,
            config.name,
            missing_values_allowed=True,
        ),
        description=formatter.format(
            sample,
            config.description,
            missing_values_allowed=True,
        ),
        system_prompt=formatter.format(
            sample,
            config.system_prompt,
            missing_values_allowed=True,
        ),
        state_schema=copy.deepcopy(config.state_schema),
        initial_state=copy.deepcopy(config.initial_state),
    )


def init_sample_environments(
    samples: list[dict],
    tools: list[ToolAttribute],
    env_configs: dict[str, ToolEnvironmentAttribute],
    formatter: Any,
    inference_engine: Any,
    inference_config: Any,
) -> list[dict[str, GeneratedToolEnvironment] | None]:
    """Create per-sample environments, reusing builds when config is identical."""
    env_tools: dict[str, list[ToolAttribute]] = {}
    for tool in tools:
        if tool.environment:
            env_tools.setdefault(tool.environment, []).append(tool)

    if not env_tools:
        return [None] * len(samples)

    result: list[dict[str, GeneratedToolEnvironment]] = [{} for _ in samples]

    for env_id, bound_tools in env_tools.items():
        config = env_configs.get(env_id)
        if not config:
            logger.warning(f"Environment config not found for '{env_id}'")
            continue

        variants: dict[
            tuple[str, str, str, str | None, str | None],
            GeneratedToolEnvironment,
        ] = {}

        for i, sample in enumerate(samples):
            formatted_config = format_environment_config(formatter, sample, config)
            signature = (
                formatted_config.name,
                formatted_config.description,
                formatted_config.system_prompt,
                (
                    json.dumps(formatted_config.state_schema, sort_keys=True)
                    if formatted_config.state_schema is not None
                    else None
                ),
                (
                    json.dumps(formatted_config.initial_state, sort_keys=True)
                    if formatted_config.initial_state is not None
                    else None
                ),
            )

            source_env = variants.get(signature)
            if source_env is None:
                registry = EnvironmentRegistry()
                if (
                    formatted_config.initial_state is not None
                    and formatted_config.state_schema is not None
                ):
                    registry.register_static(formatted_config)
                else:
                    registry.build(
                        formatted_config,
                        bound_tools,
                        inference_engine,
                        inference_config,
                        scenario_context=None,
                    )

                try:
                    source_env = registry.create_copies(env_id, 1)[0]
                except KeyError:
                    logger.warning(f"Environment '{env_id}' not built. Skipping.")
                    continue
                variants[signature] = source_env

            result[i][env_id] = copy.deepcopy(source_env)

    finalized: list[dict[str, GeneratedToolEnvironment] | None] = []
    for envs in result:
        if not envs:
            finalized.append(None)
            continue

        all_empty = all(not env.state for env in envs.values())
        if all_empty:
            logger.warning("Dropping sample: all environments have empty state.")
            finalized.append(None)
        else:
            finalized.append(envs)

    return finalized


def process_env_tool_calls(
    env_items: list[
        tuple[int, str, dict, str, GeneratedToolEnvironment, ToolAttribute]
    ],
    env_result_prompts: list[Conversation],
    turn_tool_msgs: dict[int, list[Message]],
    output_messages: list[list[dict]],
    inference_engine: Any,
    inference_config: Any,
    record_fn: Callable[..., None],
) -> list[int]:
    """Execute batched env tool calls."""
    final_results: list[str | None] = [None] * len(env_items)

    read_indices = [
        i for i, (_, _, _, _, _, tool_obj) in enumerate(env_items) if tool_obj.read_only
    ]
    write_indices = [
        i
        for i, (_, _, _, _, _, tool_obj) in enumerate(env_items)
        if not tool_obj.read_only
    ]

    if read_indices:
        _process_env_read_calls(
            env_items,
            env_result_prompts,
            read_indices,
            final_results,
            inference_engine,
            inference_config,
        )

    if write_indices:
        _process_env_write_calls(
            env_items,
            write_indices,
            final_results,
            inference_engine,
            inference_config,
        )

    activated: list[int] = []
    for i, (idx, text, tool_call, call_id, env, _tool_obj) in enumerate(env_items):
        result = final_results[i]
        if result is None:
            result = '{"error": "Failed to generate result"}'
        record_fn(
            idx,
            text,
            tool_call,
            call_id,
            result,
            turn_tool_msgs,
            output_messages,
            env_state=copy.deepcopy(env.state),
        )
        activated.append(idx)
    return activated


def _process_env_read_calls(
    env_items: list[
        tuple[int, str, dict, str, GeneratedToolEnvironment, ToolAttribute]
    ],
    env_result_prompts: list[Conversation],
    read_indices: list[int],
    final_results: list[str | None],
    inference_engine: Any,
    inference_config: Any,
) -> None:
    """Generate results for read-only environment tool calls."""
    prompts = [env_result_prompts[i] for i in read_indices]
    responses = _infer_exact(
        inference_engine, prompts, inference_config, "Read tool result generation"
    )

    failed_positions: list[int] = []
    last_raw: dict[int, str] = {}

    for pos, response in enumerate(responses):
        i = read_indices[pos]
        raw = env_items[i][4].apply_result(response)
        cleaned = clean_json_output(raw)
        if is_valid_json(cleaned):
            final_results[i] = cleaned
        else:
            last_raw[pos] = cleaned
            failed_positions.append(pos)

    for _ in range(_MAX_RESULT_RETRIES):
        if not failed_positions:
            break
        retry_prompts = [
            env_items[read_indices[pos]][4].build_result_prompt(
                env_items[read_indices[pos]][5],
                arguments=env_items[read_indices[pos]][2]["arguments"],
                retry=True,
            )
            for pos in failed_positions
        ]
        retry_responses = _infer_exact(
            inference_engine, retry_prompts, inference_config, "Read tool result retry"
        )
        still_failed: list[int] = []
        for pos, response in zip(failed_positions, retry_responses):
            i = read_indices[pos]
            raw = env_items[i][4].apply_result(response)
            cleaned = clean_json_output(raw)
            if is_valid_json(cleaned):
                final_results[i] = cleaned
            else:
                last_raw[pos] = cleaned
                still_failed.append(pos)
        failed_positions = still_failed

    for pos in failed_positions:
        i = read_indices[pos]
        logger.warning(
            f"Read result for '{env_items[i][5].name}' was not valid JSON "
            f"after {_MAX_RESULT_RETRIES} retries. Using raw text."
        )
        final_results[i] = last_raw.get(pos, "{}")


def _process_env_write_calls(
    env_items: list[
        tuple[int, str, dict, str, GeneratedToolEnvironment, ToolAttribute]
    ],
    write_indices: list[int],
    final_results: list[str | None],
    inference_engine: Any,
    inference_config: Any,
) -> None:
    """Process write tool calls: patch state first, then generate result."""
    update_prompts = [
        env_items[i][4].build_write_state_update_prompt(
            env_items[i][5], env_items[i][2]["arguments"]
        )
        for i in write_indices
    ]
    update_responses = _infer_exact(
        inference_engine,
        update_prompts,
        inference_config,
        "Write state update generation",
    )

    patch_data: dict[int, tuple[bool, list[dict[str, Any]]]] = {}
    pre_patch_states: dict[int, dict[str, Any]] = {}
    failed_indices: list[int] = []
    update_errors: dict[int, str] = {}

    for env_item_index, response in zip(write_indices, update_responses):
        env = env_items[env_item_index][4]
        pre_patch_states[env_item_index] = copy.deepcopy(env.state)
        succeeded, ops, error = env.apply_state_update_returning_patch(response)
        patch_data[env_item_index] = (succeeded, ops)
        if not succeeded:
            failed_indices.append(env_item_index)
            if error:
                update_errors[env_item_index] = error

    for _ in range(_MAX_STATE_UPDATE_RETRIES):
        if not failed_indices:
            break
        retry_prompts = [
            env_items[env_item_index][4].build_write_state_update_prompt(
                env_items[env_item_index][5],
                env_items[env_item_index][2]["arguments"],
                retry=True,
                retry_error=update_errors.get(env_item_index),
            )
            for env_item_index in failed_indices
        ]
        retry_responses = _infer_exact(
            inference_engine,
            retry_prompts,
            inference_config,
            "Write state update retry",
        )
        still_failed: list[int] = []

        for env_item_index, response in zip(failed_indices, retry_responses):
            env = env_items[env_item_index][4]
            pre_patch_states[env_item_index] = copy.deepcopy(env.state)
            succeeded, ops, error = env.apply_state_update_returning_patch(response)
            patch_data[env_item_index] = (succeeded, ops)
            if not succeeded:
                still_failed.append(env_item_index)
                if error:
                    update_errors[env_item_index] = error
            else:
                update_errors.pop(env_item_index, None)
        failed_indices = still_failed

    successful_indices: list[int] = []
    for env_item_index in write_indices:
        succeeded, _patch_ops = patch_data.get(env_item_index, (False, []))
        if succeeded:
            successful_indices.append(env_item_index)
            continue

        error_message = update_errors.get(
            env_item_index, "State update could not be applied."
        )
        logger.warning(
            f"Write state update for '{env_items[env_item_index][5].name}' "
            f"failed after {_MAX_STATE_UPDATE_RETRIES} retries."
        )
        final_results[env_item_index] = json.dumps(
            {
                "error": "state_update_failed",
                "message": error_message,
            }
        )

    if not successful_indices:
        return

    result_prompts = [
        env_items[env_item_index][4].build_write_result_prompt(
            env_items[env_item_index][5],
            env_items[env_item_index][2]["arguments"],
            patch_ops=patch_data[env_item_index][1],
            patch_succeeded=patch_data[env_item_index][0],
            pre_patch_state=pre_patch_states.get(env_item_index),
        )
        for env_item_index in successful_indices
    ]
    result_responses = _infer_exact(
        inference_engine, result_prompts, inference_config, "Write result generation"
    )

    result_failed: list[int] = []
    last_raw: dict[int, str] = {}
    result_errors: dict[int, str] = {}

    for env_item_index, response in zip(successful_indices, result_responses):
        raw = env_items[env_item_index][4].apply_result(response)
        cleaned = clean_json_output(raw)
        if is_valid_json(cleaned):
            final_results[env_item_index] = cleaned
        else:
            last_raw[env_item_index] = cleaned
            result_failed.append(env_item_index)
            result_errors[env_item_index] = (
                f"Result was not valid JSON: {cleaned[:200]}"
            )

    for _ in range(_MAX_RESULT_RETRIES):
        if not result_failed:
            break
        retry_prompts = [
            env_items[env_item_index][4].build_write_result_prompt(
                env_items[env_item_index][5],
                env_items[env_item_index][2]["arguments"],
                patch_ops=patch_data[env_item_index][1],
                patch_succeeded=patch_data[env_item_index][0],
                pre_patch_state=pre_patch_states.get(env_item_index),
                retry=True,
                retry_error=result_errors.get(env_item_index),
            )
            for env_item_index in result_failed
        ]
        retry_responses = _infer_exact(
            inference_engine, retry_prompts, inference_config, "Write result retry"
        )
        still_failed_result: list[int] = []
        for env_item_index, response in zip(result_failed, retry_responses):
            raw = env_items[env_item_index][4].apply_result(response)
            cleaned = clean_json_output(raw)
            if is_valid_json(cleaned):
                final_results[env_item_index] = cleaned
                result_errors.pop(env_item_index, None)
            else:
                last_raw[env_item_index] = cleaned
                result_errors[env_item_index] = (
                    f"Result was not valid JSON: {cleaned[:200]}"
                )
                still_failed_result.append(env_item_index)
        result_failed = still_failed_result

    for env_item_index in result_failed:
        logger.warning(
            f"Write result for '{env_items[env_item_index][5].name}' not valid JSON "
            f"after {_MAX_RESULT_RETRIES} retries. Using raw text."
        )
        final_results[env_item_index] = last_raw.get(env_item_index, "{}")
