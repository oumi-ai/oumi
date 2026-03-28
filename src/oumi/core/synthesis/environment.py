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
from typing import Any

import jsonschema

from oumi.core.configs.params.tool_params import ToolAttribute, ToolEnvironmentAttribute
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

_MAX_RESULT_RETRIES = 2  # Max retries for tool results that fail JSON parsing.


class GeneratedToolEnvironment:
    """Stateful environment for tool synthesis using a build/apply pattern.

    Maintains a JSON state document that evolves as tools read from and
    write to it. Each tool call is processed via two LLM calls:
    1. build_result_prompt / apply_result: produces the tool's output given
       current state
    2. build_state_update_prompt / apply_state_update: updates state to
       reflect the tool call (if not read_only)

    The environment does not call inference itself — it only builds prompts
    and applies responses. The caller (e.g. ConversationSynthesizer) is
    responsible for batching inference calls across environments.

    State is validated against a JSON Schema after each update.
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

    def build_state_update_prompt(
        self,
        tool: ToolAttribute,
        arguments: dict[str, Any],
        result: str,
        retry: bool = False,
    ) -> Conversation:
        """Build a few-shot prompt for generating a JSON Patch state update."""
        system_parts = [self._config.system_prompt]
        if self._state_schema:
            system_parts.append(
                f"\nState schema:\n{json.dumps(self._state_schema, indent=2)}"
            )

        # Example 1: replace (UPDATE)
        ex1_user = (
            'Current state:\n{"users": {"1": {"name": "Alice", "role": "admin"}, '
            '"2": {"name": "Bob", "role": "viewer"}}}\n\n'
            "Tool 'UpdateRole' was called with:\n"
            '{"user_id": "2", "new_role": "editor"}\n\n'
            'Tool returned:\n{"status": "success", "rows_affected": 1}\n\n'
            "Output a JSON Patch (RFC 6902) array describing ONLY the changes "
            "to the state. Each operation must have: op (add/remove/replace), "
            "path (JSON Pointer referencing dict keys, e.g. /users/2/role), "
            "and value (for add/replace). Output ONLY the JSON array."
        )
        ex1_assistant = (
            '[{"op": "replace", "path": "/users/2/role", "value": "editor"}]'
        )

        ex2_user = (
            'Current state:\n{"users": {"1": {"name": "Alice", "role": "admin"}, '
            '"2": {"name": "Bob", "role": "viewer"}}}\n\n'
            "Tool 'CreateUser' was called with:\n"
            '{"name": "Carol", "role": "editor"}\n\n'
            'Tool returned:\n{"status": "success", "id": "3"}\n\n'
            "Output a JSON Patch (RFC 6902) array describing ONLY the changes "
            "to the state. Each operation must have: op (add/remove/replace), "
            "path (JSON Pointer referencing dict keys, e.g. /users/3), "
            "and value (for add/replace). Output ONLY the JSON array."
        )
        ex2_assistant = (
            '[{"op": "add", "path": "/users/3", '
            '"value": {"name": "Carol", "role": "editor"}}]'
        )

        # Example 3: remove (DELETE)
        ex3_user = (
            'Current state:\n{"users": {"1": {"name": "Alice", "role": "admin"}, '
            '"2": {"name": "Bob", "role": "viewer"}, '
            '"3": {"name": "Carol", "role": "editor"}}}\n\n'
            "Tool 'DeleteUser' was called with:\n"
            '{"user_id": "2"}\n\n'
            'Tool returned:\n{"status": "success", "rows_affected": 1}\n\n'
            "Output a JSON Patch (RFC 6902) array describing ONLY the changes "
            "to the state. Each operation must have: op (add/remove/replace), "
            "path (JSON Pointer referencing dict keys, e.g. /users/2), "
            "and value (for add/replace). Output ONLY the JSON array."
        )
        ex3_assistant = '[{"op": "remove", "path": "/users/2"}]'

        example_path = self._build_example_path()

        user_parts = [
            f"Current state:\n{json.dumps(self._state, indent=2)}\n\n"
            f"Tool '{tool.name}' was called with:\n"
            f"{json.dumps(arguments, indent=2)}\n\n"
            f"Tool returned:\n{result}\n\n"
            "Output a JSON Patch (RFC 6902) array describing ONLY the changes "
            "to the state. Each operation must have: op (add/remove/replace), "
            "path (JSON Pointer referencing dict keys, e.g. "
            f"{example_path}), "
            "and value (for add/replace). Output ONLY the JSON array.",
        ]
        if retry:
            user_parts.append(
                "\nIMPORTANT: Your previous output was not a valid JSON Patch "
                "array. Output ONLY a JSON array of RFC 6902 patch operations."
            )

        messages = [
            Message(role=Role.SYSTEM, content="\n".join(system_parts)),
            Message(role=Role.USER, content=ex1_user),
            Message(role=Role.ASSISTANT, content=ex1_assistant),
            Message(role=Role.USER, content=ex2_user),
            Message(role=Role.ASSISTANT, content=ex2_assistant),
            Message(role=Role.USER, content=ex3_user),
            Message(role=Role.ASSISTANT, content=ex3_assistant),
            Message(role=Role.USER, content="\n".join(user_parts)),
        ]
        return Conversation(messages=messages)

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

    def apply_state_update(self, response: Conversation) -> bool:
        """Parse, validate, and apply a JSON Patch from an inference response.

        Returns True if the state was successfully updated, False otherwise.
        """
        text = self._extract_text(response)
        patch = parse_patch_response(text)
        if patch is None:
            logger.warning(f"Failed to parse JSON Patch: {text[:200]}")
            return False
        try:
            self._state = apply_json_patch(
                self._state, patch, schema=self._state_schema
            )
            return True
        except JsonPatchError as e:
            logger.warning(f"JSON Patch application failed: {e}")
            return False
        except JsonPatchValidationError as e:
            logger.warning(f"Patched state failed schema validation: {e}")
            return False

    def _validate_state(self, state: dict[str, Any]) -> bool:
        """Validate state against the schema. Returns True if valid."""
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
    """Naive English pluralization for entity names.

    Handles common suffixes. Not meant to be exhaustive — just good enough
    for collection names like tenant→tenants, category→categories.
    """
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
    """Build a dependency graph from a state schema.

    For each collection, scans its record properties for foreign-key fields
    (fields ending in `_id` whose prefix matches another collection).

    Args:
        schema: The full state JSON Schema.

    Returns:
        Dict mapping collection name to set of collection names it depends on.
    """
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
            # Don't add self-references as dependencies
            if target_collection == collection_name:
                continue
            if target_collection in collection_names:
                graph[collection_name].add(target_collection)

    return graph


def sort_into_waves(
    graph: dict[str, set[str]],
) -> list[list[str]]:
    """Topological sort of collections into parallel waves.

    Each wave contains collections whose dependencies are all satisfied
    by previous waves. Collections within a wave can be generated in parallel.

    Cycles are broken by forcing the collection with fewer inbound
    references into an earlier wave.

    Args:
        graph: Dependency graph from build_dependency_graph.

    Returns:
        List of waves. Each wave is a sorted list of collection names.
    """
    if not graph:
        return []

    remaining = {name: set(deps) for name, deps in graph.items()}
    waves: list[list[str]] = []

    while remaining:
        # Find collections with no unresolved dependencies
        ready = sorted(name for name, deps in remaining.items() if not deps)

        if not ready:
            # Cycle detected — break it by picking the collection with
            # the fewest inbound references
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
    """Build a prompt for generating one collection's records.

    Args:
        config: The environment config (name, description, system_prompt).
        collection_name: Name of the collection to populate (e.g., "tenants").
        sub_schema: JSON Schema for a single record in this collection.
        existing_state: Previously generated collections (for FK context).
        scenario_context: Optional scenario description for realism.
        record_count: How many records to generate (e.g., "3-5").
        retry_error: If retrying, the error message from the previous attempt.

    Returns:
        A Conversation with system and user messages.
    """
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

    # Foreign key instructions
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
    """Build a prompt for generating full environment state.

    Used when no state_schema is provided. The LLM generates the complete
    state from the environment description and bound tool definitions.

    Args:
        config: The environment config (name, description, system_prompt).
        tools: Tools bound to this environment.
        scenario_context: Optional scenario description for realism.
        retry_error: If retrying, the error message from the previous attempt.

    Returns:
        A Conversation with system and user messages.
    """
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
    """Validate a generated collection's data.

    Checks:
    1. Data is a dict (keyed by ID).
    2. Each record validates against the sub-schema.
    3. Foreign key fields reference existing records.

    Args:
        collection_name: Name of this collection.
        data: The generated data (should be dict[str, dict]).
        sub_schema: JSON Schema for a single record.
        existing_state: Previously generated collections for FK checks.

    Returns:
        (True, None) if valid, (False, error_message) if not.
    """
    if not isinstance(data, dict):
        return (
            False,
            f"Expected a dict keyed by ID, got {type(data).__name__}.",
        )

    # Validate each record against the sub-schema
    for record_id, record in data.items():
        try:
            jsonschema.validate(instance=record, schema=sub_schema)
        except jsonschema.ValidationError as e:
            return (
                False,
                f"Record '{record_id}' failed schema validation: {e.message}",
            )

    # Check referential integrity
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
                continue  # Self-references are fine
            if target_collection not in existing_state:
                continue  # No data to check against
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
    """Builds environments once through a layered pipeline, then copies N times.

    Phases:
    1. Schema resolution: config-provided or derived from tools.
    2. Dependency analysis: topological sort into generation waves.
    3. Per-collection population: small LLM calls per collection, wave-ordered.
    4. Assembly + validation: combine and validate full state.
    """

    def __init__(self):
        """Initialize an empty registry."""
        self._built: dict[str, GeneratedToolEnvironment] = {}

    def register_static(self, config: ToolEnvironmentAttribute) -> None:
        """Register an environment with config-provided schema and state.

        No LLM calls needed — the config has everything.
        """
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
        """Build a fully populated environment through the layered pipeline.

        Two paths:
        - If config has a state_schema: wave-based per-collection generation.
        - Otherwise: single LLM call using env description + tool definitions.

        Args:
            config: Environment configuration.
            tools: Tools bound to this environment.
            inference_engine: Engine for LLM inference calls.
            inference_config: Config for inference calls.
            scenario_context: Optional scenario for realistic data generation.
        """
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
        """Return n independent deepcopies of the built environment.

        Args:
            env_id: The environment config ID.
            n: Number of copies to produce.

        Returns:
            List of independent GeneratedToolEnvironment instances.

        Raises:
            KeyError: If env_id has not been built or registered.
        """
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
        """Generate state using wave-based per-collection prompts.

        Used when config provides an explicit state_schema.
        """
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
        """Generate state from environment description and tool definitions.

        Used when no state_schema is provided. Makes a single LLM call
        with the environment description and bound tool definitions, and
        lets the LLM decide the appropriate state structure.
        """
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

        # Retry
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

        # Retry loop
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
