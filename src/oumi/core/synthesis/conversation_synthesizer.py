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

import copy
import json
import random
import re
from typing import Any

from oumi.builders.inference_engines import build_inference_engine
from oumi.core.configs.inference_config import InferenceConfig
from oumi.core.configs.inference_engine_type import InferenceEngineType
from oumi.core.configs.params.synthesis_params import (
    GeneralSynthesisParams,
    MultiTurnAttribute,
)
from oumi.core.configs.params.tool_params import (
    ToolAttribute,
    ToolEnvironmentAttribute,
    ToolOutputStrategy,
)
from oumi.core.synthesis.attribute_formatter import AttributeFormatter
from oumi.core.synthesis.environment import (
    _MAX_RESULT_RETRIES,
    _MAX_STATE_UPDATE_RETRIES,
    EnvironmentRegistry,
    GeneratedToolEnvironment,
)
from oumi.core.synthesis.tool_executor import (
    ToolCallError,
    ToolCallParsed,
    ToolExecutor,
    _example_value,
)
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.utils.logging import logger
from oumi.utils.str_utils import extract_json


def _clean_json_output(text: str) -> str:
    """Strip markdown fences and extract clean JSON from LLM-generated tool output."""
    parsed = extract_json(text, expected_type=None)
    if parsed is not None:
        return json.dumps(parsed)
    return text


def _is_valid_json(text: str) -> bool:
    """Return True if *text* is parseable as a JSON object or array."""
    try:
        result = json.loads(text)
        return isinstance(result, (dict, list))
    except (json.JSONDecodeError, TypeError):
        return False


def _build_example_result(tool: ToolAttribute) -> str:
    """Build a realistic example result from a tool's output_schema."""
    if not tool.output_schema:
        return '{"status": "ok"}'
    props = tool.output_schema.get("properties", {})
    if not props:
        return json.dumps(_example_value(tool.output_schema))
    example = {}
    for key, schema in props.items():
        example[key] = _example_value(schema)
    return json.dumps(example)


_INCOMPLETE_FINAL_ASSISTANT_PATTERN = re.compile(
    r"(?:^|[.!?]\s+)(?:now\s+)?(?:let me|i(?:'ll| will)|i am going to|i'm going to)\b",
    re.IGNORECASE,
)


class ConversationSynthesizer:
    """Synthesizes a conversation.

    Args:
        params: The parameters for the conversation synthesizer.
        inference_config: The configuration for the inference engine.
    """

    def __init__(
        self,
        params: GeneralSynthesisParams,
        inference_config: InferenceConfig,
    ):
        """Initialize the synthesizer."""
        self._params = params
        self._formatter = AttributeFormatter(params)

        self._inference_engine = build_inference_engine(
            engine_type=inference_config.engine or InferenceEngineType.NATIVE,
            model_params=inference_config.model,
            remote_params=inference_config.remote_params,
        )
        self._inference_config = inference_config
        self._default_turn_order = [Role.USER, Role.ASSISTANT]

        self._tools_by_id: dict[str, ToolAttribute] = {}
        if params.tools:
            for tool in params.tools:
                self._tools_by_id[tool.id] = tool

        self._env_configs: dict[str, ToolEnvironmentAttribute] = {}
        if params.environments:
            for env_config in params.environments:
                self._env_configs[env_config.id] = env_config

    def _get_tools_for_multiturn(
        self, multiturn_attribute: MultiTurnAttribute
    ) -> list[ToolAttribute]:
        """Resolve tool ids from a MultiTurnAttribute to ToolAttribute objects."""
        if not multiturn_attribute.available_tools:
            return []
        tools = []
        for tool_id in multiturn_attribute.available_tools:
            tool = self._tools_by_id.get(tool_id)
            if tool:
                tools.append(tool)
            else:
                logger.warning(
                    f"Tool id '{tool_id}' referenced in "
                    f"'{multiturn_attribute.id}' not found in params.tools"
                )
        return tools

    def _format_environment_config(
        self,
        sample: dict,
        config: ToolEnvironmentAttribute,
    ) -> ToolEnvironmentAttribute:
        """Format environment prompt fields with sample-specific attributes."""
        return ToolEnvironmentAttribute(
            id=config.id,
            name=self._formatter.format(
                sample,
                config.name,
                missing_values_allowed=True,
            ),
            description=self._formatter.format(
                sample,
                config.description,
                missing_values_allowed=True,
            ),
            system_prompt=self._formatter.format(
                sample,
                config.system_prompt,
                missing_values_allowed=True,
            ),
            state_schema=copy.deepcopy(config.state_schema),
            initial_state=copy.deepcopy(config.initial_state),
        )

    def _init_sample_environments(
        self,
        samples: list[dict],
        multiturn_attribute: MultiTurnAttribute,
    ) -> list[dict[str, GeneratedToolEnvironment] | None]:
        """Create environments, reusing shared builds when config is identical.

        Args:
            samples: List of sample dicts with resolved attribute values.
            multiturn_attribute: Multi-turn attribute defining which tools
                are used.

        Returns:
            A list aligned to samples. Each entry is a dict mapping
            env_id to a GeneratedToolEnvironment, or None if no
            env-bound tools exist.
        """
        tools = self._get_tools_for_multiturn(multiturn_attribute)
        env_tools: dict[str, list[ToolAttribute]] = {}
        for tool in tools:
            if tool.environment:
                env_tools.setdefault(tool.environment, []).append(tool)

        if not env_tools:
            return [None] * len(samples)

        result: list[dict[str, GeneratedToolEnvironment]] = [{} for _ in samples]

        for env_id, bound_tools in env_tools.items():
            config = self._env_configs.get(env_id)
            if not config:
                logger.warning(f"Environment config not found for '{env_id}'")
                continue

            variants: dict[
                tuple[str, str, str, str | None, str | None],
                GeneratedToolEnvironment,
            ] = {}

            for i, sample in enumerate(samples):
                formatted_config = self._format_environment_config(sample, config)
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
                            self._inference_engine,
                            self._inference_config,
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

    def _serialize_env_states(self, envs: dict[str, GeneratedToolEnvironment]) -> str:
        """Serialize the full state of all environments as formatted JSON."""
        parts = []
        for env_id, env in envs.items():
            state_json = json.dumps(env.state, indent=2)
            parts.append(f'Environment "{env_id}":\n{state_json}')
        return "\n\n".join(parts)

    def _validate_roles(self, multiturn_attribute: MultiTurnAttribute) -> None:
        """Validate that required roles have corresponding personas.

        Args:
            multiturn_attribute: The multi-turn attribute to validate.

        Raises:
            ValueError: If a required role is missing from role_instruction_messages.
        """
        available_roles = set(multiturn_attribute.role_instruction_messages.keys())

        for role in self._default_turn_order:
            if role not in available_roles:
                raise ValueError(
                    f"Role '{role.value}' is missing from "
                    f"role_instruction_messages. Available roles: "
                    f"{[r.value for r in available_roles]}"
                )

    def synthesize(
        self,
        samples: list[dict],
        multiturn_attributes: MultiTurnAttribute,
    ) -> list[dict[str, dict | str] | None]:
        """Synthesize a multi-turn conversation.

        Order will be identical to the order of the samples.

        Args:
            samples: The samples to synthesize values for.
            multiturn_attributes: The multi-turn attribute defining conversation rules.

        Returns:
            A list aligned to the input samples. Each entry is either:
            - a dictionary containing the conversation and plan, or
            - None when the synthesized conversation is filtered out.
        """
        if not samples:
            return []

        self._validate_roles(multiturn_attributes)

        logger.info(
            f"Synthesizing {len(samples)} conversations for "
            f"attribute '{multiturn_attributes.id}'"
        )

        tools = self._get_tools_for_multiturn(multiturn_attributes)
        has_envs = any(t.environment for t in tools)
        sample_envs = None
        env_states = None
        if has_envs:
            sample_envs = self._init_sample_environments(samples, multiturn_attributes)
            env_states = [
                self._serialize_env_states(envs) if envs else None
                for envs in sample_envs
            ]
        samples = self._plan_samples(
            samples, multiturn_attributes, env_states=env_states
        )
        conversations, tool_data = self._synthesize_all_samples(
            samples, multiturn_attributes, sample_envs=sample_envs
        )
        has_tools = tool_data is not None

        records: list[dict[str, dict | str] | None] = []
        plan_key = f"{multiturn_attributes.id}_plan"
        filtered_count = 0

        for i, (sample, conversation) in enumerate(zip(samples, conversations)):
            if self._has_empty_messages(conversation):
                filtered_count += 1
                records.append(None)
                continue

            if has_tools and self._has_empty_output_messages(
                tool_data["output_messages"][i]
            ):
                filtered_count += 1
                records.append(None)
                continue

            if has_tools and tool_data["tool_call_counts"][i] == 0:
                records.append(None)
                continue

            if has_tools and not self._has_valid_final_assistant_message(
                tool_data["output_messages"][i]
            ):
                filtered_count += 1
                records.append(None)
                continue

            if has_tools and self._final_assistant_message_looks_incomplete(
                tool_data["output_messages"][i]
            ):
                filtered_count += 1
                records.append(None)
                continue

            if has_tools:
                output_msgs = tool_data["output_messages"][i]
                sys_msg = self._format_output_system_message(
                    sample, multiturn_attributes.output_system_prompt
                )
                if sys_msg:
                    output_msgs = [
                        {"role": "system", "content": sys_msg.content}
                    ] + output_msgs

                conv_dict: dict = {
                    "tools": tool_data["tool_definitions"],
                    "messages": output_msgs,
                }
            else:
                conv_dict = conversation.to_dict()

            record: dict[str, dict | str] = {
                multiturn_attributes.id: conv_dict,
                plan_key: sample["conversation_plan"],
            }
            records.append(record)

        if filtered_count > 0:
            logger.warning(
                f"Filtered out {filtered_count} conversation(s) with empty messages "
                f"out of {len(conversations)} total"
            )

        return records

    def _plan_samples(
        self,
        samples: list[dict],
        multiturn_attributes: MultiTurnAttribute,
        max_retries: int = 2,
        env_states: list[str | None] | None = None,
    ) -> list[dict]:
        """Plan the conversation samples with retry logic for failed parses.

        Args:
            samples: The conversation samples to plan.
            multiturn_attributes: The multi-turn attribute defining conversation rules.
            max_retries: Maximum number of retry attempts for failed plan parsing.
            env_states: Optional list of serialized environment states for each sample.

        Returns:
            A list of sample dicts augmented with runtime fields
            (target_turns, conversation_plan, parsed_turn_plans).
        """
        turn_order = self._default_turn_order

        augmented_samples: list[dict] = []
        for sample in samples:
            target_turns = self._select_target_turns(multiturn_attributes, turn_order)
            augmented_sample = {
                **sample,
                "target_turns": target_turns,
                "conversation_plan": "",
                "parsed_turn_plans": [""] * target_turns,
            }
            augmented_samples.append(augmented_sample)
            logger.debug(f"Planning conversation with {target_turns} turns")

        indices_to_process = list(range(len(augmented_samples)))

        for attempt in range(max_retries + 1):
            if not indices_to_process:
                break

            planner_conversations = [
                self._create_planner_prompt(
                    multiturn_attributes,
                    augmented_samples[i],
                    env_state=env_states[i] if env_states else None,
                )
                for i in indices_to_process
            ]

            plans = self._generate_plan(planner_conversations)

            failed_indices: list[int] = []
            for idx, plan in zip(indices_to_process, plans):
                augmented_sample = augmented_samples[idx]
                target_turns = augmented_sample["target_turns"]
                parsed = self._parse_plan(plan, target_turns)

                if parsed is not None:
                    augmented_sample["conversation_plan"] = plan
                    augmented_sample["parsed_turn_plans"] = parsed
                else:
                    failed_indices.append(idx)
                    if attempt < max_retries:
                        logger.warning(
                            f"Plan parsing failed for sample {idx}, "
                            f"retrying ({attempt + 1}/{max_retries})"
                        )

            indices_to_process = failed_indices

        if indices_to_process:
            logger.warning(
                f"Failed to parse plans for {len(indices_to_process)} samples "
                f"after {max_retries + 1} attempts, proceeding without plan"
            )

        return augmented_samples

    def _parse_plan(self, plan: str, target_turns: int) -> list[str] | None:
        """Parse a JSON-formatted conversation plan.

        Extracts turn instructions from JSON array. Expects format:
        [{"turn": 1, "instruction": "..."}, ...]

        Args:
            plan: The full plan text from the planner.
            target_turns: Expected number of turns.

        Returns:
            List of instruction strings (one per turn), or None if parsing failed.
        """
        if not plan:
            return None

        turns = extract_json(plan, expected_type=list)
        if turns is None:
            single = extract_json(plan, expected_type=dict)
            if single is not None:
                turns = [single]
            else:
                return None

        result = [""] * target_turns
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            turn_num = turn.get("turn")
            instruction = turn.get("instruction", "")

            if isinstance(turn_num, str):
                try:
                    turn_num = int(turn_num)
                except ValueError:
                    continue
            if isinstance(turn_num, int) and 1 <= turn_num <= target_turns:
                result[turn_num - 1] = str(instruction).strip()

        return result

    def _extract_response(
        self,
        inference_conversations: list[Conversation],
    ) -> list[str]:
        """Get the inference results from the inference conversations.

        If the inference result is not a string or the conversation is empty,
        an empty string will be returned.
        Strips whitespace to avoid API errors with trailing whitespace.
        """
        results = []
        for inference_result in inference_conversations:
            if not inference_result.messages:
                results.append("")
                continue
            content = inference_result.messages[-1].content
            if isinstance(content, str):
                results.append(content.strip())
            else:
                results.append("")
        return results

    def _infer_exact(
        self, prompts: list[Conversation], context: str
    ) -> list[Conversation]:
        """Run batched inference and enforce one response per prompt."""
        responses = self._inference_engine.infer(
            prompts, inference_config=self._inference_config
        )
        if len(responses) != len(prompts):
            raise RuntimeError(
                f"{context}: inference returned {len(responses)} results "
                f"for {len(prompts)} prompts."
            )
        return responses

    def _has_empty_messages(self, conversation: Conversation) -> bool:
        """Check if any non-system message in a conversation has empty content.

        System messages (e.g., output_system_prompt) are excluded from this check
        since they are generated by the synthesizer itself, not by inference.

        Args:
            conversation: The conversation to check.

        Returns:
            True if any non-system message has empty string content.
        """
        for message in conversation.messages:
            if message.role == Role.SYSTEM:
                continue
            if not isinstance(message.content, str) or not message.content.strip():
                return True
        return False

    @staticmethod
    def _has_empty_output_messages(output_msgs: list[dict]) -> bool:
        """Check if any user/assistant output message has empty content."""
        for msg in output_msgs:
            role = msg.get("role")
            if role in ("system", "tool"):
                continue
            if role == "assistant" and "tool_calls" in msg:
                continue
            content = msg.get("content", "")
            if isinstance(content, str) and not content.strip():
                return True
        return False

    @staticmethod
    def _sanitize_turn_text(
        role: Role,
        text: str,
        tool_executor: ToolExecutor | None,
    ) -> str:
        """Remove tool-call artifacts before storing conversational text."""
        if role == Role.ASSISTANT and tool_executor is not None:
            return ToolExecutor.sanitize_assistant_content(text)
        return text.strip()

    @staticmethod
    def _append_if_present(
        history: list[Message],
        role: Role,
        content: str,
    ) -> None:
        """Append a message only when the cleaned content is non-empty."""
        if content:
            history.append(Message(role=role, content=content))

    @staticmethod
    def _has_valid_final_assistant_message(output_msgs: list[dict]) -> bool:
        """Require the final natural-language message to come from the assistant."""
        for msg in reversed(output_msgs):
            role = msg.get("role")
            if role in ("system", "tool"):
                continue
            if role == "assistant" and "tool_calls" in msg:
                continue
            content = msg.get("content", "")
            return role == "assistant" and isinstance(content, str) and bool(
                content.strip()
            )
        return False

    @staticmethod
    def _final_assistant_message_looks_incomplete(output_msgs: list[dict]) -> bool:
        """Detect assistant endings that announce future work instead of concluding."""
        for msg in reversed(output_msgs):
            role = msg.get("role")
            if role in ("system", "tool"):
                continue
            if role == "assistant" and "tool_calls" in msg:
                continue
            if role != "assistant":
                return True
            content = msg.get("content", "")
            if not isinstance(content, str):
                return True
            return bool(_INCOMPLETE_FINAL_ASSISTANT_PATTERN.search(content.strip()))
        return True

    def _format_persona(
        self, sample: dict, persona: str, tools: list[ToolAttribute]
    ) -> Message:
        """Format the persona for the sample.

        Args:
            sample: The sample dict containing all attributes.
            persona: The persona string to format.
            tools: The list of ToolAttribute objects available to the persona.

        Returns:
            A Message with the formatted persona as a SYSTEM message.
        """
        formatted_content = self._formatter.format(
            sample,
            persona,
            missing_values_allowed=False,
        )
        if tools:
            tool_catalog = ToolExecutor.build_tool_catalog(tools)
            tool_section = (
                "\n\n## Available Tools\n\n"
                f"{tool_catalog}\n\n"
                "## RULES\n"
                "1. To call a tool, output EXACTLY: "
                '<tool_call>{"name": "...", "arguments": {...}}'
                "</tool_call>\n"
                "2. NEVER narrate or simulate a tool call. "
                "If you need data, call the tool.\n"
                "3. NEVER fabricate tool results. "
                "Every data point MUST come from an actual tool response.\n"
                "4. WAIT for the tool result before responding."
            )
            formatted_content += tool_section

            return Message(role=Role.SYSTEM, content=formatted_content)
        return Message(
            role=Role.SYSTEM,
            content=formatted_content,
        )

    @staticmethod
    def _build_tool_few_shot(tools: list[ToolAttribute]) -> list[Message]:
        """Build few-shot messages demonstrating a correct tool-call exchange.

        Creates actual message turns mirroring the exact format used by
        _record_tool_result so the model sees a realistic interaction pattern.
        """
        if not tools:
            return []

        tool = tools[0]
        example_args: dict[str, Any] = {}
        if tool.parameters:
            for pname, pschema in tool.parameters.get("properties", {}).items():
                if "example" in pschema:
                    example_args[pname] = pschema["example"]
                elif "enum" in pschema:
                    example_args[pname] = pschema["enum"][0]
                else:
                    example_args[pname] = _example_value(pschema)

        tool_call_json = json.dumps({"name": tool.name, "arguments": example_args})
        example_result = _build_example_result(tool)

        return [
            Message(
                role=Role.USER,
                content=f"Look up information using {tool.name}.",
            ),
            Message(
                role=Role.ASSISTANT,
                content=f"<tool_call>{tool_call_json}</tool_call>",
            ),
            Message(
                role=Role.USER,
                content=(f"[Tool result from {tool.name}]: {example_result}"),
            ),
            Message(
                role=Role.ASSISTANT,
                content=f"The {tool.name} result shows {example_result}.",
            ),
        ]

    @staticmethod
    def _build_tool_turn_info(
        current_turn: int,
        target_turns: int,
        turn_instruction: str,
        max_calls_reached: bool,
    ) -> str:
        """Build the turn-level user message for assistant tool turns.

        Places the format constraint in the user message (last thing the LLM
        reads) with a concrete full-turn example.
        """
        parts = [f"Turn {current_turn} of {target_turns}.\n"]
        is_final_turn = current_turn == target_turns

        if turn_instruction:
            parts.append(f"Task: {turn_instruction}\n")

        if is_final_turn:
            parts.append(
                "This is the final turn. Finish the task now. "
                "Do not announce future steps or promise another action.\n"
            )

        if max_calls_reached:
            parts.append(
                "You have used all tool calls for this turn. "
                "Respond to the user based on the information gathered so far. "
                "Do NOT output tool tags or tool JSON."
            )
            return "\n".join(parts)

        parts.append(
            "You MUST use <tool_call> tags to call tools. "
            "Do NOT narrate, describe, or simulate tool calls. "
            "Do NOT make up results — call the tool and wait for the response."
        )
        if is_final_turn:
            parts.append(
                "If you already have enough verified information, conclude "
                "directly instead of starting another step."
            )
        return "\n".join(parts)

    @staticmethod
    def _build_prose_turn_info(
        current_turn: int,
        target_turns: int,
        role: str,
        turn_instruction: str,
    ) -> str:
        """Build turn-level user message for non-tool turns (user turns)."""
        parts = [
            f"You are generating turn {current_turn} of {target_turns} as the {role}.\n"
        ]
        if turn_instruction:
            parts.append(f"For this turn: {turn_instruction}\n")
        if role == Role.USER.value.upper():
            parts.append(
                "Write like a real person: direct, concise, and do not narrate "
                "your workflow.\n"
            )
        parts.append("Generate your response for this turn.")
        return "\n".join(parts)

    def _build_role_context(
        self, sample: dict, multiturn_attribute: MultiTurnAttribute
    ) -> str:
        """Build formatted role context for the planner.

        Formats the persona strings for each role.
        The returned string has curly braces escaped ({{ and }}) so it can be
        safely embedded in another template without causing format errors.
        """
        parts = []
        for role, persona in multiturn_attribute.role_instruction_messages.items():
            formatted = self._formatter.format(
                sample, persona, missing_values_allowed=False
            )
            parts.append(f"[{role.value.upper()}]\n{formatted}")

        result = "\n\n".join(parts)
        return result.replace("{", "{{").replace("}", "}}")

    def _build_turn_order_str(self, turn_order: list[Role], target_turns: int) -> str:
        """Build a string showing which role speaks at each turn.

        Args:
            turn_order: The role sequence that repeats.
            target_turns: Total number of turns.

        Returns:
            A string like "Turn 1: USER, Turn 2: ASSISTANT, Turn 3: USER, ..."
        """
        parts = []
        for i in range(target_turns):
            role = turn_order[i % len(turn_order)]
            parts.append(f"Turn {i + 1}: {role.value.upper()}")
        return ", ".join(parts)

    def _create_planner_prompt(
        self,
        multiturn_attribute: MultiTurnAttribute,
        sample: dict,
        env_state: str | None = None,
    ) -> Conversation:
        """Create the planner prompt template with role context and turn order.

        Returns a Conversation with a one-shot example for consistent formatting.
        """
        role_context = self._build_role_context(sample, multiturn_attribute)
        turn_order = self._default_turn_order
        target_turns = sample["target_turns"]
        turn_order_str = self._build_turn_order_str(turn_order, target_turns)

        tools = self._get_tools_for_multiturn(multiturn_attribute)
        has_tools = bool(tools)

        system_prompt = (
            "You are a conversation planner. Create conversation outlines "
            "that flow logically from start to finish.\n\n"
            "IMPORTANT: Output your plan as a JSON array wrapped in ```json code "
            "fences. Each element must have: turn (number) and instruction (string).\n"
            "Your instructions MUST be specific to the role context provided. "
            "Each turn's instruction should reflect what that specific role "
            "would do at that point in the conversation."
        )

        if has_tools:
            system_prompt += (
                "\n\nThe assistant has access to tools and will use them naturally "
                "to complete tasks. Write instructions that describe WHAT the "
                "assistant should accomplish — looking up information, verifying "
                "details, or taking actions — without naming specific tools. "
                "The assistant will decide which tools to use on its own."
            )

        if has_tools:
            example_request = (
                "Plan a 4-turn conversation.\n"
                "Turn order: Turn 1: USER, Turn 2: ASSISTANT, Turn 3: USER, "
                "Turn 4: ASSISTANT\n\n"
                "The assistant has the following capabilities:\n"
                "- Look up order details and status\n"
                "- Check return eligibility for an order\n\n"
                "Role context:\n"
                "[USER]\n"
                "You are a customer who wants to return a recent order.\n\n"
                "[ASSISTANT]\n"
                "You are a support agent who helps customers with orders.\n\n"
                "Additional instructions: Help the customer with their return "
                "request by investigating the order details."
            )
            example_response = """```json
[
  {"turn": 1, "instruction": "Explain that you want to return your order and
  provide the reason"},
  {"turn": 2, "instruction": "Verify the order details and check whether a return
  is eligible based on the return policy"},
  {"turn": 3, "instruction": "Confirm you want to proceed with the return"},
  {"turn": 4, "instruction": "Process the return and provide confirmation with
  next steps"}
]
```"""
        else:
            example_request = (
                "Plan a 4-turn conversation.\n"
                "Turn order: Turn 1: USER, Turn 2: ASSISTANT, Turn 3: USER, "
                "Turn 4: ASSISTANT\n\n"
                "Role context:\n"
                "[USER]\n"
                "You are a customer who has an issue with a recent order.\n\n"
                "[ASSISTANT]\n"
                "You are a helpful support agent who resolves customer issues.\n\n"
                "Additional instructions: Focus on resolving the order issue "
                "efficiently while maintaining a polite and helpful tone."
            )
            example_response = """```json
[
  {"turn": 1, "instruction": "Greet support and explain the issue with the order"},
  {"turn": 2, "instruction": "Acknowledge the issue and ask for order details"},
  {"turn": 3, "instruction": "Provide order number and describe the problem further"},
  {"turn": 4, "instruction": "Confirm the issue and offer a resolution"}
]
```"""

        base_prompt = (
            f"Plan a {target_turns}-turn conversation.\n"
            f"Turn order: {turn_order_str}\n\n"
            "Guidelines:\n"
            "- Each turn should build on the previous turn.\n"
            f"- Pace the conversation naturally for {target_turns} turns.\n"
            "- Focus on what happens, not exact wording.\n"
            "- Instructions MUST be specific to the roles and context provided below.\n"
        )

        if has_tools:
            base_prompt += (
                "- ASSISTANT turns should actively investigate, verify, or "
                "take actions using the assistant's available capabilities.\n"
            )
            capability_summary = ToolExecutor.build_capability_summary(tools)
            if capability_summary:
                base_prompt += (
                    "\nThe assistant has the following capabilities:\n"
                    f"{capability_summary}\n"
                )

        if role_context:
            base_prompt += f"\nRole context:\n{role_context}\n"

        if multiturn_attribute.conversation_planner:
            formatted_planner = self._formatter.format(
                sample,
                multiturn_attribute.conversation_planner,
                missing_values_allowed=False,
            )
            base_prompt += f"\nAdditional instructions: {formatted_planner}\n"

        if env_state:
            base_prompt += (
                f"\nThe environment contains the following data:\n{env_state}\n\n"
                "Your plan MUST be grounded in this data. Reference actual entities, "
                "values, and relationships present in the environment. "
                "Do not invent data that is not here.\n"
            )

        base_prompt += "\nOutput ONLY the JSON array wrapped in ```json code fences. "

        return Conversation(
            messages=[
                Message(role=Role.SYSTEM, content=system_prompt),
                Message(role=Role.USER, content=example_request),
                Message(role=Role.ASSISTANT, content=example_response),
                Message(role=Role.USER, content=base_prompt),
            ],
        )

    def _generate_plan(self, planners: list[Conversation]) -> list[str]:
        """Generate plans for how the conversations should proceed.

        Args:
            planners: The planner conversation templates (already formatted).

        Returns:
            A list of plan strings, one per sample.
        """
        inference_results = self._infer_exact(
            planners,
            context="Conversation planning",
        )

        return self._extract_response(inference_results)

    def _synthesize_all_samples(
        self,
        samples: list[dict],
        multiturn_attribute: MultiTurnAttribute,
        sample_envs: list[dict[str, GeneratedToolEnvironment] | None] | None = None,
    ) -> tuple[list[Conversation], dict | None]:
        """Synthesize multi-turn conversations for all samples with batched inference.

        Returns:
            (conversations, tool_data) where tool_data is None when no tools
            are configured.
        """
        if not samples:
            return [], None

        tools = self._get_tools_for_multiturn(multiturn_attribute)

        resolved_envs: list[dict[str, GeneratedToolEnvironment] | None] = (
            sample_envs if sample_envs is not None else [None for _ in samples]
        )

        tool_executor = ToolExecutor(tools) if tools else None
        deterministic_selections = (
            [tool_executor.sample_deterministic_outputs(tools) for _ in samples]
            if tool_executor
            else []
        )
        # contains raw tool calls
        full_histories: list[list[Message]] = [[] for _ in samples]
        # only contains conversational messages, no tool calls
        conversational_histories: list[list[Message]] = [[] for _ in samples]

        output_messages: list[list[dict]] = [[] for _ in samples]
        tool_call_counts = [0] * len(samples)
        max_turns = max(s["target_turns"] for s in samples)
        max_tool_calls = multiturn_attribute.max_tool_calls_per_turn

        for turn_idx in range(max_turns):
            current_turn = turn_idx + 1
            role = self._default_turn_order[turn_idx % len(self._default_turn_order)]
            active = [i for i, s in enumerate(samples) if turn_idx < s["target_turns"]]
            if not active:
                break

            is_tool_turn = role == Role.ASSISTANT and tool_executor is not None
            turn_tool_msgs: dict[int, list[Message]] = {i: [] for i in active}
            turn_call_counts: dict[int, int] = {i: 0 for i in active}

            while active:
                prompts: list[Conversation] = []
                for i in active:
                    sample_ctx = {**samples[i], "current_turn": current_turn}
                    persona_text = multiturn_attribute.role_instruction_messages[role]

                    persona_msg = self._format_persona(
                        sample_ctx,
                        persona_text,
                        tools if is_tool_turn else [],
                    )

                    turn_instruction = ""
                    parsed_plans = samples[i].get("parsed_turn_plans", [])
                    if turn_idx < len(parsed_plans):
                        turn_instruction = parsed_plans[turn_idx]

                    if is_tool_turn:
                        turn_info = self._build_tool_turn_info(
                            current_turn=current_turn,
                            target_turns=samples[i]["target_turns"],
                            turn_instruction=turn_instruction,
                            max_calls_reached=turn_call_counts[i] >= max_tool_calls,
                        )
                    else:
                        turn_info = self._build_prose_turn_info(
                            current_turn=current_turn,
                            target_turns=samples[i]["target_turns"],
                            role=role.value.upper(),
                            turn_instruction=turn_instruction,
                        )

                    history = (
                        full_histories[i]
                        if is_tool_turn
                        else conversational_histories[i]
                    )
                    few_shot = (
                        self._build_tool_few_shot(tools)
                        if is_tool_turn and not history
                        else []
                    )
                    msgs = (
                        [persona_msg]
                        + few_shot
                        + history
                        + [Message(role=Role.USER, content=turn_info)]
                        + turn_tool_msgs[i]
                    )
                    prompts.append(Conversation(messages=msgs))

                texts = self._extract_response(
                    self._infer_exact(
                        prompts,
                        context=f"Turn {current_turn} generation",
                    )
                )

                still_active: list[int] = []
                gen_items: list[tuple[int, str, dict, str]] = []
                gen_prompts: list[Conversation] = []
                env_items: list[
                    tuple[
                        int,
                        str,
                        dict,
                        str,
                        GeneratedToolEnvironment,
                        ToolAttribute,
                    ]
                ] = []
                env_result_prompts: list[Conversation] = []

                for idx, text in zip(active, texts):
                    tc_result = None
                    if is_tool_turn and tool_executor:
                        tc_result = tool_executor.parse_and_validate_tool_call(text)
                    max_calls_reached = turn_call_counts.get(idx, 0) >= max_tool_calls

                    if max_calls_reached and isinstance(
                        tc_result, (ToolCallParsed, ToolCallError)
                    ):
                        logger.warning(
                            "Dropping over-limit tool call from assistant turn output."
                        )
                        tc_result = None

                    if isinstance(tc_result, ToolCallError) and not max_calls_reached:
                        # Inject deterministic error as tool result
                        turn_call_counts[idx] += 1
                        tool_call_counts[idx] += 1
                        call_id = f"call_{tool_call_counts[idx]:03d}"
                        error_tool_call = {
                            "name": tc_result.tool_name or "unknown",
                            "arguments": {},
                        }
                        self._record_tool_result(
                            idx,
                            text,
                            error_tool_call,
                            call_id,
                            tc_result.error_json,
                            turn_tool_msgs,
                            output_messages,
                        )
                        still_active.append(idx)
                        continue

                    tool_call = (
                        tc_result.tool_call
                        if isinstance(tc_result, ToolCallParsed)
                        else None
                    )

                    if tool_call is None:
                        full_histories[idx].extend(turn_tool_msgs[idx])
                        # Add assistant prose fragments (not tool results)
                        # to conversational history so the user sees natural
                        # language the assistant said before tool calls.
                        for msg in turn_tool_msgs[idx]:
                            if msg.role == Role.ASSISTANT:
                                cleaned_msg = self._sanitize_turn_text(
                                    msg.role,
                                    msg.content if isinstance(msg.content, str) else "",
                                    tool_executor,
                                )
                                self._append_if_present(
                                    conversational_histories[idx],
                                    msg.role,
                                    cleaned_msg,
                                )
                        cleaned_text = self._sanitize_turn_text(
                            role, text, tool_executor
                        )
                        self._append_if_present(full_histories[idx], role, cleaned_text)
                        self._append_if_present(
                            conversational_histories[idx], role, cleaned_text
                        )
                        if tool_executor:
                            if cleaned_text:
                                output_messages[idx].append(
                                    {"role": role.value, "content": cleaned_text}
                                )
                        continue

                    assert tool_executor is not None
                    turn_call_counts[idx] += 1
                    tool_call_counts[idx] += 1
                    call_id = f"call_{tool_call_counts[idx]:03d}"

                    env, tool_obj = self._resolve_env_tool(
                        tool_executor, tool_call, resolved_envs[idx]
                    )

                    if env and tool_obj:
                        env_items.append((idx, text, tool_call, call_id, env, tool_obj))
                        env_result_prompts.append(
                            env.build_result_prompt(
                                tool_obj,
                                arguments=tool_call["arguments"],
                            )
                        )
                    elif self._is_env_tool_missing_env(tool_executor, tool_call):
                        still_active.append(idx)
                    else:
                        result = tool_executor.resolve_output(
                            tool_call, deterministic_selections[idx]
                        )
                        if result is not None:
                            self._record_tool_result(
                                idx,
                                text,
                                tool_call,
                                call_id,
                                result,
                                turn_tool_msgs,
                                output_messages,
                            )
                            still_active.append(idx)
                        else:
                            ctx = full_histories[idx] + turn_tool_msgs[idx]
                            gen_items.append((idx, text, tool_call, call_id))
                            gen_prompts.append(
                                tool_executor.build_generated_simulator_prompt(
                                    tool_call, ctx
                                )
                            )
                if env_result_prompts:
                    env_activated = self._process_env_tool_calls(
                        env_items,
                        env_result_prompts,
                        turn_tool_msgs,
                        output_messages,
                    )
                    still_active.extend(env_activated)
                if gen_prompts:
                    sim_texts = self._extract_response(
                        self._infer_exact(
                            gen_prompts,
                            context=(
                                f"Turn {current_turn} generated-tool simulation"
                            ),
                        )
                    )
                    # Identify which results need retries
                    gen_failed: list[int] = []
                    gen_results: list[str | None] = [None] * len(gen_items)
                    for j, sim in enumerate(sim_texts):
                        cleaned = _clean_json_output(sim)
                        if _is_valid_json(cleaned):
                            gen_results[j] = cleaned
                        else:
                            gen_failed.append(j)

                    assert tool_executor is not None
                    for _ in range(_MAX_RESULT_RETRIES):
                        if not gen_failed:
                            break
                        retry_prompts = [
                            tool_executor.build_generated_simulator_prompt(
                                gen_items[j][2],
                                full_histories[gen_items[j][0]]
                                + turn_tool_msgs[gen_items[j][0]],
                            )
                            for j in gen_failed
                        ]
                        retry_texts = self._extract_response(
                            self._infer_exact(
                                retry_prompts,
                                context=(
                                    "Generated-tool simulation retry "
                                    f"for turn {current_turn}"
                                ),
                            )
                        )
                        still_gen_failed: list[int] = []
                        for j, sim in zip(gen_failed, retry_texts):
                            cleaned = _clean_json_output(sim)
                            if _is_valid_json(cleaned):
                                gen_results[j] = cleaned
                            else:
                                still_gen_failed.append(j)
                        gen_failed = still_gen_failed

                    for j in gen_failed:
                        logger.warning(
                            f"Generated tool result for "
                            f"'{gen_items[j][2]['name']}' was not valid "
                            f"JSON after {_MAX_RESULT_RETRIES} retries."
                        )
                        gen_results[j] = _clean_json_output(sim_texts[j])

                    for j, (idx, raw, tc, cid) in enumerate(gen_items):
                        self._record_tool_result(
                            idx,
                            raw,
                            tc,
                            cid,
                            gen_results[j] or "",
                            turn_tool_msgs,
                            output_messages,
                        )
                        still_active.append(idx)

                if not is_tool_turn:
                    break
                active = still_active

        conversations: list[Conversation] = []
        for sample, history in zip(samples, full_histories):
            out: list[Message] = []
            sys_msg = self._format_output_system_message(
                sample, multiturn_attribute.output_system_prompt
            )
            if sys_msg:
                out.append(sys_msg)
            out.extend(history)
            conversations.append(Conversation(messages=out))

        tool_data = None
        if tool_executor:
            tool_data = {
                "tool_definitions": ToolExecutor.build_tool_definitions(tools),
                "output_messages": output_messages,
                "tool_call_counts": tool_call_counts,
            }

        return conversations, tool_data

    @staticmethod
    def _resolve_env_tool(
        tool_executor: ToolExecutor,
        tool_call: dict,
        idx_envs: dict[str, GeneratedToolEnvironment] | None,
    ) -> tuple[GeneratedToolEnvironment | None, ToolAttribute | None]:
        """Look up the environment and tool object for a tool call.

        Returns (env, tool_obj) if this is an environment-bound tool with a
        matching environment, or (None, None) otherwise.
        """
        if idx_envs is None:
            return None, None
        tool_obj = tool_executor.get_tool_by_name(tool_call["name"])
        if tool_obj and tool_obj.environment:
            env = idx_envs.get(tool_obj.environment)
            if env:
                return env, tool_obj
        return None, None

    @staticmethod
    def _is_env_tool_missing_env(tool_executor: ToolExecutor, tool_call: dict) -> bool:
        """Return True if tool_call targets an ENVIRONMENT tool whose env is missing."""
        tool = tool_executor.get_tool_by_name(tool_call["name"])
        if tool and tool.output_strategy == ToolOutputStrategy.ENVIRONMENT:
            logger.warning(
                f"Environment not found for tool '{tool_call['name']}', skipping."
            )
            return True
        return False

    @staticmethod
    def _record_tool_result(
        idx: int,
        raw_text: str,
        tool_call: dict,
        call_id: str,
        result: str,
        turn_tool_msgs: dict[int, list[Message]],
        output_messages: list[list[dict]],
        env_state: dict | None = None,
    ) -> None:
        """Append a tool call + result to conversation history and output messages."""
        turn_tool_msgs[idx].append(Message(role=Role.ASSISTANT, content=raw_text))
        turn_tool_msgs[idx].append(
            Message(
                role=Role.USER,
                content=f"[Tool result from {tool_call['name']}]: {result}",
            )
        )
        output_messages[idx].append(
            ToolExecutor.format_tool_call_message(tool_call, call_id)
        )
        tool_result_msg = ToolExecutor.format_tool_result_message(
            call_id, result, tool_call["name"]
        )
        if env_state is not None:
            tool_result_msg["_environment_state"] = env_state
        output_messages[idx].append(tool_result_msg)

    def _process_env_tool_calls(
        self,
        env_items: list[
            tuple[int, str, dict, str, GeneratedToolEnvironment, ToolAttribute]
        ],
        env_result_prompts: list[Conversation],
        turn_tool_msgs: dict[int, list[Message]],
        output_messages: list[list[dict]],
    ) -> list[int]:
        """Execute batched environment tool calls: results, state updates, retries.

        Returns list of sample indices that were processed (to add to still_active).
        """
        env_responses = self._infer_exact(
            env_result_prompts,
            context="Environment tool result generation",
        )

        env_results: list[str | None] = [None] * len(env_items)
        last_raw: dict[int, str] = {}
        failed_indices: list[int] = []

        for i, ((idx, text, tool_call, call_id, env, tool_obj), response) in enumerate(
            zip(env_items, env_responses)
        ):
            raw = env.apply_result(response)
            result = _clean_json_output(raw)
            if _is_valid_json(result):
                env_results[i] = result
            else:
                last_raw[i] = result
                failed_indices.append(i)

        for _ in range(_MAX_RESULT_RETRIES):
            if not failed_indices:
                break
            retry_prompts = [
                env_items[i][4].build_result_prompt(
                    env_items[i][5],
                    arguments=env_items[i][2]["arguments"],
                    retry=True,
                )
                for i in failed_indices
            ]
            retry_responses = self._infer_exact(
                retry_prompts,
                context="Environment tool result generation retry",
            )
            still_failed: list[int] = []
            for i, response in zip(failed_indices, retry_responses):
                raw = env_items[i][4].apply_result(response)
                result = _clean_json_output(raw)
                if _is_valid_json(result):
                    env_results[i] = result
                else:
                    last_raw[i] = result
                    still_failed.append(i)
            failed_indices = still_failed

        for i in failed_indices:
            logger.warning(
                f"Tool result for '{env_items[i][5].name}' was not valid JSON "
                f"after {_MAX_RESULT_RETRIES} retries. Using raw text."
            )
            env_results[i] = last_raw[i]

        final_results: list[str] = []
        state_update_indices: list[int] = []
        state_update_prompts: list[Conversation] = []

        for i, (idx, text, tool_call, call_id, env, tool_obj) in enumerate(env_items):
            result = env_results[i]
            assert result is not None
            final_results.append(result)
            if not tool_obj.read_only and i not in failed_indices:
                state_update_indices.append(i)
                state_update_prompts.append(
                    env.build_state_update_prompt(
                        tool_obj,
                        arguments=tool_call["arguments"],
                        result=result,
                    )
                )

        if state_update_prompts:
            self._apply_state_updates(
                env_items, final_results, state_update_indices, state_update_prompts
            )

        activated: list[int] = []
        for i, (idx, text, tool_call, call_id, env, tool_obj) in enumerate(env_items):
            self._record_tool_result(
                idx,
                text,
                tool_call,
                call_id,
                final_results[i],
                turn_tool_msgs,
                output_messages,
                env_state=copy.deepcopy(env.state),
            )
            activated.append(idx)
        return activated

    def _apply_state_updates(
        self,
        env_items: list[
            tuple[int, str, dict, str, GeneratedToolEnvironment, ToolAttribute]
        ],
        env_results: list[str],
        state_update_indices: list[int],
        state_update_prompts: list[Conversation],
    ) -> None:
        """Apply state updates with batched retries on failure."""
        update_responses = self._infer_exact(
            state_update_prompts,
            context="Environment state update generation",
        )

        failed: list[int] = []
        for ui, response in zip(state_update_indices, update_responses):
            env = env_items[ui][4]
            if not env.apply_state_update(response):
                failed.append(ui)

        for _ in range(_MAX_STATE_UPDATE_RETRIES):
            if not failed:
                break
            retry_prompts = [
                env_items[ui][4].build_state_update_prompt(
                    env_items[ui][5],
                    arguments=env_items[ui][2]["arguments"],
                    result=env_results[ui],
                    retry=True,
                )
                for ui in failed
            ]
            retry_responses = self._infer_exact(
                retry_prompts,
                context="Environment state update generation retry",
            )
            still_failed = []
            for ui, response in zip(failed, retry_responses):
                if not env_items[ui][4].apply_state_update(response):
                    still_failed.append(ui)
            failed = still_failed

        for ui in failed:
            logger.warning(
                f"State update failed after {_MAX_STATE_UPDATE_RETRIES} retries "
                f"for tool '{env_items[ui][5].name}'. Keeping previous state."
            )

    def _select_target_turns(
        self, multiturn_attribute: MultiTurnAttribute, turn_order: list[Role]
    ) -> int:
        min_turns = multiturn_attribute.min_turns
        max_turns = multiturn_attribute.max_turns
        target_turns = random.randint(min_turns, max_turns)
        if Role.ASSISTANT not in turn_order:
            return target_turns

        def role_at(turn_count: int) -> Role:
            return turn_order[(turn_count - 1) % len(turn_order)]

        if role_at(target_turns) == Role.ASSISTANT:
            return target_turns
        for turn_count in range(target_turns + 1, max_turns + 1):
            if role_at(turn_count) == Role.ASSISTANT:
                return turn_count
        for turn_count in range(target_turns - 1, min_turns - 1, -1):
            if role_at(turn_count) == Role.ASSISTANT:
                return turn_count
        return target_turns

    def _format_output_system_message(
        self,
        sample: dict,
        system_message: str | None,
    ) -> Message | None:
        if system_message is None:
            return None
        formatted_content = self._formatter.format(
            sample,
            system_message,
        )
        return Message(role=Role.SYSTEM, content=formatted_content.strip())
