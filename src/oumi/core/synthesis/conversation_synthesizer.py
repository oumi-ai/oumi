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
    GeneratedToolEnvironment,
)
from oumi.core.synthesis.tool_executor import (
    ToolCallError,
    ToolCallParsed,
    ToolExecutor,
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

    def _init_sample_environments(
        self,
        samples: list[dict],
        multiturn_attribute: MultiTurnAttribute,
    ) -> list[dict[str, GeneratedToolEnvironment] | None]:
        """Create and initialize per-sample environments with batched LLM calls.

        Args:
            samples: List of sample dicts, each containing resolved attribute values.
            multiturn_attribute: The multi-turn attribute defining which tools are used.

        Returns:
            A list aligned to samples. Each entry is a dict mapping env_id to a
            fresh GeneratedToolEnvironment, or None if no env-bound tools exist.
        """
        tools = self._get_tools_for_multiturn(multiturn_attribute)
        env_tools: dict[str, list[ToolAttribute]] = {}
        for tool in tools:
            if tool.environment:
                env_tools.setdefault(tool.environment, []).append(tool)

        if not env_tools:
            return [None] * len(samples)

        scenario_parts = []
        for role, instruction in multiturn_attribute.role_instruction_messages.items():
            scenario_parts.append(f"{role.value}: {instruction}")
        scenario_template = "\n".join(scenario_parts) if scenario_parts else None
        sample_envs: list[dict[str, GeneratedToolEnvironment]] = []
        for _ in samples:
            envs: dict[str, GeneratedToolEnvironment] = {}
            for env_id in env_tools:
                config = self._env_configs.get(env_id)
                if not config:
                    logger.warning(f"Environment config not found for '{env_id}'")
                    continue
                envs[env_id] = GeneratedToolEnvironment(config=config)
            sample_envs.append(envs)

        def _resolve_scenario(sample: dict) -> str | None:
            if scenario_template is None:
                return None
            return self._formatter.format(
                sample, scenario_template, missing_values_allowed=True
            )

        scenario_contexts = [_resolve_scenario(s) for s in samples]
        schema_pairs: list[tuple[int, str]] = []
        schema_prompts: list = []
        for i, envs in enumerate(sample_envs):
            ctx = scenario_contexts[i]
            for env_id, env in envs.items():
                bound_tools = env_tools[env_id]
                schema_pairs.append((i, env_id))
                schema_prompts.append(env.build_schema_prompt(bound_tools, ctx))

        if schema_prompts:
            schema_responses = self._inference_engine.infer(
                schema_prompts, inference_config=self._inference_config
            )
            # Track which (pair_index) still need retries
            failed_indices: list[int] = []
            for j, ((i, env_id), response) in enumerate(
                zip(schema_pairs, schema_responses)
            ):
                env = sample_envs[i][env_id]
                if not env.apply_schema(response):
                    failed_indices.append(j)

            for _ in range(_MAX_STATE_UPDATE_RETRIES):
                if not failed_indices:
                    break
                retry_prompts = [
                    sample_envs[schema_pairs[j][0]][
                        schema_pairs[j][1]
                    ].build_schema_prompt(
                        env_tools[schema_pairs[j][1]],
                        scenario_contexts[schema_pairs[j][0]],
                    )
                    for j in failed_indices
                ]
                retry_responses = self._inference_engine.infer(
                    retry_prompts, inference_config=self._inference_config
                )
                still_failed: list[int] = []
                for j, response in zip(failed_indices, retry_responses):
                    i, env_id = schema_pairs[j]
                    env = sample_envs[i][env_id]
                    if not env.apply_schema(response):
                        still_failed.append(j)
                failed_indices = still_failed

            # Fallback: any still-failed schemas get permissive type
            for j in failed_indices:
                i, env_id = schema_pairs[j]
                logger.warning(
                    f"Failed to generate valid schema for env '{env_id}' "
                    f"sample {i} after {_MAX_STATE_UPDATE_RETRIES} retries. "
                    f"Using permissive schema."
                )
                sample_envs[i][env_id].set_schema({"type": "object"})

            # Ensure any env that never got a schema attempt also has a fallback
            for envs in sample_envs:
                for env_id, env in envs.items():
                    if env._state_schema is None:
                        env.set_schema({"type": "object"})
        state_pairs: list[tuple[int, str]] = []
        state_prompts: list = []
        for i, envs in enumerate(sample_envs):
            ctx = scenario_contexts[i]
            for env_id, env in envs.items():
                state_pairs.append((i, env_id))
                state_prompts.append(env.build_initial_state_prompt(ctx))

        if state_prompts:
            state_responses = self._inference_engine.infer(
                state_prompts, inference_config=self._inference_config
            )
            failed_indices: list[int] = []
            for j, ((i, env_id), response) in enumerate(
                zip(state_pairs, state_responses)
            ):
                env = sample_envs[i][env_id]
                if not env.apply_initial_state(response):
                    failed_indices.append(j)

            for _ in range(_MAX_STATE_UPDATE_RETRIES):
                if not failed_indices:
                    break
                retry_prompts = [
                    sample_envs[state_pairs[j][0]][
                        state_pairs[j][1]
                    ].build_initial_state_prompt(
                        scenario_contexts[state_pairs[j][0]]
                    )
                    for j in failed_indices
                ]
                retry_responses = self._inference_engine.infer(
                    retry_prompts, inference_config=self._inference_config
                )
                still_failed: list[int] = []
                for j, response in zip(failed_indices, retry_responses):
                    i, env_id = state_pairs[j]
                    env = sample_envs[i][env_id]
                    if not env.apply_initial_state(response):
                        still_failed.append(j)
                failed_indices = still_failed

            # Fallback for exhausted retries
            for j in failed_indices:
                i, env_id = state_pairs[j]
                env = sample_envs[i][env_id]
                if env._last_parsed_state is not None:
                    env.set_state(env._last_parsed_state, validate=False)
                    logger.warning(
                        f"Using parsed-but-schema-invalid initial state for "
                        f"env '{env_id}' (sample {i})."
                    )
                else:
                    env.set_state({})
                    logger.warning(
                        f"Initial state generation failed completely for "
                        f"env '{env_id}' (sample {i}). Using empty state."
                    )

        # Kill samples where ALL environments have completely empty state
        # (i.e., state generation failed entirely and fell back to {}).
        # Samples with schema-invalid but parseable state are kept — they
        # have enough data for the environment LLM to work with.
        result: list[dict[str, GeneratedToolEnvironment] | None] = []
        for envs in sample_envs:
            if not envs:
                result.append(None)
                continue
            all_empty = all(not env.state for env in envs.values())
            if all_empty:
                logger.warning(
                    "Dropping sample: all environments have empty state "
                    "after exhausting retries. This sample cannot produce "
                    "tool calls."
                )
                result.append(None)
            else:
                result.append(envs)
        return result

    def _summarize_envs(self, envs: dict[str, GeneratedToolEnvironment]) -> str:
        """Concatenate state summaries from all environments."""
        parts = []
        for env_id, env in envs.items():
            parts.append(f'Environment "{env_id}":\n{env.summarize_state()}')
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
        env_summaries = None
        if has_envs:
            sample_envs = self._init_sample_environments(samples, multiturn_attributes)
            env_summaries = [
                self._summarize_envs(envs) if envs else None for envs in sample_envs
            ]
        samples = self._plan_samples(
            samples, multiturn_attributes, env_summaries=env_summaries
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
        env_summaries: list[str | None] | None = None,
    ) -> list[dict]:
        """Plan the conversation samples with retry logic for failed parses.

        Args:
            samples: The conversation samples to plan.
            multiturn_attributes: The multi-turn attribute defining conversation rules.
            max_retries: Maximum number of retry attempts for failed plan parsing.
            env_summaries: Optional list of environment summaries for each sample.

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
                    env_summary=env_summaries[i] if env_summaries else None,
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
                "\n\nYou have access to the following tools. Use them to look up "
                "information and perform actions — do not guess or fabricate data.\n\n"
                f"Tools:\n{tool_catalog}\n\n"
                "To use a tool, output:\n"
                '<tool_call>{"name": "ToolName", "arguments": '
                '{"param": "value"}}</tool_call>\n\n'
                "After receiving a tool result, you may call another tool "
                "or respond to the user.\n\n"
                "IMPORTANT RULES:\n"
                "1. NEVER claim you performed an action without using the "
                "<tool_call> tag. Every action must go through a tool call.\n"
                "2. Base ALL responses on actual tool results. If tool results "
                "contradict your expectations, trust the tool results.\n"
                "3. Do NOT fabricate data, statistics, or results. Only reference "
                "information returned by tools.\n"
                "4. When using a tool, output the <tool_call> tag clearly.\n"
                "5. CRITICAL: You MUST use <tool_call> tags for EVERY database "
                "operation or tool invocation. Do NOT narrate or describe running "
                "queries in prose — actually call the tool using the tag. "
                "Do NOT fabricate query results. Every data point must come from "
                "a real tool result.\n"
                "6. If you want to look something up or run a query, you MUST "
                "output a <tool_call> tag. A response that describes tool results "
                "without a preceding <tool_call> tag is INVALID.\n"
                "7. Each tool operation MUST have its own <tool_call> tag. "
                "You may call multiple tools in a single message, but each "
                "must be a separate <tool_call> invocation — never narrate "
                "tool operations in prose."
            )
            formatted_content += tool_section

            return Message(role=Role.SYSTEM, content=formatted_content)
        return Message(
            role=Role.SYSTEM,
            content=formatted_content,
        )

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
        env_summary: str | None = None,
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

        if env_summary:
            base_prompt += (
                f"\nThe environment currently contains:\n{env_summary}\n\n"
                "Your plan MUST work with this data. Do not reference tables, "
                "fields, or entities that are not present in the environment.\n"
            )

        base_prompt += (
            "\nOutput ONLY the JSON array wrapped in ```json code fences. "
            "No other text."
        )

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
        inference_results = self._inference_engine.infer(
            planners,
            inference_config=self._inference_config,
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

        # Full history includes tool call tags and results — used for
        # assistant prompts so the LLM sees in-context tool-call examples.
        full_histories: list[list[Message]] = [[] for _ in samples]
        # Conversational history omits tool mechanics — used for user
        # prompts so the user LLM only sees natural language exchanges.
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

                    target = samples[i]["target_turns"]
                    turn_info = (
                        f"You are generating turn {current_turn} of {target} "
                        f"as the {role.value.upper()}.\n\n"
                    )
                    if turn_instruction:
                        turn_info += f"For this turn: {turn_instruction}\n\n"
                    if is_tool_turn and turn_call_counts[i] >= max_tool_calls:
                        turn_info += (
                            "You have reached the maximum number of tool calls "
                            "for this turn. Respond directly to the user based "
                            "on the information gathered so far."
                        )
                    else:
                        turn_info += (
                            "Generate ONLY your response for this turn. "
                            "Stay in character."
                        )

                    history = (
                        full_histories[i]
                        if is_tool_turn
                        else conversational_histories[i]
                    )
                    msgs = (
                        [persona_msg]
                        + history
                        + [Message(role=Role.USER, content=turn_info)]
                        + turn_tool_msgs[i]
                    )
                    prompts.append(Conversation(messages=msgs))

                texts = self._extract_response(
                    self._inference_engine.infer(
                        prompts, inference_config=self._inference_config
                    )
                )
                if len(texts) != len(active):
                    raise RuntimeError(
                        f"Inference returned {len(texts)} results "
                        f"for {len(active)} prompts"
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
                        if turn_call_counts.get(idx, 0) < max_tool_calls:
                            tc_result = tool_executor.parse_and_validate_tool_call(text)

                    if isinstance(tc_result, ToolCallError):
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
                                conversational_histories[idx].append(msg)
                        full_histories[idx].append(Message(role=role, content=text))
                        conversational_histories[idx].append(
                            Message(role=role, content=text)
                        )
                        if tool_executor:
                            content = ToolExecutor.strip_tool_tags(text)
                            content = ToolExecutor.strip_bare_tool_json(content)
                            output_messages[idx].append(
                                {"role": role.value, "content": content}
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
                        self._inference_engine.infer(
                            gen_prompts, inference_config=self._inference_config
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
                            self._inference_engine.infer(
                                retry_prompts,
                                inference_config=self._inference_config,
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
        # Preserve the full raw text (including <tool_call> tags) in the
        # agent-facing history so the LLM sees its own prior tool call format
        # as in-context examples. Without this, the LLM "forgets" the
        # <tool_call> tag convention after enough context accumulates.
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
        env_responses = self._inference_engine.infer(
            env_result_prompts,
            inference_config=self._inference_config,
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

        # Retry failed results with an explicit retry hint
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
            retry_responses = self._inference_engine.infer(
                retry_prompts,
                inference_config=self._inference_config,
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

        # Collect valid results for state updates
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
            self._apply_state_updates_batched(
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

    def _apply_state_updates_batched(
        self,
        env_items: list[
            tuple[int, str, dict, str, GeneratedToolEnvironment, ToolAttribute]
        ],
        env_results: list[str],
        state_update_indices: list[int],
        state_update_prompts: list[Conversation],
    ) -> None:
        """Apply state updates with batched retries on failure."""
        update_responses = self._inference_engine.infer(
            state_update_prompts,
            inference_config=self._inference_config,
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
            retry_responses = self._inference_engine.infer(
                retry_prompts,
                inference_config=self._inference_config,
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
