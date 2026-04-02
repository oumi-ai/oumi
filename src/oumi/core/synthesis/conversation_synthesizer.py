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
)
from oumi.core.synthesis.attribute_formatter import AttributeFormatter
from oumi.core.synthesis.environment import (
    _MAX_RESULT_RETRIES,
    GeneratedToolEnvironment,
    init_sample_environments,
    is_env_tool_missing_env,
    process_env_tool_calls,
    resolve_env_tool,
    serialize_env_states,
)
from oumi.core.synthesis.tool_executor import (
    ToolCallError,
    ToolCallParsed,
    ToolExecutor,
    clean_json_output,
    is_valid_json,
)
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.utils.logging import logger
from oumi.utils.str_utils import extract_json


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
            sample_envs = init_sample_environments(
                samples,
                tools,
                self._env_configs,
                self._formatter,
                self._inference_engine,
                self._inference_config,
            )
            env_states = [
                serialize_env_states(envs) if envs else None for envs in sample_envs
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
                validation_error = (
                    self._validate_parsed_turn_plans(parsed)
                    if parsed is not None
                    else "plan was not valid JSON"
                )

                if parsed is not None and validation_error is None:
                    augmented_sample["conversation_plan"] = plan
                    augmented_sample["parsed_turn_plans"] = parsed
                else:
                    failed_indices.append(idx)
                    if attempt < max_retries:
                        logger.warning(
                            f"Plan generation failed for sample {idx}: "
                            f"{validation_error or 'unknown validation error'}, "
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

    @staticmethod
    def _validate_parsed_turn_plans(
        parsed_turn_plans: list[str],
    ) -> str | None:
        """Return a validation error when planner instructions are unsafe."""
        for turn_idx, instruction in enumerate(parsed_turn_plans, start=1):
            cleaned = instruction.strip()
            if not cleaned:
                return f"missing instruction for turn {turn_idx}"
        return None

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
            return (
                role == "assistant"
                and isinstance(content, str)
                and bool(content.strip())
            )
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

    def _build_role_context(
        self, sample: dict, multiturn_attribute: MultiTurnAttribute
    ) -> str:
        """Build formatted role context for the planner.

        Formats the persona strings for each role exactly as the planner
        should see them.
        """
        parts = []
        for role, persona in multiturn_attribute.role_instruction_messages.items():
            formatted = self._formatter.format(
                sample, persona, missing_values_allowed=False
            )
            parts.append(f"[{role.value.upper()}]\n{formatted}")
        return "\n\n".join(parts)

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
            "IMPORTANT: Output your plan as a JSON array only. Do not use "
            "markdown fences or surrounding prose. Each element must have: "
            "turn (number) and instruction (string).\n"
            "Your instructions MUST be specific to the role context provided. "
            "Each turn's instruction should reflect what that specific role "
            "would do at that point in the conversation."
        )

        if has_tools:
            system_prompt += (
                "\n\nThe assistant has access to tools and will use them/ naturally "
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
            example_response = json.dumps(
                [
                    {
                        "turn": 1,
                        "instruction": (
                            "Explain that you want to return your order and "
                            "provide the reason"
                        ),
                    },
                    {
                        "turn": 2,
                        "instruction": (
                            "Verify the order details and check whether a return "
                            "is eligible based on the return policy"
                        ),
                    },
                    {
                        "turn": 3,
                        "instruction": ("Confirm you want to proceed with the return"),
                    },
                    {
                        "turn": 4,
                        "instruction": (
                            "Process the return and provide confirmation with "
                            "next steps"
                        ),
                    },
                ],
                indent=2,
            )
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
            example_response = json.dumps(
                [
                    {
                        "turn": 1,
                        "instruction": (
                            "Greet support and explain the issue with the order"
                        ),
                    },
                    {
                        "turn": 2,
                        "instruction": (
                            "Acknowledge the issue and ask for order details"
                        ),
                    },
                    {
                        "turn": 3,
                        "instruction": (
                            "Provide order number and describe the problem further"
                        ),
                    },
                    {
                        "turn": 4,
                        "instruction": ("Confirm the issue and offer a resolution"),
                    },
                ],
                indent=2,
            )

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
                f"\nEnvironment state (ground truth):\n{env_state}\n\n"
                "This is the actual state the tools operate on. Use the real "
                "entity identifiers (ids, names, values) from this state in "
                "your plan so that tool calls will succeed. The USER should "
                "refer to entities naturally but the ASSISTANT must use exact "
                "ids from this state when calling tools.\n"
            )

        base_prompt += (
            "\nOutput ONLY the JSON array. No markdown fences. No surrounding prose."
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

        # contains raw tool calls for agent
        full_histories: list[list[Message]] = [[] for _ in samples]
        # only contains conversational messages, for the user
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
                        turn_info = ToolExecutor.build_tool_turn_info(
                            current_turn=current_turn,
                            target_turns=samples[i]["target_turns"],
                            turn_instruction=turn_instruction,
                            max_calls_reached=turn_call_counts[i] >= max_tool_calls,
                        )
                    else:
                        turn_info = ToolExecutor.build_prose_turn_info(
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
                        ToolExecutor.build_tool_few_shot(tools)
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
                        turn_call_counts[idx] += 1
                        tool_call_counts[idx] += 1
                        call_id = f"call_{tool_call_counts[idx]:03d}"
                        error_tool_call = {
                            "name": tc_result.tool_name or "unknown",
                            "arguments": {},
                        }
                        ToolExecutor.record_tool_result(
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

                    env, tool_obj = resolve_env_tool(
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
                    elif is_env_tool_missing_env(tool_executor, tool_call):
                        error_msg = (
                            f"Error: environment not available for tool "
                            f"'{tool_call['name']}'"
                        )
                        ToolExecutor.record_tool_result(
                            idx,
                            text,
                            tool_call,
                            call_id,
                            error_msg,
                            turn_tool_msgs,
                            output_messages,
                        )
                        still_active.append(idx)
                    else:
                        result = tool_executor.resolve_output(
                            tool_call, deterministic_selections[idx]
                        )
                        if result is not None:
                            ToolExecutor.record_tool_result(
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
                    env_activated = process_env_tool_calls(
                        env_items,
                        env_result_prompts,
                        turn_tool_msgs,
                        output_messages,
                        inference_engine=self._inference_engine,
                        inference_config=self._inference_config,
                        record_fn=ToolExecutor.record_tool_result,
                    )
                    still_active.extend(env_activated)
                if gen_prompts:
                    sim_texts = self._extract_response(
                        self._infer_exact(
                            gen_prompts,
                            context=(f"Turn {current_turn} generated-tool simulation"),
                        )
                    )
                    # Identify which results need retries
                    gen_failed: list[int] = []
                    gen_results: list[str | None] = [None] * len(gen_items)
                    for j, sim in enumerate(sim_texts):
                        cleaned = clean_json_output(sim)
                        if is_valid_json(cleaned):
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
                            cleaned = clean_json_output(sim)
                            if is_valid_json(cleaned):
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
                        gen_results[j] = clean_json_output(sim_texts[j])

                    for j, (idx, raw, tc, cid) in enumerate(gen_items):
                        ToolExecutor.record_tool_result(
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
