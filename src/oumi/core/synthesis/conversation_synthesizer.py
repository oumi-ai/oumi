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
    _MAX_STATE_UPDATE_RETRIES,
    GeneratedToolEnvironment,
)
from oumi.core.synthesis.tool_executor import ToolExecutor
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.utils.logging import logger
from oumi.utils.str_utils import extract_json


def _clean_json_output(text: str) -> str:
    """Strip markdown fences and extract clean JSON from LLM-generated tool output."""
    parsed = extract_json(text, expected_type=None)
    if parsed is not None:
        return json.dumps(parsed)
    return text


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

    def _create_environments(
        self, multiturn_attribute: MultiTurnAttribute
    ) -> dict[str, GeneratedToolEnvironment] | None:
        """Create and initialize GeneratedToolEnvironment instances.

        Called once per multiturn synthesis. Returns None if no
        environment-bound tools are used.
        """
        tools = self._get_tools_for_multiturn(multiturn_attribute)
        env_tools: dict[str, list[ToolAttribute]] = {}
        for tool in tools:
            if tool.environment:
                env_tools.setdefault(tool.environment, []).append(tool)

        if not env_tools:
            return None

        scenario_parts = []
        for role, instruction in multiturn_attribute.role_instruction_messages.items():
            scenario_parts.append(f"{role.value}: {instruction}")
        scenario_context = "\n".join(scenario_parts) if scenario_parts else None

        environments: dict[str, GeneratedToolEnvironment] = {}
        for env_id, bound_tools in env_tools.items():
            config = self._env_configs.get(env_id)
            if not config:
                logger.warning(f"Environment config not found for '{env_id}'")
                continue
            env = GeneratedToolEnvironment(
                config=config,
                inference_engine=self._inference_engine,
                inference_config=self._inference_config,
            )
            env.initialize(bound_tools, scenario_context=scenario_context)
            environments[env_id] = env

        return environments if environments else None

    def _init_sample_environments(
        self,
        samples: list[dict],
        multiturn_attribute: MultiTurnAttribute,
    ) -> list[dict[str, GeneratedToolEnvironment] | None]:
        """Create and initialize per-sample environments with batched LLM calls.

        Unlike _create_environments() which creates one shared set of environments,
        this method creates fresh GeneratedToolEnvironment instances for each sample
        and initializes them using two batched inference calls across all samples:
        one for schema generation and one for initial state generation.

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

        # Build scenario context template from role instruction messages
        scenario_parts = []
        for role, instruction in multiturn_attribute.role_instruction_messages.items():
            scenario_parts.append(f"{role.value}: {instruction}")
        scenario_template = "\n".join(scenario_parts) if scenario_parts else None

        # Create fresh environments for each sample
        sample_envs: list[dict[str, GeneratedToolEnvironment]] = []
        for _ in samples:
            envs: dict[str, GeneratedToolEnvironment] = {}
            for env_id in env_tools:
                config = self._env_configs.get(env_id)
                if not config:
                    logger.warning(f"Environment config not found for '{env_id}'")
                    continue
                envs[env_id] = GeneratedToolEnvironment(
                    config=config,
                    inference_engine=self._inference_engine,
                    inference_config=self._inference_config,
                )
            sample_envs.append(envs)

        # Resolve per-sample scenario contexts
        def _resolve_scenario(sample: dict) -> str | None:
            if scenario_template is None:
                return None
            return self._formatter.format(
                sample, scenario_template, missing_values_allowed=True
            )

        scenario_contexts = [_resolve_scenario(s) for s in samples]

        # Batch 1 — Schema generation
        # Build prompts for all (sample, env) pairs that need a schema
        schema_pairs: list[tuple[int, str]] = []  # (sample_idx, env_id)
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
            # Apply responses; collect failures for retry
            retry_schema_pairs: list[tuple[int, str]] = []
            retry_schema_prompts: list = []
            for (i, env_id), response in zip(schema_pairs, schema_responses):
                env = sample_envs[i][env_id]
                if not env.apply_schema(response):
                    for _ in range(_MAX_STATE_UPDATE_RETRIES):
                        retry_schema_pairs.append((i, env_id))
                        retry_schema_prompts.append(
                            env.build_schema_prompt(env_tools[env_id], scenario_contexts[i])
                        )
                        break  # one retry prompt per failure; batched below

            if retry_schema_prompts:
                retry_responses = self._inference_engine.infer(
                    retry_schema_prompts, inference_config=self._inference_config
                )
                for (i, env_id), response in zip(retry_schema_pairs, retry_responses):
                    env = sample_envs[i][env_id]
                    if not env.apply_schema(response):
                        logger.warning(
                            f"Failed to generate valid schema for env '{env_id}' "
                            f"sample {i}. Using permissive schema."
                        )
                        env._state_schema = {"type": "object"}

            # Ensure every env has a schema (fallback for any that still lack one)
            for envs in sample_envs:
                for env_id, env in envs.items():
                    if env._state_schema is None:
                        env._state_schema = {"type": "object"}

        # Batch 2 — Initial state generation (only after ALL schemas are set)
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
            retry_state_pairs: list[tuple[int, str]] = []
            retry_state_prompts: list = []
            for (i, env_id), response in zip(state_pairs, state_responses):
                env = sample_envs[i][env_id]
                if not env.apply_initial_state(response):
                    for _ in range(_MAX_STATE_UPDATE_RETRIES):
                        retry_state_pairs.append((i, env_id))
                        retry_state_prompts.append(
                            env.build_initial_state_prompt(scenario_contexts[i])
                        )
                        break

            if retry_state_prompts:
                retry_responses = self._inference_engine.infer(
                    retry_state_prompts, inference_config=self._inference_config
                )
                for (i, env_id), response in zip(retry_state_pairs, retry_responses):
                    env = sample_envs[i][env_id]
                    if not env.apply_initial_state(response):
                        logger.warning(
                            f"Initial state generation failed for env '{env_id}' "
                            f"sample {i}. Using empty dict."
                        )
                        env._state = {}

        return [envs if envs else None for envs in sample_envs]

    def _summarize_envs(
        self, envs: dict[str, GeneratedToolEnvironment]
    ) -> str:
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

        samples = self._plan_samples(samples, multiturn_attributes)
        conversations, tool_data = self._synthesize_all_samples(
            samples, multiturn_attributes
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
                "IMPORTANT: Never claim you performed an action without using "
                "the <tool_call> tag. Every action must go through a tool call."
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
        The prompt instructs the model to output JSON wrapped in code fences.
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
    ) -> tuple[list[Conversation], dict | None]:
        """Synthesize multi-turn conversations for all samples with batched inference.

        For ASSISTANT turns with tools, the agent loops: generate → parse for
        tool calls → resolve output → generate again, until a natural language
        response. Non-tool turns run through the same loop exactly once.

        Returns:
            (conversations, tool_data) where tool_data is None when no tools
            are configured.
        """
        if not samples:
            return [], None

        tools = self._get_tools_for_multiturn(multiturn_attribute)

        template_envs = (
            self._create_environments(multiturn_attribute) if tools else None
        )

        sample_envs: list[dict[str, GeneratedToolEnvironment] | None] = []
        for _ in samples:
            if template_envs:
                sample_envs.append(
                    {eid: copy.deepcopy(env) for eid, env in template_envs.items()}
                )
            else:
                sample_envs.append(None)

        tool_executor = ToolExecutor(tools) if tools else None
        deterministic_selections = (
            [tool_executor.sample_deterministic_outputs(tools) for _ in samples]
            if tool_executor
            else []
        )

        histories: list[list[Message]] = [[] for _ in samples]
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

                    msgs = (
                        [persona_msg]
                        + histories[i]
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
                # Collect ENVIRONMENT tool calls for batched inference
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
                    tool_call = None
                    if is_tool_turn and tool_executor:
                        if turn_call_counts.get(idx, 0) < max_tool_calls:
                            tool_call = tool_executor.parse_tool_call(text)

                    if tool_call is None:
                        histories[idx].extend(turn_tool_msgs[idx])
                        histories[idx].append(Message(role=role, content=text))
                        if tool_executor:
                            content = (
                                ToolExecutor.strip_tool_tags(text)
                                if is_tool_turn
                                else text
                            )
                            output_messages[idx].append(
                                {"role": role.value, "content": content}
                            )
                    else:
                        assert tool_executor is not None
                        turn_call_counts[idx] += 1
                        tool_call_counts[idx] += 1
                        call_id = f"call_{tool_call_counts[idx]:03d}"

                        # Check if this is an ENVIRONMENT tool
                        env = None
                        tool_obj = None
                        idx_envs = sample_envs[idx]
                        if idx_envs is not None:
                            tool_obj = tool_executor.get_tool_by_name(tool_call["name"])
                            if tool_obj and tool_obj.environment:
                                env = idx_envs.get(tool_obj.environment)

                        if env and tool_obj:
                            # Queue for batched env processing
                            env_items.append(
                                (idx, text, tool_call, call_id, env, tool_obj)
                            )
                            env_result_prompts.append(
                                env.build_result_prompt(
                                    tool_obj,
                                    arguments=tool_call["arguments"],
                                )
                            )
                            continue

                        # Guard: ENVIRONMENT tools must not fall through
                        # to DETERMINISTIC/GENERATED paths
                        env_tool = tool_executor.get_tool_by_name(tool_call["name"])
                        if (
                            env_tool
                            and env_tool.output_strategy
                            == ToolOutputStrategy.ENVIRONMENT
                        ):
                            logger.warning(
                                f"Environment not found for tool "
                                f"'{tool_call['name']}', skipping."
                            )
                            still_active.append(idx)
                            continue

                        result = tool_executor.resolve_output(
                            tool_call, deterministic_selections[idx]
                        )
                        if result is not None:
                            turn_tool_msgs[idx].append(
                                Message(role=Role.ASSISTANT, content=text)
                            )
                            turn_tool_msgs[idx].append(
                                Message(
                                    role=Role.USER,
                                    content=(
                                        f"[Tool result from "
                                        f"{tool_call['name']}]: "
                                        f"{result}"
                                    ),
                                )
                            )
                            output_messages[idx].append(
                                ToolExecutor.format_tool_call_message(
                                    tool_call, call_id
                                )
                            )
                            output_messages[idx].append(
                                ToolExecutor.format_tool_result_message(
                                    call_id, result, tool_call["name"]
                                )
                            )
                            still_active.append(idx)
                        else:
                            ctx = histories[idx] + turn_tool_msgs[idx]
                            gen_items.append((idx, text, tool_call, call_id))
                            gen_prompts.append(
                                tool_executor.build_generated_simulator_prompt(
                                    tool_call, ctx
                                )
                            )

                # Batched ENVIRONMENT tool result generation
                if env_result_prompts:
                    env_responses = self._inference_engine.infer(
                        env_result_prompts,
                        inference_config=self._inference_config,
                    )
                    # Collect state update prompts for non-read-only tools
                    state_update_indices: list[int] = []
                    state_update_prompts: list[Conversation] = []
                    env_results: list[str] = []

                    for i, (
                        (idx, text, tool_call, call_id, env, tool_obj),
                        response,
                    ) in enumerate(zip(env_items, env_responses)):
                        result = env.apply_result(response)
                        env_results.append(result)

                        if not tool_obj.read_only:
                            state_update_indices.append(i)
                            state_update_prompts.append(
                                env.build_state_update_prompt(
                                    tool_obj,
                                    arguments=tool_call["arguments"],
                                    result=result,
                                )
                            )

                    # Batched state updates
                    if state_update_prompts:
                        update_responses = self._inference_engine.infer(
                            state_update_prompts,
                            inference_config=self._inference_config,
                        )
                        # Track failures for retry
                        failed: list[tuple[int, int]] = []
                        for ui, response in zip(state_update_indices, update_responses):
                            env = env_items[ui][4]
                            if not env.apply_state_update(response):
                                failed.append((ui, state_update_indices.index(ui)))

                        # Retry failed state updates
                        for _ in range(1, _MAX_STATE_UPDATE_RETRIES):
                            if not failed:
                                break
                            retry_prompts = []
                            for ui, _ in failed:
                                (
                                    idx,
                                    text,
                                    tool_call,
                                    call_id,
                                    env,
                                    tool_obj,
                                ) = env_items[ui]
                                retry_prompts.append(
                                    env.build_state_update_prompt(
                                        tool_obj,
                                        arguments=tool_call["arguments"],
                                        result=env_results[ui],
                                        retry=True,
                                    )
                                )
                            retry_responses = self._inference_engine.infer(
                                retry_prompts,
                                inference_config=self._inference_config,
                            )
                            still_failed = []
                            for (ui, _), response in zip(failed, retry_responses):
                                env = env_items[ui][4]
                                if not env.apply_state_update(response):
                                    still_failed.append((ui, 0))
                            failed = still_failed

                        if failed:
                            for ui, _ in failed:
                                env = env_items[ui][4]
                                tool_obj = env_items[ui][5]
                                logger.warning(
                                    f"State update failed after "
                                    f"{_MAX_STATE_UPDATE_RETRIES} retries "
                                    f"for tool '{tool_obj.name}'. "
                                    f"Keeping previous state."
                                )

                    # Now append all env results to histories
                    for i, (
                        idx,
                        text,
                        tool_call,
                        call_id,
                        env,
                        tool_obj,
                    ) in enumerate(env_items):
                        result = env_results[i]
                        turn_tool_msgs[idx].append(
                            Message(role=Role.ASSISTANT, content=text)
                        )
                        turn_tool_msgs[idx].append(
                            Message(
                                role=Role.USER,
                                content=(
                                    f"[Tool result from {tool_call['name']}]: {result}"
                                ),
                            )
                        )
                        output_messages[idx].append(
                            ToolExecutor.format_tool_call_message(tool_call, call_id)
                        )
                        tool_result_msg = ToolExecutor.format_tool_result_message(
                            call_id, result, tool_call["name"]
                        )
                        tool_result_msg["_environment_state"] = copy.deepcopy(env.state)
                        output_messages[idx].append(tool_result_msg)
                        still_active.append(idx)

                # Batched GENERATED tool simulation
                if gen_prompts:
                    sim_texts = self._extract_response(
                        self._inference_engine.infer(
                            gen_prompts, inference_config=self._inference_config
                        )
                    )
                    for (idx, raw, tc, cid), sim in zip(gen_items, sim_texts):
                        sim = _clean_json_output(sim)
                        turn_tool_msgs[idx].append(
                            Message(role=Role.ASSISTANT, content=raw)
                        )
                        turn_tool_msgs[idx].append(
                            Message(
                                role=Role.USER,
                                content=f"[Tool result from {tc['name']}]: {sim}",
                            )
                        )
                        output_messages[idx].append(
                            ToolExecutor.format_tool_call_message(tc, cid)
                        )
                        output_messages[idx].append(
                            ToolExecutor.format_tool_result_message(
                                cid, sim, tc["name"]
                            )
                        )
                        still_active.append(idx)

                if not is_tool_turn:
                    break
                active = still_active

        conversations: list[Conversation] = []
        for sample, history in zip(samples, histories):
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
