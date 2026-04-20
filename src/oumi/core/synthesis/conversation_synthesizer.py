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

import dataclasses
import json
import random
import re

from oumi.builders.inference_engines import build_inference_engine
from oumi.core.configs.environment_config import EnvironmentConfig
from oumi.core.configs.inference_config import InferenceConfig
from oumi.core.configs.inference_engine_type import InferenceEngineType
from oumi.core.configs.params.guided_decoding_params import GuidedDecodingParams
from oumi.core.configs.params.synthesis_params import (
    GeneralSynthesisParams,
    MultiTurnAttribute,
)
from oumi.core.synthesis.attribute_formatter import AttributeFormatter
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.environments import (
    GroundingFact,
    Tool,
    ToolArgumentError,
    ToolError,
)
from oumi.environments.base_tool import describe_grounding_default
from oumi.utils.logging import logger
from oumi.utils.str_utils import extract_json, repair_json_braces

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

_FORCED_FINALIZE_NUDGE = (
    "You have reached the tool-call limit. Do NOT emit any more "
    "<tool_call> blocks. Based on the information gathered so far, "
    "provide your final response to the user now."
)


def _strip_tool_call_blocks(text: str) -> str:
    return _TOOL_CALL_RE.sub("", text)


def _close_dangling_tool_call(text: str) -> str:
    """Re-append ``</tool_call>`` when stop-sequence inference stripped it."""
    opens = text.count("<tool_call>")
    closes = text.count("</tool_call>")
    if opens > closes:
        return text + "</tool_call>"
    return text


def _truncate_after_last_tool_call(text: str) -> str:
    """Return the text prefix up to and including the LAST ``</tool_call>``."""
    last_close = text.rfind("</tool_call>")
    if last_close == -1:
        return text
    return text[: last_close + len("</tool_call>")]


def _canonicalize_tool_call_bodies(text: str) -> str:
    """Replace each ``<tool_call>`` body with a canonical JSON re-serialization.

    Models occasionally emit malformed JSON inside tool calls — most commonly
    an extra trailing ``}`` (``}}}``) or a missing close when the stop
    sequence fires early. Without this pass, the malformation is persisted
    verbatim into the output dataset (it would only be repaired transiently
    during tool execution). Downstream consumers of the dataset would then
    need to re-implement the repair.

    For each ``<tool_call>...</tool_call>`` block, this function attempts to
    brace-repair the body and, on success, substitutes it with the compact
    ``json.dumps`` form. Bodies that cannot be repaired are left untouched;
    the tool-executor surfaces a structured error message for those.
    """

    def _replace(match: re.Match[str]) -> str:
        body = match.group(1).strip()
        repaired = repair_json_braces(body)
        if repaired is None:
            return match.group(0)
        try:
            parsed = json.loads(repaired)
        except json.JSONDecodeError:
            return match.group(0)
        return f"<tool_call>{json.dumps(parsed)}</tool_call>"

    return _TOOL_CALL_RE.sub(_replace, text)


def _tool_result_message(content: str) -> Message:
    """Wrap a tool output as a user-role message with a <tool_result> marker."""
    return Message(role=Role.USER, content=f"<tool_result>{content}</tool_result>")


def _tool_error_msg(error: str) -> Message:
    return _tool_result_message(json.dumps({"error": error}))


_PLANNER_JSON_SCHEMA: dict = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "turn": {"type": "integer", "minimum": 1},
            "instruction": {"type": "string"},
        },
        "required": ["turn", "instruction"],
        "additionalProperties": False,
    },
}


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
        environment_config: EnvironmentConfig | None = None,
    ):
        """Initialize the synthesizer."""
        self._params = params
        self._environment_config = environment_config
        self._formatter = AttributeFormatter(params)

        self._inference_engine = build_inference_engine(
            engine_type=inference_config.engine or InferenceEngineType.NATIVE,
            model_params=inference_config.model,
            remote_params=inference_config.remote_params,
        )
        self._inference_config = inference_config
        self._default_turn_order = [Role.USER, Role.ASSISTANT]

    def _resolve_available_tools(
        self, multiturn_attribute: MultiTurnAttribute
    ) -> list[Tool]:
        """Resolve tools for a multiturn attribute from selected environments."""
        if self._environment_config is None:
            return []
        return self._environment_config.resolve_tools(
            environment_ids=multiturn_attribute.available_environments or None,
            tool_ids=multiturn_attribute.available_tools or None,
        )

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

    def _validate_tool_configuration(
        self, multiturn_attribute: MultiTurnAttribute
    ) -> None:
        """Validate that tool declarations have a backing environment_config."""
        declares_tools = bool(multiturn_attribute.available_tools) or bool(
            multiturn_attribute.available_environments
        )
        if declares_tools and self._environment_config is None:
            raise ValueError(
                f"MultiTurnAttribute '{multiturn_attribute.id}' declares "
                f"available_tools/available_environments but no environment_config "
                f"was provided to ConversationSynthesizer."
            )
        for role, persona in multiturn_attribute.role_instruction_messages.items():
            if not isinstance(persona, str):
                continue
            if "{grounding_facts}" in persona and role in (
                Role.USER,
                Role.ASSISTANT,
            ):
                logger.warning(
                    "MultiTurnAttribute '%s' references {grounding_facts} in "
                    "the %s persona template. grounding is planner-only; "
                    "placing {grounding_facts} in user/assistant templates "
                    "defeats its purpose and may leak env state to roles that "
                    "should not see it.",
                    multiturn_attribute.id,
                    role.value,
                )

    def _format_tool_block(self, multiturn_attribute: MultiTurnAttribute) -> str:
        """Build a tool-usage instruction block for the assistant persona."""
        available_tools = self._resolve_available_tools(multiturn_attribute)
        if not available_tools:
            return ""

        lines = [
            "You have access to the following tools.",
            "",
            "When you need information from a tool, emit EXACTLY ONE tool call",
            "in this format and then STOP — do not write any text after the",
            "closing </tool_call> tag:",
            "",
            "<tool_call>",
            '{"name": "<tool_name>", "arguments": {...}}',
            "</tool_call>",
            "",
            "Strict rules:",
            "- Emit only the tool call. Do NOT include prose, explanation, or",
            "  a fabricated answer after </tool_call>. The tool result will be",
            "  delivered in the next turn — only then should you respond.",
            "- Never invent fields like titles, names, IDs, or dates. State",
            "  only facts that came back from a tool.",
            "- If more than one tool call is needed, issue them one at a time",
            "  across successive turns, waiting for each result before deciding",
            "  the next call.",
            "- When you have all the information you need, reply with a plain",
            "  natural-language message (no <tool_call> block).",
            "",
            "Available tools:",
        ]
        for tool in available_tools:
            schema = tool.to_llm_schema()
            lines.append("<tool>")
            lines.append(json.dumps(schema, indent=2))
            lines.append("</tool>")
            lines.append("")
            if tool.output_schema is not None:
                lines.append(
                    "  Output schema: "
                    f"{json.dumps(tool.output_schema.to_dict(), sort_keys=True)}"
                )
            lines.append("")

        return "\n".join(lines).rstrip()

    def _format_role_persona(
        self,
        sample: dict,
        persona: str,
        role: Role,
        multiturn_attribute: MultiTurnAttribute | None = None,
    ) -> str:
        """Format a role persona and append assistant tool context when available."""
        formatted_content = self._formatter.format(
            sample,
            persona,
            missing_values_allowed=False,
        )
        if role == Role.ASSISTANT and multiturn_attribute is not None:
            tool_block = self._format_tool_block(multiturn_attribute)
            if tool_block:
                formatted_content = f"{formatted_content}\n\n{tool_block}"
        return formatted_content

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
        self._validate_tool_configuration(multiturn_attributes)

        logger.info(
            f"Synthesizing {len(samples)} conversations for "
            f"attribute '{multiturn_attributes.id}'"
        )
        available_tools = self._resolve_available_tools(multiturn_attributes)
        if available_tools:
            logger.debug(
                "Resolved tools for '%s': %s",
                multiturn_attributes.id,
                [tool.id for tool in available_tools],
            )

        self._attach_grounding_facts(samples, multiturn_attributes)
        samples = self._plan_samples(samples, multiturn_attributes)
        conversations = self._synthesize_all_samples(samples, multiturn_attributes)

        records: list[dict[str, dict | str] | None] = []
        plan_key = f"{multiturn_attributes.id}_plan"
        filtered_count = 0
        for sample, conversation in zip(samples, conversations):
            if self._has_empty_messages(conversation):
                filtered_count += 1
                records.append(None)
                continue
            record: dict[str, dict | str] = {
                multiturn_attributes.id: conversation.to_dict(),
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
        self,
        sample: dict,
        persona: str,
        role: Role,
        multiturn_attribute: MultiTurnAttribute | None = None,
    ) -> Message:
        """Format the persona for the sample.

        Args:
            sample: The sample dict containing all attributes.
            persona: The persona string to format.
            role: The role for this persona.
            multiturn_attribute: Optional multiturn config for assistant tool context.

        Returns:
            A Message with the formatted persona as a SYSTEM message.
        """
        formatted_content = self._format_role_persona(
            sample,
            persona,
            role,
            multiturn_attribute=multiturn_attribute,
        )
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
            formatted = self._format_role_persona(
                sample,
                persona,
                role,
                multiturn_attribute=multiturn_attribute,
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
        self, multiturn_attribute: MultiTurnAttribute, sample: dict
    ) -> Conversation:
        """Create the planner prompt template with role context and turn order."""
        role_context = self._build_role_context(sample, multiturn_attribute)
        turn_order = self._default_turn_order
        target_turns = sample["target_turns"]
        turn_order_str = self._build_turn_order_str(turn_order, target_turns)

        system_prompt = (
            "You are a conversation planner. Create conversation outlines "
            "that flow logically from start to finish.\n\n"
            "IMPORTANT: Output your plan as a raw JSON array. "
            "Do not use markdown or code fences. "
            "Each element must have: turn (number) and instruction (string).\n"
            "Your instructions MUST be specific to the role context provided. "
            "Each turn's instruction should reflect what that specific role "
            "would do at that point in the conversation."
        )

        example_request = (
            "Plan a 4-turn conversation.\n"
            "Turn order: Turn 1: USER, Turn 2: ASSISTANT, Turn 3: USER, "
            "Turn 4: ASSISTANT\n\n"
            "Role context:\n"
            "[USER]\n"
            "You are a customer who has an issue with a recent order.\n\n"
            "[ASSISTANT]\n"
            "You are a helpful support agent who resolves customer issues.\n\n"
            "Ground this plan in these specific entities:\n"
            '- order_id="ORD-4421", item="laptop stand", status="delayed"\n\n'
            "Additional instructions: Focus on resolving the order issue "
            "efficiently while maintaining a polite and helpful tone."
        )
        example_response = (
            "[\n"
            '  {"turn": 1, "instruction": "Greet support and explain that '
            'order ORD-4421 has not arrived"},\n'
            '  {"turn": 2, "instruction": "Acknowledge the issue and ask '
            'for details on order ORD-4421"},\n'
            '  {"turn": 3, "instruction": "Describe that the laptop stand '
            'from order ORD-4421 is late"},\n'
            '  {"turn": 4, "instruction": "Confirm the delay on order '
            'ORD-4421 and offer a resolution"}\n'
            "]"
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

        if role_context:
            base_prompt += f"\nRole context:\n{role_context}\n"

        grounding_facts = sample.get("grounding_facts") or []
        if grounding_facts:
            describer_env = next(
                (
                    env
                    for env in (
                        self._environment_config.environments
                        if self._environment_config
                        else []
                    )
                    if env.grounding is not None
                ),
                None,
            )
            if describer_env is not None:
                block = describer_env.describe_grounding(grounding_facts)
            else:
                block = describe_grounding_default(grounding_facts)
            base_prompt += (
                "\nGround this plan in these specific entities:\n"
                f"{block}\n"
                "Every turn instruction that references one of these entities "
                "MUST spell out the concrete identifier verbatim (e.g. write "
                "'book B007', not 'the book'). The user persona cannot see "
                "this list — only text you write into each instruction "
                "reaches them.\n"
            )

        if multiturn_attribute.conversation_planner:
            formatted_planner = self._formatter.format(
                sample,
                multiturn_attribute.conversation_planner,
                missing_values_allowed=False,
            )
            base_prompt += f"\nAdditional instructions: {formatted_planner}\n"

        base_prompt += "\nOutput ONLY the JSON array. No markdown. No other text."

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
            inference_config=self._planner_inference_config(),
        )

        return self._extract_response(inference_results)

    def _planner_inference_config(self) -> InferenceConfig:
        """Create an inference config for planner calls with JSON guided decoding."""
        return dataclasses.replace(
            self._inference_config,
            generation=dataclasses.replace(
                self._inference_config.generation,
                guided_decoding=GuidedDecodingParams(json=_PLANNER_JSON_SCHEMA),
            ),
        )

    def _synthesize_all_samples(
        self,
        samples: list[dict],
        multiturn_attribute: MultiTurnAttribute,
    ) -> list[Conversation]:
        """Synthesize multi-turn conversations for all samples with batched inference.

        Args:
            samples: List of sample dicts with runtime fields (target_turns,
                conversation_plan).
            multiturn_attribute: The multi-turn attribute defining conversation rules.

        Returns:
            List of Conversation objects, one per sample.
        """
        if not samples:
            return []

        histories: list[list[Message]] = [[] for _ in samples]
        user_histories: list[list[Message]] = [[] for _ in samples]
        tool_use = len(self._resolve_available_tools(multiturn_attribute)) > 0
        max_turns = max(sample["target_turns"] for sample in samples)

        turn_order = self._default_turn_order
        for turn_idx in range(max_turns):
            current_turn = turn_idx + 1
            role = turn_order[turn_idx % len(turn_order)]

            sample_indices = [
                i
                for i, sample in enumerate(samples)
                if turn_idx < sample["target_turns"]
            ]
            if not sample_indices:
                break

            if role == Role.ASSISTANT and tool_use:
                per_sample_msgs = self._run_assistant_turn(
                    samples=samples,
                    sample_indices=sample_indices,
                    histories=histories,
                    current_turn=current_turn,
                    multiturn_attribute=multiturn_attribute,
                )
                for i, msgs in zip(sample_indices, per_sample_msgs):
                    histories[i].extend(msgs)
                    user_histories[i].append(msgs[-1])
                continue

            history_source = (
                user_histories if (role == Role.USER and tool_use) else histories
            )
            prompts = [
                self._build_turn_prompt(
                    samples[i],
                    role,
                    history_source[i],
                    current_turn,
                    multiturn_attribute,
                )
                for i in sample_indices
            ]
            generated_texts = self._extract_response(
                self._inference_engine.infer(
                    prompts,
                    inference_config=self._inference_config,
                )
            )
            if len(generated_texts) != len(prompts):
                raise RuntimeError(
                    f"Inference engine returned {len(generated_texts)} results "
                    f"but {len(prompts)} prompts were submitted. "
                    f"This may indicate an inference engine error."
                )
            for i, text in zip(sample_indices, generated_texts):
                msg = Message(role=role, content=text)
                histories[i].append(msg)
                user_histories[i].append(msg)

        conversations: list[Conversation] = []
        for sample, history in zip(samples, histories):
            output_messages: list[Message] = []
            output_message = self._format_output_system_message(
                sample, multiturn_attribute.output_system_prompt
            )
            if output_message:
                output_messages.append(output_message)
            output_messages.extend(history)
            conversations.append(Conversation(messages=output_messages))

        return conversations

    def _build_turn_prompt(
        self,
        sample: dict,
        role: Role,
        history: list[Message],
        current_turn: int,
        multiturn_attribute: MultiTurnAttribute,
        trailing: list[Message] | None = None,
    ) -> Conversation:
        """Build the inference prompt for one sample at one turn."""
        target_turns = sample["target_turns"]
        parsed_turn_plans = sample.get("parsed_turn_plans", [])
        turn_idx = current_turn - 1
        turn_instruction = (
            parsed_turn_plans[turn_idx] if turn_idx < len(parsed_turn_plans) else ""
        )

        persona_msg = self._format_persona(
            {**sample, "current_turn": current_turn},
            multiturn_attribute.role_instruction_messages[role],
            role,
            multiturn_attribute=multiturn_attribute,
        )

        turn_info = (
            f"You are generating turn {current_turn} of {target_turns} "
            f"as the {role.value.upper()}.\n\n"
        )
        if turn_instruction:
            turn_info += f"For this turn: {turn_instruction}\n\n"
        turn_info += "Generate ONLY your response for this turn. Stay in character."

        messages = [persona_msg, *history, Message(role=Role.USER, content=turn_info)]
        if trailing:
            messages.extend(trailing)
        return Conversation(messages=messages)

    def _run_assistant_turn(
        self,
        samples: list[dict],
        sample_indices: list[int],
        histories: list[list[Message]],
        current_turn: int,
        multiturn_attribute: MultiTurnAttribute,
    ) -> list[list[Message]]:
        """Run the inner tool-call loop for the assistant role at one turn."""
        cap = multiturn_attribute.max_tool_calls_per_turn
        turn_messages: dict[int, list[Message]] = {i: [] for i in sample_indices}
        done: dict[int, bool] = {i: False for i in sample_indices}
        tool_count: dict[int, int] = {i: 0 for i in sample_indices}

        while True:
            active = [i for i in sample_indices if not done[i] and tool_count[i] < cap]
            if not active:
                break

            prompts = [
                self._build_turn_prompt(
                    samples[i],
                    Role.ASSISTANT,
                    histories[i] + turn_messages[i],
                    current_turn,
                    multiturn_attribute,
                )
                for i in active
            ]
            texts = self._extract_response(
                self._inference_engine.infer(
                    prompts,
                    inference_config=self._assistant_inference_config(),
                )
            )
            for i, text in zip(active, texts):
                text = _close_dangling_tool_call(text)
                text = _truncate_after_last_tool_call(text)
                text = _canonicalize_tool_call_bodies(text)
                turn_messages[i].append(Message(role=Role.ASSISTANT, content=text))
                if self._is_final_response(text):
                    done[i] = True
                    continue
                tool_messages = self._execute_tool_calls(text)
                turn_messages[i].extend(tool_messages)
                tool_count[i] += len(tool_messages)

        stragglers = [i for i in sample_indices if not done[i]]
        if stragglers:
            nudge = Message(role=Role.USER, content=_FORCED_FINALIZE_NUDGE)
            prompts = [
                self._build_turn_prompt(
                    samples[i],
                    Role.ASSISTANT,
                    histories[i] + turn_messages[i],
                    current_turn,
                    multiturn_attribute,
                    trailing=[nudge],
                )
                for i in stragglers
            ]
            texts = self._extract_response(
                self._inference_engine.infer(
                    prompts,
                    inference_config=self._inference_config,
                )
            )
            for i, text in zip(stragglers, texts):
                final = _strip_tool_call_blocks(text).strip()
                turn_messages[i].append(Message(role=Role.ASSISTANT, content=final))

        return [turn_messages[i] for i in sample_indices]

    def _assistant_inference_config(self) -> InferenceConfig:
        """Build the inference config used for assistant turns in the tool loop."""
        existing_stops = list(self._inference_config.generation.stop_strings or [])
        if "</tool_call>" not in existing_stops:
            existing_stops = [*existing_stops, "</tool_call>"]
        return dataclasses.replace(
            self._inference_config,
            generation=dataclasses.replace(
                self._inference_config.generation,
                stop_strings=existing_stops,
            ),
        )

    def _is_final_response(self, text: str) -> bool:
        return _TOOL_CALL_RE.search(text) is None

    def _execute_tool_calls(self, response_text: str) -> list[Message]:
        return [
            self._run_single_tool_call(match.group(1).strip())
            for match in _TOOL_CALL_RE.finditer(response_text)
        ]

    def _make_grounding_rng(self, seed: int | None, sample_index: int) -> random.Random:
        """Build the RNG used for sampling grounding facts for one sample."""
        if seed is None:
            return random.Random()
        return random.Random(seed + sample_index)

    def _attach_grounding_facts(
        self,
        samples: list[dict],
        multiturn_attribute: MultiTurnAttribute,
    ) -> None:
        """Attach per-sample grounding facts drawn from grounded envs in scope."""
        if self._environment_config is None:
            return

        scoped_env_ids = (
            set(multiturn_attribute.available_environments)
            if multiturn_attribute.available_environments
            else {env.id for env in self._environment_config.environments}
        )
        grounding_envs = [
            env
            for env in self._environment_config.environments
            if env.id in scoped_env_ids and env.grounding is not None
        ]
        if not grounding_envs:
            return

        warned_envs: set[str] = set()
        for sample_index, sample in enumerate(samples):
            facts: list[GroundingFact] = []
            for env in grounding_envs:
                assert env.grounding is not None
                rng = self._make_grounding_rng(env.grounding.seed, sample_index)
                sampled = env.sample_grounding(n=env.grounding.sample_size, rng=rng)
                if (
                    len(sampled) < env.grounding.sample_size
                    and env.id not in warned_envs
                ):
                    logger.warning(
                        "Grounding sample_size=%d exceeds pool size for "
                        "environment '%s'; truncating to %d facts.",
                        env.grounding.sample_size,
                        env.id,
                        len(sampled),
                    )
                    warned_envs.add(env.id)
                facts.extend(sampled)
            sample["grounding_facts"] = facts

    def _run_single_tool_call(self, body: str) -> Message:
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as e:
            # Retry via brace repair to recover from stop-sequence truncation
            # (missing closing braces) and over-emission (trailing }}).
            repaired = repair_json_braces(body)
            if repaired is None:
                return _tool_error_msg(f"Malformed tool_call JSON: {e}")
            logger.debug(
                "Tool-call JSON repaired (len %d -> %d).", len(body), len(repaired)
            )
            parsed = json.loads(repaired)
        if not isinstance(parsed, dict):
            return _tool_error_msg("tool_call body must be a JSON object")

        name = parsed.get("name")
        arguments = parsed.get("arguments", {})

        if not isinstance(name, str) or not name:
            return _tool_error_msg("tool_call missing 'name'")
        if not isinstance(arguments, dict):
            return _tool_error_msg("tool_call 'arguments' must be an object")

        assert self._environment_config is not None

        tool = self._environment_config.get_tool(name)
        if tool is None:
            return _tool_error_msg(f"Unknown tool '{name}'")

        try:
            tool.validate_arguments(arguments)
        except ToolArgumentError as e:
            return _tool_error_msg(f"Invalid arguments for tool '{name}': {e}")

        try:
            result = self._environment_config.step(name, arguments)
        except ToolError as e:
            return _tool_error_msg(str(e))
        except KeyError:
            return _tool_error_msg(f"Unknown tool '{name}'")
        except Exception as e:  # noqa: BLE001
            return _tool_error_msg(f"Tool '{name}' raised: {e}")

        output = result.output

        if output is None:
            content = ""
        elif isinstance(output, str):
            content = output
        else:
            content = json.dumps(output)
        return _tool_result_message(content)

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
