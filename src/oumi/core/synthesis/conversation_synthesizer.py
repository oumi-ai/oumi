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
import re

from oumi.builders.inference_engines import build_inference_engine
from oumi.core.configs.inference_config import InferenceConfig
from oumi.core.configs.inference_engine_type import InferenceEngineType
from oumi.core.configs.params.synthesis_params import (
    GeneralSynthesisParams,
    MultiTurnAttribute,
)
from oumi.core.synthesis.attribute_formatter import AttributeFormatter
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.utils.logging import logger


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

    def _validate_turn_order(self, multiturn_attribute: MultiTurnAttribute) -> None:
        """Validate that all roles in turn_order have corresponding personas.

        Args:
            multiturn_attribute: The multi-turn attribute to validate.

        Raises:
            ValueError: If a role in turn_order is missing from
            role_instruction_messages.
        """
        turn_order = multiturn_attribute.turn_order or self._default_turn_order
        available_roles = set(multiturn_attribute.role_instruction_messages.keys())

        for role in turn_order:
            if role not in available_roles:
                raise ValueError(
                    f"Role '{role.value}' in turn_order is missing from "
                    f"role_instruction_messages. Available roles: "
                    f"{[r.value for r in available_roles]}"
                )

    def synthesize(
        self,
        samples: list[dict],
        multiturn_attributes: MultiTurnAttribute,
    ) -> list[dict[str, dict | str]]:
        """Synthesize a multi-turn conversation.

        Order will be identical to the order of the samples.

        Args:
            samples: The samples to synthesize values for.
            multiturn_attributes: The multi-turn attribute defining conversation rules.

        Returns:
            A list of dictionaries containing the conversation and the plan.
        """
        if not samples:
            return []

        self._validate_turn_order(multiturn_attributes)

        logger.info(
            f"Synthesizing {len(samples)} conversations for "
            f"attribute '{multiturn_attributes.id}'"
        )

        samples = self._plan_samples(samples, multiturn_attributes)
        conversations = self._synthesize_all_samples(samples, multiturn_attributes)

        records: list[dict[str, dict | str]] = []
        plan_key = f"{multiturn_attributes.id}_plan"
        for sample, conversation in zip(samples, conversations):
            record: dict[str, dict | str] = {
                multiturn_attributes.id: conversation.to_dict(),
                plan_key: sample["conversation_plan"],
            }
            records.append(record)

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
            (target_turns, turn_order, conversation_plan, parsed_turn_plans).
        """
        turn_order = multiturn_attributes.turn_order or self._default_turn_order

        augmented_samples: list[dict] = []
        for sample in samples:
            target_turns = self._select_target_turns(multiturn_attributes, turn_order)
            augmented_sample = {
                **sample,
                "target_turns": target_turns,
                "turn_order": turn_order,
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

        turns = self._extract_json_from_plan(plan)
        if turns is None:
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

    def _extract_json_from_plan(self, plan: str) -> list | None:
        """Extract and parse JSON array from plan text.

        Tries code-fenced JSON first, then raw JSON array

        Args:
            plan: The full plan text.

        Returns:
            Parsed JSON list, or None if extraction/parsing failed.
        """
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", plan)
        if json_match:
            try:
                result = json.loads(json_match.group(1))
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass
        bracket_start = plan.find("[")

        if bracket_start != -1:
            substring = plan[bracket_start:]
            for end_pos in range(len(substring), 0, -1):
                candidate = substring[:end_pos]
                if not candidate.rstrip().endswith("]"):
                    continue
                try:
                    result = json.loads(candidate)
                    if isinstance(result, list):
                        return result
                except json.JSONDecodeError:
                    continue

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

    def _format_persona(self, sample: dict, persona: str, role: Role) -> Message:
        """Format the persona for the sample.

        Args:
            sample: The sample dict containing all attributes.
            persona: The persona string to format.
            role: The role for this persona.

        Returns:
            A Message with the formatted persona as a SYSTEM message.
        """
        formatted_content = self._formatter.format(
            sample,
            persona,
            missing_values_allowed=False,
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
        self, multiturn_attribute: MultiTurnAttribute, sample: dict
    ) -> Conversation:
        """Create the planner prompt template with role context and turn order.

        Returns a Conversation with a one-shot example for consistent formatting.
        The prompt instructs the model to output JSON wrapped in code fences.
        """
        role_context = self._build_role_context(sample, multiturn_attribute)
        turn_order = sample["turn_order"]
        target_turns = sample["target_turns"]
        turn_order_str = self._build_turn_order_str(turn_order, target_turns)

        system_prompt = (
            "You are a conversation planner. Create conversation outlines "
            "that flow logically from start to finish.\n\n"
            "IMPORTANT: Output your plan as a JSON array wrapped in ```json code "
            "fences. Each element must have: turn (number) and instruction (string).\n"
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

        if role_context:
            base_prompt += f"\nRole context:\n{role_context}\n"

        if multiturn_attribute.conversation_planner:
            formatted_planner = self._formatter.format(
                sample,
                multiturn_attribute.conversation_planner,
                missing_values_allowed=False,
            )
            base_prompt += f"\nAdditional instructions: {formatted_planner}\n"

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
    ) -> list[Conversation]:
        """Synthesize multi-turn conversations for all samples with batched inference.

        Args:
            samples: List of sample dicts with runtime fields (target_turns, turn_order,
                conversation_plan).
            multiturn_attribute: The multi-turn attribute defining conversation rules.

        Returns:
            List of Conversation objects, one per sample.
        """
        if not samples:
            return []

        histories: list[list[Message]] = [[] for _ in samples]
        max_turns = max(sample["target_turns"] for sample in samples)

        for turn_idx in range(max_turns):
            current_turn = turn_idx + 1

            prompts: list[Conversation] = []
            sample_indices: list[int] = []
            roles_for_turn: list[Role] = []

            for i, sample in enumerate(samples):
                if turn_idx >= sample["target_turns"]:
                    continue

                turn_order = sample["turn_order"]
                role = turn_order[turn_idx % len(turn_order)]
                roles_for_turn.append(role)

                prompt_messages: list[Message] = []
                sample_with_turn = {**sample, "current_turn": current_turn}

                persona = multiturn_attribute.role_instruction_messages[role]
                formatted_persona = self._format_persona(
                    sample_with_turn, persona, role
                )
                prompt_messages.append(formatted_persona)
                prompt_messages.extend(histories[i])

                target_turns = sample["target_turns"]
                parsed_turn_plans = sample.get("parsed_turn_plans", [])

                turn_instruction = ""
                if turn_idx < len(parsed_turn_plans):
                    turn_instruction = parsed_turn_plans[turn_idx]

                turn_info = (
                    f"You are generating turn {current_turn} of {target_turns} "
                    f"as the {role.value.upper()}.\n\n"
                )
                if turn_instruction:
                    turn_info += f"For this turn: {turn_instruction}\n\n"
                turn_info += (
                    "Generate ONLY your response for this turn. Stay in character."
                )
                prompt_messages.append(Message(role=Role.USER, content=turn_info))

                prompts.append(Conversation(messages=prompt_messages))
                sample_indices.append(i)

            if not prompts:
                break

            inference_results = self._inference_engine.infer(
                prompts,
                inference_config=self._inference_config,
            )

            generated_texts = self._extract_response(inference_results)

            if len(generated_texts) != len(prompts):
                raise RuntimeError(
                    f"Inference engine returned {len(generated_texts)} results "
                    f"but {len(prompts)} prompts were submitted. "
                    f"This may indicate an inference engine error."
                )

            for idx, generated_text, role in zip(
                sample_indices, generated_texts, roles_for_turn
            ):
                histories[idx].append(Message(role=role, content=generated_text))

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
