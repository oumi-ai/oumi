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

import random

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

        logger.info(
            f"Synthesizing {len(samples)} conversations for "
            f"attribute '{multiturn_attributes.id}'"
        )

        samples = self._plan_samples(samples, multiturn_attributes)
        records: list[dict[str, dict | str]] = []
        plan_key = f"{multiturn_attributes.id}_plan"
        for sample in samples:
            conversation = self._synthesize_sample(sample, multiturn_attributes)
            record: dict[str, dict | str] = {
                multiturn_attributes.id: conversation.to_dict(),
                plan_key: sample["conversation_plan"],
            }
            records.append(record)

        return records

    def _plan_samples(
        self, samples: list[dict], multiturn_attributes: MultiTurnAttribute
    ) -> list[dict]:
        """Plan the conversation samples.

        Args:
            samples: The conversation samples to plan.
            multiturn_attributes: The multi-turn attribute defining conversation rules.

        Returns:
            A list of sample dicts augmented with runtime fields
            (target_turns, turn_order, conversation_plan).
        """
        turn_order = multiturn_attributes.turn_order or self._default_turn_order

        augmented_samples: list[dict] = []
        planner_conversations: list[Conversation] = []

        for sample in samples:
            target_turns = self._select_target_turns(multiturn_attributes, turn_order)

            augmented_sample = {
                **sample,
                "target_turns": target_turns,
                "turn_order": turn_order,
            }
            augmented_samples.append(augmented_sample)

            planner_conv = self._create_planner_prompt(
                multiturn_attributes, augmented_sample
            )
            planner_conversations.append(planner_conv)

            logger.debug(f"Planning conversation with {target_turns} turns")

        plans = self._generate_plan(augmented_samples, planner_conversations)

        for augmented_sample, plan in zip(augmented_samples, plans):
            augmented_sample["conversation_plan"] = plan

        return augmented_samples

    def _extract_response(
        self,
        inference_conversations: list[Conversation],
    ) -> list[str]:
        """Get the inference results from the inference conversations.

        If the inference result is not a string, an empty string will be returned.
        Strips whitespace to avoid API errors with trailing whitespace.
        """
        return [
            inference_result.messages[-1].content.strip()
            if isinstance(inference_result.messages[-1].content, str)
            else ""
            for inference_result in inference_conversations
        ]

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

    def _create_planner_prompt(
        self, multiturn_attribute: MultiTurnAttribute, sample: dict
    ) -> Conversation:
        """Create the planner prompt template with role context.

        Returns a Conversation.
        """
        role_context = self._build_role_context(sample, multiturn_attribute)

        base_prompt = (
            "Plan a {target_turns}-turn conversation. "
            "Create a turn-by-turn plan. Format each turn as: "
            "Turn N: [USER/ASSISTANT] Brief description of what happens. "
            "Guidelines: "
            "- Ensure the conversation flows naturally and logically. "
            "- Each turn should build on or respond to the previous turn. "
            "- Pace the conversation for {target_turns} turns. "
            "- Focus on the purpose of each turn, not exact wording. "
        )

        if role_context:
            base_prompt += f"\n\nRole context:\n{role_context}"

        if multiturn_attribute.conversation_planner:
            base_prompt += (
                "\n\nAdditional instructions: "
                f"{multiturn_attribute.conversation_planner}"
            )

        base_prompt += "\n\nOutput only the plan, nothing else."

        return Conversation(
            messages=[
                Message(
                    role=Role.SYSTEM,
                    content=(
                        "You are a conversation planner. Create conversation outlines "
                        "that flow logically from start to finish."
                    ),
                ),
                Message(role=Role.USER, content=base_prompt),
            ],
        )

    def _generate_plan(
        self, samples: list[dict], planners: list[Conversation]
    ) -> list[str]:
        """Generate plans for how the conversations should proceed.

        Args:
            samples: The sample dicts containing all attributes including target_turns.
            planners: The planner conversation templates.

        Returns:
            A list of plan strings, one per sample.
        """
        formatted_planners = []
        for sample, planner in zip(samples, planners):
            new_messages = []
            for msg in planner.messages:
                if isinstance(msg.content, str):
                    formatted = self._formatter.format(sample, msg.content)
                    new_messages.append(Message(role=msg.role, content=formatted))
                else:
                    new_messages.append(msg)
            formatted_planners.append(Conversation(messages=new_messages))

        inference_results = self._inference_engine.infer(
            formatted_planners,
            inference_config=self._inference_config,
        )

        return self._extract_response(inference_results)

    def _synthesize_sample(
        self,
        sample: dict,
        multiturn_attribute: MultiTurnAttribute,
    ) -> Conversation:
        """Synthesize a single multi-turn conversation for one sample.

        Args:
            sample: The sample dict containing all attributes and runtime fields.
            multiturn_attribute: The multi-turn attribute defining conversation rules.

        Returns:
            Conversation object with generated messages.
        """
        history: list[Message] = []
        target_turns = sample["target_turns"]
        turn_order = sample["turn_order"]
        conversation_plan = sample.get("conversation_plan", "")

        for turn_idx in range(target_turns):
            current_turn = turn_idx + 1

            role = turn_order[turn_idx % len(turn_order)]
            prompt_messages: list[Message] = []

            sample_with_turn = {**sample, "current_turn": current_turn}

            persona = multiturn_attribute.role_instruction_messages[role]
            formatted_persona = self._format_persona(sample_with_turn, persona, role)
            prompt_messages.append(formatted_persona)

            prompt_messages.extend(history)
            turn_info = (
                f"You are generating turn {current_turn} of {target_turns} "
                f"as the {role.value.upper()}.\n\n"
            )
            if conversation_plan:
                turn_info += f"Follow this conversation plan:\n{conversation_plan}\n\n"
            turn_info += (
                "Generate ONLY your response for this turn. "
                "Stay in character and follow the plan for this specific turn."
            )
            prompt_messages.append(Message(role=Role.USER, content=turn_info))

            inference_results = self._inference_engine.infer(
                [Conversation(messages=prompt_messages)],
                inference_config=self._inference_config,
            )

            generated_text = self._extract_response(inference_results)[0]
            history.append(Message(role=role, content=generated_text))

        output_messages: list[Message] = []
        output_message = self._format_output_system_message(
            sample, multiturn_attribute.output_system_prompt
        )
        if output_message:
            output_messages.append(output_message)
        output_messages.extend(history)
        return Conversation(messages=output_messages)

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
