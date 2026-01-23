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
    TextMessage,
)
from oumi.core.synthesis.attribute_formatter import AttributeFormatter
from oumi.core.types.conversation import Conversation, Message, Role


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
            A list of dictionaries containing the conversation and optionally the plan.
        """
        if not samples:
            return []

        records: list[dict[str, dict | str]] = []
        for sample in samples:
            conversation, plan = self._synthesize_sample(sample, multiturn_attributes)
            record: dict[str, dict | str] = {
                multiturn_attributes.id: conversation.to_dict()
            }
            if plan:
                record["conversation_plan"] = plan
            records.append(record)

        return records

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

    def _format_instructions(
        self,
        sample: dict,
        instruction_messages: list[TextMessage],
    ) -> Conversation:
        """Format the instructions for the sample."""
        new_messages = []
        for turn in instruction_messages:
            if not isinstance(turn.content, str):
                new_messages.append(turn)
                continue

            formatted_content = self._formatter.format(
                sample,
                turn.content,
                missing_values_allowed=False,
            )
            new_message = Message(
                role=turn.role,
                content=formatted_content,
            )
            new_messages.append(new_message)

        return Conversation(messages=new_messages)

    def _generate_plan(
        self,
        sample: dict,
        multiturn_attribute: MultiTurnAttribute,
    ) -> str:
        """Generate a plan for how the conversation should proceed.

        Args:
            sample: The sample dict containing attributes (including target_turns).
            multiturn_attribute: The multi-turn attribute with conversation_planner.

        Returns:
            The generated conversation plan as a string, or empty string if no planner.
        """
        if not multiturn_attribute.conversation_planner:
            return ""

        inference_results = self._inference_engine.infer(
            [
                self._format_instructions(
                    sample,
                    multiturn_attribute.conversation_planner.instruction_messages,
                )
            ],
            inference_config=self._inference_config,
        )

        return self._extract_response(inference_results)[0]

    def _synthesize_sample(
        self,
        sample: dict,
        multiturn_attribute: MultiTurnAttribute,
    ) -> tuple[Conversation, str]:
        """Synthesize a single multi-turn conversation for one sample.

        Args:
            sample: The sample dict containing all attributes.
            multiturn_attribute: The multi-turn attribute defining conversation rules.

        Returns:
            A tuple of (Conversation object with generated messages, plan string).
        """
        history: list[Message] = []
        target_turns = self._select_target_turns(multiturn_attribute)

        sample_with_context = {**sample, "target_turns": target_turns}

        plan = self._generate_plan(sample_with_context, multiturn_attribute)
        sample_with_context["conversation_plan"] = plan

        for turn_idx in range(target_turns):
            sample_with_context["current_turn"] = turn_idx + 1

            role = multiturn_attribute.turn_order[
                turn_idx % len(multiturn_attribute.turn_order)
            ]
            prompt_messages: list[Message] = []
            prompt_messages.extend(
                self._format_messages(
                    sample_with_context, multiturn_attribute.system_messages
                )
            )
            persona_message = self._persona_system_message(
                sample_with_context, multiturn_attribute, role
            )
            if persona_message:
                prompt_messages.append(persona_message)
            prompt_messages.extend(history)

            instruction = multiturn_attribute.turn_instructions[role]
            instruction_msg = self._format_message(sample_with_context, instruction)
            prompt_messages.append(instruction_msg)

            inference_results = self._inference_engine.infer(
                [Conversation(messages=prompt_messages)],
                inference_config=self._inference_config,
            )
            generated_text = self._extract_response(inference_results)[0]
            history.append(Message(role=role, content=generated_text))

        return Conversation(messages=history), plan

    def _select_target_turns(self, multiturn_attribute: MultiTurnAttribute) -> int:
        min_turns = multiturn_attribute.min_turns
        max_turns = multiturn_attribute.max_turns
        target_turns = random.randint(min_turns, max_turns)
        if Role.ASSISTANT not in multiturn_attribute.turn_order:
            return target_turns

        def role_at(turn_count: int) -> Role:
            return multiturn_attribute.turn_order[
                (turn_count - 1) % len(multiturn_attribute.turn_order)
            ]

        if role_at(target_turns) == Role.ASSISTANT:
            return target_turns
        for turn_count in range(target_turns + 1, max_turns + 1):
            if role_at(turn_count) == Role.ASSISTANT:
                return turn_count
        for turn_count in range(target_turns - 1, min_turns - 1, -1):
            if role_at(turn_count) == Role.ASSISTANT:
                return turn_count
        return target_turns

    def _format_messages(
        self,
        sample: dict,
        instruction_messages: list[TextMessage] | None,
    ) -> list[Message]:
        if not instruction_messages:
            return []
        return [
            self._format_message(sample, message) for message in instruction_messages
        ]

    def _format_message(self, sample: dict, message: TextMessage) -> Message:
        if not isinstance(message.content, str):
            return Message(role=message.role, content=message.content)
        formatted_content = self._formatter.format(
            sample,
            message.content,
            missing_values_allowed=False,
        )
        return Message(role=message.role, content=formatted_content.strip())

    def _persona_system_message(
        self,
        sample: dict,
        multiturn_attribute: MultiTurnAttribute,
        role: Role,
    ) -> Message | None:
        persona = None
        if role == Role.USER:
            persona = multiturn_attribute.user_persona
        elif role == Role.ASSISTANT:
            persona = multiturn_attribute.assistant_persona
        if persona is None:
            return None
        formatted_prompt = self._formatter.format(
            sample,
            persona.system_prompt,
            missing_values_allowed=False,
        )
        return Message(role=Role.SYSTEM, content=formatted_prompt.strip())
