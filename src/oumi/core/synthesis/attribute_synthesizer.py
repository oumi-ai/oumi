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


from oumi.core.configs.params.synthesis_params import (
    GeneralSynthesisParams,
    GeneratedAttribute,
)
from oumi.core.synthesis.attribute_formatter import AttributeFormatter
from oumi.core.types.conversation import Conversation


class AttributeSynthesizer:
    """Synthesizes values for a generated attribute based on the given samples.

    Args:
        params: The parameters for the attribute synthesizer.
    """

    def __init__(self, params: GeneralSynthesisParams):
        """Initialize the synthesizer."""
        self._params = params
        self._formatter = AttributeFormatter(params)

    def synthesize(
        self,
        samples: list[dict],
        generated_attribute: GeneratedAttribute,
    ) -> list[Conversation]:
        """Synthesize values for the generated attribute."""
        inference_conversations: list[Conversation] = []
        for sample in samples:
            inference_conversations.append(
                self._format_instructions(
                    sample,
                    generated_attribute.instruction_messages,
                )
            )

        # TODO: Run inference

        # TODO: Post-process inference results

        # TODO: Return inference results
        return inference_conversations

    def _format_instructions(
        self,
        sample: dict,
        instruction_messages: Conversation,
    ) -> Conversation:
        """Format the instructions for the sample."""
        new_messages = []
        for turn in instruction_messages.messages:
            if not isinstance(turn.content, str):
                new_messages.append(turn)
                continue

            formatted_content = self._formatter.format(
                sample,
                turn.content,
                missing_values_allowed=False,
            )

            # Create new Message with formatted content
            new_message = turn.model_copy(
                deep=True,
                update={"content": formatted_content},
            )
            new_messages.append(new_message)

        # Create new conversation with formatted messages
        new_conversation = Conversation(
            messages=new_messages,
            conversation_id=instruction_messages.conversation_id,
            metadata=instruction_messages.metadata,
        )
        return new_conversation
