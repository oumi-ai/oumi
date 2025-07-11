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

import string

from oumi.core.configs.params.synthesis_params import (
    GeneralSynthesisParams,
    GeneratedAttribute,
)
from oumi.core.types.conversation import Conversation


class _AttributeValueInfo:
    """Information about a value of a permutable attribute.

    Used to format the instructions for a sample.
    """

    def __init__(self, value_name: str, value_description: str):
        """Initialize the attribute value info."""
        self._value_name = value_name
        self.description = value_description

    def __str__(self) -> str:
        return self._value_name


class _AttributeInfo:
    """Information about a permutable attribute.

    Used to format the instructions for a sample.
    """

    def __init__(
        self,
        attribute_id: str,
        attribute_name: str,
        attribute_description: str,
        value_name: str,
        value_description: str,
    ):
        """Initialize the attribute value info."""
        self.attribute_id = attribute_id
        self._attribute_name = attribute_name
        self.description = attribute_description
        self.value = _AttributeValueInfo(value_name, value_description)

    def __str__(self) -> str:
        return self._attribute_name


class AttributeSynthesizer:
    """Synthesizes attributes based on the given samples and GeneratedAttribute.

    Args:
        params: The parameters for the attribute synthesizer.
    """

    def __init__(self, params: GeneralSynthesisParams):
        """Initialize the synthesizer."""
        self._params = params
        self._permutable_attribute_map = (
            {perm_attr.id: perm_attr for perm_attr in params.permutable_attributes}
            if params.permutable_attributes
            else {}
        )

    def synthesize(
        self,
        samples: list[dict],
        generated_attribute: GeneratedAttribute,
    ) -> list[Conversation]:
        """Synthesize a value for the generated attribute."""
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
        attr_values = {}
        for attribute_id, attribute_value in sample.items():
            if self._is_permutable_attribute(attribute_id):
                value_id = attribute_value
                attr_values[attribute_id] = self._get_permutable_attribute_value_info(
                    attribute_id, value_id
                )
            else:
                attr_values[attribute_id] = attribute_value

        new_messages = []
        for turn in instruction_messages.messages:
            if not isinstance(turn.content, str):
                new_messages.append(turn)
                continue

            formatted_content = turn.content.format(**attr_values)

            field_names = [
                v[1]
                for v in string.Formatter().parse(formatted_content)
                if v[1] is not None
            ]
            if len(field_names) > 0:
                raise ValueError(
                    f"Format string {formatted_content} contains "
                    f"unresolved fields: {field_names}"
                )

            # Create new Message with formatted content
            new_message = turn.model_copy(update={"content": formatted_content})
            new_messages.append(new_message)

        # Create new conversation with formatted messages
        new_conversation = Conversation(
            messages=new_messages,
            conversation_id=instruction_messages.conversation_id,
            metadata=instruction_messages.metadata,
        )
        return new_conversation

    def _is_permutable_attribute(self, attribute_id: str) -> bool:
        """Check if the attribute is a permutable attribute."""
        return attribute_id in self._permutable_attribute_map

    def _get_permutable_attribute_value_info(
        self, attribute_id: str, attribute_value_id: str
    ) -> _AttributeInfo:
        """Get the instruction for a permutable attribute."""
        attribute = self._permutable_attribute_map[attribute_id]
        attribute_id = attribute.id
        attribute_name = attribute.attribute
        attribute_desc = attribute.description
        values = attribute.possible_values
        for value in values:
            if value.id == attribute_value_id:
                value_name = value.value
                value_description = value.description
                return _AttributeInfo(
                    attribute_id=attribute_id,
                    attribute_name=attribute_name,
                    attribute_description=attribute_desc,
                    value_name=value_name,
                    value_description=value_description,
                )

        raise ValueError(
            f"Attribute value {attribute_value_id} not found for "
            f"attribute {attribute_id}"
        )
