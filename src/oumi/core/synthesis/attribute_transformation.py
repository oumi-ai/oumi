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


import uuid
from typing import Any, Union

from oumi.core.configs.params.synthesis_params import (
    GeneralSynthesisParams,
    TransformationStrategy,
    TransformationType,
    TransformedAttribute,
)
from oumi.core.synthesis.attribute_formatter import AttributeFormatter
from oumi.core.types.conversation import Conversation, Message

SampleValue = Union[str, list[str], dict[str, str], Conversation]


class AttributeTransformer:
    """Transforms attributes of a dataset plan to a particular format."""

    def __init__(self, params: GeneralSynthesisParams):
        """Initializes the attribute transformer.

        Args:
            params: The general synthesis parameters containing the transformed
            attributes.
        """
        self._formatter = AttributeFormatter(params)
        self._transformed_attributes = (
            params.transformed_attributes if params.transformed_attributes else []
        )

    def transform(
        self,
        samples: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Transforms attributes of a dataset plan to a particular format.

        Args:
            samples: The samples to add transformed attributes to, using the values in
            each sample as the input to the transformation.

        Returns:
            The samples with the transformed attributes added.
        """
        for attribute in self._transformed_attributes:
            transformed_attribute_id = attribute.id
            for sample in samples:
                sample[transformed_attribute_id] = self._transform_attribute(
                    sample,
                    attribute,
                )

        return samples

    def _transform_attribute(
        self,
        sample: dict[str, Any],
        attribute: TransformedAttribute,
    ) -> SampleValue:
        """Transforms an attribute of a sample to a particular format."""
        strategy = attribute.get_strategy()
        if strategy.type == TransformationType.STRING:
            assert strategy.string_transform is not None  # Validated in __post_init__
            return self._transform_string(sample, strategy.string_transform)
        elif strategy.type == TransformationType.LIST:
            return self._transform_list(sample, strategy)
        elif strategy.type == TransformationType.DICT:
            return self._transform_dict(sample, strategy)
        elif strategy.type == TransformationType.CHAT:
            return self._transform_chat(sample, strategy, attribute.id)
        else:
            raise ValueError(f"Unsupported transformation strategy: {strategy.type}")

    def _transform_string(
        self,
        sample: dict[str, SampleValue],
        transform: str,
    ) -> str:
        """Transforms a string attribute of a sample to a particular format."""
        string_sample = {k: v for k, v in sample.items() if isinstance(v, str)}
        formatted_string = self._formatter.format(
            string_sample,
            transform,
            missing_values_allowed=False,
        )
        return formatted_string

    def _transform_list(
        self,
        sample: dict[str, SampleValue],
        transform: TransformationStrategy,
    ) -> list[str]:
        """Transforms a list attribute of a sample to a particular format."""
        assert transform.list_transform is not None
        return [self._transform_string(sample, e) for e in transform.list_transform]

    def _transform_dict(
        self,
        sample: dict[str, SampleValue],
        transform: TransformationStrategy,
    ) -> dict[str, str]:
        """Transforms a dict attribute of a sample to a particular format."""
        assert transform.dict_transform is not None  # Validated in __post_init__
        return {
            k: self._transform_string(sample, v)
            for k, v in transform.dict_transform.items()
        }

    def _transform_chat(
        self,
        sample: dict[str, SampleValue],
        transform: TransformationStrategy,
        attribute_id: str,
    ) -> dict[str, Any]:
        """Transforms a chat attribute of a sample to a particular format."""
        assert transform.chat_transform is not None  # Validated in __post_init__
        messages = []
        for message in transform.chat_transform.messages:
            content = message.content
            if not isinstance(content, str):
                raise ValueError(
                    "ChatTransform.transforms.messages.content must be a string."
                )

            formatted_content = self._transform_string(sample, content)
            messages.append(Message(role=message.role, content=formatted_content))

        transformed_metadata = {}
        if transform.chat_transform.metadata:
            # Create a TransformationStrategy for the metadata dict transformation
            metadata_transform = TransformationStrategy(
                type=TransformationType.DICT,
                dict_transform=transform.chat_transform.metadata,
            )
            transformed_metadata = self._transform_dict(sample, metadata_transform)

        new_conv_id = transform.chat_transform.conversation_id
        if not transform.chat_transform.conversation_id:
            new_conv_id = f"{attribute_id}-{uuid.uuid4()}"

        return Conversation(
            messages=messages,
            conversation_id=new_conv_id,
            metadata=transformed_metadata,
        ).to_dict()
