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
from typing import Union

from oumi.core.configs.params.synthesis_params import (
    ChatTransform,
    DictTransform,
    GeneralSynthesisParams,
    ListTransform,
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
        samples: list[dict[str, SampleValue]],
    ) -> list[dict[str, SampleValue]]:
        """Transforms attributes of a dataset plan to a particular format.

        Args:
            samples: The samples to add transformed attributes to, using the values in
            each sample as the input to the transformation.

        Returns:
            The samples with the transformed attributes added.
        """
        for attribute in self._transformed_attributes:
            transformed_attribute_id = attribute.id
            sample_transforms = self._apply_transform_to_samples(samples, attribute)
            for sample, transformed_attribute_value in zip(samples, sample_transforms):
                sample[transformed_attribute_id] = transformed_attribute_value

        return samples

    def _apply_transform_to_samples(
        self,
        samples: list[dict[str, SampleValue]],
        attribute: TransformedAttribute,
    ) -> list[SampleValue]:
        """Applies a transform to a list of samples."""
        transformed_samples = []
        for sample in samples:
            transformed_sample = self._transform_attribute(sample, attribute)
            transformed_samples.append(transformed_sample)
        return transformed_samples

    def _transform_attribute(
        self,
        sample: dict[str, SampleValue],
        attribute: TransformedAttribute,
    ) -> SampleValue:
        """Transforms an attribute of a sample to a particular format."""
        if isinstance(attribute.transformation_strategy, str):
            return self._transform_string(sample, attribute.transformation_strategy)
        if isinstance(attribute.transformation_strategy, ListTransform):
            return self._transform_list(sample, attribute.transformation_strategy)
        elif isinstance(attribute.transformation_strategy, DictTransform):
            return self._transform_dict(sample, attribute.transformation_strategy)
        elif isinstance(attribute.transformation_strategy, ChatTransform):
            return self._transform_chat(
                sample, attribute.transformation_strategy, attribute.id
            )
        else:
            raise ValueError(
                "Unsupported transformation strategy: "
                f"{attribute.transformation_strategy}"
            )

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
        transform: ListTransform,
    ) -> list[str]:
        """Transforms a list attribute of a sample to a particular format."""
        result = []
        for element_transform in transform.element_transforms:
            result.append(self._transform_string(sample, element_transform))
        return result

    def _transform_dict(
        self,
        sample: dict[str, SampleValue],
        transform: DictTransform,
    ) -> dict[str, str]:
        """Transforms a dict attribute of a sample to a particular format."""
        result = {}
        for key, value_transform in transform.transforms.items():
            result[key] = self._transform_string(sample, value_transform)
        return result

    def _transform_chat(
        self,
        sample: dict[str, SampleValue],
        transform: ChatTransform,
        attribute_id: str,
    ) -> Conversation:
        """Transforms a chat attribute of a sample to a particular format."""
        messages = []
        for message in transform.transforms.messages:
            content = message.content
            if not isinstance(content, str):
                raise ValueError(
                    "ChatTransform.transforms.messages.content must be a string."
                )

            formatted_content = self._transform_string(sample, content)
            messages.append(Message(role=message.role, content=formatted_content))

        transformed_metadata = {}
        if transform.transforms.metadata:
            metadata_transform = DictTransform(transform.transforms.metadata)
            transformed_metadata = self._transform_dict(sample, metadata_transform)

        new_conv_id = transform.transforms.conversation_id
        if not transform.transforms.conversation_id:
            new_conv_id = f"{attribute_id}-{uuid.uuid4()}"

        return Conversation(
            messages=messages,
            conversation_id=new_conv_id,
            metadata=transformed_metadata,
        )
