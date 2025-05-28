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

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union

from oumi.core.configs.params.base_params import BaseParams
from oumi.core.types.conversation import Conversation


@dataclass
class DatasetSource:
    """Dataset to be used in synthesis."""

    path: str
    """Path to the dataset source."""

    attribute_map: Optional[dict[str, str]] = None
    """Map of attributes to be used in synthesis. Will use original attribute names if
    not specified."""

    def __post_init__(self):
        """Verifies/populates params."""
        if not self.path:
            raise ValueError("DatasetSource.path cannot be empty.")

        if self.path.endswith(".jsonl"):
            pass
        elif self.path.endswith(".json"):
            pass
        elif self.path.endswith(".csv"):
            pass
        elif self.path.endswith(".parquet"):
            pass
        elif self.path.endswith(".tsv"):
            pass
        else:
            raise ValueError(f"Unsupported dataset file type: {self.path}")


class SegmentationStrategy(str, Enum):
    """Segmentation strategies."""

    TOKENS = "tokens"
    """Segment the document via tokens."""


@dataclass
class DocumentSegmentationParams:
    """Segmentation parameters to be used when segmenting the document."""

    id: str
    """ID to be used when referencing the document segment during synthesis."""

    segmentation_strategy: SegmentationStrategy = SegmentationStrategy.TOKENS
    """Type of segmentation to be used."""

    segment_length: int = 2048
    """Length of each segment, dependent on the segmentation strategy."""

    segment_overlap: int = 0
    """Overlap between segments. Must be less than segment_length."""

    keep_original_text: bool = False
    """Whether to keep the original text of the document."""

    def __post_init__(self):
        """Verifies/populates params."""
        if self.segment_length <= 0:
            raise ValueError("Segment length must be positive.")
        if self.segment_overlap < 0:
            raise ValueError("Segment overlap must be non-negative.")
        if self.segment_overlap >= self.segment_length:
            raise ValueError("Segment overlap must be less than segment length.")


@dataclass
class DocumentSource:
    """Documents to be used in synthesis."""

    path: str
    """Path to the document source."""

    id: str
    """ID to be used when referencing the document during synthesis."""

    segmentation_params: Optional[DocumentSegmentationParams] = None
    """Segmentation parameters to be used when segmenting the document."""

    def __post_init__(self):
        """Verifies/populates params."""
        if not self.path:
            raise ValueError("DocumentSource.path cannot be empty.")
        if not self.id:
            raise ValueError("DocumentSource.id cannot be empty.")


@dataclass
class ExampleSource:
    """In-line examples to be used in synthesis."""

    examples: list[dict[str, Any]]
    """Examples to be used in synthesis."""

    def __post_init__(self):
        """Verifies/populates params."""
        if not self.examples:
            raise ValueError("ExampleSource.examples cannot be empty.")

        keys = self.examples[0].keys()
        for example in self.examples:
            if example.keys() != keys:
                raise ValueError("All examples must have the same keys.")


@dataclass
class PermutableAttributeValue:
    """Value to be used for the attribute."""

    id: str
    """ID to be used when referencing the attribute value during synthesis."""

    value: str
    """Value to be used for the attribute."""

    description: str
    """Description of the attribute value.
    Referenced as <<attribute_id.value.description>>"""

    sample_rate: Optional[float] = None
    """Sample rate for the attribute value. If not specified, will assume uniform
    sampling among possible values."""

    def __post_init__(self):
        """Verifies/populates params."""
        if not self.id:
            raise ValueError("PermutableAttributeValue.id cannot be empty.")
        if not self.value:
            raise ValueError("PermutableAttributeValue.value cannot be empty.")
        if not self.description:
            raise ValueError("PermutableAttributeValue.description cannot be empty.")
        if self.sample_rate is not None and (
            self.sample_rate < 0 or self.sample_rate > 1
        ):
            raise ValueError(
                "PermutableAttributeValue.sample_rate must be between 0 and 1."
            )


@dataclass
class PermutableAttribute:
    """Attributes to be varied across the dataset."""

    id: str
    """ID to be used when referencing the attribute during synthesis."""

    attribute: str
    """Plaintext name of the attribute. Referenced as <<atribute_id>>"""

    description: str
    """Description of the attribute. Referenced as <<attribute_id.description>>"""

    possible_values: list[PermutableAttributeValue]
    """Type of the attribute."""

    def __post_init__(self):
        """Verifies/populates params."""
        if not self.id:
            raise ValueError("PermutableAttribute.id cannot be empty.")
        if not self.attribute:
            raise ValueError("PermutableAttribute.attribute cannot be empty.")
        if not self.description:
            raise ValueError("PermutableAttribute.description cannot be empty.")
        if not self.possible_values:
            raise ValueError("PermutableAttribute.possible_values cannot be empty.")

        value_ids = []
        sample_rates = []
        for value in self.possible_values:
            value_ids.append(value.id)
            sample_rates.append(value.sample_rate)

        value_ids_set = set(value_ids)
        if len(value_ids) != len(value_ids_set):
            raise ValueError(
                "PermutableAttribute.possible_values must have unique IDs."
            )

        # Normalize sample rates
        normalized_sample_rates = []
        undefined_sample_rate_count = 0
        defined_sample_rate = 0.0
        for sample_rate in sample_rates:
            if sample_rate is not None:
                defined_sample_rate += sample_rate
            else:
                undefined_sample_rate_count += 1

            if defined_sample_rate > 1.0:
                raise ValueError("PermutableAttribute.possible_values must sum to 1.0.")

        # Assign remaining sample rate to undefined sample rates
        remaining_sample_rate = 1.0 - defined_sample_rate
        for sample_rate in sample_rates:
            if sample_rate is None:
                normalized_sample_rates.append(
                    remaining_sample_rate / undefined_sample_rate_count
                )
            else:
                normalized_sample_rates.append(sample_rate)

        # Update sample rates
        for i, sample_rate in enumerate(normalized_sample_rates):
            self.possible_values[i].sample_rate = sample_rate


@dataclass
class AttributeCombination:
    """Sampling rates for combinations of attributes."""

    combination: dict[str, str]
    """Combination of attribute values to be used."""

    sample_rate: float
    """Sample rate for the combination."""

    def __post_init__(self):
        """Verifies/populates params."""
        if self.sample_rate < 0 or self.sample_rate > 1:
            raise ValueError(
                "AttributeCombination.sample_rate must be between 0 and 1."
            )
        if not self.combination:
            raise ValueError("AttributeCombination.combination cannot be empty.")

        for key, value in self.combination.items():
            if not key:
                raise ValueError(
                    "AttributeCombination.combination key cannot be empty."
                )
            if not value:
                raise ValueError(
                    "AttributeCombination.combination value cannot be empty."
                )

        if len(self.combination.keys()) <= 1:
            raise ValueError(
                "AttributeCombination.combination must have at least two keys."
            )


@dataclass
class GeneratedAttributePostprocessingParams:
    """Postprocessing parameters for generated attributes."""

    cut_prefix: Optional[str] = None
    """Cut off value before and including prefix."""

    cut_suffix: Optional[str] = None
    """Cut off value after and including suffix."""

    regex: Optional[str] = None
    """Regex to be used to pull out the value from the generated text."""

    strip_whitespace: bool = True
    """Whether to strip whitespace from the value."""

    added_prefix: Optional[str] = None
    """Prefix to be added to the value."""

    added_suffix: Optional[str] = None
    """Suffix to be added to the value."""


@dataclass
class GeneratedAttribute:
    """Attributes to be generated."""

    id: str
    """ID to be used when referencing the attribute during synthesis."""

    instruction_messages: Conversation
    """List of messages providing instructions for generating this attribute."""

    postprocessing_params: Optional[GeneratedAttributePostprocessingParams] = None
    """Postprocessing parameters for the generated attribute."""

    def __post_init__(self):
        """Verifies/populates params."""
        if not self.id:
            raise ValueError("GeneratedAttribute.id cannot be empty.")
        if not self.instruction_messages:
            raise ValueError("GeneratedAttribute.instruction_messages cannot be empty.")


@dataclass
class ListTransform:
    """Create a new attribute which is a list of strings."""

    element_transforms: list[str]
    """List of transforms for each element of the list."""

    def __post_init__(self):
        """Verifies/populates params."""
        if not self.element_transforms:
            raise ValueError("ListTransform.element_transforms cannot be empty.")


@dataclass
class DictTransform:
    """Create a new attribute which is a dictionary of strings."""

    transforms: dict[str, str]
    """Mapping of dictionary keys to their corresponding transforms."""

    def __post_init__(self):
        """Verifies/populates params."""
        if not self.transforms:
            raise ValueError("DictTransform.transforms cannot be empty.")


@dataclass
class ChatTransform:
    """Transform of an attribute using a chat."""

    transforms: Conversation
    """List of transforms for chat messages."""

    def __post_init__(self):
        """Verifies/populates params."""
        messages = self.transforms.messages
        if not messages or len(messages) == 0:
            raise ValueError("ChatTransform.transforms must have at least one message.")

        for message in messages:
            content = message.content
            if not isinstance(content, str):
                raise ValueError(
                    "ChatTransform.transforms message content must be a string."
                )

            if not content:
                raise ValueError(
                    "ChatTransform.transforms message content cannot be empty."
                )


TransformationStrategy = Union[str, ListTransform, DictTransform, ChatTransform]


@dataclass
class TransformedAttribute:
    """Transformation of existing attributes."""

    id: str
    """ID to be used when referencing the transformed attribute during synthesis."""

    transformation_strategy: TransformationStrategy
    """Strategy to be used for the transformation."""

    def __post_init__(self):
        """Verifies/populates params."""
        if not self.id:
            raise ValueError("TransformedAttribute.id cannot be empty.")


@dataclass
class GeneralSynthesisParams(BaseParams):
    """General synthesis parameters."""

    input_data: Optional[list[DatasetSource]] = None
    """Data to be used in synthesis"""

    input_documents: Optional[list[DocumentSource]] = None
    """Documents to be used in synthesis"""

    input_examples: Optional[list[ExampleSource]] = None
    """In-line examples to be used in synthesis"""

    permutable_attributes: Optional[list[PermutableAttribute]] = None
    """Attributes to be varied across the dataset"""

    combination_sampling: Optional[list[AttributeCombination]] = None
    """Sampling rates for combinations of attributes"""

    generated_attributes: Optional[list[GeneratedAttribute]] = None
    """Attributes to be generated"""

    transformed_attributes: Optional[list[TransformedAttribute]] = None
    """Transformation of existing attributes"""

    passthrough_attributes: Optional[list[str]] = None
    """When specified, will ONLY pass through these attributes in final output.
    If left unspecified, all attributes are saved."""

    def __post_init__(self):
        """Verifies/populates params."""
        if isinstance(self.input_data, list) and len(self.input_data) == 0:
            raise ValueError("GeneralSynthesisParams.input_data cannot be empty.")
        if isinstance(self.input_documents, list) and len(self.input_documents) == 0:
            raise ValueError("GeneralSynthesisParams.input_documents cannot be empty.")
        if isinstance(self.input_examples, list) and len(self.input_examples) == 0:
            raise ValueError("GeneralSynthesisParams.input_examples cannot be empty.")
        if (
            isinstance(self.permutable_attributes, list)
            and len(self.permutable_attributes) == 0
        ):
            raise ValueError(
                "GeneralSynthesisParams.permutable_attributes cannot be empty."
            )
        if (
            isinstance(self.combination_sampling, list)
            and len(self.combination_sampling) == 0
        ):
            raise ValueError(
                "GeneralSynthesisParams.combination_sampling cannot be empty."
            )
        if (
            isinstance(self.generated_attributes, list)
            and len(self.generated_attributes) == 0
        ):
            raise ValueError(
                "GeneralSynthesisParams.generated_attributes cannot be empty."
            )
        if (
            isinstance(self.transformed_attributes, list)
            and len(self.transformed_attributes) == 0
        ):
            raise ValueError(
                "GeneralSynthesisParams.transformed_attributes cannot be empty."
            )
        if (
            isinstance(self.passthrough_attributes, list)
            and len(self.passthrough_attributes) == 0
        ):
            raise ValueError(
                "GeneralSynthesisParams.passthrough_attributes cannot be empty."
            )

        attribute_ids = set()
        if self.input_data:
            for dataset_source in self.input_data:
                if dataset_source.attribute_map:
                    for new_key in dataset_source.attribute_map.values():
                        if new_key in attribute_ids:
                            raise ValueError(
                                f"GeneralSynthesisParams contains duplicate attribute "
                                f"IDs: {new_key}"
                            )
                        attribute_ids.add(new_key)

        if self.input_documents:
            for document_source in self.input_documents:
                doc_key = document_source.id
                if doc_key in attribute_ids:
                    raise ValueError(
                        f"GeneralSynthesisParams contains duplicate attribute "
                        f"IDs: {doc_key}"
                    )
                attribute_ids.add(doc_key)

                if document_source.segmentation_params:
                    seg_key = document_source.segmentation_params.id
                    if seg_key in attribute_ids:
                        raise ValueError(
                            f"GeneralSynthesisParams contains duplicate attribute "
                            f"IDs: {seg_key}"
                        )
                    attribute_ids.add(seg_key)

        if self.input_examples:
            for example_source in self.input_examples:
                example_keys = example_source.examples[0].keys()
                for new_key in example_keys:
                    if new_key in attribute_ids:
                        raise ValueError(
                            f"GeneralSynthesisParams contains duplicate attribute "
                            f"IDs: {new_key}"
                        )
                    attribute_ids.add(new_key)

        if self.permutable_attributes:
            for permutable_attribute in self.permutable_attributes:
                attribute_id = permutable_attribute.id
                if attribute_id in attribute_ids:
                    raise ValueError(
                        f"GeneralSynthesisParams contains duplicate attribute "
                        f"IDs: {attribute_id}"
                    )

                attribute_ids.add(attribute_id)

        if self.generated_attributes:
            for generated_attribute in self.generated_attributes:
                attribute_id = generated_attribute.id
                if attribute_id in attribute_ids:
                    raise ValueError(
                        f"GeneralSynthesisParams contains duplicate attribute "
                        f"IDs: {attribute_id}"
                    )
                attribute_ids.add(attribute_id)

        if self.transformed_attributes:
            for transformed_attribute in self.transformed_attributes:
                attribute_id = transformed_attribute.id
                if attribute_id in attribute_ids:
                    raise ValueError(
                        f"GeneralSynthesisParams contains duplicate attribute "
                        f"IDs: {attribute_id}"
                    )
                attribute_ids.add(attribute_id)

        if self.combination_sampling:
            sample_rates = [
                combination.sample_rate for combination in self.combination_sampling
            ]
            if sum(sample_rates) > 1.0:
                raise ValueError(
                    "GeneralSynthesisParams.combination_sampling sample rates must be "
                    "less than or equal to 1.0."
                )
