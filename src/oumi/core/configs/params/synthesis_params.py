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

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from oumi.core.configs.params.base_params import BaseParams
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.core.types.exceptions import InvalidParameterValueError, MissingParameterError

_SUPPORTED_DATASET_FILE_TYPES = {".jsonl", ".json", ".csv", ".parquet", ".tsv"}


@dataclass
class TextMessage:
    """Text-only message to make it usable in omegaconf."""

    role: Role
    content: str

    def to_message(self) -> Message:
        """Convert to a Message."""
        return Message(role=self.role, content=self.content)


@dataclass
class TextConversation:
    """Text-only conversation to make it usable in omegaconf."""

    messages: list[TextMessage]

    conversation_id: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_conversation(self) -> Conversation:
        """Convert to a Conversation."""
        return Conversation(
            messages=[message.to_message() for message in self.messages],
            conversation_id=self.conversation_id,
            metadata=self.metadata,
        )


@dataclass
class DatasetSource:
    """Dataset to be used in synthesis."""

    path: str
    """Path to the dataset source."""

    hf_split: str | None = None
    """Split of the huggingface dataset to be used in synthesis."""

    hf_revision: str | None = None
    """Revision of the huggingface dataset to be used in synthesis."""

    attribute_map: dict[str, str] | None = None
    """Map of attributes to be used in synthesis.
    Will use the existing keys in the dataset if not specified."""

    def __post_init__(self):
        """Verifies/populates params."""
        if not self.path:
            raise MissingParameterError(
                "DatasetSource.path is required. "
                "Provide a path to a dataset file (e.g., 'data.jsonl') or "
                "a HuggingFace dataset (e.g., 'hf:dataset_name')."
            )

        file_path = Path(self.path)
        prefix = self.path.split(":")[0]
        if prefix == "hf" or prefix == "oumi":
            return
        if file_path.suffix.lower() not in _SUPPORTED_DATASET_FILE_TYPES:
            raise InvalidParameterValueError(
                f"Unsupported dataset file type: '{file_path.suffix}'. "
                f"Supported types: {_SUPPORTED_DATASET_FILE_TYPES}"
            )


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

    tokenizer: str = "openai-community/gpt2"
    """Tokenizer to be used for segmentation.

    Tokenizers can be specified by their HuggingFace Hub ID or by direct file path.
    If not specified, will use the GPT-2 tokenizer from the HuggingFace Hub."""

    segment_length: int = 2048
    """Length of each segment, dependent on the segmentation strategy."""

    segment_overlap: int = 0
    """Overlap between segments. Must be less than segment_length."""

    keep_original_text: bool = False
    """Whether to keep the original text of the document."""

    def __post_init__(self):
        """Verifies/populates params."""
        if self.segment_length <= 0:
            raise InvalidParameterValueError(
                f"segment_length must be positive, got {self.segment_length}."
            )
        if self.segment_overlap < 0:
            raise InvalidParameterValueError(
                f"segment_overlap must be non-negative, got {self.segment_overlap}."
            )
        if self.segment_overlap >= self.segment_length:
            raise InvalidParameterValueError(
                f"segment_overlap ({self.segment_overlap}) must be less than "
                f"segment_length ({self.segment_length})."
            )
        if self.segmentation_strategy == SegmentationStrategy.TOKENS:
            if not self.tokenizer:
                raise MissingParameterError(
                    "tokenizer is required when segmentation_strategy is TOKENS."
                )


@dataclass
class DocumentSource:
    """Documents to be used in synthesis."""

    path: str
    """Path to the document source."""

    id: str
    """ID to be used when referencing the document during synthesis."""

    segmentation_params: DocumentSegmentationParams | None = None
    """Segmentation parameters to be used when segmenting the document."""

    def __post_init__(self):
        """Verifies/populates params."""
        if not self.path:
            raise MissingParameterError(
                "DocumentSource.path is required. "
                "Provide a path to a document file to use in synthesis."
            )
        if not self.id:
            raise MissingParameterError(
                "DocumentSource.id is required. "
                "Provide a unique ID to reference this document during synthesis."
            )


@dataclass
class ExampleSource:
    """In-line examples to be used in synthesis."""

    examples: list[dict[str, Any]]
    """Examples to be used in synthesis."""

    def __post_init__(self):
        """Verifies/populates params."""
        if not self.examples:
            raise MissingParameterError(
                "ExampleSource.examples is required. "
                "Provide a list of example dictionaries to use in synthesis."
            )

        keys = set(self.examples[0].keys())
        for i, example in enumerate(self.examples):
            example_keys = set(example.keys())
            if example_keys != keys:
                missing = keys - example_keys
                extra = example_keys - keys
                raise InvalidParameterValueError(
                    f"All examples must have the same keys. "
                    f"Example at index {i} has inconsistent keys. "
                    f"Expected keys: {sorted(keys)}. "
                    + (f"Missing: {sorted(missing)}. " if missing else "")
                    + (f"Extra: {sorted(extra)}." if extra else "")
                )


@dataclass
class SampledAttributeValue:
    """Value to be sampled for the attribute."""

    id: str
    """ID to be used when referencing the attribute value during synthesis."""

    name: str
    """Plaintext name of the attribute value.
    Referenced as {attribute_id}"""

    description: str
    """Description of the attribute value.
    Referenced as {attribute_id.description}"""

    sample_rate: float | None = None
    """Sample rate for the attribute value. If not specified, will assume uniform
    sampling among possible values."""

    def __post_init__(self):
        """Verifies/populates params."""
        if not self.id:
            raise MissingParameterError(
                "SampledAttributeValue.id is required. "
                "Provide a unique ID to reference this value during synthesis."
            )
        if not self.name:
            raise MissingParameterError(
                f"SampledAttributeValue.name is required for value '{self.id}'. "
                "Provide a plaintext name for this attribute value."
            )
        if not self.description:
            raise MissingParameterError(
                f"SampledAttributeValue.description is required for value '{self.id}'. "
                "Provide a description of this attribute value."
            )
        if self.sample_rate is not None and (
            self.sample_rate < 0 or self.sample_rate > 1
        ):
            raise InvalidParameterValueError(
                f"SampledAttributeValue '{self.id}' has invalid sample_rate={self.sample_rate}. "
                "sample_rate must be between 0 and 1 (inclusive)."
            )


@dataclass
class SampledAttribute:
    """Attributes to be sampled across the dataset."""

    id: str
    """ID to be used when referencing the attribute during synthesis."""

    name: str
    """Plaintext name of the attribute. Referenced as {id.parent}"""

    description: str
    """Description of the attribute. Referenced as {id.parent.description}"""

    possible_values: list[SampledAttributeValue]
    """Values to be sampled for the attribute."""

    def get_value_distribution(self) -> dict[str, float]:
        """Get the distribution of attribute values."""
        value_distribution = {}
        for value in self.possible_values:
            value_distribution[value.id] = value.sample_rate
        return value_distribution

    def __post_init__(self):
        """Verifies/populates params."""
        if not self.id:
            raise MissingParameterError(
                "SampledAttribute.id is required. "
                "Provide a unique ID to reference this attribute during synthesis."
            )
        if not self.name:
            raise MissingParameterError(
                f"SampledAttribute.name is required for attribute '{self.id}'. "
                "Provide a plaintext name for this attribute."
            )
        if not self.description:
            raise MissingParameterError(
                f"SampledAttribute.description is required for attribute '{self.id}'. "
                "Provide a description of this attribute."
            )
        if not self.possible_values:
            raise MissingParameterError(
                f"SampledAttribute.possible_values is required for attribute '{self.id}'. "
                "Provide a list of SampledAttributeValue objects defining possible values."
            )

        value_ids = []
        sample_rates = []
        for value in self.possible_values:
            value_ids.append(value.id)
            sample_rates.append(value.sample_rate)

        value_ids_set = set(value_ids)
        if len(value_ids) != len(value_ids_set):
            duplicates = [vid for vid in value_ids if value_ids.count(vid) > 1]
            raise InvalidParameterValueError(
                f"SampledAttribute '{self.id}' has duplicate value IDs: {set(duplicates)}. "
                "Each possible_value must have a unique ID."
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

        if defined_sample_rate > 1.0 and not math.isclose(defined_sample_rate, 1.0):
            raise InvalidParameterValueError(
                f"SampledAttribute '{self.id}' has sample_rates that sum to {defined_sample_rate:.3f}. "
                "The total of all defined sample_rates must be at most 1.0."
            )

        # Assign remaining sample rate to undefined sample rates
        remaining_sample_rate = max(0.0, 1.0 - defined_sample_rate)
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
            raise InvalidParameterValueError(
                f"AttributeCombination has invalid sample_rate={self.sample_rate}. "
                "sample_rate must be between 0 and 1 (inclusive)."
            )
        if not self.combination:
            raise MissingParameterError(
                "AttributeCombination.combination is required. "
                "Provide a dict mapping attribute IDs to their values."
            )

        for key, value in self.combination.items():
            if not key:
                raise InvalidParameterValueError(
                    f"AttributeCombination.combination has an empty key. "
                    f"All keys must be non-empty attribute IDs. Got combination: {self.combination}"
                )
            if not value:
                raise InvalidParameterValueError(
                    f"AttributeCombination.combination['{key}'] has an empty value. "
                    "All values must be non-empty attribute value IDs."
                )

        if len(self.combination.keys()) <= 1:
            raise InvalidParameterValueError(
                f"AttributeCombination.combination must have at least two keys. "
                f"Got {len(self.combination)} key(s): {list(self.combination.keys())}. "
                "A combination requires multiple attributes to be meaningful."
            )


@dataclass
class GeneratedAttributePostprocessingParams:
    """Postprocessing parameters for generated attributes."""

    id: str
    """ID to be used when referencing the postprocessing parameters during synthesis."""

    keep_original_text_attribute: bool = True
    """Whether to keep the original text of the generated attribute.
    If True, the original text will be returned as an attribute.
    If False, the original text will be discarded."""

    cut_prefix: str | None = None
    """Cut off value before and including prefix."""

    cut_suffix: str | None = None
    """Cut off value after and including suffix."""

    regex: str | None = None
    """Regex to be used to pull out the value from the generated text."""

    strip_whitespace: bool = True
    """Whether to strip whitespace from the value."""

    added_prefix: str | None = None
    """Prefix to be added to the value."""

    added_suffix: str | None = None
    """Suffix to be added to the value."""

    def __post_init__(self):
        """Verifies/populates params."""
        if not self.id:
            raise MissingParameterError(
                "GeneratedAttributePostprocessingParams.id is required. "
                "Provide a unique ID for the postprocessed output attribute."
            )

        if self.regex:
            try:
                re.compile(self.regex)
            except Exception as e:
                raise InvalidParameterValueError(
                    f"GeneratedAttributePostprocessingParams '{self.id}' has invalid regex pattern: "
                    f"'{self.regex}'. Error: {e}"
                )


@dataclass
class GeneratedAttribute:
    """Attributes to be generated."""

    id: str
    """ID to be used when referencing the attribute during synthesis."""

    instruction_messages: list[TextMessage]
    """List of messages providing instructions for generating this attribute."""

    postprocessing_params: GeneratedAttributePostprocessingParams | None = None
    """Postprocessing parameters for the generated attribute."""

    def __post_init__(self):
        """Verifies/populates params."""
        if not self.id:
            raise MissingParameterError(
                "GeneratedAttribute.id is required. "
                "Provide a unique ID to reference this generated attribute."
            )
        if not self.instruction_messages:
            raise MissingParameterError(
                f"GeneratedAttribute '{self.id}' requires instruction_messages. "
                "Provide a list of TextMessage objects defining the generation prompt."
            )
        if self.postprocessing_params:
            if self.id == self.postprocessing_params.id:
                raise InvalidParameterValueError(
                    f"GeneratedAttribute.id ('{self.id}') and postprocessing_params.id "
                    f"('{self.postprocessing_params.id}') cannot be the same. "
                    "Use different IDs for the raw and postprocessed outputs."
                )


class TransformationType(str, Enum):
    """Types of transformation strategies."""

    STRING = "string"
    LIST = "list"
    DICT = "dict"
    CHAT = "chat"


@dataclass
class TransformationStrategy:
    """Discriminated union for transformation strategies that works with OmegaConf."""

    type: TransformationType
    """The type of transformation strategy."""

    # For string transformations
    string_transform: str | None = None
    """String transformation template (used when type=STRING)."""

    # For list transformations
    list_transform: list[str] | None = None
    """List of transforms for each element (used when type=LIST)."""

    # For dict transformations
    dict_transform: dict[str, str] | None = None
    """Mapping of dictionary keys to their transforms (used when type=DICT)."""

    # For chat transformations
    chat_transform: TextConversation | None = None
    """Chat transform for chat messages (used when type=CHAT)."""

    def __post_init__(self):
        """Verifies/populates params based on the type."""
        if self.type == TransformationType.STRING:
            if self.string_transform is None or self.string_transform == "":
                raise MissingParameterError(
                    "TransformationStrategy with type=STRING requires string_transform. "
                    "Provide a template string like '{attribute1} - {attribute2}'."
                )
            # Clear other fields
            self.list_transform = None
            self.dict_transform = None
            self.chat_transform = None

        elif self.type == TransformationType.LIST:
            if not self.list_transform or len(self.list_transform) == 0:
                raise MissingParameterError(
                    "TransformationStrategy with type=LIST requires list_transform. "
                    "Provide a list of template strings."
                )
            # Clear other fields
            self.string_transform = None
            self.dict_transform = None
            self.chat_transform = None

        elif self.type == TransformationType.DICT:
            if not self.dict_transform or len(self.dict_transform) == 0:
                raise MissingParameterError(
                    "TransformationStrategy with type=DICT requires dict_transform. "
                    "Provide a dict mapping output keys to template strings."
                )
            # Clear other fields
            self.string_transform = None
            self.list_transform = None
            self.chat_transform = None

        elif self.type == TransformationType.CHAT:
            if not self.chat_transform or len(self.chat_transform.messages) == 0:
                raise MissingParameterError(
                    "TransformationStrategy with type=CHAT requires chat_transform. "
                    "Provide a TextConversation with at least one message."
                )

            messages = self.chat_transform.messages
            for i, message in enumerate(messages):
                content = message.content
                if not isinstance(content, str):
                    raise InvalidParameterValueError(
                        f"chat_transform message at index {i} has non-string content "
                        f"(type: {type(content).__name__}). Message content must be a string."
                    )
                if not content:
                    raise InvalidParameterValueError(
                        f"chat_transform message at index {i} has empty content. "
                        "Message content cannot be empty."
                    )

            # Clear other fields
            self.string_transform = None
            self.list_transform = None
            self.dict_transform = None


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
            raise MissingParameterError(
                "TransformedAttribute.id is required. "
                "Provide a unique ID to reference this transformed attribute."
            )

        if not isinstance(self.transformation_strategy, TransformationStrategy):
            raise InvalidParameterValueError(
                f"TransformedAttribute '{self.id}' has invalid transformation_strategy. "
                f"Expected TransformationStrategy, got {type(self.transformation_strategy).__name__}."
            )

    def get_strategy(self) -> TransformationStrategy:
        """Get the strategy for the transformation."""
        return self.transformation_strategy


@dataclass
class GeneralSynthesisParams(BaseParams):
    """General synthesis parameters."""

    input_data: list[DatasetSource] | None = None
    """Datasets whose rows and columns will be used in synthesis.

    Rows will be enumerated during sampling, and columns can be referenced as attributes
    when generating new attributes."""

    input_documents: list[DocumentSource] | None = None
    """Documents to be used in synthesis.

    Documents will be enumerated during sampling, and both documents and document
    segments can be referenced as attributes when generating new attributes."""

    input_examples: list[ExampleSource] | None = None
    """In-line examples to be used in synthesis.

    Examples will be enumerated during sampling, and attributes can be referenced as
    attributes when generating new attributes."""

    sampled_attributes: list[SampledAttribute] | None = None
    """Attributes to be varied across the dataset.

    Attributes each have a set of possible values which will be randomly sampled
    according to their sample rate. If no sample rate is specified, a uniform
    distribution is used. Sample rates must sum to <= 1.0. Any attributes that do not
    have a sample rate will be given a uniform sample rate equal to whatever remains.

    For example, if there are 3 attributes with sample rates of 0.5, 0.3, and 0.2,
    the total sample rate is 1.0. The first attribute will be sampled 50% of the time,
    the second attribute will be sampled 30% of the time, and the third attribute will
    be sampled 20% of the time. If the last two attributes have no sample rate, they
    will be sampled 25% of the time each as (1.0 - 0.5) / 2 = 0.25."""

    combination_sampling: list[AttributeCombination] | None = None
    """Sampling rates for combinations of attributes.

    Each combination is a dictionary of attribute IDs to their values. The sample rate
    is the probability of sampling this combination. The sample rate of all combinations
    must sum to <= 1.0."""

    generated_attributes: list[GeneratedAttribute] | None = None
    """Attributes to be generated.

    Generated attributes are created by running a chat with the model. The chat is
    specified by a list of messages. The messages will be populated with attribute
    values specific to that data point. The output of the chat is the generated
    attribute.

    For example, if one of the previous attributes is "name", and you use the following
    instruction messages::

        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "How do you pronounce the name {name}?"}
        ]

    Then assuming your data point has a value of "Oumi" for the "name" attribute, the
    chat will be run with the following messages::

        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "How do you pronounce the name Oumi?"}
        ]

    The model's response to these messages will be the value of the "name" attribute
    for that data point."""

    transformed_attributes: list[TransformedAttribute] | None = None
    """Transformation of existing attributes.

    Transformed attributes involve no model interaction and instead are for the
    convenience of transforming parts of your data into a new form.

    For example, if you have "prompt" and "response" attributes, you can create a
    "chat" attribute by transforming the "prompt" and "response" attributes into a
    chat message::

        [
            {"role": "user", "content": "{prompt}"},
            {"role": "assistant", "content": "{response}"}
        ]

    """

    passthrough_attributes: list[str] | None = None
    """When specified, will ONLY pass through these attributes in final output.
    If left unspecified, all attributes are saved. If an attribute is specified in
    passthrough_attributes but doesn't exist, it will be ignored."""

    def _check_attribute_ids(self, attribute_ids: set[str], id: str):
        """Check if the attribute ID is already in the set."""
        if id in attribute_ids:
            raise InvalidParameterValueError(
                f"Duplicate attribute ID: '{id}'. All attribute IDs must be unique "
                "across all data sources, sampled attributes, generated attributes, "
                "and transformed attributes."
            )
        attribute_ids.add(id)

    def _check_dataset_source_attribute_ids(self, all_attribute_ids: set[str]) -> None:
        """Check attribute IDs from dataset sources for uniqueness."""
        if self.input_data is None:
            return

        if len(self.input_data) == 0:
            raise InvalidParameterValueError(
                "GeneralSynthesisParams.input_data cannot be an empty list. "
                "Either remove it entirely (set to None) or add at least one DatasetSource."
            )

        for dataset_source in self.input_data:
            if dataset_source.attribute_map:
                for new_key in dataset_source.attribute_map.values():
                    self._check_attribute_ids(all_attribute_ids, new_key)

    def _check_document_source_attribute_ids(self, all_attribute_ids: set[str]) -> None:
        """Check attribute IDs from document sources for uniqueness."""
        if self.input_documents is None:
            return

        if len(self.input_documents) == 0:
            raise InvalidParameterValueError(
                "GeneralSynthesisParams.input_documents cannot be an empty list. "
                "Either remove it entirely (set to None) or add at least one DocumentSource."
            )

        for document_source in self.input_documents:
            if not document_source.segmentation_params:
                continue

            seg_key = document_source.segmentation_params.id
            self._check_attribute_ids(all_attribute_ids, seg_key)

    def _check_example_source_attribute_ids(self, all_attribute_ids: set[str]) -> None:
        """Check attribute IDs from example sources for uniqueness."""
        if self.input_examples is None:
            return

        if len(self.input_examples) == 0:
            raise InvalidParameterValueError(
                "GeneralSynthesisParams.input_examples cannot be an empty list. "
                "Either remove it entirely (set to None) or add at least one ExampleSource."
            )

        for example_source in self.input_examples:
            example_keys = example_source.examples[0].keys()
            for new_key in example_keys:
                self._check_attribute_ids(all_attribute_ids, new_key)

    def _check_sampled_attribute_ids(self, all_attribute_ids: set[str]) -> None:
        """Check attribute IDs from sampled attributes for uniqueness."""
        if self.sampled_attributes is None:
            return

        if len(self.sampled_attributes) == 0:
            raise InvalidParameterValueError(
                "GeneralSynthesisParams.sampled_attributes cannot be an empty list. "
                "Either remove it entirely (set to None) or add at least one SampledAttribute."
            )

        for sampled_attribute in self.sampled_attributes:
            attribute_id = sampled_attribute.id
            self._check_attribute_ids(all_attribute_ids, attribute_id)

    def _check_generated_attribute_ids(self, all_attribute_ids: set[str]) -> None:
        """Check attribute IDs from generated attributes for uniqueness."""
        if self.generated_attributes is None:
            return

        if len(self.generated_attributes) == 0:
            raise InvalidParameterValueError(
                "GeneralSynthesisParams.generated_attributes cannot be an empty list. "
                "Either remove it entirely (set to None) or add at least one GeneratedAttribute."
            )

        for generated_attribute in self.generated_attributes:
            attribute_id = generated_attribute.id
            self._check_attribute_ids(all_attribute_ids, attribute_id)
            if generated_attribute.postprocessing_params:
                postprocessing_id = generated_attribute.postprocessing_params.id
                self._check_attribute_ids(all_attribute_ids, postprocessing_id)

    def _check_transformed_attribute_ids(self, all_attribute_ids: set[str]) -> None:
        """Check attribute IDs from transformed attributes for uniqueness."""
        if self.transformed_attributes is None:
            return

        if len(self.transformed_attributes) == 0:
            raise InvalidParameterValueError(
                "GeneralSynthesisParams.transformed_attributes cannot be an empty list. "
                "Either remove it entirely (set to None) or add at least one TransformedAttribute."
            )

        for transformed_attribute in self.transformed_attributes:
            attribute_id = transformed_attribute.id
            self._check_attribute_ids(all_attribute_ids, attribute_id)

    def _check_combination_sampling_sample_rates(self) -> None:
        """Validate that the combination sample rates are <= 1.0."""
        if self.combination_sampling is None:
            return

        if len(self.combination_sampling) == 0:
            raise InvalidParameterValueError(
                "GeneralSynthesisParams.combination_sampling cannot be an empty list. "
                "Either remove it entirely (set to None) or add at least one AttributeCombination."
            )

        sample_rates = [
            combination.sample_rate for combination in self.combination_sampling
        ]
        total_rate = sum(sample_rates)
        if total_rate > 1.0:
            raise InvalidParameterValueError(
                f"combination_sampling sample_rates sum to {total_rate:.3f}, "
                f"but must be at most 1.0. Individual rates: {sample_rates}"
            )

    def _check_passthrough_attribute_ids(self) -> None:
        """Validate that passthrough attributes are non-empty when defined."""
        if self.passthrough_attributes is None:
            return

        if len(self.passthrough_attributes) == 0:
            raise InvalidParameterValueError(
                "GeneralSynthesisParams.passthrough_attributes cannot be an empty list. "
                "Either remove it entirely (set to None) to pass through all attributes, "
                "or specify which attribute IDs to include in the output."
            )

    def __post_init__(self):
        """Verifies/populates params."""
        all_attribute_ids = set()
        self._check_dataset_source_attribute_ids(all_attribute_ids)
        self._check_document_source_attribute_ids(all_attribute_ids)
        self._check_example_source_attribute_ids(all_attribute_ids)
        self._check_sampled_attribute_ids(all_attribute_ids)
        self._check_generated_attribute_ids(all_attribute_ids)
        self._check_transformed_attribute_ids(all_attribute_ids)
        self._check_passthrough_attribute_ids()
        self._check_combination_sampling_sample_rates()
