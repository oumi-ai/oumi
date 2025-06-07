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

from unittest.mock import Mock

import pytest

from oumi.core.configs.params.synthesis_params import (
    AttributeCombination,
    ChatTransform,
    DatasetSource,
    DictTransform,
    DocumentSegmentationParams,
    DocumentSource,
    ExampleSource,
    GeneralSynthesisParams,
    GeneratedAttribute,
    ListTransform,
    PermutableAttribute,
    PermutableAttributeValue,
    SegmentationStrategy,
    TransformedAttribute,
)
from oumi.core.types.conversation import Conversation, Message, Role


def test_dataset_source_valid():
    # Test valid file types
    valid_paths = [
        "data.jsonl",
        "data.json",
        "data.csv",
        "data.parquet",
        "data.tsv",
    ]
    for path in valid_paths:
        ds = DatasetSource(path=path)
        assert ds.path == path

    ds = DatasetSource(path="data.jsonl", attribute_map={"original_name": "new_name"})
    assert ds.attribute_map == {"original_name": "new_name"}


def test_dataset_source_invalid():
    with pytest.raises(ValueError, match="DatasetSource.path cannot be empty."):
        DatasetSource(path="")

    with pytest.raises(ValueError, match="Unsupported dataset file type:"):
        DatasetSource(path="data.txt")


def test_document_segmentation_params_valid():
    # Test default values
    params = DocumentSegmentationParams(id="test")
    assert params.id == "test"
    assert params.segmentation_strategy == SegmentationStrategy.TOKENS
    assert params.segment_length == 2048
    assert params.segment_overlap == 0
    assert params.keep_original_text is False

    # Test custom values
    params = DocumentSegmentationParams(
        id="test",
        segmentation_strategy=SegmentationStrategy.TOKENS,
        segment_length=1024,
        segment_overlap=256,
        keep_original_text=True,
    )
    assert params.segment_length == 1024
    assert params.segment_overlap == 256
    assert params.keep_original_text is True


def test_document_segmentation_params_invalid():
    # Test overlap >= length
    with pytest.raises(
        ValueError, match="Segment overlap must be less than segment length"
    ):
        DocumentSegmentationParams(id="test", segment_length=100, segment_overlap=100)

    # Test negative overlap
    with pytest.raises(ValueError, match="Segment overlap must be non-negative"):
        DocumentSegmentationParams(id="test", segment_overlap=-1)

    # Test non-positive length
    with pytest.raises(ValueError, match="Segment length must be positive"):
        DocumentSegmentationParams(id="test", segment_length=0)


def test_document_source_valid():
    # Test minimal valid case
    doc = DocumentSource(path="doc.txt", id="test")
    assert doc.path == "doc.txt"
    assert doc.id == "test"
    assert doc.segmentation_params is None

    # Test with segmentation params
    seg_params = DocumentSegmentationParams(id="seg_test")
    doc = DocumentSource(path="doc.txt", id="test", segmentation_params=seg_params)
    assert doc.segmentation_params == seg_params


def test_document_source_invalid():
    # Test empty path
    with pytest.raises(ValueError, match="DocumentSource.path cannot be empty"):
        DocumentSource(path="", id="test")

    # Test empty id
    with pytest.raises(ValueError, match="DocumentSource.id cannot be empty"):
        DocumentSource(path="doc.txt", id="")


def test_example_source_valid():
    # Test valid examples
    examples = [
        {"field1": "value1", "field2": "value2"},
        {"field1": "value3", "field2": "value4"},
    ]
    es = ExampleSource(examples=examples)
    assert es.examples == examples


def test_example_source_invalid():
    # Test empty examples
    with pytest.raises(ValueError, match="ExampleSource.examples cannot be empty"):
        ExampleSource(examples=[])

    # Test inconsistent keys
    with pytest.raises(ValueError, match="All examples must have the same keys"):
        ExampleSource(
            examples=[
                {"field1": "value1", "field2": "value2"},
                {"field1": "value3", "field3": "value4"},
            ]
        )


def test_permutable_attribute_value_valid():
    # Test minimal valid case
    value = PermutableAttributeValue(
        id="test", value="test_value", description="test description"
    )
    assert value.id == "test"
    assert value.value == "test_value"
    assert value.description == "test description"
    assert value.sample_rate is None

    # Test with sample rate
    value = PermutableAttributeValue(
        id="test", value="test_value", description="test description", sample_rate=0.5
    )
    assert value.sample_rate == 0.5


def test_permutable_attribute_value_invalid():
    # Test empty id
    with pytest.raises(ValueError, match="PermutableAttributeValue.id cannot be empty"):
        PermutableAttributeValue(id="", value="test", description="test")

    # Test empty value
    with pytest.raises(
        ValueError, match="PermutableAttributeValue.value cannot be empty"
    ):
        PermutableAttributeValue(id="test", value="", description="test")

    # Test empty description
    with pytest.raises(
        ValueError, match="PermutableAttributeValue.description cannot be empty"
    ):
        PermutableAttributeValue(id="test", value="test", description="")

    # Test invalid sample rate
    with pytest.raises(
        ValueError, match="PermutableAttributeValue.sample_rate must be between 0 and 1"
    ):
        PermutableAttributeValue(
            id="test", value="test", description="test", sample_rate=1.5
        )


def test_permutable_attribute_valid():
    # Test valid case with uniform sampling
    values = [
        PermutableAttributeValue(id="v1", value="value1", description="desc1"),
        PermutableAttributeValue(id="v2", value="value2", description="desc2"),
    ]
    attr = PermutableAttribute(
        id="test",
        attribute="test_attr",
        description="test description",
        possible_values=values,
    )
    assert attr.id == "test"
    assert attr.attribute == "test_attr"
    assert attr.description == "test description"
    assert len(attr.possible_values) == 2
    # Check that sample rates were normalized
    assert all(v.sample_rate == 0.5 for v in attr.possible_values)

    # Test with explicit sample rates
    values = [
        PermutableAttributeValue(
            id="v1", value="value1", description="desc1", sample_rate=0.3
        ),
        PermutableAttributeValue(
            id="v2", value="value2", description="desc2", sample_rate=0.7
        ),
    ]
    attr = PermutableAttribute(
        id="test",
        attribute="test_attr",
        description="test description",
        possible_values=values,
    )
    assert attr.possible_values[0].sample_rate == 0.3
    assert attr.possible_values[1].sample_rate == 0.7


def test_permutable_attribute_invalid():
    # Test empty id
    with pytest.raises(ValueError, match="PermutableAttribute.id cannot be empty"):
        PermutableAttribute(
            id="",
            attribute="test",
            description="test",
            possible_values=[
                PermutableAttributeValue(id="v1", value="value1", description="desc1")
            ],
        )

    # Test empty attribute
    with pytest.raises(
        ValueError, match="PermutableAttribute.attribute cannot be empty"
    ):
        PermutableAttribute(
            id="test",
            attribute="",
            description="test",
            possible_values=[
                PermutableAttributeValue(id="v1", value="value1", description="desc1")
            ],
        )

    # Test empty description
    with pytest.raises(
        ValueError, match="PermutableAttribute.description cannot be empty"
    ):
        PermutableAttribute(
            id="test",
            attribute="test",
            description="",
            possible_values=[
                PermutableAttributeValue(id="v1", value="value1", description="desc1")
            ],
        )

    # Test empty possible values
    with pytest.raises(
        ValueError, match="PermutableAttribute.possible_values cannot be empty"
    ):
        PermutableAttribute(
            id="test", attribute="test", description="test", possible_values=[]
        )

    # Test duplicate value ids
    with pytest.raises(
        ValueError, match="PermutableAttribute.possible_values must have unique IDs"
    ):
        PermutableAttribute(
            id="test",
            attribute="test",
            description="test",
            possible_values=[
                PermutableAttributeValue(id="v1", value="value1", description="desc1"),
                PermutableAttributeValue(id="v1", value="value2", description="desc2"),
            ],
        )

    # Test sample rates sum > 1
    with pytest.raises(
        ValueError, match="PermutableAttribute.possible_values must sum to 1.0"
    ):
        PermutableAttribute(
            id="test",
            attribute="test",
            description="test",
            possible_values=[
                PermutableAttributeValue(
                    id="v1", value="value1", description="desc1", sample_rate=0.6
                ),
                PermutableAttributeValue(
                    id="v2", value="value2", description="desc2", sample_rate=0.6
                ),
            ],
        )


def test_attribute_combination_valid():
    # Test valid case with two keys
    combo = AttributeCombination(
        combination={"attr1": "value1", "attr2": "value2"}, sample_rate=0.5
    )
    assert combo.combination == {"attr1": "value1", "attr2": "value2"}
    assert combo.sample_rate == 0.5

    # Test valid case with more than two keys
    combo = AttributeCombination(
        combination={"attr1": "value1", "attr2": "value2", "attr3": "value3"},
        sample_rate=0.5,
    )
    assert combo.combination == {
        "attr1": "value1",
        "attr2": "value2",
        "attr3": "value3",
    }
    assert combo.sample_rate == 0.5


def test_attribute_combination_invalid():
    # Test invalid sample rate
    with pytest.raises(
        ValueError, match="AttributeCombination.sample_rate must be between 0 and 1"
    ):
        AttributeCombination(combination={"attr1": "value1"}, sample_rate=1.5)

    # Test single key combination
    with pytest.raises(
        ValueError, match="AttributeCombination.combination must have at least two keys"
    ):
        AttributeCombination(combination={"attr1": "value1"}, sample_rate=0.5)

    # Test empty combination
    with pytest.raises(
        ValueError, match="AttributeCombination.combination cannot be empty"
    ):
        AttributeCombination(combination={}, sample_rate=0.5)

    # Test empty key
    with pytest.raises(
        ValueError, match="AttributeCombination.combination key cannot be empty"
    ):
        AttributeCombination(combination={"": "value1"}, sample_rate=0.5)

    # Test empty value
    with pytest.raises(
        ValueError, match="AttributeCombination.combination value cannot be empty"
    ):
        AttributeCombination(combination={"attr1": ""}, sample_rate=0.5)


def test_generated_attribute_valid():
    # Test valid case
    messages = [
        Message(role=Role.SYSTEM, content="System message"),
        Message(role=Role.USER, content="User message"),
    ]
    conversation = Conversation(messages=messages)

    attr = GeneratedAttribute(id="test", instruction_messages=conversation)
    assert attr.id == "test"
    assert attr.instruction_messages == conversation
    assert attr.postprocessing_params is None


def test_generated_attribute_invalid():
    # Test empty id
    messages = [Message(role=Role.SYSTEM, content="System message")]
    conversation = Conversation(messages=messages)

    with pytest.raises(ValueError, match="GeneratedAttribute.id cannot be empty"):
        GeneratedAttribute(id="", instruction_messages=conversation)

    # Test None instruction messages
    with pytest.raises(
        ValueError, match="GeneratedAttribute.instruction_messages cannot be empty"
    ):
        GeneratedAttribute(
            id="test",
            instruction_messages=None,  # type: ignore
        )


def test_list_transform_valid():
    # Test valid case
    transform = ListTransform(element_transforms=["transform1", "transform2"])
    assert transform.element_transforms == ["transform1", "transform2"]


def test_list_transform_invalid():
    # Test empty transforms
    with pytest.raises(
        ValueError, match="ListTransform.element_transforms cannot be empty"
    ):
        ListTransform(element_transforms=[])


def test_dict_transform_valid():
    # Test valid case
    transform = DictTransform(transforms={"key1": "transform1", "key2": "transform2"})
    assert transform.transforms == {"key1": "transform1", "key2": "transform2"}


def test_dict_transform_invalid():
    # Test empty transforms
    with pytest.raises(ValueError, match="DictTransform.transforms cannot be empty"):
        DictTransform(transforms={})


def test_chat_transform_valid():
    # Test valid case
    messages = [
        Message(role=Role.SYSTEM, content="System message"),
        Message(role=Role.USER, content="User message"),
    ]
    conversation = Conversation(messages=messages)

    transform = ChatTransform(transforms=conversation)
    assert transform.transforms == conversation


def test_chat_transform_invalid():
    # Test empty messages list
    empty_conversation = Conversation(messages=[])
    with pytest.raises(
        ValueError, match="ChatTransform.transforms must have at least one message"
    ):
        ChatTransform(transforms=empty_conversation)

    # Test non-string message content
    mock_message = Mock(spec=Message)
    mock_message.content = 123
    mock_message.role = Role.SYSTEM
    conversation = Conversation(messages=[mock_message])
    with pytest.raises(
        ValueError, match="ChatTransform.transforms message content must be a string"
    ):
        ChatTransform(transforms=conversation)

    # Test empty message content
    conversation = Conversation(messages=[Message(role=Role.SYSTEM, content="")])
    with pytest.raises(
        ValueError, match="ChatTransform.transforms message content cannot be empty"
    ):
        ChatTransform(transforms=conversation)


def test_transformed_attribute_valid():
    # Test valid case
    attr = TransformedAttribute(
        id="test",
        transformation_strategy=ListTransform(element_transforms=["transform1"]),
    )
    assert attr.id == "test"
    assert isinstance(attr.transformation_strategy, ListTransform)


def test_transformed_attribute_invalid():
    # Test empty id
    with pytest.raises(ValueError, match="TransformedAttribute.id cannot be empty"):
        TransformedAttribute(
            id="",
            transformation_strategy=ListTransform(element_transforms=["transform1"]),
        )


def test_general_synthesis_params_valid():
    # Test minimal valid case
    params = GeneralSynthesisParams()
    assert params.input_data is None
    assert params.input_documents is None
    assert params.input_examples is None
    assert params.permutable_attributes is None
    assert params.combination_sampling is None
    assert params.generated_attributes is None
    assert params.transformed_attributes is None
    assert params.passthrough_attributes is None

    # Test with all optional fields
    params = GeneralSynthesisParams(
        input_data=[
            DatasetSource(path="data.jsonl", attribute_map={"attr1": "new_attr1"})
        ],
        input_documents=[DocumentSource(path="doc.txt", id="doc1")],
        input_examples=[ExampleSource(examples=[{"field1": "value1"}])],
        permutable_attributes=[
            PermutableAttribute(
                id="attr1",
                attribute="test_attr",
                description="test desc",
                possible_values=[
                    PermutableAttributeValue(
                        id="v1", value="value1", description="desc1", sample_rate=0.5
                    )
                ],
            )
        ],
        combination_sampling=[
            AttributeCombination(
                combination={"attr1": "value1", "attr2": "value2"}, sample_rate=0.5
            )
        ],
        generated_attributes=[
            GeneratedAttribute(
                id="gen1",
                instruction_messages=Conversation(
                    messages=[Message(role=Role.SYSTEM, content="System message")]
                ),
            )
        ],
        transformed_attributes=[
            TransformedAttribute(
                id="trans1",
                transformation_strategy=ListTransform(
                    element_transforms=["transform1"]
                ),
            )
        ],
        passthrough_attributes=["attr1", "attr2"],
    )
    assert params.combination_sampling is not None
    assert len(params.combination_sampling) == 1
    assert sum(c.sample_rate for c in params.combination_sampling) <= 1.0

    # Test valid combination sampling rates sum <= 1.0
    params = GeneralSynthesisParams(
        combination_sampling=[
            AttributeCombination(
                combination={"attr1": "val1", "attr2": "val2"}, sample_rate=0.5
            ),
            AttributeCombination(
                combination={"attr1": "val3", "attr2": "val4"}, sample_rate=0.4
            ),
        ]
    )
    assert params.combination_sampling is not None
    assert len(params.combination_sampling) == 2
    assert sum(c.sample_rate for c in params.combination_sampling) <= 1.0


def test_general_synthesis_params_invalid():
    # Test empty lists
    with pytest.raises(
        ValueError, match="GeneralSynthesisParams.input_data cannot be empty."
    ):
        GeneralSynthesisParams(input_data=[])

    with pytest.raises(
        ValueError, match="GeneralSynthesisParams.input_documents cannot be empty."
    ):
        GeneralSynthesisParams(input_documents=[])

    with pytest.raises(
        ValueError, match="GeneralSynthesisParams.input_examples cannot be empty."
    ):
        GeneralSynthesisParams(input_examples=[])

    with pytest.raises(
        ValueError,
        match="GeneralSynthesisParams.permutable_attributes cannot be empty.",
    ):
        GeneralSynthesisParams(permutable_attributes=[])

    with pytest.raises(
        ValueError, match="GeneralSynthesisParams.combination_sampling cannot be empty."
    ):
        GeneralSynthesisParams(combination_sampling=[])

    with pytest.raises(
        ValueError, match="GeneralSynthesisParams.generated_attributes cannot be empty."
    ):
        GeneralSynthesisParams(generated_attributes=[])

    with pytest.raises(
        ValueError,
        match="GeneralSynthesisParams.transformed_attributes cannot be empty.",
    ):
        GeneralSynthesisParams(transformed_attributes=[])

    with pytest.raises(
        ValueError,
        match="GeneralSynthesisParams.passthrough_attributes cannot be empty.",
    ):
        GeneralSynthesisParams(passthrough_attributes=[])

    # Test duplicate attribute IDs
    with pytest.raises(
        ValueError, match="GeneralSynthesisParams contains duplicate attribute IDs"
    ):
        GeneralSynthesisParams(
            input_data=[
                DatasetSource(path="data1.jsonl", attribute_map={"attr1": "new_attr1"}),
                DatasetSource(path="data2.jsonl", attribute_map={"attr1": "new_attr1"}),
            ]
        )

    # Test combination sampling rates sum > 1.0
    with pytest.raises(
        ValueError,
        match="GeneralSynthesisParams.combination_sampling sample rates must be "
        "less than or equal to 1.0",
    ):
        GeneralSynthesisParams(
            combination_sampling=[
                AttributeCombination(
                    combination={"attr1": "val1", "attr2": "val2"}, sample_rate=0.6
                ),
                AttributeCombination(
                    combination={"attr1": "val3", "attr2": "val4"}, sample_rate=0.5
                ),
            ]
        )
