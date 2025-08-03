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
from unittest.mock import patch

import pytest

from oumi.core.configs.params.synthesis_params import (
    GeneralSynthesisParams,
    TextConversation,
    TextMessage,
    TransformationStrategy,
    TransformationType,
    TransformedAttribute,
)
from oumi.core.synthesis.attribute_transformation import (
    AttributeTransformer,
    SampleValue,
)
from oumi.core.types.conversation import Role


@pytest.fixture
def basic_samples() -> list[dict[str, SampleValue]]:
    """Basic samples for testing transformations."""
    return [
        {"name": "Alice", "age": "25", "city": "New York"},
        {"name": "Bob", "age": "30", "city": "San Francisco"},
        {"name": "Charlie", "age": "35", "city": "Chicago"},
    ]


def test_transform_with_no_transformed_attributes(basic_samples):
    """Test transform with no transformed attributes returns original samples."""
    params_with_no_transformed_attributes = GeneralSynthesisParams()
    transformer = AttributeTransformer(params_with_no_transformed_attributes)
    result = transformer.transform(basic_samples)

    # Should return the same samples unchanged
    assert result == basic_samples


def test_transform_with_empty_samples():
    """Test transform with empty samples list."""
    params = GeneralSynthesisParams(
        transformed_attributes=[
            TransformedAttribute(
                id="greeting",
                transformation_strategy=TransformationStrategy(
                    type=TransformationType.STRING, string_transform="Hello {name}!"
                ),
            )
        ]
    )
    transformer = AttributeTransformer(params)
    result = transformer.transform([])

    assert result == []


def test_transform_string_strategy(basic_samples):
    """Test transform with string transformation strategy."""
    params = GeneralSynthesisParams(
        transformed_attributes=[
            TransformedAttribute(
                id="greeting",
                transformation_strategy=TransformationStrategy(
                    type=TransformationType.STRING, string_transform="Hello {name}!"
                ),
            )
        ]
    )
    transformer = AttributeTransformer(params)
    results = transformer.transform(basic_samples)

    expected_results = [
        {"greeting": "Hello Alice!", "name": "Alice", "age": "25", "city": "New York"},
        {"greeting": "Hello Bob!", "name": "Bob", "age": "30", "city": "San Francisco"},
        {
            "greeting": "Hello Charlie!",
            "name": "Charlie",
            "age": "35",
            "city": "Chicago",
        },
    ]

    assert results == expected_results


def test_transform_string_strategy_multiple_placeholders(basic_samples):
    """Test transform with string strategy using multiple placeholders."""
    params = GeneralSynthesisParams(
        transformed_attributes=[
            TransformedAttribute(
                id="bio",
                transformation_strategy=TransformationStrategy(
                    type=TransformationType.STRING,
                    string_transform="My name is {name} and I am {age} years old.",
                ),
            )
        ]
    )
    transformer = AttributeTransformer(params)
    result = transformer.transform(basic_samples)

    assert len(result) == 3
    assert result[0]["bio"] == "My name is Alice and I am 25 years old."
    assert result[1]["bio"] == "My name is Bob and I am 30 years old."
    assert result[2]["bio"] == "My name is Charlie and I am 35 years old."


def test_transform_list_strategy(basic_samples):
    """Test transform with list transformation strategy."""
    list_transform = TransformationStrategy(
        type=TransformationType.LIST,
        list_transform=[
            "Hello {name}!",
            "You are {age} years old.",
            "You live in {city}.",
        ],
    )
    params = GeneralSynthesisParams(
        transformed_attributes=[
            TransformedAttribute(id="facts", transformation_strategy=list_transform)
        ]
    )
    transformer = AttributeTransformer(params)
    result = transformer.transform(basic_samples)

    assert len(result) == 3
    assert result[0]["facts"] == [
        "Hello Alice!",
        "You are 25 years old.",
        "You live in New York.",
    ]
    assert result[1]["facts"] == [
        "Hello Bob!",
        "You are 30 years old.",
        "You live in San Francisco.",
    ]
    assert result[2]["facts"] == [
        "Hello Charlie!",
        "You are 35 years old.",
        "You live in Chicago.",
    ]


def test_transform_dict_strategy(basic_samples):
    """Test transform with dict transformation strategy."""
    dict_transform = TransformationStrategy(
        type=TransformationType.DICT,
        dict_transform={
            "greeting": "Hello {name}!",
            "age_info": "Age: {age}",
            "location": "Lives in {city}",
        },
    )
    params = GeneralSynthesisParams(
        transformed_attributes=[
            TransformedAttribute(
                id="person_info", transformation_strategy=dict_transform
            )
        ]
    )
    transformer = AttributeTransformer(params)
    result = transformer.transform(basic_samples)

    assert len(result) == 3
    assert result[0]["person_info"] == {
        "greeting": "Hello Alice!",
        "age_info": "Age: 25",
        "location": "Lives in New York",
    }
    assert result[1]["person_info"] == {
        "greeting": "Hello Bob!",
        "age_info": "Age: 30",
        "location": "Lives in San Francisco",
    }
    assert result[2]["person_info"] == {
        "greeting": "Hello Charlie!",
        "age_info": "Age: 35",
        "location": "Lives in Chicago",
    }


def test_transform_chat_strategy(basic_samples):
    """Test transform with chat transformation strategy."""
    chat_conversation = TextConversation(
        messages=[
            TextMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
            TextMessage(
                role=Role.USER, content="Tell me about {name} who is {age} years old."
            ),
            TextMessage(
                role=Role.ASSISTANT, content="I can help you learn about {name}."
            ),
        ],
        metadata={"name": "{name}"},
        conversation_id="test-conversation",
    )
    chat_transform = TransformationStrategy(
        type=TransformationType.CHAT, chat_transform=chat_conversation
    )
    params = GeneralSynthesisParams(
        transformed_attributes=[
            TransformedAttribute(
                id="conversation", transformation_strategy=chat_transform
            )
        ]
    )
    transformer = AttributeTransformer(params)
    result = transformer.transform(basic_samples)

    assert len(result) == 3

    # Check first sample
    conv = result[0]["conversation"]
    assert isinstance(conv, dict)
    assert len(conv["messages"]) == 3
    assert conv["messages"][0]["role"] == Role.SYSTEM
    assert conv["messages"][0]["content"] == "You are a helpful assistant."
    assert conv["messages"][1]["role"] == Role.USER
    assert conv["messages"][1]["content"] == "Tell me about Alice who is 25 years old."
    assert conv["messages"][2]["role"] == Role.ASSISTANT
    assert conv["messages"][2]["content"] == "I can help you learn about Alice."
    assert conv["conversation_id"] == "test-conversation"
    assert conv["metadata"] == {"name": "Alice"}


def test_transform_chat_strategy_with_auto_generated_id(basic_samples):
    """Test transform with chat strategy that auto-generates conversation ID."""
    chat_conversation = TextConversation(
        messages=[TextMessage(role=Role.USER, content="Hello {name}!")],
        conversation_id=None,  # No conversation ID provided
    )
    chat_transform = TransformationStrategy(
        type=TransformationType.CHAT, chat_transform=chat_conversation
    )
    params = GeneralSynthesisParams(
        transformed_attributes=[
            TransformedAttribute(id="chat_attr", transformation_strategy=chat_transform)
        ]
    )
    transformer = AttributeTransformer(params)

    with patch("uuid.uuid4") as mock_uuid:
        mock_uuid.return_value = uuid.UUID("12345678-1234-5678-1234-567812345678")
        result = transformer.transform(basic_samples)

    assert len(result) == 3
    conv = result[0]["chat_attr"]
    assert isinstance(conv, dict)
    assert conv["conversation_id"] == "chat_attr-12345678-1234-5678-1234-567812345678"


def test_transform_multiple_attributes(basic_samples):
    """Test transform with multiple transformed attributes."""
    params = GeneralSynthesisParams(
        transformed_attributes=[
            TransformedAttribute(
                id="greeting",
                transformation_strategy=TransformationStrategy(
                    type=TransformationType.STRING, string_transform="Hello {name}!"
                ),
            ),
            TransformedAttribute(
                id="age_info",
                transformation_strategy=TransformationStrategy(
                    type=TransformationType.STRING, string_transform="Age: {age}"
                ),
            ),
            TransformedAttribute(
                id="location_list",
                transformation_strategy=TransformationStrategy(
                    type=TransformationType.LIST,
                    list_transform=["City: {city}", "Name: {name}"],
                ),
            ),
        ]
    )
    transformer = AttributeTransformer(params)
    result = transformer.transform(basic_samples)

    assert len(result) == 3

    # Check first sample
    assert result[0]["greeting"] == "Hello Alice!"
    assert result[0]["age_info"] == "Age: 25"
    assert result[0]["location_list"] == ["City: New York", "Name: Alice"]

    # Original attributes should still be present
    assert result[0]["name"] == "Alice"
    assert result[0]["age"] == "25"
    assert result[0]["city"] == "New York"


def test_transform_with_non_string_values_in_sample():
    """Test transform with samples containing non-string values."""
    samples = [  # type: ignore
        {
            "name": "Alice",
            "age": "25",
            "scores": [95, 87, 92],
            "metadata": {"active": True},
        },
        {
            "name": "Bob",
            "age": "30",
            "scores": [88, 91, 85],
            "metadata": {"active": False},
        },
    ]
    params = GeneralSynthesisParams(
        transformed_attributes=[
            TransformedAttribute(
                id="greeting",
                transformation_strategy=TransformationStrategy(
                    type=TransformationType.STRING, string_transform="Hello {name}!"
                ),
            )
        ]
    )
    transformer = AttributeTransformer(params)
    result = transformer.transform(samples)  # type: ignore

    assert len(result) == 2
    assert result[0]["greeting"] == "Hello Alice!"
    assert result[1]["greeting"] == "Hello Bob!"

    # Non-string values should be preserved
    assert result[0]["scores"] == [95, 87, 92]
    assert result[0]["metadata"] == {"active": True}
    assert result[1]["scores"] == [88, 91, 85]
    assert result[1]["metadata"] == {"active": False}


def test_transform_preserves_original_sample_order():
    """Test that transform preserves the order of samples."""
    samples: list[dict[str, SampleValue]] = [
        {"name": f"Person{i}", "id": str(i)} for i in range(100)
    ]
    params = GeneralSynthesisParams(
        transformed_attributes=[
            TransformedAttribute(
                id="greeting",
                transformation_strategy=TransformationStrategy(
                    type=TransformationType.STRING, string_transform="Hello {name}!"
                ),
            )
        ]
    )
    transformer = AttributeTransformer(params)
    result = transformer.transform(samples)

    assert len(result) == 100
    for i, sample in enumerate(result):
        assert sample["name"] == f"Person{i}"
        assert sample["id"] == str(i)
        assert sample["greeting"] == f"Hello Person{i}!"
