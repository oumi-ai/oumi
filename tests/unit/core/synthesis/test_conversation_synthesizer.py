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

"""Tests for ConversationSynthesizer."""

import json
from unittest.mock import Mock, patch

import pytest

from oumi.core.configs.inference_config import InferenceConfig
from oumi.core.configs.inference_engine_type import InferenceEngineType
from oumi.core.configs.params.model_params import ModelParams
from oumi.core.configs.params.remote_params import RemoteParams
from oumi.core.configs.params.synthesis_params import (
    GeneralSynthesisParams,
    MultiTurnAttribute,
    SampledAttribute,
    SampledAttributeValue,
)
from oumi.core.synthesis.conversation_synthesizer import (
    ConversationSynthesizer,
    _clean_json_output,
)
from oumi.core.types.conversation import Conversation, Message, Role


@pytest.fixture
def mock_inference_config():
    """Create a mock inference config."""
    mock = Mock(spec=InferenceConfig)
    mock.engine = InferenceEngineType.NATIVE
    mock.model = Mock(spec=ModelParams)
    mock.remote_params = Mock(spec=RemoteParams)
    return mock


@pytest.fixture
def mock_general_synthesis_params():
    """Create mock GeneralSynthesisParams for testing."""
    return GeneralSynthesisParams(
        sampled_attributes=[
            SampledAttribute(
                id="style",
                name="Writing Style",
                description="The style of writing to use",
                possible_values=[
                    SampledAttributeValue(
                        id="formal",
                        name="Formal",
                        description="A formal writing style",
                    ),
                    SampledAttributeValue(
                        id="casual",
                        name="Casual",
                        description="A casual writing style",
                    ),
                ],
            ),
        ],
    )


@pytest.fixture
def mock_multiturn_attribute():
    """Create a mock multiturn attribute for testing.

    Uses placeholders: {customer_type}, {issue}
    """
    return MultiTurnAttribute(
        id="multiturn_conversation",
        min_turns=2,
        max_turns=16,
        role_instruction_messages={
            Role.USER: "You are a {customer_type} customer with issue: {issue}.",
            Role.ASSISTANT: "You are a helpful support agent.",
        },
        conversation_planner="Plan a {target_turns}-turn conversation about {issue}.",
        output_system_prompt="This is a customer support conversation about {issue}.",
    )


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_synthesize_returns_list_of_dicts(
    mock_build_inference_engine,
    mock_general_synthesis_params,
    mock_multiturn_attribute,
    mock_inference_config,
):
    """Test that synthesize returns a list of dictionaries."""
    mock_inference_engine = Mock()
    mock_build_inference_engine.return_value = mock_inference_engine

    def infer_side_effect(conversations, **kwargs):
        return [
            Conversation(messages=[Message(role=Role.ASSISTANT, content="Response")])
            for _ in conversations
        ]

    mock_inference_engine.infer.side_effect = infer_side_effect

    synthesizer = ConversationSynthesizer(
        mock_general_synthesis_params,
        mock_inference_config,
    )

    samples = [
        {"customer_type": "frustrated", "issue": "billing problem"},
        {"customer_type": "friendly", "issue": "product question"},
    ]
    result = synthesizer.synthesize(samples, mock_multiturn_attribute)

    assert isinstance(result, list)
    assert len(result) == len(samples)
    plan_key = f"{mock_multiturn_attribute.id}_plan"
    for item in result:
        assert isinstance(item, dict)
        assert mock_multiturn_attribute.id in item
        assert plan_key in item


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_synthesize_with_empty_samples(
    mock_build_inference_engine,
    mock_general_synthesis_params,
    mock_multiturn_attribute,
    mock_inference_config,
):
    """Test that synthesize returns empty list for empty samples."""
    mock_build_inference_engine.return_value = Mock()

    synthesizer = ConversationSynthesizer(
        mock_general_synthesis_params,
        mock_inference_config,
    )

    result = synthesizer.synthesize([], mock_multiturn_attribute)

    assert result == []


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_output_system_prompt_prepended_to_conversation(
    mock_build_inference_engine,
    mock_general_synthesis_params,
    mock_multiturn_attribute,
    mock_inference_config,
):
    """Test that output_system_prompt is formatted and prepended to conversation."""
    mock_inference_engine = Mock()
    mock_build_inference_engine.return_value = mock_inference_engine
    mock_inference_engine.infer.return_value = [
        Conversation(messages=[Message(role=Role.ASSISTANT, content="Response")])
    ]

    synthesizer = ConversationSynthesizer(
        mock_general_synthesis_params,
        mock_inference_config,
    )

    samples = [{"customer_type": "frustrated", "issue": "billing problem"}]
    result = synthesizer.synthesize(samples, mock_multiturn_attribute)

    assert len(result) == 1
    record = result[0]
    assert record is not None
    conversation = record[mock_multiturn_attribute.id]
    assert isinstance(conversation, dict)
    messages = conversation["messages"]

    # First message should be the formatted output system message
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == (
        "This is a customer support conversation about billing problem."
    )

    # Subsequent messages should be the conversation history
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_conversation_plan_uses_namespaced_key(
    mock_build_inference_engine,
    mock_general_synthesis_params,
    mock_multiturn_attribute,
    mock_inference_config,
):
    """Test that conversation_plan is stored under a namespaced key."""
    mock_inference_engine = Mock()
    mock_build_inference_engine.return_value = mock_inference_engine
    mock_inference_engine.infer.return_value = [
        Conversation(messages=[Message(role=Role.ASSISTANT, content="Response")])
    ]

    synthesizer = ConversationSynthesizer(
        mock_general_synthesis_params,
        mock_inference_config,
    )

    samples = [{"customer_type": "friendly", "issue": "product question"}]
    result = synthesizer.synthesize(samples, mock_multiturn_attribute)

    assert len(result) == 1
    # Plan should be returned under a namespaced key: {attribute_id}_plan
    plan_key = f"{mock_multiturn_attribute.id}_plan"
    record = result[0]
    assert record is not None
    assert plan_key in record


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_synthesize_without_conversation_planner(
    mock_build_inference_engine,
    mock_inference_config,
):
    """Test that synthesize works without a user planner override."""
    mock_inference_engine = Mock()
    mock_build_inference_engine.return_value = mock_inference_engine
    mock_inference_engine.infer.return_value = [
        Conversation(messages=[Message(role=Role.ASSISTANT, content="Response")])
    ]

    multiturn_attr = MultiTurnAttribute(
        id="test_conversation",
        min_turns=2,
        max_turns=2,
        role_instruction_messages={
            Role.USER: "You are a user in a conversation.",
            Role.ASSISTANT: "You are an assistant in a conversation.",
        },
    )

    synthesizer = ConversationSynthesizer(
        GeneralSynthesisParams(),
        mock_inference_config,
    )

    result = synthesizer.synthesize([{}], multiturn_attr)

    assert len(result) == 1
    plan_key = f"{multiturn_attr.id}_plan"
    record = result[0]
    assert record is not None
    assert plan_key in record
    assert multiturn_attr.id in record


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_turn_order_is_user_then_assistant(
    mock_build_inference_engine,
    mock_inference_config,
):
    """Test that turn order is [USER, ASSISTANT]."""
    mock_inference_engine = Mock()
    mock_build_inference_engine.return_value = mock_inference_engine
    mock_inference_engine.infer.return_value = [
        Conversation(messages=[Message(role=Role.ASSISTANT, content="Response")])
    ]

    multiturn_attr = MultiTurnAttribute(
        id="test_conversation",
        min_turns=2,
        max_turns=2,
        role_instruction_messages={
            Role.USER: "You are a user",
            Role.ASSISTANT: "You are an assistant",
        },
    )

    synthesizer = ConversationSynthesizer(
        GeneralSynthesisParams(),
        mock_inference_config,
    )

    result = synthesizer.synthesize([{}], multiturn_attr)

    record = result[0]
    assert record is not None
    conversation = record[multiturn_attr.id]
    assert isinstance(conversation, dict)
    messages = conversation["messages"]

    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_format_persona_replaces_placeholders(
    mock_build_inference_engine,
    mock_general_synthesis_params,
    mock_multiturn_attribute,
    mock_inference_config,
):
    """Test that _format_instructions replaces placeholders with sample values."""
    mock_build_inference_engine.return_value = Mock()

    synthesizer = ConversationSynthesizer(
        mock_general_synthesis_params,
        mock_inference_config,
    )

    sample = {"customer_type": "frustrated", "issue": "billing"}
    result = synthesizer._format_persona(
        sample,
        "You are a {customer_type} customer with issue: {issue}.",
        [],
    )

    assert isinstance(result, Message)
    assert result.role == Role.SYSTEM
    assert result.content == "You are a frustrated customer with issue: billing."


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_build_role_context_formats_personas(
    mock_build_inference_engine,
    mock_inference_config,
):
    """Test that _build_role_context formats personas with sample values."""
    mock_build_inference_engine.return_value = Mock()

    synthesizer = ConversationSynthesizer(
        GeneralSynthesisParams(),
        mock_inference_config,
    )

    multiturn_attr = MultiTurnAttribute(
        id="test_conversation",
        min_turns=2,
        max_turns=2,
        role_instruction_messages={
            Role.USER: "You are a {customer_type} customer.",
            Role.ASSISTANT: "You are a helpful agent.",
        },
    )

    sample = {"customer_type": "frustrated"}
    result = synthesizer._build_role_context(sample, multiturn_attr)

    assert "[USER]" in result
    assert "You are a frustrated customer." in result
    assert "[ASSISTANT]" in result
    assert "You are a helpful agent." in result


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_planner_prompt_includes_role_context(
    mock_build_inference_engine,
    mock_inference_config,
):
    """Test that planner prompt includes formatted role context."""
    mock_build_inference_engine.return_value = Mock()

    synthesizer = ConversationSynthesizer(
        GeneralSynthesisParams(),
        mock_inference_config,
    )

    multiturn_attr = MultiTurnAttribute(
        id="test_conversation",
        min_turns=2,
        max_turns=2,
        role_instruction_messages={
            Role.USER: "You are a {customer_type} customer.",
            Role.ASSISTANT: "You are a helpful agent.",
        },
    )

    sample = {
        "customer_type": "frustrated",
        "target_turns": 4,
    }

    planner = synthesizer._create_planner_prompt(multiturn_attr, sample)

    assert len(planner.messages) == 4
    assert planner.messages[0].role == Role.SYSTEM
    assert planner.messages[1].role == Role.USER
    assert planner.messages[2].role == Role.ASSISTANT
    assert planner.messages[3].role == Role.USER

    user_message = planner.messages[3].content
    assert "Role context:" in user_message
    assert "You are a frustrated customer." in user_message
    assert "You are a helpful agent." in user_message

    example_response = planner.messages[2].content
    assert "```json" in example_response
    assert '"turn": 1' in example_response
    assert '"instruction"' in example_response


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_parse_plan_extracts_turn_instructions(
    mock_build_inference_engine,
    mock_inference_config,
):
    """Test that _parse_plan correctly extracts turn-by-turn instructions."""
    mock_build_inference_engine.return_value = Mock()

    synthesizer = ConversationSynthesizer(
        GeneralSynthesisParams(),
        mock_inference_config,
    )

    plan = """```json
[
  {"turn": 1, "instruction": "Greet support and explain the issue."},
  {"turn": 2, "instruction": "Acknowledge and ask for details."},
  {"turn": 3, "instruction": "Provide order number."},
  {"turn": 4, "instruction": "Offer a resolution."}
]
```"""

    result = synthesizer._parse_plan(plan, target_turns=4)

    assert result is not None
    assert len(result) == 4
    assert result[0] == "Greet support and explain the issue."
    assert result[1] == "Acknowledge and ask for details."
    assert result[2] == "Provide order number."
    assert result[3] == "Offer a resolution."


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_parse_plan_handles_empty_plan(
    mock_build_inference_engine,
    mock_inference_config,
):
    """Test that _parse_plan returns None for empty plan."""
    mock_build_inference_engine.return_value = Mock()

    synthesizer = ConversationSynthesizer(
        GeneralSynthesisParams(),
        mock_inference_config,
    )

    result = synthesizer._parse_plan("", target_turns=3)

    assert result is None


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_parse_plan_handles_missing_turns(
    mock_build_inference_engine,
    mock_inference_config,
):
    """Test that _parse_plan handles plans with missing turn numbers."""
    mock_build_inference_engine.return_value = Mock()

    synthesizer = ConversationSynthesizer(
        GeneralSynthesisParams(),
        mock_inference_config,
    )

    plan = """```json
[
  {"turn": 1, "instruction": "First message."},
  {"turn": 3, "instruction": "Third message."}
]
```"""

    result = synthesizer._parse_plan(plan, target_turns=3)

    assert result is not None
    assert len(result) == 3
    assert result[0] == "First message."
    assert result[1] == ""
    assert result[2] == "Third message."


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_parse_plan_handles_raw_json(
    mock_build_inference_engine,
    mock_inference_config,
):
    """Test that _parse_plan handles raw JSON without code fences."""
    mock_build_inference_engine.return_value = Mock()

    synthesizer = ConversationSynthesizer(
        GeneralSynthesisParams(),
        mock_inference_config,
    )

    plan = """[
  {"turn": 1, "instruction": "First instruction"},
  {"turn": 2, "instruction": "Second instruction"}
]"""

    result = synthesizer._parse_plan(plan, target_turns=2)

    assert result is not None
    assert len(result) == 2
    assert result[0] == "First instruction"
    assert result[1] == "Second instruction"


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_parse_plan_handles_invalid_json(
    mock_build_inference_engine,
    mock_inference_config,
):
    """Test that _parse_plan returns None for invalid JSON."""
    mock_build_inference_engine.return_value = Mock()

    synthesizer = ConversationSynthesizer(
        GeneralSynthesisParams(),
        mock_inference_config,
    )

    plan = "This is not valid JSON at all"

    result = synthesizer._parse_plan(plan, target_turns=2)

    assert result is None


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_parse_plan_handles_string_turn_numbers(
    mock_build_inference_engine,
    mock_inference_config,
):
    """Test that _parse_plan handles turn numbers as strings (common LLM output)."""
    mock_build_inference_engine.return_value = Mock()

    synthesizer = ConversationSynthesizer(
        GeneralSynthesisParams(),
        mock_inference_config,
    )
    plan = """```json
[
  {"turn": "1", "instruction": "First instruction"},
  {"turn": "2", "instruction": "Second instruction"}
]
```"""

    result = synthesizer._parse_plan(plan, target_turns=2)

    assert result is not None
    assert len(result) == 2
    assert result[0] == "First instruction"
    assert result[1] == "Second instruction"


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_validate_roles_raises_on_missing_role(
    mock_build_inference_engine,
    mock_inference_config,
):
    """Test that _validate_roles raises ValueError for missing roles."""
    mock_build_inference_engine.return_value = Mock()

    synthesizer = ConversationSynthesizer(
        GeneralSynthesisParams(),
        mock_inference_config,
    )

    multiturn_attr = MultiTurnAttribute(
        id="test_conversation",
        min_turns=2,
        max_turns=2,
        role_instruction_messages={
            Role.USER: "You are a user",
            Role.ASSISTANT: "You are an assistant",
        },
    )
    multiturn_attr.role_instruction_messages = {Role.USER: "You are a user"}

    with pytest.raises(ValueError) as exc_info:
        synthesizer._validate_roles(multiturn_attr)

    assert "assistant" in str(exc_info.value).lower()
    assert "missing" in str(exc_info.value).lower()


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_validate_roles_passes_for_valid_config(
    mock_build_inference_engine,
    mock_inference_config,
):
    """Test that _validate_roles passes for valid configuration."""
    mock_build_inference_engine.return_value = Mock()

    synthesizer = ConversationSynthesizer(
        GeneralSynthesisParams(),
        mock_inference_config,
    )

    multiturn_attr = MultiTurnAttribute(
        id="test_conversation",
        min_turns=2,
        max_turns=2,
        role_instruction_messages={
            Role.USER: "You are a user",
            Role.ASSISTANT: "You are an assistant",
        },
    )

    synthesizer._validate_roles(multiturn_attr)


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_parse_plan_extracts_json_with_surrounding_prose(
    mock_build_inference_engine,
    mock_inference_config,
):
    """Test that _parse_plan extracts JSON when LLM wraps it in prose."""
    mock_build_inference_engine.return_value = Mock()
    synthesizer = ConversationSynthesizer(
        GeneralSynthesisParams(), mock_inference_config
    )
    plan = (
        "Sure! Here is a conversation plan:\n"
        "```json\n"
        "[\n"
        '  {"turn": 1, "instruction": "Ask about the product"},\n'
        '  {"turn": 2, "instruction": "Provide product details"}\n'
        "]\n"
        "```\n"
        "Let me know if you need any changes."
    )
    result = synthesizer._parse_plan(plan, target_turns=2)
    assert result is not None
    assert result[0] == "Ask about the product"
    assert result[1] == "Provide product details"


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_parse_plan_extracts_raw_json_without_fences(
    mock_build_inference_engine,
    mock_inference_config,
):
    """Test that _parse_plan extracts raw JSON without code fences."""
    mock_build_inference_engine.return_value = Mock()
    synthesizer = ConversationSynthesizer(
        GeneralSynthesisParams(), mock_inference_config
    )
    plan = (
        "Here is the plan:\n"
        '[{"turn": 1, "instruction": "Greet"}, '
        '{"turn": 2, "instruction": "Respond"}]\n'
        "That should work."
    )
    result = synthesizer._parse_plan(plan, target_turns=2)
    assert result is not None
    assert result[0] == "Greet"
    assert result[1] == "Respond"


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_parse_plan_handles_single_dict_json(
    mock_build_inference_engine,
    mock_inference_config,
):
    """Test that _parse_plan handles LLM returning a single dict instead of a list."""
    mock_build_inference_engine.return_value = Mock()
    synthesizer = ConversationSynthesizer(
        GeneralSynthesisParams(), mock_inference_config
    )
    plan = '```json\n{"turn": 1, "instruction": "Only turn"}\n```'
    result = synthesizer._parse_plan(plan, target_turns=1)
    assert result is not None
    assert result[0] == "Only turn"


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_parse_plan_returns_none_for_malformed_text(
    mock_build_inference_engine,
    mock_inference_config,
):
    """Test that _parse_plan returns None for non-JSON text."""
    mock_build_inference_engine.return_value = Mock()
    synthesizer = ConversationSynthesizer(
        GeneralSynthesisParams(), mock_inference_config
    )
    result = synthesizer._parse_plan(
        "I'm sorry, I can't create a plan right now.", target_turns=2
    )
    assert result is None


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_has_empty_messages_detects_empty_content(
    mock_build_inference_engine,
    mock_inference_config,
):
    """Test that _has_empty_messages returns True when a message has empty content."""
    mock_build_inference_engine.return_value = Mock()
    synthesizer = ConversationSynthesizer(
        GeneralSynthesisParams(), mock_inference_config
    )

    conversation = Conversation(
        messages=[
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content=""),
        ]
    )
    assert synthesizer._has_empty_messages(conversation) is True

    conversation_ws = Conversation(
        messages=[
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="   "),
        ]
    )
    assert synthesizer._has_empty_messages(conversation_ws) is True


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_has_empty_messages_passes_valid_conversation(
    mock_build_inference_engine,
    mock_inference_config,
):
    """Test that _has_empty_messages returns False when all messages have content."""
    mock_build_inference_engine.return_value = Mock()
    synthesizer = ConversationSynthesizer(
        GeneralSynthesisParams(), mock_inference_config
    )

    conversation = Conversation(
        messages=[
            Message(role=Role.SYSTEM, content="System prompt"),
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="Hi there!"),
        ]
    )
    assert synthesizer._has_empty_messages(conversation) is False


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_has_empty_messages_ignores_system_messages(
    mock_build_inference_engine,
    mock_inference_config,
):
    """Test that _has_empty_messages ignores system messages."""
    mock_build_inference_engine.return_value = Mock()
    synthesizer = ConversationSynthesizer(
        GeneralSynthesisParams(), mock_inference_config
    )

    conversation = Conversation(
        messages=[
            Message(role=Role.SYSTEM, content=""),
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="Hi!"),
        ]
    )
    assert synthesizer._has_empty_messages(conversation) is False


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_synthesize_filters_conversations_with_empty_messages(
    mock_build_inference_engine,
    mock_inference_config,
):
    """Test that filtered conversations keep input/output index alignment."""
    mock_build_inference_engine.return_value = Mock()

    multiturn_attr = MultiTurnAttribute(
        id="test_conversation",
        min_turns=2,
        max_turns=2,
        role_instruction_messages={
            Role.USER: "You are a user.",
            Role.ASSISTANT: "You are an assistant.",
        },
    )

    synthesizer = ConversationSynthesizer(
        GeneralSynthesisParams(), mock_inference_config
    )

    with (
        patch.object(synthesizer, "_synthesize_all_samples") as mock_synth,
        patch.object(synthesizer, "_plan_samples") as mock_plan,
    ):
        mock_plan.return_value = [
            {"conversation_plan": "plan1"},
            {"conversation_plan": "plan2"},
        ]
        mock_synth.return_value = (
            [
                Conversation(
                    messages=[
                        Message(role=Role.USER, content="Hello"),
                        Message(role=Role.ASSISTANT, content="Hi!"),
                    ]
                ),
                Conversation(
                    messages=[
                        Message(role=Role.USER, content="Hello"),
                        Message(role=Role.ASSISTANT, content=""),
                    ]
                ),
            ],
            None,
        )

        samples = [{"key": "val1"}, {"key": "val2"}]
        result = synthesizer.synthesize(samples, multiturn_attr)

    assert len(result) == 2
    assert result[1] is None
    record = result[0]
    assert record is not None
    conversation = record["test_conversation"]
    assert isinstance(conversation, dict)
    assert conversation["messages"][0]["content"] == "Hello"
    assert conversation["messages"][1]["content"] == "Hi!"


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_synthesize_filters_all_conversations_returns_empty(
    mock_build_inference_engine,
    mock_inference_config,
):
    """Test synthesize returns all-None list when all conversations are filtered."""
    mock_build_inference_engine.return_value = Mock()

    multiturn_attr = MultiTurnAttribute(
        id="test_conversation",
        min_turns=2,
        max_turns=2,
        role_instruction_messages={
            Role.USER: "You are a user.",
            Role.ASSISTANT: "You are an assistant.",
        },
    )

    synthesizer = ConversationSynthesizer(
        GeneralSynthesisParams(), mock_inference_config
    )

    with (
        patch.object(synthesizer, "_synthesize_all_samples") as mock_synth,
        patch.object(synthesizer, "_plan_samples") as mock_plan,
    ):
        mock_plan.return_value = [
            {"conversation_plan": "plan1"},
            {"conversation_plan": "plan2"},
        ]
        mock_synth.return_value = (
            [
                Conversation(
                    messages=[
                        Message(role=Role.USER, content=""),
                        Message(role=Role.ASSISTANT, content="Response"),
                    ]
                ),
                Conversation(
                    messages=[
                        Message(role=Role.USER, content="Hello"),
                        Message(role=Role.ASSISTANT, content=""),
                    ]
                ),
            ],
            None,
        )

        samples = [{"key": "val1"}, {"key": "val2"}]
        result = synthesizer.synthesize(samples, multiturn_attr)

    assert result == [None, None]


def _make_tool_params():
    from oumi.core.configs.params.tool_params import (
        DeterministicToolOutput,
        GeneratedToolOutput,
        ToolAttribute,
        ToolOutputStrategy,
    )

    return GeneralSynthesisParams(
        tools=[
            ToolAttribute(
                id="search",
                name="SearchOrders",
                description="Look up order details",
                parameters={
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string", "description": "Order ID"}
                    },
                    "required": ["order_id"],
                },
                output_strategy=ToolOutputStrategy.GENERATED,
                generated_output=GeneratedToolOutput(
                    instruction="Return order details."
                ),
            ),
            ToolAttribute(
                id="escalate",
                name="Escalate",
                description="Escalate to specialist",
                parameters={
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string", "description": "Reason"}
                    },
                    "required": ["reason"],
                },
                output_strategy=ToolOutputStrategy.DETERMINISTIC,
                deterministic_outputs=[
                    DeterministicToolOutput(
                        values={"ticket_id": "ESC-001", "status": "escalated"}
                    ),
                ],
            ),
        ],
    )


def _make_tool_multiturn():
    return MultiTurnAttribute(
        id="tool_conversation",
        min_turns=2,
        max_turns=2,
        available_tools=["search", "escalate"],
        max_tool_calls_per_turn=3,
        role_instruction_messages={
            Role.USER: "You are a customer with issue: {issue}.",
            Role.ASSISTANT: "You are a support agent.",
        },
        output_system_prompt="You are a support agent.",
    )


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_agentic_turn_with_tool_call_and_response(
    mock_build_inference_engine,
    mock_inference_config,
):
    mock_engine = Mock()
    mock_build_inference_engine.return_value = mock_engine

    def infer_side_effect(conversations, **kwargs):
        results = []
        for conv in conversations:
            last_content = conv.messages[-1].content if conv.messages else ""

            if "conversation planner" in str(conv.messages[0].content).lower():
                results.append(
                    Conversation(
                        messages=[
                            Message(
                                role=Role.ASSISTANT,
                                content=(
                                    '```json\n[{"turn": 1, "instruction":'
                                    ' "Ask about order"}, {"turn": 2,'
                                    ' "instruction": "Look up the order"}'
                                    "]\n```"
                                ),
                            )
                        ]
                    )
                )
            elif "simulating the tool" in str(conv.messages[0].content).lower():
                results.append(
                    Conversation(
                        messages=[
                            Message(
                                role=Role.ASSISTANT,
                                content='{"order_id": "ORD-001", "status": "shipped"}',
                            )
                        ]
                    )
                )
            elif "ASSISTANT" in str(last_content) and not any(
                "[Tool result from" in str(m.content) for m in conv.messages
            ):
                results.append(
                    Conversation(
                        messages=[
                            Message(
                                role=Role.ASSISTANT,
                                content=(
                                    '<tool_call>{"name": "SearchOrders",'
                                    ' "arguments": {"order_id": "ORD-001"}}'
                                    "</tool_call>"
                                ),
                            )
                        ]
                    )
                )
            else:
                results.append(
                    Conversation(
                        messages=[
                            Message(
                                role=Role.ASSISTANT,
                                content="Your order ORD-001 has been shipped!",
                            )
                        ]
                    )
                )
        return results

    mock_engine.infer.side_effect = infer_side_effect

    params = _make_tool_params()
    mt_attr = _make_tool_multiturn()
    synthesizer = ConversationSynthesizer(params, mock_inference_config)

    result = synthesizer.synthesize([{"issue": "order tracking"}], mt_attr)

    assert len(result) == 1
    record = result[0]
    assert record is not None

    conv_dict = record["tool_conversation"]
    assert isinstance(conv_dict, dict)
    assert len(conv_dict["tools"]) == 2
    assert conv_dict["tools"][0]["function"]["name"] == "SearchOrders"
    assert conv_dict["tools"][1]["function"]["name"] == "Escalate"

    messages = conv_dict["messages"]
    assert len(messages) > 0
    assert messages[0]["role"] == "system"
    assert any(m.get("tool_calls") is not None for m in messages)
    assert any(m.get("role") == "tool" for m in messages)


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_agentic_turn_deterministic_tool(
    mock_build_inference_engine,
    mock_inference_config,
):
    mock_engine = Mock()
    mock_build_inference_engine.return_value = mock_engine

    def infer_side_effect(conversations, **kwargs):
        results = []
        for conv in conversations:
            if "conversation planner" in str(conv.messages[0].content).lower():
                results.append(
                    Conversation(
                        messages=[
                            Message(
                                role=Role.ASSISTANT,
                                content=(
                                    '```json\n[{"turn": 1, "instruction":'
                                    ' "Ask"}, {"turn": 2, "instruction":'
                                    ' "Escalate"}]\n```'
                                ),
                            )
                        ]
                    )
                )
            elif "customer" in str(conv.messages[0].content).lower():
                # User turn — generate a normal user message
                results.append(
                    Conversation(
                        messages=[
                            Message(
                                role=Role.ASSISTANT,
                                content="I have a complex problem I need help with.",
                            )
                        ]
                    )
                )
            elif not any("[Tool result from" in str(m.content) for m in conv.messages):
                results.append(
                    Conversation(
                        messages=[
                            Message(
                                role=Role.ASSISTANT,
                                content=(
                                    '<tool_call>{"name": "Escalate",'
                                    ' "arguments": {"reason":'
                                    ' "complex issue"}}</tool_call>'
                                ),
                            )
                        ]
                    )
                )
            else:
                results.append(
                    Conversation(
                        messages=[
                            Message(
                                role=Role.ASSISTANT,
                                content="I've escalated your issue.",
                            )
                        ]
                    )
                )
        return results

    mock_engine.infer.side_effect = infer_side_effect

    params = _make_tool_params()
    mt_attr = _make_tool_multiturn()
    synthesizer = ConversationSynthesizer(params, mock_inference_config)

    result = synthesizer.synthesize([{"issue": "complex problem"}], mt_attr)

    assert len(result) == 1
    assert result[0] is not None
    conv_dict = result[0]["tool_conversation"]
    assert isinstance(conv_dict, dict)

    messages = conv_dict["messages"]
    tool_results = [m for m in messages if m.get("role") == "tool"]
    assert len(tool_results) >= 1
    content = json.loads(tool_results[0]["content"])
    assert content["ticket_id"] == "ESC-001"
    assert content["status"] == "escalated"


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_agentic_zero_tool_calls_filtered(
    mock_build_inference_engine,
    mock_inference_config,
):
    mock_engine = Mock()
    mock_build_inference_engine.return_value = mock_engine
    mock_engine.infer.side_effect = lambda convs, **kw: [
        Conversation(
            messages=[Message(role=Role.ASSISTANT, content="Just a text response")]
        )
        for _ in convs
    ]

    params = _make_tool_params()
    mt_attr = _make_tool_multiturn()
    synthesizer = ConversationSynthesizer(params, mock_inference_config)

    result = synthesizer.synthesize([{"issue": "simple question"}], mt_attr)

    assert len(result) == 1
    assert result[0] is None


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_tool_catalog_injected_into_assistant_persona(
    mock_build_inference_engine,
    mock_inference_config,
):
    mock_build_inference_engine.return_value = Mock()

    params = _make_tool_params()
    synthesizer = ConversationSynthesizer(params, mock_inference_config)

    tools = synthesizer._get_tools_for_multiturn(_make_tool_multiturn())
    persona_msg = synthesizer._format_persona(
        {"issue": "test"}, "You are a support agent.", tools
    )

    assert "SearchOrders" in persona_msg.content
    assert "Escalate" in persona_msg.content
    assert "<tool_call>" in persona_msg.content


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_planner_prompt_includes_tool_guidance(
    mock_build_inference_engine,
    mock_inference_config,
):
    mock_build_inference_engine.return_value = Mock()

    params = _make_tool_params()
    synthesizer = ConversationSynthesizer(params, mock_inference_config)

    mt_attr = _make_tool_multiturn()
    planner = synthesizer._create_planner_prompt(
        mt_attr, {"issue": "test", "target_turns": 4}
    )

    system_content = str(planner.messages[0].content)
    assert "tools" in system_content.lower()
    assert "WHAT" in system_content

    user_content = str(planner.messages[3].content)
    assert "The assistant has the following capabilities:" in user_content
    assert "Look up order details" in user_content
    assert "Escalate to specialist" in user_content
    capability_section = user_content.split("capabilities:\n")[-1].split(
        "\nRole context:"
    )[0]
    assert "SearchOrders" not in capability_section
    assert "available capabilities" in user_content


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_non_tool_conversation_unchanged(
    mock_build_inference_engine,
    mock_inference_config,
):
    mock_engine = Mock()
    mock_build_inference_engine.return_value = mock_engine
    mock_engine.infer.return_value = [
        Conversation(messages=[Message(role=Role.ASSISTANT, content="Response")])
    ]

    params = GeneralSynthesisParams()
    mt_attr = MultiTurnAttribute(
        id="test_conversation",
        min_turns=2,
        max_turns=2,
        role_instruction_messages={
            Role.USER: "You are a user",
            Role.ASSISTANT: "You are an assistant",
        },
    )

    synthesizer = ConversationSynthesizer(params, mock_inference_config)
    result = synthesizer.synthesize([{}], mt_attr)

    assert len(result) == 1
    assert result[0] is not None
    conv = result[0]["test_conversation"]
    assert isinstance(conv, dict)
    assert "tools" not in conv
    assert "messages" in conv


def test_clean_json_output_strips_markdown_fences():
    raw = '```json\n{"key": "value"}\n```'
    assert _clean_json_output(raw) == '{"key": "value"}'


def test_clean_json_output_passes_clean_json():
    raw = '{"key": "value"}'
    assert _clean_json_output(raw) == '{"key": "value"}'


def test_clean_json_output_returns_raw_on_failure():
    raw = "not json at all"
    assert _clean_json_output(raw) == "not json at all"


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_init_sample_environments_per_sample(
    mock_build_inference_engine,
    mock_inference_config,
):
    """Each sample should get its own environment with resolved context."""
    from oumi.core.configs.params.tool_params import (
        ToolAttribute,
        ToolEnvironmentAttribute,
        ToolOutputStrategy,
    )

    mock_engine = Mock()
    mock_build_inference_engine.return_value = mock_engine

    schema = json.dumps({"type": "object", "properties": {"data": {"type": "string"}}})
    state1 = json.dumps({"data": "healthcare"})
    state2 = json.dumps({"data": "ecommerce"})

    call_count = 0

    def infer_side_effect(conversations, **kwargs):
        nonlocal call_count
        results = []
        for _ in conversations:
            call_count += 1
            if call_count <= 2:
                text = schema
            elif call_count == 3:
                text = state1
            else:
                text = state2
            results.append(
                Conversation(messages=[Message(role=Role.ASSISTANT, content=text)])
            )
        return results

    mock_engine.infer.side_effect = infer_side_effect

    env_config = ToolEnvironmentAttribute(
        id="db",
        name="Database",
        description="A database",
        system_prompt="You manage a database.",
    )
    tool = ToolAttribute(
        id="query",
        name="Query",
        description="Run query",
        output_strategy=ToolOutputStrategy.ENVIRONMENT,
        environment="db",
        read_only=True,
    )

    params = GeneralSynthesisParams(
        tools=[tool],
        environments=[env_config],
    )
    multiturn = MultiTurnAttribute(
        id="conv",
        min_turns=2,
        max_turns=2,
        available_tools=["query"],
        role_instruction_messages={
            Role.USER: "You are a analyst.",
            Role.ASSISTANT: "You are a database assistant.",
        },
    )
    synthesizer = ConversationSynthesizer(params, mock_inference_config)

    from types import SimpleNamespace

    samples = [
        {"domain": SimpleNamespace(name="Healthcare")},
        {"domain": SimpleNamespace(name="E-Commerce")},
    ]
    result = synthesizer._init_sample_environments(samples, multiturn)

    assert len(result) == 2
    assert result[0] is not None
    assert result[1] is not None
    assert result[0]["db"] is not result[1]["db"]
    # Batching: 2 batched infer calls (schemas + states), not 2N serial
    assert mock_engine.infer.call_count == 2


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_planner_prompt_includes_env_summary(
    mock_build_inference_engine,
    mock_inference_config,
):
    """When env_summary is provided, planner prompt should contain it."""
    mock_build_inference_engine.return_value = Mock()

    params = GeneralSynthesisParams()
    multiturn = MultiTurnAttribute(
        id="conv",
        min_turns=2,
        max_turns=2,
        role_instruction_messages={
            Role.USER: "You are a user.",
            Role.ASSISTANT: "You are an assistant.",
        },
    )
    synthesizer = ConversationSynthesizer(params, mock_inference_config)

    sample = {"target_turns": 2}
    env_summary = 'Environment "database":\ntables: users (10 items), orders (5 items)'

    prompt = synthesizer._create_planner_prompt(
        multiturn, sample, env_summary=env_summary
    )

    last_user_msg = prompt.messages[-1].content
    assert "tables: users" in last_user_msg
    assert "MUST work with this data" in last_user_msg


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_format_persona_includes_grounding_rules(
    mock_build_inference_engine,
    mock_inference_config,
):
    """When tools are provided, persona should include grounding instructions."""
    from oumi.core.configs.params.tool_params import (
        ToolAttribute,
        ToolEnvironmentAttribute,
        ToolOutputStrategy,
    )

    mock_build_inference_engine.return_value = Mock()

    env_config = ToolEnvironmentAttribute(
        id="db",
        name="Database",
        description="A database",
        system_prompt="You manage a database.",
    )
    tool = ToolAttribute(
        id="query",
        name="Query",
        description="Run a query",
        output_strategy=ToolOutputStrategy.ENVIRONMENT,
        environment="db",
        read_only=True,
    )
    params = GeneralSynthesisParams(tools=[tool], environments=[env_config])
    synthesizer = ConversationSynthesizer(params, mock_inference_config)

    result = synthesizer._format_persona({}, "You are an assistant.", [tool])

    content = result.content
    assert "trust the tool results" in content
    assert "Do NOT fabricate" in content
    assert "output the <tool_call> tag clearly" in content
