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
from oumi.core.synthesis.conversation_synthesizer import ConversationSynthesizer
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
        Role.USER,
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


# ── Agentic Tool Tests ───────────────────────────────────────────────────


def _make_tool_params():
    """Create GeneralSynthesisParams with tools for agentic tests."""
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
    """Create a MultiTurnAttribute with tools."""
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
    """Test that an agentic turn with a tool call produces correct output format."""
    mock_engine = Mock()
    mock_build_inference_engine.return_value = mock_engine

    call_count = [0]

    def infer_side_effect(conversations, **kwargs):
        results = []
        for conv in conversations:
            call_count[0] += 1
            last_content = conv.messages[-1].content if conv.messages else ""

            # Planning call
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
            # Tool simulator call
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
            # First assistant turn iteration: make a tool call
            elif (
                "ASSISTANT" in str(last_content)
                and call_count[0] <= 10
                and not any(m.role == Role.TOOL for m in conv.messages)
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
            # After tool result, respond naturally
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

    samples = [{"issue": "order tracking"}]
    result = synthesizer.synthesize(samples, mt_attr)

    assert len(result) == 1
    record = result[0]
    assert record is not None

    conv_dict = record["tool_conversation"]
    assert isinstance(conv_dict, dict)

    # Should have "tools" key with tool definitions
    assert "tools" in conv_dict
    tools_list = conv_dict["tools"]
    assert len(tools_list) == 2
    assert tools_list[0]["function"]["name"] == "SearchOrders"
    assert tools_list[1]["function"]["name"] == "Escalate"

    # Should have "messages" key
    assert "messages" in conv_dict
    messages = conv_dict["messages"]
    assert len(messages) > 0

    # Check system message is first
    assert messages[0]["role"] == "system"

    # Check there's at least one tool_calls message and one tool result
    has_tool_call = any(
        m.get("tool_calls") is not None for m in messages if isinstance(m, dict)
    )
    has_tool_result = any(
        m.get("role") == "tool" for m in messages if isinstance(m, dict)
    )
    assert has_tool_call, "Should have at least one tool call message"
    assert has_tool_result, "Should have at least one tool result message"


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_agentic_turn_deterministic_tool(
    mock_build_inference_engine,
    mock_inference_config,
):
    """Test that DETERMINISTIC tools resolve without extra LLM calls."""
    mock_engine = Mock()
    mock_build_inference_engine.return_value = mock_engine

    def infer_side_effect(conversations, **kwargs):
        results = []
        for conv in conversations:
            # Planning call
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
            # First assistant turn: escalate (deterministic tool)
            elif not any(m.role == Role.TOOL for m in conv.messages):
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
            # After tool result: respond
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

    samples = [{"issue": "complex problem"}]
    result = synthesizer.synthesize(samples, mt_attr)

    assert len(result) == 1
    record = result[0]
    assert record is not None
    conv_dict = record["tool_conversation"]

    # Find the tool result message
    messages = conv_dict["messages"]  # type: ignore[index]
    tool_results = [
        m
        for m in messages
        if m.get("role") == "tool"  # type: ignore[union-attr]
    ]
    assert len(tool_results) >= 1
    # Deterministic output should contain our canned values
    import json

    content = json.loads(tool_results[0]["content"])  # type: ignore[index]
    assert content["ticket_id"] == "ESC-001"
    assert content["status"] == "escalated"


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_agentic_zero_tool_calls_filtered(
    mock_build_inference_engine,
    mock_inference_config,
):
    """Test that conversations with tools but zero tool calls are filtered out."""
    mock_engine = Mock()
    mock_build_inference_engine.return_value = mock_engine

    def infer_side_effect(conversations, **kwargs):
        return [
            Conversation(
                messages=[Message(role=Role.ASSISTANT, content="Just a text response")]
            )
            for _ in conversations
        ]

    mock_engine.infer.side_effect = infer_side_effect

    params = _make_tool_params()
    mt_attr = _make_tool_multiturn()
    synthesizer = ConversationSynthesizer(params, mock_inference_config)

    samples = [{"issue": "simple question"}]
    result = synthesizer.synthesize(samples, mt_attr)

    # Should be filtered out (None) because no tools were called
    assert len(result) == 1
    assert result[0] is None


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_tool_catalog_injected_into_assistant_persona(
    mock_build_inference_engine,
    mock_inference_config,
):
    """Test that the tool catalog is auto-injected into ASSISTANT persona."""
    mock_build_inference_engine.return_value = Mock()

    params = _make_tool_params()
    synthesizer = ConversationSynthesizer(params, mock_inference_config)

    tools = synthesizer._get_tools_for_multiturn(_make_tool_multiturn())
    persona_msg = synthesizer._format_persona_with_tools(
        {"issue": "test"}, "You are a support agent.", tools
    )

    content = persona_msg.content
    assert "SearchOrders" in content
    assert "Escalate" in content
    assert "<tool_call>" in content
    assert "do not guess or fabricate data" in content


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_planner_prompt_includes_tool_guidance(
    mock_build_inference_engine,
    mock_inference_config,
):
    """Test that planner prompt mentions tool availability when tools exist."""
    mock_build_inference_engine.return_value = Mock()

    params = _make_tool_params()
    synthesizer = ConversationSynthesizer(params, mock_inference_config)

    mt_attr = _make_tool_multiturn()
    sample = {"issue": "test", "target_turns": 4}
    planner = synthesizer._create_planner_prompt(mt_attr, sample)

    system_content = str(planner.messages[0].content)
    assert "tools" in system_content.lower()
    assert "WHAT" in system_content

    user_content = str(planner.messages[3].content)
    assert "looking up information" in user_content


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_non_tool_conversation_unchanged(
    mock_build_inference_engine,
    mock_inference_config,
):
    """Test that non-tool conversations produce the same output format as before."""
    mock_engine = Mock()
    mock_build_inference_engine.return_value = mock_engine
    mock_engine.infer.return_value = [
        Conversation(messages=[Message(role=Role.ASSISTANT, content="Response")])
    ]

    # No tools defined
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
    record = result[0]
    assert record is not None
    conv = record["test_conversation"]
    assert isinstance(conv, dict)
    # Should NOT have "tools" key
    assert "tools" not in conv
    # Should have "messages" key
    assert "messages" in conv
