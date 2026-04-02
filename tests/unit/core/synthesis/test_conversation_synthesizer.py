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
from oumi.core.synthesis.conversation_synthesizer import ConversationSynthesizer
from oumi.core.synthesis.environment import (
    _process_env_write_calls,
    init_sample_environments,
)
from oumi.core.synthesis.tool_executor import ToolExecutor, clean_json_output
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
def test_build_role_context_preserves_json_braces(
    mock_build_inference_engine,
    mock_inference_config,
):
    """Role context should preserve literal JSON snippets for the planner."""
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
            Role.USER: "Use this payload exactly: {payload}",
            Role.ASSISTANT: "You are a helpful agent.",
        },
    )

    sample = {"payload": '{"schema": {"type": "object"}}'}
    result = synthesizer._build_role_context(sample, multiturn_attr)

    assert '{"schema": {"type": "object"}}' in result
    assert "{{" not in result


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
    assert isinstance(example_response, str)
    parsed_example = json.loads(example_response)
    assert parsed_example[0]["turn"] == 1
    assert "instruction" in parsed_example[0]


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
def test_plan_samples_retries_on_invalid_planner_instructions(
    mock_build_inference_engine,
    mock_inference_config,
):
    """Planner retries when instructions leak tool syntax or tool names."""
    mock_engine = Mock()
    mock_build_inference_engine.return_value = mock_engine

    invalid_plan = json.dumps(
        [
            {
                "turn": 1,
                "instruction": (
                    '<tool_call>{"name": "SearchOrders", '
                    '"arguments": {"order_id": "ORD-001"}}</tool_call>'
                ),
            },
            {
                "turn": 2,
                "instruction": "Use SearchOrders to summarize the result.",
            },
        ]
    )
    valid_plan = json.dumps(
        [
            {"turn": 1, "instruction": "Describe the order problem."},
            {
                "turn": 2,
                "instruction": (
                    "Investigate the order details and respond with the outcome."
                ),
            },
        ]
    )
    mock_engine.infer.side_effect = [
        [Conversation(messages=[Message(role=Role.ASSISTANT, content=invalid_plan)])],
        [Conversation(messages=[Message(role=Role.ASSISTANT, content=valid_plan)])],
    ]

    synthesizer = ConversationSynthesizer(_make_tool_params(), mock_inference_config)
    planned = synthesizer._plan_samples(
        [{"issue": "order tracking"}],
        _make_tool_multiturn(),
        max_retries=1,
    )

    assert mock_engine.infer.call_count == 2
    assert planned[0]["parsed_turn_plans"] == [
        "Describe the order problem.",
        "Investigate the order details and respond with the outcome.",
    ]


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
            elif "MUST use <tool_call>" in str(last_content) and not any(
                "[Tool result from SearchOrders]: {" in str(m.content)
                for m in conv.messages
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
                                    ' "Route the issue for specialist support"}]\n```'
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
    assert "NEVER fabricate tool results" in persona_msg.content


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

    parsed_example = json.loads(str(planner.messages[2].content))
    assert parsed_example[0]["turn"] == 1
    assert "instruction" in parsed_example[0]


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
    assert clean_json_output(raw) == '{"key": "value"}'


def test_clean_json_output_passes_clean_json():
    raw = '{"key": "value"}'
    assert clean_json_output(raw) == '{"key": "value"}'


def test_clean_json_output_returns_raw_on_failure():
    raw = "not json at all"
    assert clean_json_output(raw) == "not json at all"


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_init_sample_environments_per_sample(
    mock_build_inference_engine,
    mock_inference_config,
):
    """Each sample should get its own independent environment copy via registry."""
    from oumi.core.configs.params.tool_params import (
        ToolAttribute,
        ToolEnvironmentAttribute,
        ToolOutputStrategy,
    )

    mock_engine = Mock()
    mock_build_inference_engine.return_value = mock_engine

    # Use static env (initial_state + state_schema) so no LLM calls are needed
    # for environment initialization — the registry uses register_static path.
    env_config = ToolEnvironmentAttribute(
        id="db",
        name="Database",
        description="A database",
        system_prompt="You manage a database.",
        state_schema={"type": "object", "properties": {"data": {"type": "string"}}},
        initial_state={"data": "seed"},
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
    synthesizer = ConversationSynthesizer(params, mock_inference_config)

    from types import SimpleNamespace

    samples = [
        {"domain": SimpleNamespace(name="Healthcare")},
        {"domain": SimpleNamespace(name="E-Commerce")},
    ]
    result = init_sample_environments(
        samples,
        [tool],
        synthesizer._env_configs,
        synthesizer._formatter,
        synthesizer._inference_engine,
        synthesizer._inference_config,
    )

    assert len(result) == 2
    assert result[0] is not None
    assert result[1] is not None
    # Each sample gets an independent deepcopy — not the same object
    assert result[0]["db"] is not result[1]["db"]
    # Registry uses register_static: no LLM inference calls needed
    assert mock_engine.infer.call_count == 0


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_init_env_registry_build_path_used(
    mock_build_inference_engine,
    mock_inference_config,
):
    """When env has no static state, registry.build() is called via LLM inference."""
    from oumi.core.configs.params.tool_params import (
        ToolAttribute,
        ToolEnvironmentAttribute,
        ToolOutputStrategy,
    )

    mock_engine = Mock()
    mock_build_inference_engine.return_value = mock_engine

    # Return valid collection data so the state is non-empty
    mock_engine.infer.return_value = [
        Conversation(
            messages=[
                Message(
                    role=Role.ASSISTANT,
                    content='{"1": {"name": "Alice"}, "2": {"name": "Bob"}}',
                )
            ]
        )
    ]

    env_config = ToolEnvironmentAttribute(
        id="db",
        name="Database",
        description="A database",
        system_prompt="You manage a database.",
        # Provide state_schema so registry knows what to generate
        state_schema={
            "type": "object",
            "properties": {
                "users": {
                    "type": "object",
                    "additionalProperties": {"type": "object"},
                }
            },
        },
        # No initial_state → goes through registry.build() path
    )
    tool = ToolAttribute(
        id="query",
        name="Query",
        description="Run query",
        output_strategy=ToolOutputStrategy.ENVIRONMENT,
        environment="db",
        read_only=True,
    )
    params = GeneralSynthesisParams(tools=[tool], environments=[env_config])
    synthesizer = ConversationSynthesizer(params, mock_inference_config)

    from types import SimpleNamespace

    result = init_sample_environments(
        [{"domain": SimpleNamespace(name="Test")}],
        [tool],
        synthesizer._env_configs,
        synthesizer._formatter,
        synthesizer._inference_engine,
        synthesizer._inference_config,
    )

    # Registry called infer at least once to build the environment
    assert mock_engine.infer.call_count >= 1

    # Sample should survive — env has non-empty state from the LLM response
    assert result[0] is not None
    env = result[0]["db"]
    assert env.state != {}


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_init_env_kills_sample_on_total_state_failure(
    mock_build_inference_engine,
    mock_inference_config,
):
    """If state generation produces no parseable JSON after retries, sample is None."""
    from oumi.core.configs.params.tool_params import (
        ToolAttribute,
        ToolEnvironmentAttribute,
        ToolOutputStrategy,
    )

    mock_engine = Mock()
    mock_build_inference_engine.return_value = mock_engine

    def infer_side_effect(conversations, **kwargs):
        # All state generation attempts fail with non-JSON
        return [
            Conversation(
                messages=[
                    Message(
                        role=Role.ASSISTANT,
                        content="I cannot generate state data",
                    )
                ]
            )
            for _ in conversations
        ]

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
    params = GeneralSynthesisParams(tools=[tool], environments=[env_config])
    synthesizer = ConversationSynthesizer(params, mock_inference_config)

    from types import SimpleNamespace

    result = init_sample_environments(
        [{"domain": SimpleNamespace(name="Test")}],
        [tool],
        synthesizer._env_configs,
        synthesizer._formatter,
        synthesizer._inference_engine,
        synthesizer._inference_config,
    )

    # Sample should be killed (None) because state was completely unparseable
    assert result[0] is None


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_init_env_static_config_used_without_llm(
    mock_build_inference_engine,
    mock_inference_config,
):
    """When both initial_state and state_schema are provided, no LLM calls are made."""
    from oumi.core.configs.params.tool_params import (
        ToolAttribute,
        ToolEnvironmentAttribute,
        ToolOutputStrategy,
    )

    mock_engine = Mock()
    mock_build_inference_engine.return_value = mock_engine

    static_state = {"users": {"1": {"name": "Alice"}, "2": {"name": "Bob"}}}
    env_config = ToolEnvironmentAttribute(
        id="db",
        name="Database",
        description="A database",
        system_prompt="You manage a database.",
        state_schema={
            "type": "object",
            "properties": {"users": {"type": "object"}},
        },
        initial_state=static_state,
    )
    tool = ToolAttribute(
        id="query",
        name="Query",
        description="Run query",
        output_strategy=ToolOutputStrategy.ENVIRONMENT,
        environment="db",
        read_only=True,
    )
    params = GeneralSynthesisParams(tools=[tool], environments=[env_config])
    synthesizer = ConversationSynthesizer(params, mock_inference_config)

    from types import SimpleNamespace

    result = init_sample_environments(
        [{"domain": SimpleNamespace(name="Test")}],
        [tool],
        synthesizer._env_configs,
        synthesizer._formatter,
        synthesizer._inference_engine,
        synthesizer._inference_config,
    )

    # Static path: no LLM inference calls needed
    assert mock_engine.infer.call_count == 0

    # Sample should survive with the static state
    assert result[0] is not None
    env = result[0]["db"]
    assert env.state == static_state


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_init_sample_environments_reuses_shared_environment_build(
    mock_build_inference_engine,
    mock_inference_config,
):
    """When env config is identical, build once and copy across samples."""
    from oumi.core.configs.params.tool_params import (
        ToolAttribute,
        ToolEnvironmentAttribute,
        ToolOutputStrategy,
    )

    mock_engine = Mock()
    mock_build_inference_engine.return_value = mock_engine

    def infer_side_effect(conversations, **kwargs):
        return [
            Conversation(
                messages=[
                    Message(
                        role=Role.ASSISTANT,
                        content='{"1": {"name": "Alice"}}',
                    )
                ]
            )
            for _ in conversations
        ]

    mock_engine.infer.side_effect = infer_side_effect

    env_config = ToolEnvironmentAttribute(
        id="db",
        name="Database",
        description="A database",
        system_prompt="You manage a database.",
        state_schema={
            "type": "object",
            "properties": {
                "users": {
                    "type": "object",
                    "additionalProperties": {"type": "object"},
                }
            },
        },
    )
    tool = ToolAttribute(
        id="query",
        name="Query",
        description="Run query",
        output_strategy=ToolOutputStrategy.ENVIRONMENT,
        environment="db",
        read_only=True,
    )
    params = GeneralSynthesisParams(tools=[tool], environments=[env_config])
    synthesizer = ConversationSynthesizer(params, mock_inference_config)

    from types import SimpleNamespace

    result = init_sample_environments(
        [
            {"domain": SimpleNamespace(name="Healthcare")},
            {"domain": SimpleNamespace(name="E-Commerce")},
        ],
        [tool],
        synthesizer._env_configs,
        synthesizer._formatter,
        synthesizer._inference_engine,
        synthesizer._inference_config,
    )

    assert result[0] is not None
    assert result[1] is not None
    assert mock_engine.infer.call_count == 1


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_init_sample_environments_rebuilds_when_env_config_varies(
    mock_build_inference_engine,
    mock_inference_config,
):
    """Env config format varies per sample: each variant built separately."""
    from oumi.core.configs.params.tool_params import (
        ToolAttribute,
        ToolEnvironmentAttribute,
        ToolOutputStrategy,
    )

    seen_prompts: list[str] = []
    mock_engine = Mock()
    mock_build_inference_engine.return_value = mock_engine

    def infer_side_effect(conversations, **kwargs):
        for conv in conversations:
            seen_prompts.append(str(conv.messages[0].content))
        return [
            Conversation(
                messages=[
                    Message(
                        role=Role.ASSISTANT,
                        content='{"1": {"name": "Alice"}}',
                    )
                ]
            )
            for _ in conversations
        ]

    mock_engine.infer.side_effect = infer_side_effect

    env_config = ToolEnvironmentAttribute(
        id="db",
        name="Database",
        description="A {domain.name} database",
        system_prompt="You manage a {domain.name} database.",
        state_schema={
            "type": "object",
            "properties": {
                "users": {
                    "type": "object",
                    "additionalProperties": {"type": "object"},
                }
            },
        },
    )
    tool = ToolAttribute(
        id="query",
        name="Query",
        description="Run query",
        output_strategy=ToolOutputStrategy.ENVIRONMENT,
        environment="db",
        read_only=True,
    )
    params = GeneralSynthesisParams(tools=[tool], environments=[env_config])

    synthesizer = ConversationSynthesizer(params, mock_inference_config)

    from types import SimpleNamespace

    result = init_sample_environments(
        [
            {"domain": SimpleNamespace(name="Healthcare")},
            {"domain": SimpleNamespace(name="E-Commerce")},
        ],
        [tool],
        synthesizer._env_configs,
        synthesizer._formatter,
        synthesizer._inference_engine,
        synthesizer._inference_config,
    )

    assert result[0] is not None
    assert result[1] is not None
    assert mock_engine.infer.call_count == 2
    assert any("Healthcare" in prompt for prompt in seen_prompts)
    assert any("E-Commerce" in prompt for prompt in seen_prompts)


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_process_env_write_calls_updates_state_before_generating_result(
    mock_build_inference_engine,
    mock_inference_config,
):
    from oumi.core.configs.params.tool_params import (
        ToolAttribute,
        ToolEnvironmentAttribute,
        ToolOutputStrategy,
    )
    from oumi.core.synthesis.environment import GeneratedToolEnvironment

    mock_build_inference_engine.return_value = Mock()

    env = GeneratedToolEnvironment(
        ToolEnvironmentAttribute(
            id="filesystem",
            name="Filesystem",
            description="A filesystem",
            system_prompt="You manage a filesystem.",
            state_schema={
                "type": "object",
                "properties": {"files": {"type": "object"}},
            },
            initial_state={"files": {"a.txt": "old"}},
        )
    )
    tool = ToolAttribute(
        id="write_file",
        name="WriteFile",
        description="Write a file",
        output_strategy=ToolOutputStrategy.ENVIRONMENT,
        environment="filesystem",
        read_only=False,
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "path": {"type": "string"},
            },
        },
    )

    call_count = 0

    def mock_infer(prompts, inference_config=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return [
                Conversation(
                    messages=[
                        Message(
                            role=Role.ASSISTANT,
                            content=(
                                '{"patch": [{"op": "replace", '
                                '"path": "/files/a.txt", "value": "new"}], '
                                '"error": null}'
                            ),
                        )
                    ]
                )
            ]
        assert env.state["files"]["a.txt"] == "new"
        return [
            Conversation(
                messages=[
                    Message(
                        role=Role.ASSISTANT,
                        content='{"status": "success", "path": "a.txt"}',
                    )
                ]
            )
        ]

    mock_engine = Mock()
    mock_engine.infer = mock_infer
    mock_config = Mock()

    final_results: list[str | None] = [None]
    env_items = [
        (
            0,
            '<tool_call>{"name": "WriteFile", "arguments": {"path": "a.txt", '
            '"content": "new"}}</tool_call>',
            {"name": "WriteFile", "arguments": {"path": "a.txt", "content": "new"}},
            "call_001",
            env,
            tool,
        )
    ]

    _process_env_write_calls(env_items, [0], final_results, mock_engine, mock_config)

    assert env.state["files"]["a.txt"] == "new"
    assert json.loads(final_results[0] or "{}") == {
        "status": "success",
        "path": "a.txt",
    }
    assert call_count == 2


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_process_env_write_calls_returns_error_without_result_generation(
    mock_build_inference_engine,
    mock_inference_config,
):
    from oumi.core.configs.params.tool_params import (
        ToolAttribute,
        ToolEnvironmentAttribute,
        ToolOutputStrategy,
    )
    from oumi.core.synthesis.environment import (
        _MAX_STATE_UPDATE_RETRIES,
        GeneratedToolEnvironment,
    )

    mock_build_inference_engine.return_value = Mock()

    env = GeneratedToolEnvironment(
        ToolEnvironmentAttribute(
            id="filesystem",
            name="Filesystem",
            description="A filesystem",
            system_prompt="You manage a filesystem.",
            state_schema={
                "type": "object",
                "properties": {"files": {"type": "object"}},
            },
            initial_state={"files": {"a.txt": "old"}},
        )
    )
    tool = ToolAttribute(
        id="write_file",
        name="WriteFile",
        description="Write a file",
        output_strategy=ToolOutputStrategy.ENVIRONMENT,
        environment="filesystem",
        read_only=False,
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    )

    mock_engine = Mock()
    mock_engine.infer.return_value = [
        Conversation(
            messages=[
                Message(
                    role=Role.ASSISTANT,
                    content='{"patch": [], "error": "File is locked."}',
                )
            ]
        )
    ]
    mock_config = Mock()

    final_results: list[str | None] = [None]
    env_items = [
        (
            0,
            '<tool_call>{"name": "WriteFile", "arguments": {"path": "a.txt", '
            '"content": "new"}}</tool_call>',
            {"name": "WriteFile", "arguments": {"path": "a.txt", "content": "new"}},
            "call_001",
            env,
            tool,
        )
    ]

    _process_env_write_calls(env_items, [0], final_results, mock_engine, mock_config)

    assert env.state["files"]["a.txt"] == "old"
    assert json.loads(final_results[0] or "{}") == {
        "error": "state_update_failed",
        "message": "File is locked.",
    }
    assert mock_engine.infer.call_count == 1 + _MAX_STATE_UPDATE_RETRIES


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_planner_prompt_includes_env_state(
    mock_build_inference_engine,
    mock_inference_config,
):
    """When env_state is provided, planner prompt should contain it."""
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
    env_state = 'Environment "database":\ntables: users (10 items), orders (5 items)'

    prompt = synthesizer._create_planner_prompt(multiturn, sample, env_state=env_state)

    last_user_msg = prompt.messages[-1].content
    assert "Environment state (ground truth):" in last_user_msg
    assert "tables: users" in last_user_msg
    assert isinstance(last_user_msg, str)
    assert "exact ids from this state" in last_user_msg.lower()


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
    assert "Query" in content
    assert "Run a query" in content
    assert "NEVER fabricate tool results" in content


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_format_persona_system_msg_has_tools_not_format_rules(
    mock_build_inference_engine,
    mock_inference_config,
):
    """System message should list tools but NOT contain format rules."""
    from oumi.core.configs.params.tool_params import (
        ToolAttribute,
        ToolEnvironmentAttribute,
        ToolOutputStrategy,
    )

    mock_build_inference_engine.return_value = Mock()

    tool = ToolAttribute(
        id="query",
        name="RunQuery",
        description="Execute a SQL query",
        output_strategy=ToolOutputStrategy.ENVIRONMENT,
        environment="db",
        read_only=True,
    )
    env_config = ToolEnvironmentAttribute(
        id="db",
        name="Database",
        description="A database",
        system_prompt="You manage a database.",
    )
    params = GeneralSynthesisParams(tools=[tool], environments=[env_config])
    synthesizer = ConversationSynthesizer(params, mock_inference_config)

    result = synthesizer._format_persona({}, "You are an assistant.", [tool])
    content = result.content

    assert "RunQuery" in content
    assert "Execute a SQL query" in content
    assert "IMPORTANT RULES:" not in content
    assert "CRITICAL:" not in content


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_over_limit_tool_call_is_stripped_from_assistant_output(
    mock_build_inference_engine,
    mock_inference_config,
):
    """Tool call emitted after per-turn limit should not leak into output."""
    mock_engine = Mock()
    mock_build_inference_engine.return_value = mock_engine
    call_count = 0

    def infer_side_effect(conversations, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return [
                Conversation(
                    messages=[
                        Message(
                            role=Role.ASSISTANT,
                            content=(
                                '```json\n[{"turn": 1, "instruction": "Ask about '
                                'the order"}, {"turn": 2, "instruction": '
                                '"Investigate and conclude"}]\n```'
                            ),
                        )
                    ]
                )
            ]
        if call_count == 2:
            return [
                Conversation(
                    messages=[
                        Message(
                            role=Role.ASSISTANT,
                            content="Please check order ORD-001.",
                        )
                    ]
                )
            ]
        if call_count == 3:
            return [
                Conversation(
                    messages=[
                        Message(
                            role=Role.ASSISTANT,
                            content=(
                                '<tool_call>{"name": "SearchOrders", '
                                '"arguments": {"order_id": "ORD-001"}}'
                                "</tool_call>"
                            ),
                        )
                    ]
                )
            ]
        if call_count == 4:
            return [
                Conversation(
                    messages=[
                        Message(
                            role=Role.ASSISTANT,
                            content='{"order_id": "ORD-001", "status": "delivered"}',
                        )
                    ]
                )
            ]
        return [
            Conversation(
                messages=[
                    Message(
                        role=Role.ASSISTANT,
                        content=(
                            "Based on the current results, the order is delivered.\n\n"
                            '{"name": "SearchOrders", '
                            '"arguments": {"order_id": "ORD-001"}}'
                        ),
                    )
                ]
            )
        ]

    mock_engine.infer.side_effect = infer_side_effect

    params = _make_tool_params()
    mt_attr = MultiTurnAttribute(
        id="tool_conversation",
        min_turns=2,
        max_turns=2,
        available_tools=["search", "escalate"],
        max_tool_calls_per_turn=1,
        role_instruction_messages={
            Role.USER: "You are a customer with issue: {issue}.",
            Role.ASSISTANT: "You are a support agent.",
        },
        output_system_prompt="You are a support agent.",
    )
    synthesizer = ConversationSynthesizer(params, mock_inference_config)

    result = synthesizer.synthesize([{"issue": "check order"}], mt_attr)

    assert result[0] is not None
    conv = result[0]["tool_conversation"]
    assert isinstance(conv, dict)
    messages = conv["messages"]
    assistant_messages = [
        m for m in messages if m.get("role") == "assistant" and m.get("content")
    ]
    assert assistant_messages[-1]["content"] == (
        "Based on the current results, the order is delivered."
    )
    assert "SearchOrders" not in assistant_messages[-1]["content"]


def test_has_valid_final_assistant_message():
    valid = [
        {"role": "user", "content": "Please summarize."},
        {"role": "assistant", "content": "Perfect! Now let me verify the changes."},
    ]
    assert ConversationSynthesizer._has_valid_final_assistant_message(valid) is True

    invalid = [
        {"role": "user", "content": "Please summarize."},
    ]
    assert ConversationSynthesizer._has_valid_final_assistant_message(invalid) is False


def test_build_tool_turn_info_contains_format_instruction():
    """Turn info user message should contain tool-call reminder."""
    turn_info = ToolExecutor.build_tool_turn_info(
        current_turn=2,
        target_turns=6,
        turn_instruction="Explore the database schema",
        max_calls_reached=False,
    )

    assert "<tool_call>" in turn_info
    assert "MUST" in turn_info
    assert "Explore the database schema" in turn_info
    assert "Stay in character" not in turn_info
    assert "Generate ONLY" not in turn_info


def test_build_tool_turn_info_max_calls_no_format():
    """When max tool calls reached, turn info should request prose response."""
    turn_info = ToolExecutor.build_tool_turn_info(
        current_turn=2,
        target_turns=6,
        turn_instruction="Summarize findings",
        max_calls_reached=True,
    )

    assert "<tool_call>" not in turn_info
    assert "Respond" in turn_info or "respond" in turn_info


def test_build_tool_turn_info_final_turn_no_future_steps():
    """Final tool turn should tell the model to conclude, not defer work."""
    turn_info = ToolExecutor.build_tool_turn_info(
        current_turn=4,
        target_turns=4,
        turn_instruction="Wrap up the investigation",
        max_calls_reached=False,
    )

    assert "final turn" in turn_info.lower()
    assert "future" in turn_info.lower() or "conclude" in turn_info.lower()


def test_build_prose_turn_info_no_tool_format():
    """Prose turn info should not contain tool format instructions."""
    turn_info = ToolExecutor.build_prose_turn_info(
        current_turn=1,
        target_turns=4,
        role="USER",
        turn_instruction="Describe the problem",
    )

    assert "USER" in turn_info
    assert "Describe the problem" in turn_info
    assert "<tool_call>" not in turn_info


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_generate_plan_raises_on_response_count_mismatch(
    mock_build_inference_engine,
    mock_inference_config,
):
    """Planner should fail fast if batched inference returns wrong result count."""
    mock_engine = Mock()
    mock_engine.infer.return_value = []
    mock_build_inference_engine.return_value = mock_engine
    synthesizer = ConversationSynthesizer(
        GeneralSynthesisParams(), mock_inference_config
    )

    planner = Conversation(
        messages=[Message(role=Role.USER, content="Plan something.")]
    )
    with pytest.raises(RuntimeError, match="Conversation planning"):
        synthesizer._generate_plan([planner])


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_user_turn_history_excludes_tool_artifacts(
    mock_build_inference_engine,
    mock_inference_config,
):
    """User-role LLM must never see tool calls or tool results in its history.

    The synthesis loop maintains two separate histories:
    - full_histories: includes raw tool calls + tool results (for assistant turns)
    - conversational_histories: sanitized prose only (for user turns)

    This test verifies that when the user-role LLM is prompted after an assistant
    turn that made a tool call, none of the history messages passed to it contain
    tool call tags, tool result strings, or raw JSON tool payloads.
    """
    mock_engine = Mock()
    mock_build_inference_engine.return_value = mock_engine

    # Conversations seen by the user-turn LLM will be captured here
    user_turn_conversations: list[Conversation] = []

    def infer_side_effect(conversations, **kwargs):
        results = []
        for conv in conversations:
            first_content = str(conv.messages[0].content) if conv.messages else ""
            last_content = str(conv.messages[-1].content) if conv.messages else ""

            if "conversation planner" in first_content.lower():
                # Planner call: return a 4-turn plan (USER, ASST, USER, ASST)
                results.append(
                    Conversation(
                        messages=[
                            Message(
                                role=Role.ASSISTANT,
                                content=(
                                    "```json\n"
                                    '[{"turn": 1, "instruction": "Ask about order"},'
                                    ' {"turn": 2, "instruction": "Look up the order"},'
                                    ' {"turn": 3, "instruction": "Follow up"},'
                                    ' {"turn": 4, "instruction": "Confirm and close"}]'
                                    "\n```"
                                ),
                            )
                        ]
                    )
                )
            elif "simulating the tool" in first_content.lower():
                # Tool simulator call: return valid JSON result
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
            elif (
                "customer" in first_content.lower()
                and "[Tool result from" not in last_content
                and "<tool_call>" not in last_content
            ):
                # This is a USER turn prompt — capture it for later assertions
                user_turn_conversations.extend(conversations)
                results.append(
                    Conversation(
                        messages=[
                            Message(
                                role=Role.ASSISTANT,
                                content="I need help with my order ORD-001.",
                            )
                        ]
                    )
                )
            elif any(
                "[Tool result from SearchOrders]:" in str(m.content)
                for m in conv.messages
            ):
                # Assistant prose turn after tool result
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
            elif "MUST use <tool_call>" in last_content:
                # Assistant tool-call turn
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
                                content="Your order has been resolved.",
                            )
                        ]
                    )
                )
        return results

    mock_engine.infer.side_effect = infer_side_effect

    params = _make_tool_params()
    mt_attr = MultiTurnAttribute(
        id="tool_conversation",
        min_turns=4,
        max_turns=4,
        available_tools=["search", "escalate"],
        max_tool_calls_per_turn=3,
        role_instruction_messages={
            Role.USER: "You are a customer with issue: {issue}.",
            Role.ASSISTANT: "You are a support agent.",
        },
        output_system_prompt="You are a support agent.",
    )
    synthesizer = ConversationSynthesizer(params, mock_inference_config)

    result = synthesizer.synthesize([{"issue": "order tracking"}], mt_attr)

    # Synthesis should have produced a result
    assert result[0] is not None

    # We must have captured at least one user-turn conversation (turn 3)
    assert len(user_turn_conversations) >= 1, (
        "Expected at least one user-turn conversation to be captured"
    )

    # Verify none of the history messages passed to the user LLM contain tool artifacts
    tool_artifact_strings = [
        "<tool_call>",
        "</tool_call>",
        "[Tool result from",
    ]
    for conv in user_turn_conversations:
        # The history messages are everything between the persona (first) and
        # the turn instruction (last). Check all non-system messages.
        for msg in conv.messages:
            if msg.role == Role.SYSTEM:
                continue
            content = msg.content if isinstance(msg.content, str) else ""
            for artifact in tool_artifact_strings:
                assert artifact not in content, (
                    f"User-turn history contains tool artifact {artifact!r} "
                    f"in {msg.role.value} message: {content[:200]!r}"
                )
