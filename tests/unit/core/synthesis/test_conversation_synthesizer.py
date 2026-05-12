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

import logging
import random
from unittest.mock import MagicMock, Mock, patch

import pytest

from oumi.core.configs.environment_config import EnvironmentConfig
from oumi.core.configs.inference_config import InferenceConfig
from oumi.core.configs.inference_engine_type import InferenceEngineType
from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.configs.params.generation_params import GenerationParams
from oumi.core.configs.params.model_params import ModelParams
from oumi.core.configs.params.remote_params import RemoteParams
from oumi.core.configs.params.synthesis_params import (
    GeneralSynthesisParams,
    MultiTurnAttribute,
    SampledAttribute,
    SampledAttributeValue,
)
from oumi.core.configs.params.tool_params import ToolParams
from oumi.core.synthesis.conversation_synthesizer import ConversationSynthesizer
from oumi.core.types.conversation import (
    PLANNER_JSON_SCHEMA,
    Conversation,
    Message,
    Role,
)
from oumi.core.types.tool_call import FunctionCall, ToolCall, ToolResult
from oumi.environments.base_environment import BaseEnvironment


@pytest.fixture
def mock_inference_config():
    """Create a real InferenceConfig.

    Uses a real config rather than ``Mock(spec=InferenceConfig)`` so that
    ``dataclasses.replace`` works in ``_planner_inference_config``. The engine
    is set to OPENAI so the synthesizer's tool-calling capability check passes
    when grounding tests pass an environment with tools.
    """
    return InferenceConfig(
        engine=InferenceEngineType.OPENAI,
        model=Mock(spec=ModelParams),
        remote_params=Mock(spec=RemoteParams),
        generation=GenerationParams(),
    )


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
    assert isinstance(user_message, str)
    assert "Role context:" in user_message
    assert "You are a frustrated customer." in user_message
    assert "You are a helpful agent." in user_message

    example_response = planner.messages[2].content
    assert isinstance(example_response, str)
    # Wrapped object form, no markdown fences (matches PLANNER_JSON_SCHEMA).
    assert example_response.startswith('{"turns":')
    assert "```" not in example_response
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

    plan = (
        '{"turns": ['
        '{"turn": 1, "instruction": "Greet support and explain the issue."},'
        '{"turn": 2, "instruction": "Acknowledge and ask for details."},'
        '{"turn": 3, "instruction": "Provide order number."},'
        '{"turn": 4, "instruction": "Offer a resolution."}'
        "]}"
    )

    result = synthesizer._parse_plan(plan, target_turns=4)

    assert result is not None
    assert len(result) == 4
    assert result[0] == "Greet support and explain the issue."
    assert result[1] == "Acknowledge and ask for details."
    assert result[2] == "Provide order number."
    assert result[3] == "Offer a resolution."


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_parse_plan_unwraps_object_form(
    mock_build_inference_engine,
    mock_inference_config,
):
    """``_parse_plan`` extracts turns from the ``{"turns": [...]}`` object form.

    This is the primary shape enforced by ``PLANNER_JSON_SCHEMA`` for guided
    decoding, so the happy path must be covered directly.
    """
    mock_build_inference_engine.return_value = Mock()

    synthesizer = ConversationSynthesizer(
        GeneralSynthesisParams(),
        mock_inference_config,
    )

    plan = (
        '{"turns": ['
        '{"turn": 1, "instruction": "Greet"},'
        '{"turn": 2, "instruction": "Answer"}'
        "]}"
    )
    result = synthesizer._parse_plan(plan, target_turns=2)
    assert result == ["Greet", "Answer"]


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_generate_plan_uses_planner_only_guided_decoding(
    mock_build_inference_engine,
    mock_general_synthesis_params,
    mock_inference_config,
):
    """Planner calls get JSON guided decoding without affecting turn generation."""
    mock_inference_engine = Mock()
    mock_build_inference_engine.return_value = mock_inference_engine

    planner_result = Conversation(
        messages=[
            Message(
                role=Role.ASSISTANT,
                content=(
                    '{"turns": ['
                    '{"turn": 1, "instruction": "Ask"},'
                    '{"turn": 2, "instruction": "Answer"}'
                    "]}"
                ),
            )
        ]
    )
    turn_result = Conversation(messages=[Message(role=Role.ASSISTANT, content="Turn")])
    mock_inference_engine.infer.side_effect = [
        [planner_result],
        [turn_result],
        [turn_result],
    ]

    synthesizer = ConversationSynthesizer(
        mock_general_synthesis_params,
        mock_inference_config,
    )

    result = synthesizer.synthesize(
        [{"customer_type": "friendly", "issue": "product question"}],
        MultiTurnAttribute(
            id="test_conversation",
            min_turns=2,
            max_turns=2,
            role_instruction_messages={
                Role.USER: "You are a {customer_type} customer with issue: {issue}.",
                Role.ASSISTANT: "You are a helpful support agent.",
            },
            conversation_planner="Plan a conversation about {issue}.",
        ),
    )

    assert result[0] is not None
    planner_call = mock_inference_engine.infer.call_args_list[0].kwargs[
        "inference_config"
    ]
    turn_call = mock_inference_engine.infer.call_args_list[1].kwargs["inference_config"]

    # Planner config is a fresh copy with guided decoding set.
    assert planner_call is not mock_inference_config
    assert planner_call.generation is not mock_inference_config.generation
    assert planner_call.generation.guided_decoding is not None
    assert planner_call.generation.guided_decoding.json == PLANNER_JSON_SCHEMA
    # Turn calls use the original config — no guided decoding.
    assert turn_call is mock_inference_config
    assert turn_call.generation.guided_decoding is None
    # Original config is left untouched.
    assert mock_inference_config.generation.guided_decoding is None


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

    plan = (
        '{"turns": ['
        '{"turn": 1, "instruction": "First message."},'
        '{"turn": 3, "instruction": "Third message."}'
        "]}"
    )

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

    plan = (
        '{"turns": ['
        '{"turn": 1, "instruction": "First instruction"},'
        '{"turn": 2, "instruction": "Second instruction"}'
        "]}"
    )

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
    plan = (
        '{"turns": ['
        '{"turn": "1", "instruction": "First instruction"},'
        '{"turn": "2", "instruction": "Second instruction"}'
        "]}"
    )

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
    """Tolerate stray prose around the wrapped object — `extract_json` finds it."""
    mock_build_inference_engine.return_value = Mock()
    synthesizer = ConversationSynthesizer(
        GeneralSynthesisParams(), mock_inference_config
    )
    plan = (
        "Sure! Here is a conversation plan:\n"
        '{"turns": ['
        '{"turn": 1, "instruction": "Ask about the product"},'
        '{"turn": 2, "instruction": "Provide product details"}'
        "]}\n"
        "Let me know if you need any changes."
    )
    result = synthesizer._parse_plan(plan, target_turns=2)
    assert result is not None
    assert result[0] == "Ask about the product"
    assert result[1] == "Provide product details"


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
def test_has_empty_messages_skips_assistant_tool_call_messages(
    mock_build_inference_engine,
    mock_general_synthesis_params,
    mock_inference_config,
):
    """Assistant messages with tool_calls but content=None must NOT be flagged empty."""
    mock_build_inference_engine.return_value = Mock()
    synthesizer = ConversationSynthesizer(
        mock_general_synthesis_params,
        mock_inference_config,
    )

    conversation = Conversation(
        messages=[
            Message(role=Role.USER, content="Hello"),
            Message(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[
                    ToolCall(
                        id="c1",
                        function=FunctionCall(name="t", arguments="{}"),
                    )
                ],
            ),
            Message(role=Role.TOOL, tool_call_id="c1", content='{"result": 1}'),
            Message(role=Role.ASSISTANT, content="Final answer"),
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
        mock_synth.return_value = [
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
        ]

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
        mock_synth.return_value = [
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
        ]

        samples = [{"key": "val1"}, {"key": "val2"}]
        result = synthesizer.synthesize(samples, multiturn_attr)

    assert result == [None, None]


# --- Grounding helpers ---


def _make_synthesizer(mock_inference_config, environment_config=None):
    """Build a synthesizer with the inference engine builder patched out."""
    with patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine"):
        return ConversationSynthesizer(
            GeneralSynthesisParams(),
            mock_inference_config,
            environment_config=environment_config,
        )


def _grounded_env_params(
    env_id: str = "env1",
    tool_id: str = "lookup",
    n_entries: int = 10,
    sample_size: int = 3,
    seed: int | None = None,
):
    from oumi.core.configs.params.environment_params import EnvironmentParams
    from oumi.core.configs.params.grounding_params import (
        GroundingConfig,
        ToolGroundingConfig,
    )
    from oumi.core.configs.params.tool_params import ToolParams

    return EnvironmentParams(
        id=env_id,
        name=env_id,
        description=f"env {env_id}",
        env_type="deterministic",
        tools=[ToolParams(id=tool_id, name=tool_id, description="Look up an id.")],
        env_kwargs={
            "lookup_table": {
                tool_id: [
                    {"input": {"id": str(i)}, "output": {"title": f"title-{i}"}}
                    for i in range(n_entries)
                ]
            }
        },
        grounding=GroundingConfig(
            sample_size=sample_size,
            seed=seed,
            tools={tool_id: ToolGroundingConfig(fields=["id", "title"])},
        ),
    )


def _grounded_env_config(**env_kwargs):
    from oumi.core.configs.environment_config import EnvironmentConfig

    return EnvironmentConfig(environments=[_grounded_env_params(**env_kwargs)])


def _ungrounded_env_config():
    from oumi.core.configs.environment_config import EnvironmentConfig
    from oumi.core.configs.params.environment_params import EnvironmentParams
    from oumi.core.configs.params.tool_params import ToolParams

    return EnvironmentConfig(
        environments=[
            EnvironmentParams(
                id="env1",
                name="env1",
                description="ungrounded env",
                env_type="deterministic",
                tools=[ToolParams(id="lookup", name="lookup", description="Look up.")],
                env_kwargs={
                    "lookup_table": {
                        "lookup": [{"input": {"id": "1"}, "output": {"title": "t"}}]
                    }
                },
            )
        ]
    )


def _grounding_attr(
    available_envs: list[str] | None = None,
    available_tools: list[str] | None = None,
):
    return MultiTurnAttribute(
        id="t",
        min_turns=2,
        max_turns=2,
        role_instruction_messages={
            Role.USER: "u",
            Role.ASSISTANT: "a",
        },
        available_environments=available_envs or [],
        available_tools=available_tools or [],
    )


# --- _make_grounding_rng ---


def test_make_grounding_rng_unseeded_returns_fresh_random(mock_inference_config):

    synth = _make_synthesizer(mock_inference_config)
    rng = synth._make_grounding_rng(seed=None, sample_index=0)
    assert isinstance(rng, random.Random)


def test_make_grounding_rng_seeded_is_reproducible(mock_inference_config):
    synth = _make_synthesizer(mock_inference_config)
    rng_a = synth._make_grounding_rng(seed=42, sample_index=3)
    rng_b = synth._make_grounding_rng(seed=42, sample_index=3)
    assert [rng_a.random() for _ in range(5)] == [rng_b.random() for _ in range(5)]


def test_make_grounding_rng_seeded_varies_across_sample_indices(mock_inference_config):
    synth = _make_synthesizer(mock_inference_config)
    rng_0 = synth._make_grounding_rng(seed=42, sample_index=0)
    rng_1 = synth._make_grounding_rng(seed=42, sample_index=1)
    assert [rng_0.random() for _ in range(5)] != [rng_1.random() for _ in range(5)]


# --- _attach_grounding_facts ---


def test_attach_grounding_facts_noop_without_env_config(mock_inference_config):
    synth = _make_synthesizer(mock_inference_config)
    samples = [{"a": 1}, {"b": 2}]
    synth._attach_grounding_facts(samples, _grounding_attr())
    assert "grounding_facts" not in samples[0]
    assert "grounding_facts" not in samples[1]


def test_attach_grounding_facts_noop_when_no_env_has_grounding(mock_inference_config):
    synth = _make_synthesizer(
        mock_inference_config, environment_config=_ungrounded_env_config()
    )
    samples = [{}, {}]
    synth._attach_grounding_facts(
        samples, _grounding_attr(available_envs=["env1"], available_tools=["lookup"])
    )
    assert "grounding_facts" not in samples[0]
    assert "grounding_facts" not in samples[1]


def test_attach_grounding_facts_populates_samples(mock_inference_config):
    from oumi.core.configs.params.grounding_params import GroundingFact

    env_config = _grounded_env_config(n_entries=10, sample_size=3, seed=42)
    synth = _make_synthesizer(mock_inference_config, environment_config=env_config)
    samples = [{}, {}, {}]
    synth._attach_grounding_facts(
        samples,
        _grounding_attr(available_envs=["env1"], available_tools=["lookup"]),
    )
    for sample in samples:
        assert "grounding_facts" in sample
        assert len(sample["grounding_facts"]) == 3
        for fact in sample["grounding_facts"]:
            assert isinstance(fact, GroundingFact)


def test_attach_grounding_facts_seeded_is_reproducible(mock_inference_config):
    synth_a = _make_synthesizer(
        mock_inference_config,
        environment_config=_grounded_env_config(n_entries=20, sample_size=4, seed=7),
    )
    synth_b = _make_synthesizer(
        mock_inference_config,
        environment_config=_grounded_env_config(n_entries=20, sample_size=4, seed=7),
    )
    samples_a = [{}, {}, {}]
    samples_b = [{}, {}, {}]
    attr = _grounding_attr(available_envs=["env1"], available_tools=["lookup"])
    synth_a._attach_grounding_facts(samples_a, attr)
    synth_b._attach_grounding_facts(samples_b, attr)
    for a, b in zip(samples_a, samples_b):
        assert [f.data["id"] for f in a["grounding_facts"]] == [
            f.data["id"] for f in b["grounding_facts"]
        ]


def test_attach_grounding_facts_seeded_different_samples_differ(mock_inference_config):
    env_config = _grounded_env_config(n_entries=50, sample_size=3, seed=7)
    synth = _make_synthesizer(mock_inference_config, environment_config=env_config)
    samples = [{}, {}]
    synth._attach_grounding_facts(
        samples,
        _grounding_attr(available_envs=["env1"], available_tools=["lookup"]),
    )
    ids_0 = sorted(f.data["id"] for f in samples[0]["grounding_facts"])
    ids_1 = sorted(f.data["id"] for f in samples[1]["grounding_facts"])
    assert ids_0 != ids_1


def test_attach_grounding_facts_respects_available_environments_scoping(
    mock_inference_config,
):
    from oumi.core.configs.environment_config import EnvironmentConfig

    env_a = _grounded_env_params(
        env_id="env_a", tool_id="tool_a", n_entries=5, sample_size=2, seed=1
    )
    env_b = _grounded_env_params(
        env_id="env_b", tool_id="tool_b", n_entries=5, sample_size=2, seed=2
    )
    env_config = EnvironmentConfig(environments=[env_a, env_b])
    synth = _make_synthesizer(mock_inference_config, environment_config=env_config)
    samples = [{}]
    synth._attach_grounding_facts(
        samples,
        _grounding_attr(available_envs=["env_a"], available_tools=["tool_a"]),
    )
    assert len(samples[0]["grounding_facts"]) == 2


def test_attach_grounding_facts_filters_by_available_tools(mock_inference_config):
    from oumi.core.configs.environment_config import EnvironmentConfig
    from oumi.core.configs.params.environment_params import EnvironmentParams
    from oumi.core.configs.params.grounding_params import (
        GroundingConfig,
        ToolGroundingConfig,
    )
    from oumi.core.configs.params.tool_params import ToolParams

    env_params = EnvironmentParams(
        id="env",
        name="Env",
        description="d",
        env_type="deterministic",
        tools=[
            ToolParams(id="lookup_a", name="A", description="d"),
            ToolParams(id="lookup_b", name="B", description="d"),
        ],
        env_kwargs={
            "lookup_table": {
                "lookup_a": [{"input": {"id": "A1"}, "output": {"v": "from_a"}}],
                "lookup_b": [{"input": {"id": "B1"}, "output": {"v": "from_b"}}],
            }
        },
        grounding=GroundingConfig(
            sample_size=10,
            seed=0,
            tools={
                "lookup_a": ToolGroundingConfig(fields=["id", "v"]),
                "lookup_b": ToolGroundingConfig(fields=["id", "v"]),
            },
        ),
    )
    env_config = EnvironmentConfig(environments=[env_params])
    multiturn = MultiTurnAttribute(
        id="mt",
        min_turns=2,
        max_turns=2,
        role_instruction_messages={
            Role.USER: "user persona",
            Role.ASSISTANT: "assistant persona",
        },
        available_environments=["env"],
        available_tools=["lookup_a"],
    )
    synth = _make_synthesizer(mock_inference_config, environment_config=env_config)
    samples = [{}, {}]
    synth._attach_grounding_facts(samples, multiturn)
    for sample in samples:
        facts = sample["grounding_facts"]
        assert len(facts) == 1
        assert facts[0].data == {"id": "A1", "v": "from_a"}


def test_attach_grounding_facts_concatenates_across_multiple_envs(
    mock_inference_config,
):
    from oumi.core.configs.environment_config import EnvironmentConfig

    env_a = _grounded_env_params(
        env_id="env_a", tool_id="tool_a", n_entries=5, sample_size=2, seed=1
    )
    env_b = _grounded_env_params(
        env_id="env_b", tool_id="tool_b", n_entries=5, sample_size=3, seed=2
    )
    env_config = EnvironmentConfig(environments=[env_a, env_b])
    synth = _make_synthesizer(mock_inference_config, environment_config=env_config)
    samples = [{}]
    synth._attach_grounding_facts(
        samples, _grounding_attr(available_tools=["tool_a", "tool_b"])
    )
    assert len(samples[0]["grounding_facts"]) == 5


def test_attach_grounding_facts_truncation_emits_logger_warning(
    mock_inference_config, caplog
):

    env_config = _grounded_env_config(n_entries=2, sample_size=5, seed=1)
    synth = _make_synthesizer(mock_inference_config, environment_config=env_config)
    samples = [{}, {}]
    with caplog.at_level(logging.WARNING, logger="oumi"):
        synth._attach_grounding_facts(
            samples,
            _grounding_attr(available_envs=["env1"], available_tools=["lookup"]),
        )
    truncation_records = [
        rec for rec in caplog.records if "sample_size" in rec.getMessage()
    ]
    assert len(truncation_records) == 1
    assert "env1" in truncation_records[0].getMessage()


# --- Planner prompt grounding injection ---


def test_create_planner_prompt_injects_grounding_block_when_facts_present(
    mock_inference_config,
):
    from oumi.core.configs.params.grounding_params import GroundingFact

    env_config = _grounded_env_config(n_entries=10, sample_size=2, seed=1)
    synth = _make_synthesizer(mock_inference_config, environment_config=env_config)
    attr = MultiTurnAttribute(
        id="t",
        min_turns=2,
        max_turns=2,
        role_instruction_messages={
            Role.USER: "You are a user.",
            Role.ASSISTANT: "You are an assistant.",
        },
        available_environments=["env1"],
        available_tools=["lookup"],
    )
    sample = {
        "target_turns": 2,
        "conversation_plan": "",
        "parsed_turn_plans": [""] * 2,
        "grounding_facts": [
            GroundingFact(data={"id": "42", "title": "Dune"}),
            GroundingFact(data={"id": "7", "title": "LotR"}),
        ],
    }
    conversation = synth._create_planner_prompt(attr, sample)
    planner_user_msg = conversation.messages[-1].content
    assert isinstance(planner_user_msg, str)
    assert "Ground this plan in these specific entities" in planner_user_msg
    assert '- id="42", title="Dune"' in planner_user_msg
    assert '- id="7", title="LotR"' in planner_user_msg
    assert "Grounding rules (role-aware)" in planner_user_msg


def test_create_planner_prompt_includes_state_aware_branching_nudge(
    mock_inference_config,
):
    from oumi.core.configs.params.grounding_params import GroundingFact

    synth = _make_synthesizer(mock_inference_config)
    attr = MultiTurnAttribute(
        id="t",
        min_turns=4,
        max_turns=4,
        role_instruction_messages={
            Role.USER: "You are a user.",
            Role.ASSISTANT: "You are an assistant.",
        },
    )
    sample = {
        "target_turns": 4,
        "conversation_plan": "",
        "parsed_turn_plans": [""] * 4,
        "grounding_facts": [
            GroundingFact(data={"book_id": "B001", "status": "borrowed"}),
        ],
    }
    conversation = synth._create_planner_prompt(attr, sample)
    planner_user_msg = conversation.messages[-1].content
    assert isinstance(planner_user_msg, str)
    assert "preconditions" in planner_user_msg
    assert "recovery flow" in planner_user_msg
    assert "happy-path" in planner_user_msg


def test_create_planner_prompt_no_grounding_block_when_facts_absent(
    mock_inference_config,
):
    synth = _make_synthesizer(
        mock_inference_config, environment_config=_ungrounded_env_config()
    )
    attr = _grounding_attr(available_envs=["env1"], available_tools=["lookup"])
    sample = {
        "target_turns": 2,
        "conversation_plan": "",
        "parsed_turn_plans": [""] * 2,
    }
    conversation = synth._create_planner_prompt(attr, sample)
    planner_user_msg = conversation.messages[-1].content
    assert isinstance(planner_user_msg, str)
    assert "Ground this plan" not in planner_user_msg


def test_create_planner_prompt_empty_grounding_facts_omits_block(mock_inference_config):
    synth = _make_synthesizer(
        mock_inference_config, environment_config=_ungrounded_env_config()
    )
    attr = _grounding_attr(available_envs=["env1"], available_tools=["lookup"])
    sample = {
        "target_turns": 2,
        "conversation_plan": "",
        "parsed_turn_plans": [""] * 2,
        "grounding_facts": [],
    }
    conversation = synth._create_planner_prompt(attr, sample)
    planner_user_msg = conversation.messages[-1].content
    assert isinstance(planner_user_msg, str)
    assert "Ground this plan" not in planner_user_msg


# --- synthesize integration ---


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_synthesize_invokes_attach_grounding_facts(
    mock_build_inference_engine, mock_inference_config
):
    """End-to-end: synthesize() calls _attach_grounding_facts before planning."""
    plan_json = (
        "```json\n"
        '[{"turn": 1, "instruction": "a"}, {"turn": 2, "instruction": "b"}]\n'
        "```"
    )
    mock_inference_engine = Mock()
    mock_build_inference_engine.return_value = mock_inference_engine
    call_count = {"n": 0}

    def infer_side_effect(conversations, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return [
                Conversation(messages=[Message(role=Role.ASSISTANT, content=plan_json)])
                for _ in conversations
            ]
        return [
            Conversation(messages=[Message(role=Role.ASSISTANT, content="response")])
            for _ in conversations
        ]

    mock_inference_engine.infer.side_effect = infer_side_effect
    env_config = _grounded_env_config(n_entries=10, sample_size=2, seed=5)
    synth = ConversationSynthesizer(
        GeneralSynthesisParams(),
        mock_inference_config,
        environment_config=env_config,
    )
    attr = MultiTurnAttribute(
        id="t",
        min_turns=2,
        max_turns=2,
        role_instruction_messages={
            Role.USER: "You are a user.",
            Role.ASSISTANT: "You are an assistant.",
        },
        available_environments=["env1"],
        available_tools=["lookup"],
    )
    samples = [{}]
    result = synth.synthesize(samples, attr)
    assert "grounding_facts" in samples[0]
    assert len(samples[0]["grounding_facts"]) == 2
    assert len(result) == 1


# --- {grounding_facts} placeholder misuse warning ---


def test_warn_on_grounding_placeholder_warns_in_user_persona(
    mock_inference_config, caplog
):

    synth = _make_synthesizer(mock_inference_config)
    attr = MultiTurnAttribute(
        id="t",
        min_turns=2,
        max_turns=2,
        role_instruction_messages={
            Role.USER: "You are a user interested in {grounding_facts}.",
            Role.ASSISTANT: "You are an assistant.",
        },
    )
    with caplog.at_level(logging.WARNING, logger="oumi"):
        synth._warn_on_grounding_placeholder(attr)
    warnings = [rec for rec in caplog.records if "grounding_facts" in rec.getMessage()]
    assert len(warnings) >= 1
    assert "user" in warnings[0].getMessage().lower()


def test_warn_on_grounding_placeholder_warns_in_assistant_persona(
    mock_inference_config, caplog
):

    synth = _make_synthesizer(mock_inference_config)
    attr = MultiTurnAttribute(
        id="t",
        min_turns=2,
        max_turns=2,
        role_instruction_messages={
            Role.USER: "You are a user.",
            Role.ASSISTANT: "You know these entities: {grounding_facts}.",
        },
    )
    with caplog.at_level(logging.WARNING, logger="oumi"):
        synth._warn_on_grounding_placeholder(attr)
    warnings = [rec for rec in caplog.records if "grounding_facts" in rec.getMessage()]
    assert len(warnings) >= 1
    assert "assistant" in warnings[0].getMessage().lower()


def test_warn_on_grounding_placeholder_no_warning_when_placeholder_absent(
    mock_inference_config, caplog
):

    synth = _make_synthesizer(mock_inference_config)
    attr = MultiTurnAttribute(
        id="t",
        min_turns=2,
        max_turns=2,
        role_instruction_messages={
            Role.USER: "You are a user.",
            Role.ASSISTANT: "You are an assistant.",
        },
    )
    with caplog.at_level(logging.WARNING, logger="oumi"):
        synth._warn_on_grounding_placeholder(attr)
    grounding_warnings = [
        rec for rec in caplog.records if "grounding_facts" in rec.getMessage()
    ]
    assert grounding_warnings == []


# ---------------------------------------------------------------------------
# Tests for init-time engine validation
# ---------------------------------------------------------------------------


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_init_raises_on_unsupported_engine_with_tools(
    mock_build_inference_engine,
    mock_general_synthesis_params,
):
    """Unsupported engine + env with tools → ValueError at init."""
    mock_build_inference_engine.return_value = Mock()

    env_config = MagicMock(spec=EnvironmentConfig)
    env_config.all_tools = [ToolParams(id="my_tool", name="My Tool", description="x")]

    inference_config = InferenceConfig(
        engine=InferenceEngineType.LLAMACPP,
        model=Mock(spec=ModelParams),
        generation=GenerationParams(),
    )

    with pytest.raises(ValueError, match="native tool-calling"):
        ConversationSynthesizer(
            mock_general_synthesis_params,
            inference_config,
            environment_config=env_config,
        )


@patch("oumi.core.synthesis.tool_router.build_environment")
@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_init_no_error_on_supported_engine_with_tools(
    mock_build_inference_engine,
    mock_build_environment,
    mock_general_synthesis_params,
):
    """Supported engine + env with tools → no error at init."""
    mock_build_inference_engine.return_value = Mock()
    mock_build_environment.return_value = Mock()

    env_config = MagicMock(spec=EnvironmentConfig)
    env_config.all_tools = [ToolParams(id="my_tool", name="My Tool", description="x")]
    env_config.environments = []
    env_config.tool_environment_map = {}

    inference_config = InferenceConfig(
        engine=InferenceEngineType.OPENAI,
        model=Mock(spec=ModelParams),
        remote_params=Mock(spec=RemoteParams),
        generation=GenerationParams(),
    )

    ConversationSynthesizer(
        mock_general_synthesis_params,
        inference_config,
        environment_config=env_config,
    )


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_init_no_error_on_unsupported_engine_without_tools(
    mock_build_inference_engine,
    mock_general_synthesis_params,
):
    """Unsupported engine but no env tools → no error at init."""
    mock_build_inference_engine.return_value = Mock()

    inference_config = InferenceConfig(
        engine=InferenceEngineType.LLAMACPP,
        model=Mock(spec=ModelParams),
        generation=GenerationParams(),
    )

    ConversationSynthesizer(
        mock_general_synthesis_params,
        inference_config,
        environment_config=None,
    )


@patch("oumi.core.synthesis.tool_router.build_environment")
@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_synthesize_attaches_tools_to_assistant_prompt(
    mock_build_inference_engine,
    mock_build_environment,
    mock_general_synthesis_params,
):
    """Assistant prompts must have Conversation.tools populated when env has tools."""
    # Capture only the turn prompts (not the planner prompts).
    # The planner infer call uses a config with guided_decoding set; turn prompts don't.
    turn_prompts: list[Conversation] = []

    def capturing_infer(prompts, inference_config=None):
        if (
            inference_config is None
            or inference_config.generation.guided_decoding is None
        ):
            turn_prompts.extend(prompts)
        return [
            Conversation(messages=[Message(role=Role.ASSISTANT, content="ok")])
            for _ in prompts
        ]

    mock_engine = Mock()
    mock_engine.infer.side_effect = capturing_infer
    mock_build_inference_engine.return_value = mock_engine
    mock_build_environment.return_value = Mock()

    env_config = MagicMock(spec=EnvironmentConfig)
    env_config.all_tools = [ToolParams(id="lookup", name="lookup", description="x")]
    env_config.environments = []
    env_config.tool_environment_map = {}

    inference_config = InferenceConfig(
        engine=InferenceEngineType.OPENAI,
        model=Mock(spec=ModelParams),
        remote_params=Mock(spec=RemoteParams),
        generation=GenerationParams(),
    )

    multiturn_attr = MultiTurnAttribute(
        id="dialog",
        min_turns=2,
        max_turns=2,
        role_instruction_messages={
            Role.USER: "user",
            Role.ASSISTANT: "assistant",
        },
    )

    synth = ConversationSynthesizer(
        mock_general_synthesis_params,
        inference_config,
        environment_config=env_config,
    )
    with patch.object(
        synth,
        "_resolve_available_tools",
        return_value=env_config.all_tools,
    ):
        synth.synthesize(
            samples=[{"target_turns": 2, "parsed_turn_plans": []}],
            multiturn_attributes=multiturn_attr,
        )

    assert len(turn_prompts) == 2
    user_prompt, assistant_prompt = turn_prompts[0], turn_prompts[1]
    assert user_prompt.tools is None
    assert assistant_prompt.tools is not None
    assert len(assistant_prompt.tools) == 1
    assert assistant_prompt.tools[0].function.name == "lookup"


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_synthesize_no_tools_when_env_has_none(
    mock_build_inference_engine,
    mock_general_synthesis_params,
):
    """If env has no tools, no prompt gets tools attached."""
    captured_prompts: list[Conversation] = []

    def capturing_infer(prompts, inference_config=None):
        captured_prompts.extend(prompts)
        return [
            Conversation(messages=[Message(role=Role.ASSISTANT, content="ok")])
            for _ in prompts
        ]

    mock_engine = Mock()
    mock_engine.infer.side_effect = capturing_infer
    mock_build_inference_engine.return_value = mock_engine

    inference_config = InferenceConfig(
        engine=InferenceEngineType.OPENAI,
        model=Mock(spec=ModelParams),
        remote_params=Mock(spec=RemoteParams),
        generation=GenerationParams(),
    )

    multiturn_attr = MultiTurnAttribute(
        id="dialog",
        min_turns=2,
        max_turns=2,
        role_instruction_messages={
            Role.USER: "user",
            Role.ASSISTANT: "assistant",
        },
    )

    synth = ConversationSynthesizer(
        mock_general_synthesis_params,
        inference_config,
        environment_config=None,
    )
    synth.synthesize(
        samples=[{"target_turns": 2, "parsed_turn_plans": []}],
        multiturn_attributes=multiturn_attr,
    )

    assert all(p.tools is None for p in captured_prompts)


# ---------------------------------------------------------------------------
# Tests for the assistant tool-call loop
# ---------------------------------------------------------------------------


def _make_env_config(env_id: str, tool_id: str) -> MagicMock:
    """Build a MagicMock(spec=EnvironmentConfig) with a single tool/env."""
    env_params = EnvironmentParams(
        id=env_id,
        name="x",
        description="x",
        env_type="deterministic",
        tools=[],
    )
    env_config = MagicMock(spec=EnvironmentConfig)
    env_config.environments = [env_params]
    env_config.all_tools = [ToolParams(id=tool_id, name="x", description="x")]
    env_config.tool_environment_map = {tool_id: env_id}
    return env_config


@patch("oumi.core.synthesis.tool_router.build_environment")
@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_dispatch_tool_calls_routes_through_env(
    mock_build_inference_engine,
    mock_build_environment,
    mock_general_synthesis_params,
):
    """A valid ToolCall produces a Role.TOOL message via batched env.step()."""
    mock_build_inference_engine.return_value = Mock()

    fake_env = Mock(spec=BaseEnvironment)
    fake_env.step.return_value = [ToolResult(output={"city": "Paris"})]
    mock_build_environment.return_value = fake_env

    env_config = _make_env_config("weather", "get_weather")

    inference_config = InferenceConfig(
        engine=InferenceEngineType.OPENAI,
        model=Mock(spec=ModelParams),
        remote_params=Mock(spec=RemoteParams),
        generation=GenerationParams(),
    )

    synth = ConversationSynthesizer(
        mock_general_synthesis_params,
        inference_config,
        environment_config=env_config,
    )

    tc = ToolCall(
        id="call_1",
        function=FunctionCall(name="get_weather", arguments='{"city": "Paris"}'),
    )
    [msg] = synth._dispatch_tool_calls([tc])

    assert msg.role == Role.TOOL
    assert msg.tool_call_id == "call_1"
    assert msg.content == '{"city": "Paris"}'
    fake_env.step.assert_called_once_with([("get_weather", {"city": "Paris"})])


@patch("oumi.core.synthesis.tool_router.build_environment")
@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_dispatch_tool_calls_folds_same_env_calls_into_one_step(
    mock_build_inference_engine,
    mock_build_environment,
    mock_general_synthesis_params,
):
    """Multiple calls to the same env collapse into one batched step()."""
    mock_build_inference_engine.return_value = Mock()

    fake_env = Mock(spec=BaseEnvironment)
    fake_env.step.return_value = [
        ToolResult(output={"i": 0}),
        ToolResult(output={"i": 1}),
        ToolResult(output={"i": 2}),
    ]
    mock_build_environment.return_value = fake_env

    env_config = _make_env_config("e", "t")
    inference_config = InferenceConfig(
        engine=InferenceEngineType.OPENAI,
        model=Mock(spec=ModelParams),
        remote_params=Mock(spec=RemoteParams),
        generation=GenerationParams(),
    )
    synth = ConversationSynthesizer(
        mock_general_synthesis_params,
        inference_config,
        environment_config=env_config,
    )

    tcs = [
        ToolCall(id=f"c{i}", function=FunctionCall(name="t", arguments=f'{{"i": {i}}}'))
        for i in range(3)
    ]
    msgs = synth._dispatch_tool_calls(tcs)
    assert [m.tool_call_id for m in msgs] == ["c0", "c1", "c2"]
    assert [m.content for m in msgs] == ['{"i": 0}', '{"i": 1}', '{"i": 2}']
    fake_env.step.assert_called_once_with(
        [("t", {"i": 0}), ("t", {"i": 1}), ("t", {"i": 2})]
    )


@patch("oumi.core.synthesis.tool_router.build_environment")
@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_dispatch_tool_calls_handles_malformed_arguments(
    mock_build_inference_engine,
    mock_build_environment,
    mock_general_synthesis_params,
):
    """Malformed JSON in function.arguments → error TOOL message, env not called."""
    mock_build_inference_engine.return_value = Mock()
    fake_env = Mock(spec=BaseEnvironment)
    mock_build_environment.return_value = fake_env

    env_config = _make_env_config("e", "t")
    inference_config = InferenceConfig(
        engine=InferenceEngineType.OPENAI,
        model=Mock(spec=ModelParams),
        remote_params=Mock(spec=RemoteParams),
        generation=GenerationParams(),
    )
    synth = ConversationSynthesizer(
        mock_general_synthesis_params,
        inference_config,
        environment_config=env_config,
    )

    tc = ToolCall(
        id="call_x",
        function=FunctionCall(name="t", arguments="{not valid json"),
    )
    [msg] = synth._dispatch_tool_calls([tc])
    assert msg.role == Role.TOOL
    assert msg.tool_call_id == "call_x"
    assert "arguments are not valid JSON" in str(msg.content)
    fake_env.step.assert_not_called()


@patch("oumi.core.synthesis.tool_router.build_environment")
@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_dispatch_tool_calls_handles_non_dict_arguments(
    mock_build_inference_engine,
    mock_build_environment,
    mock_general_synthesis_params,
):
    """Tool arguments that parse as JSON but aren't a dict yield an error."""
    mock_build_inference_engine.return_value = Mock()

    fake_env = Mock(spec=BaseEnvironment)
    mock_build_environment.return_value = fake_env

    env_config = _make_env_config("e", "t")

    inference_config = InferenceConfig(
        engine=InferenceEngineType.OPENAI,
        model=Mock(spec=ModelParams),
        remote_params=Mock(spec=RemoteParams),
        generation=GenerationParams(),
    )
    synth = ConversationSynthesizer(
        mock_general_synthesis_params,
        inference_config,
        environment_config=env_config,
    )

    tc = ToolCall(id="c", function=FunctionCall(name="t", arguments="[1, 2, 3]"))
    [msg] = synth._dispatch_tool_calls([tc])
    assert msg.role == Role.TOOL
    assert msg.tool_call_id == "c"
    assert "must be a JSON object" in str(msg.content)
    fake_env.step.assert_not_called()


@patch("oumi.core.synthesis.tool_router.build_environment")
@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_dispatch_tool_calls_handles_unknown_tool(
    mock_build_inference_engine,
    mock_build_environment,
    mock_general_synthesis_params,
):
    """Unknown tool name → error TOOL message, env not invoked."""
    mock_build_inference_engine.return_value = Mock()
    fake_env = Mock(spec=BaseEnvironment)
    mock_build_environment.return_value = fake_env

    env_config = _make_env_config("e", "known_tool")
    inference_config = InferenceConfig(
        engine=InferenceEngineType.OPENAI,
        model=Mock(spec=ModelParams),
        remote_params=Mock(spec=RemoteParams),
        generation=GenerationParams(),
    )
    synth = ConversationSynthesizer(
        mock_general_synthesis_params,
        inference_config,
        environment_config=env_config,
    )

    tc = ToolCall(
        id="c",
        function=FunctionCall(name="ghost_tool", arguments="{}"),
    )
    [msg] = synth._dispatch_tool_calls([tc])
    assert msg.role == Role.TOOL
    assert "Unknown tool 'ghost_tool'" in str(msg.content)
    fake_env.step.assert_not_called()


@patch("oumi.core.synthesis.tool_router.build_environment")
@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_dispatch_tool_calls_handles_env_exception_with_per_call_fallback(
    mock_build_inference_engine,
    mock_build_environment,
    mock_general_synthesis_params,
):
    """Batched step() raising → fall back to per-call dispatch for attribution."""
    mock_build_inference_engine.return_value = Mock()
    fake_env = Mock(spec=BaseEnvironment)
    fake_env.step.side_effect = RuntimeError("boom")
    mock_build_environment.return_value = fake_env

    env_config = _make_env_config("e", "t")
    inference_config = InferenceConfig(
        engine=InferenceEngineType.OPENAI,
        model=Mock(spec=ModelParams),
        remote_params=Mock(spec=RemoteParams),
        generation=GenerationParams(),
    )
    synth = ConversationSynthesizer(
        mock_general_synthesis_params,
        inference_config,
        environment_config=env_config,
    )

    tc = ToolCall(id="c", function=FunctionCall(name="t", arguments="{}"))
    [msg] = synth._dispatch_tool_calls([tc])
    assert msg.role == Role.TOOL
    assert "Tool 't' raised: boom" in str(msg.content)


@patch("oumi.core.synthesis.tool_router.build_environment")
@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_assistant_turn_loops_on_tool_calls(
    mock_build_inference_engine,
    mock_build_environment,
    mock_general_synthesis_params,
):
    """Assistant turn loops: tool_calls -> dispatch -> re-infer -> final text."""
    fake_env = Mock(spec=BaseEnvironment)
    fake_env.step.side_effect = lambda calls: [
        ToolResult(output={"answer": 42}) for _ in calls
    ]
    mock_build_environment.return_value = fake_env

    turn_call_count = {"n": 0}

    def scripted_infer(prompts, inference_config=None):
        # Planner inference uses guided_decoding; turn inference doesn't.
        if (
            inference_config is not None
            and inference_config.generation.guided_decoding is not None
        ):
            return [
                Conversation(
                    messages=[Message(role=Role.ASSISTANT, content='{"turns": []}')]
                )
                for _ in prompts
            ]
        turn_call_count["n"] += 1
        last_msg = prompts[0].messages[-1]
        last_text = last_msg.content if isinstance(last_msg.content, str) else ""
        if "USER" in last_text:
            return [
                Conversation(messages=[Message(role=Role.ASSISTANT, content="hello")])
                for _ in prompts
            ]
        if turn_call_count["n"] == 2:
            return [
                Conversation(
                    messages=[
                        Message(
                            role=Role.ASSISTANT,
                            content=None,
                            tool_calls=[
                                ToolCall(
                                    id="c1",
                                    function=FunctionCall(name="t", arguments="{}"),
                                )
                            ],
                        )
                    ]
                )
                for _ in prompts
            ]
        return [
            Conversation(messages=[Message(role=Role.ASSISTANT, content="done")])
            for _ in prompts
        ]

    mock_engine = Mock()
    mock_engine.infer.side_effect = scripted_infer
    mock_build_inference_engine.return_value = mock_engine

    env_config = _make_env_config("e", "t")

    multiturn_attr = MultiTurnAttribute(
        id="dialog",
        min_turns=2,
        max_turns=2,
        role_instruction_messages={
            Role.USER: "user",
            Role.ASSISTANT: "assistant",
        },
        max_consecutive_tool_turns=5,
    )
    inference_config = InferenceConfig(
        engine=InferenceEngineType.OPENAI,
        model=Mock(spec=ModelParams),
        remote_params=Mock(spec=RemoteParams),
        generation=GenerationParams(),
    )
    synth = ConversationSynthesizer(
        mock_general_synthesis_params,
        inference_config,
        environment_config=env_config,
    )
    with patch.object(
        synth, "_resolve_available_tools", return_value=env_config.all_tools
    ):
        result = synth.synthesize(
            samples=[{"target_turns": 2, "parsed_turn_plans": []}],
            multiturn_attributes=multiturn_attr,
        )

    assert len(result) == 1
    record = result[0]
    assert record is not None
    conv = record["dialog"]
    assert isinstance(conv, dict)
    msgs = conv["messages"]
    roles = [m["role"] for m in msgs]
    assert "tool" in roles, f"Expected a tool message in {roles}"
    last_tool = max(i for i, m in enumerate(msgs) if m["role"] == "tool")
    assert any(
        m["role"] == "assistant" and m.get("content") == "done"
        for m in msgs[last_tool + 1 :]
    ), f"Expected final assistant text after tool message: {msgs}"
    assert fake_env.step.call_count == 1


@patch("oumi.core.synthesis.tool_router.build_environment")
@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_assistant_turn_caps_at_max_consecutive_tool_turns_then_finalizes(
    mock_build_inference_engine,
    mock_build_environment,
    mock_general_synthesis_params,
):
    """When max_consecutive_tool_turns is hit, the nudge forces a final text answer."""
    fake_env = Mock(spec=BaseEnvironment)
    fake_env.step.side_effect = lambda calls: [ToolResult(output="ok") for _ in calls]
    mock_build_environment.return_value = fake_env

    def scripted_infer(prompts, inference_config=None):
        if (
            inference_config is not None
            and inference_config.generation.guided_decoding is not None
        ):
            return [
                Conversation(
                    messages=[Message(role=Role.ASSISTANT, content='{"turns": []}')]
                )
                for _ in prompts
            ]
        last_msg = prompts[0].messages[-1]
        last_text = last_msg.content if isinstance(last_msg.content, str) else ""
        if last_text.startswith("Stop calling tools"):
            return [
                Conversation(
                    messages=[
                        Message(role=Role.ASSISTANT, content="forced final answer")
                    ]
                )
                for _ in prompts
            ]
        if "USER" in last_text:
            return [
                Conversation(messages=[Message(role=Role.ASSISTANT, content="hello")])
                for _ in prompts
            ]
        return [
            Conversation(
                messages=[
                    Message(
                        role=Role.ASSISTANT,
                        content=None,
                        tool_calls=[
                            ToolCall(
                                id="loop",
                                function=FunctionCall(name="t", arguments="{}"),
                            )
                        ],
                    )
                ]
            )
            for _ in prompts
        ]

    mock_engine = Mock()
    mock_engine.infer.side_effect = scripted_infer
    mock_build_inference_engine.return_value = mock_engine

    env_config = _make_env_config("e", "t")

    multiturn_attr = MultiTurnAttribute(
        id="dialog",
        min_turns=2,
        max_turns=2,
        role_instruction_messages={
            Role.USER: "user",
            Role.ASSISTANT: "assistant",
        },
        max_consecutive_tool_turns=2,
    )
    inference_config = InferenceConfig(
        engine=InferenceEngineType.OPENAI,
        model=Mock(spec=ModelParams),
        remote_params=Mock(spec=RemoteParams),
        generation=GenerationParams(),
    )
    synth = ConversationSynthesizer(
        mock_general_synthesis_params,
        inference_config,
        environment_config=env_config,
    )
    with patch.object(
        synth, "_resolve_available_tools", return_value=env_config.all_tools
    ):
        result = synth.synthesize(
            samples=[{"target_turns": 2, "parsed_turn_plans": []}],
            multiturn_attributes=multiturn_attr,
        )

    assert len(result) == 1
    record = result[0]
    assert record is not None
    conv = record["dialog"]
    assert isinstance(conv, dict)
    msgs = conv["messages"]
    contents = [m.get("content") for m in msgs]
    assert "forced final answer" in contents, (
        f"Expected nudge to produce final answer, got contents: {contents}"
    )
    assert fake_env.step.call_count == 2


@patch("oumi.core.synthesis.tool_router.build_environment")
@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_assistant_turn_dispatches_parallel_batch_unrestricted(
    mock_build_inference_engine,
    mock_build_environment,
    mock_general_synthesis_params,
):
    """A parallel tool_calls batch is folded into one step() invocation."""
    fake_env = Mock(spec=BaseEnvironment)
    fake_env.step.side_effect = lambda calls: [
        ToolResult(output={"ok": True}) for _ in calls
    ]
    mock_build_environment.return_value = fake_env

    assistant_turn_count = {"n": 0}

    def scripted_infer(prompts, inference_config=None):
        if (
            inference_config is not None
            and inference_config.generation.guided_decoding is not None
        ):
            return [
                Conversation(
                    messages=[Message(role=Role.ASSISTANT, content='{"turns": []}')]
                )
                for _ in prompts
            ]
        last_msg = prompts[0].messages[-1]
        last_text = last_msg.content if isinstance(last_msg.content, str) else ""
        if "USER" in last_text:
            return [
                Conversation(messages=[Message(role=Role.ASSISTANT, content="hello")])
                for _ in prompts
            ]
        assistant_turn_count["n"] += 1
        if assistant_turn_count["n"] == 1:
            return [
                Conversation(
                    messages=[
                        Message(
                            role=Role.ASSISTANT,
                            content=None,
                            tool_calls=[
                                ToolCall(
                                    id=f"c{i}",
                                    function=FunctionCall(name="t", arguments="{}"),
                                )
                                for i in range(5)
                            ],
                        )
                    ]
                )
                for _ in prompts
            ]
        return [
            Conversation(messages=[Message(role=Role.ASSISTANT, content="done")])
            for _ in prompts
        ]

    mock_engine = Mock()
    mock_engine.infer.side_effect = scripted_infer
    mock_build_inference_engine.return_value = mock_engine

    env_config = _make_env_config("e", "t")

    multiturn_attr = MultiTurnAttribute(
        id="dialog",
        min_turns=2,
        max_turns=2,
        role_instruction_messages={
            Role.USER: "user",
            Role.ASSISTANT: "assistant",
        },
        max_consecutive_tool_turns=2,
    )
    inference_config = InferenceConfig(
        engine=InferenceEngineType.OPENAI,
        model=Mock(spec=ModelParams),
        remote_params=Mock(spec=RemoteParams),
        generation=GenerationParams(),
    )
    synth = ConversationSynthesizer(
        mock_general_synthesis_params,
        inference_config,
        environment_config=env_config,
    )
    with patch.object(
        synth, "_resolve_available_tools", return_value=env_config.all_tools
    ):
        synth.synthesize(
            samples=[{"target_turns": 2, "parsed_turn_plans": []}],
            multiturn_attributes=multiturn_attr,
        )

    # 5 parallel tool calls fold into a single batched step() invocation.
    assert fake_env.step.call_count == 1
    [call] = fake_env.step.call_args_list
    assert len(call.args[0]) == 5


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_synthesizer_attaches_inference_to_synthetic_env(
    mock_build_inference_engine,
    mock_general_synthesis_params,
):
    """SyntheticEnvironments built via _tool_dispatch get attach_inference()."""
    from oumi.environments.synthetic_environment import SyntheticEnvironment

    mock_engine = Mock()
    mock_build_inference_engine.return_value = mock_engine

    env_params = EnvironmentParams(
        id="docs",
        name="Docs",
        description="Synthetic docs env",
        env_type="synthetic",
        tools=[
            ToolParams(
                id="lookup",
                name="Lookup",
                description="Look up docs.",
                parameters={
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            )
        ],
        env_kwargs={"system_prompt": "Simulate the lookup tool."},
    )
    env_config = EnvironmentConfig(environments=[env_params])

    inference_config = InferenceConfig(
        engine=InferenceEngineType.OPENAI,
        model=Mock(spec=ModelParams),
        remote_params=Mock(spec=RemoteParams),
        generation=GenerationParams(),
    )
    synth = ConversationSynthesizer(
        mock_general_synthesis_params,
        inference_config,
        environment_config=env_config,
    )

    assert synth._router is not None
    env = synth._router.tool_to_env["lookup"]
    assert isinstance(env, SyntheticEnvironment)
    assert env._engine is mock_engine
    assert env._base_inference_config is inference_config
