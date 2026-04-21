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
from oumi.core.configs.params.tool_params import (
    ToolParams,
    ToolSchema,
)
from oumi.core.synthesis.conversation_synthesizer import ConversationSynthesizer
from oumi.core.types.conversation import (
    PLANNER_JSON_SCHEMA,
    Conversation,
    Message,
    Role,
)
from oumi.environments.deterministic_tool import (
    DeterministicTool,
    DeterministicToolOutput,
)


def _unwrap_tool_result(content: object) -> str:
    """Strip the ``<tool_result>...</tool_result>`` wrapper from a tool-result
    message so assertions can inspect the inner JSON/text payload.

    The synthesizer emits tool results as ``Role.USER`` messages wrapped in
    ``<tool_result>`` markers (keeps the conversation portable across provider
    APIs that disallow a ``tool`` role). Tests use this helper to peek at the
    underlying content. ``content`` is typed as ``object`` so callers can
    pass ``Message.content`` (a ``str | list[ContentItem]``) without first
    narrowing the type; the runtime assertion makes str-ness explicit.
    """
    prefix = "<tool_result>"
    suffix = "</tool_result>"
    assert isinstance(content, str), f"expected str content, got {type(content)}"
    assert content.startswith(prefix) and content.endswith(suffix), (
        f"Expected <tool_result>-wrapped content, got: {content!r}"
    )
    return content[len(prefix) : -len(suffix)]


@pytest.fixture
def mock_inference_config():
    """Create a mock inference config."""
    return InferenceConfig(
        engine=InferenceEngineType.NATIVE,
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
def test_format_persona_injects_tools_for_assistant_only(
    mock_build_inference_engine,
    mock_inference_config,
):
    """Test assistant persona gets tool context without mutating config."""
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
            Role.USER: "You are a user.",
            Role.ASSISTANT: "You are a helpful agent.",
        },
    )
    tool = ToolParams.create(
        {
            "id": "lookup_order",
            "name": "Lookup Order",
            "description": "Look up an order by id.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        }
    )

    with patch.object(synthesizer, "_resolve_available_tools", return_value=[tool]):
        assistant_message = synthesizer._format_persona(
            {},
            multiturn_attr.role_instruction_messages[Role.ASSISTANT],
            Role.ASSISTANT,
            multiturn_attribute=multiturn_attr,
        )
        user_message = synthesizer._format_persona(
            {},
            multiturn_attr.role_instruction_messages[Role.USER],
            Role.USER,
            multiturn_attribute=multiturn_attr,
        )

    assistant_content = assistant_message.content
    user_content = user_message.content
    assert isinstance(assistant_content, str)
    assert isinstance(user_content, str)
    assert "You have access to the following tools." in assistant_content
    assert '"name": "lookup_order"' in assistant_content
    assert '"display_name": "Lookup Order"' in assistant_content
    assert '"description": "Look up an order by id."' in assistant_content
    assert '"order_id"' in assistant_content
    assert "You have access to the following tools." not in user_content
    assert (
        multiturn_attr.role_instruction_messages[Role.ASSISTANT]
        == "You are a helpful agent."
    )


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
def test_build_role_context_includes_tools_for_assistant(
    mock_build_inference_engine,
    mock_inference_config,
):
    """Test planner role context includes assistant tool awareness."""
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
            Role.USER: "You are a customer.",
            Role.ASSISTANT: "You are a helpful agent.",
        },
    )
    tool = ToolParams.create(
        {
            "id": "check_status",
            "name": "Check Status",
            "description": "Check order status.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }
    )

    with patch.object(synthesizer, "_resolve_available_tools", return_value=[tool]):
        result = synthesizer._build_role_context({}, multiturn_attr)

    assert "[ASSISTANT]" in result
    assert "You have access to the following tools." in result
    assert '"name": "check_status"' in result
    assert '"display_name": "Check Status"' in result
    assert '"description": "Check order status."' in result


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
    assert example_response.startswith('{"turns":')
    assert '"turn": 1' in example_response
    assert '"instruction"' in example_response
    system_content = planner.messages[0].content
    assert isinstance(system_content, str)
    assert "raw JSON object" in system_content
    assert "```json" not in system_content
    assert "No markdown" in user_message


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_generate_plan_uses_planner_only_guided_decoding(
    mock_build_inference_engine,
    mock_general_synthesis_params,
    mock_multiturn_attribute,
    mock_inference_config,
):
    """Test planner calls get JSON guided decoding without affecting turn generation."""
    mock_inference_engine = Mock()
    mock_build_inference_engine.return_value = mock_inference_engine

    planner_result = Conversation(
        messages=[
            Message(
                role=Role.ASSISTANT,
                content=(
                    '[{"turn": 1, "instruction": "Ask"}, '
                    '{"turn": 2, "instruction": "Answer"}]'
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

    assert planner_call is not mock_inference_config
    assert planner_call.generation is not mock_inference_config.generation
    assert planner_call.generation.guided_decoding is not None
    assert planner_call.generation.guided_decoding.json == PLANNER_JSON_SCHEMA
    assert turn_call is mock_inference_config
    assert turn_call.generation.guided_decoding is None
    assert mock_inference_config.generation.guided_decoding is None


@patch("oumi.core.synthesis.conversation_synthesizer.build_inference_engine")
def test_parse_plan_unwraps_object_form(
    mock_build_inference_engine,
    mock_inference_config,
):
    """Test that _parse_plan extracts turns from the {"turns": [...]} object form.

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


# --- _make_grounding_rng ---


def test_make_grounding_rng_unseeded_returns_fresh_random(mock_inference_config):
    import random as _random

    synth = _make_synthesizer(mock_inference_config)
    rng = synth._make_grounding_rng(seed=None, sample_index=0)
    assert isinstance(rng, _random.Random)


def test_make_grounding_rng_seeded_is_reproducible(mock_inference_config):
    synth = _make_synthesizer(mock_inference_config)
    rng_a = synth._make_grounding_rng(seed=42, sample_index=3)
    rng_b = synth._make_grounding_rng(seed=42, sample_index=3)
    assert [rng_a.random() for _ in range(5)] == [rng_b.random() for _ in range(5)]


def test_make_grounding_rng_seeded_varies_across_sample_indices(
    mock_inference_config,
):
    synth = _make_synthesizer(mock_inference_config)
    rng_0 = synth._make_grounding_rng(seed=42, sample_index=0)
    rng_1 = synth._make_grounding_rng(seed=42, sample_index=1)
    assert [rng_0.random() for _ in range(5)] != [rng_1.random() for _ in range(5)]


# --- _attach_grounding_facts ---


def _grounded_env_params(
    env_id: str = "env1",
    tool_id: str = "lookup",
    n_entries: int = 10,
    sample_size: int = 3,
    seed: int | None = None,
):
    from oumi.core.configs.params.environment_params import EnvironmentParams
    from oumi.core.configs.params.grounding_params import GroundingConfig
    from oumi.environments.deterministic_tool import (
        DeterministicTool,
        DeterministicToolOutput,
    )

    outputs = [
        DeterministicToolOutput(
            input={"id": str(i)},
            output={"title": f"title-{i}"},
        )
        for i in range(n_entries)
    ]
    return EnvironmentParams(
        id=env_id,
        name=env_id,
        description=f"env {env_id}",
        env_type="deterministic",
        grounding=GroundingConfig(sample_size=sample_size, seed=seed),
        tools=[
            DeterministicTool(
                id=tool_id,
                name=tool_id,
                description="Look up an id.",
                deterministic_outputs=outputs,
            )
        ],
    )


def _grounded_env_config(**env_kwargs):
    from oumi.core.configs.environment_config import EnvironmentConfig

    return EnvironmentConfig(environments=[_grounded_env_params(**env_kwargs)])


def _ungrounded_env_config():
    from oumi.core.configs.environment_config import EnvironmentConfig
    from oumi.core.configs.params.environment_params import EnvironmentParams
    from oumi.environments.deterministic_tool import (
        DeterministicTool,
        DeterministicToolOutput,
    )

    return EnvironmentConfig(
        environments=[
            EnvironmentParams(
                id="env1",
                name="env1",
                description="ungrounded env",
                env_type="deterministic",
                tools=[
                    DeterministicTool(
                        id="lookup",
                        name="lookup",
                        description="Look up an id.",
                        deterministic_outputs=[
                            DeterministicToolOutput(
                                input={"id": "1"}, output={"title": "t"}
                            )
                        ],
                    )
                ],
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


def test_attach_grounding_facts_noop_without_env_config(mock_inference_config):
    synth = _make_synthesizer(mock_inference_config)
    samples = [{"a": 1}, {"b": 2}]

    synth._attach_grounding_facts(samples, _grounding_attr())

    assert "grounding_facts" not in samples[0]
    assert "grounding_facts" not in samples[1]


def test_attach_grounding_facts_noop_when_no_env_has_grounding(
    mock_inference_config,
):
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
            assert "id" in fact.data
            assert "title" in fact.data


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


def test_attach_grounding_facts_seeded_different_samples_differ(
    mock_inference_config,
):
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

    # Only env_a contributes facts (sample_size=2).
    assert len(samples[0]["grounding_facts"]) == 2


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

    # available_environments=None -> all envs in config are in scope
    synth._attach_grounding_facts(
        samples, _grounding_attr(available_tools=["tool_a", "tool_b"])
    )

    assert len(samples[0]["grounding_facts"]) == 5  # 2 + 3


def test_attach_grounding_facts_truncation_emits_logger_warning(
    mock_inference_config, caplog
):
    import logging

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
    # Exactly one warning per env per synthesize invocation, even with 2 samples.
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
    assert "USER turn instructions MAY inline concrete identifiers" in planner_user_msg
    assert "ASSISTANT turn instructions MUST NOT pre-resolve" in planner_user_msg
    assert "user persona cannot see this list" in planner_user_msg


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


def test_create_planner_prompt_empty_grounding_facts_omits_block(
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
        "grounding_facts": [],
    }

    conversation = synth._create_planner_prompt(attr, sample)
    planner_user_msg = conversation.messages[-1].content
    assert isinstance(planner_user_msg, str)
    assert "Ground this plan" not in planner_user_msg


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
            # Planner call.
            return [
                Conversation(messages=[Message(role=Role.ASSISTANT, content=plan_json)])
                for _ in conversations
            ]
        # Subsequent calls are turn generations.
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

    # The sample dict passed in should have grounding_facts attached.
    assert "grounding_facts" in samples[0]
    assert len(samples[0]["grounding_facts"]) == 2
    # Basic regression: result shape is preserved.
    assert len(result) == 1


# --- {grounding_facts} placeholder misuse warning ---


def test_warn_on_grounding_placeholder_warns_in_user_persona(
    mock_inference_config, caplog
):
    import logging

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
    import logging

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
    import logging

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
# Tool-call loop tests
# ---------------------------------------------------------------------------


def _tool_env_params(*, tool_id: str = "lookup") -> EnvironmentParams:
    """Build an EnvironmentParams wrapping a deterministic tool."""
    return EnvironmentParams(
        id="env1",
        name="Env",
        description="Test env",
        env_type="deterministic",
        tools=[
            DeterministicTool(
                id=tool_id,
                name="Lookup",
                description="Look up an id.",
                deterministic_outputs=[
                    DeterministicToolOutput(
                        input={"id": "01"}, output={"status": "ok"}
                    ),
                ],
            )
        ],
    )


def _tool_env_config(*, tool_id: str = "lookup") -> EnvironmentConfig:
    """Build an EnvironmentConfig wrapping a deterministic environment."""
    return EnvironmentConfig(environments=[_tool_env_params(tool_id=tool_id)])


def _typed_tool_env_config() -> EnvironmentConfig:
    """Env with a typed parameter schema so we can exercise validation."""
    env = EnvironmentParams(
        id="env1",
        name="Env",
        description="Test env",
        env_type="deterministic",
        tools=[
            DeterministicTool(
                id="lookup",
                name="Lookup",
                description="Look up a policy.",
                parameters=ToolSchema(
                    type="object",
                    properties={
                        "policy_id": ToolSchema(type="string"),
                        "limit": ToolSchema(type="integer"),
                    },
                    required=["policy_id"],
                ),
                deterministic_outputs=[
                    DeterministicToolOutput(
                        input={"policy_id": "p1", "limit": 5},
                        output={"policy": "ok"},
                    ),
                ],
            )
        ],
    )
    return EnvironmentConfig(environments=[env])


def _scripted_inference_engine(responses_per_call: list[list[str]]) -> Mock:
    """Inference engine mock returning one batched response set per call."""
    engine = Mock()
    iterator = iter(responses_per_call)

    def infer_side_effect(conversations, **kwargs):
        batch = next(iterator)
        assert len(batch) == len(conversations), (
            f"scripted batch size {len(batch)} != prompt count {len(conversations)}"
        )
        return [
            Conversation(messages=[Message(role=Role.ASSISTANT, content=text)])
            for text in batch
        ]

    engine.infer.side_effect = infer_side_effect
    return engine


def _make_synthesizer(
    mock_inference_config,
    *,
    environment_config: EnvironmentConfig | None = None,
    inference_engine: Mock | None = None,
) -> ConversationSynthesizer:
    with patch(
        "oumi.core.synthesis.conversation_synthesizer.build_inference_engine"
    ) as mock_build:
        mock_build.return_value = inference_engine or Mock()
        return ConversationSynthesizer(
            GeneralSynthesisParams(),
            mock_inference_config,
            environment_config=environment_config,
        )


def _tool_multiturn_attr(tool_id: str = "lookup", cap: int = 50) -> MultiTurnAttribute:
    return MultiTurnAttribute(
        id="tool_conv",
        min_turns=2,
        max_turns=2,
        role_instruction_messages={
            Role.USER: "You are a user.",
            Role.ASSISTANT: "You are an assistant.",
        },
        available_environments=["env1"],
        available_tools=[tool_id],
        max_tool_calls_per_turn=cap,
    )


# --- _execute_tool_calls ---


def test_execute_tool_calls_happy_path(mock_inference_config):
    env_config = _tool_env_config()
    synth = _make_synthesizer(mock_inference_config, environment_config=env_config)
    dispatch = synth._build_tool_dispatch(_tool_multiturn_attr())
    response = '<tool_call>{"name": "lookup", "arguments": {"id": "01"}}</tool_call>'

    messages = synth._execute_tool_calls(response, dispatch)

    assert len(messages) == 1
    assert messages[0].role == Role.USER
    assert json.loads(_unwrap_tool_result(messages[0].content)) == {"status": "ok"}


def test_execute_tool_calls_multiple_blocks_in_order(mock_inference_config):
    env_config = _tool_env_config()
    synth = _make_synthesizer(mock_inference_config, environment_config=env_config)
    dispatch = synth._build_tool_dispatch(_tool_multiturn_attr())
    response = (
        '<tool_call>{"name": "lookup", "arguments": {"id": "01"}}</tool_call>'
        "some prose between"
        '<tool_call>{"name": "lookup", "arguments": {"id": "99"}}</tool_call>'
    )

    messages = synth._execute_tool_calls(response, dispatch)

    assert len(messages) == 2
    assert json.loads(_unwrap_tool_result(messages[0].content)) == {"status": "ok"}
    err = json.loads(_unwrap_tool_result(messages[1].content))["error"]
    assert "No deterministic output matches" in err
    assert '"id": "01"' in err  # configured input is surfaced for self-correction


def test_execute_tool_calls_passes_through_string_output(mock_inference_config):
    from oumi.core.configs.params.tool_params import ToolResult

    env_config = _tool_env_config()
    synth = _make_synthesizer(mock_inference_config, environment_config=env_config)
    dispatch = synth._build_tool_dispatch(_tool_multiturn_attr())
    dispatch["lookup"].step = Mock(return_value=ToolResult(output="hello"))

    messages = synth._execute_tool_calls(
        '<tool_call>{"name": "lookup", "arguments": {"id": "01"}}</tool_call>',
        dispatch,
    )
    assert _unwrap_tool_result(messages[0].content) == "hello"


def test_execute_tool_calls_malformed_json(mock_inference_config):
    synth = _make_synthesizer(
        mock_inference_config, environment_config=_tool_env_config()
    )
    dispatch = synth._build_tool_dispatch(_tool_multiturn_attr())
    messages = synth._execute_tool_calls("<tool_call>not json</tool_call>", dispatch)
    assert len(messages) == 1
    assert "Malformed" in json.loads(_unwrap_tool_result(messages[0].content))["error"]


def test_execute_tool_calls_non_object_body(mock_inference_config):
    synth = _make_synthesizer(
        mock_inference_config, environment_config=_tool_env_config()
    )
    dispatch = synth._build_tool_dispatch(_tool_multiturn_attr())
    messages = synth._execute_tool_calls("<tool_call>[1, 2]</tool_call>", dispatch)
    assert (
        "must be a JSON object"
        in json.loads(_unwrap_tool_result(messages[0].content))["error"]
    )


def test_execute_tool_calls_missing_name(mock_inference_config):
    synth = _make_synthesizer(
        mock_inference_config, environment_config=_tool_env_config()
    )
    dispatch = synth._build_tool_dispatch(_tool_multiturn_attr())
    messages = synth._execute_tool_calls(
        '<tool_call>{"arguments": {}}</tool_call>', dispatch
    )
    assert (
        "missing 'name'"
        in json.loads(_unwrap_tool_result(messages[0].content))["error"]
    )


def test_execute_tool_calls_non_dict_arguments(mock_inference_config):
    synth = _make_synthesizer(
        mock_inference_config, environment_config=_tool_env_config()
    )
    dispatch = synth._build_tool_dispatch(_tool_multiturn_attr())
    messages = synth._execute_tool_calls(
        '<tool_call>{"name": "lookup", "arguments": "oops"}</tool_call>', dispatch
    )
    assert (
        "must be an object"
        in json.loads(_unwrap_tool_result(messages[0].content))["error"]
    )


def test_execute_tool_calls_unknown_tool(mock_inference_config):
    synth = _make_synthesizer(
        mock_inference_config, environment_config=_tool_env_config()
    )
    dispatch = synth._build_tool_dispatch(_tool_multiturn_attr())
    messages = synth._execute_tool_calls(
        '<tool_call>{"name": "nope", "arguments": {}}</tool_call>', dispatch
    )
    assert (
        "Unknown tool 'nope'"
        in json.loads(_unwrap_tool_result(messages[0].content))["error"]
    )


def test_execute_tool_calls_missing_required_argument(mock_inference_config):
    synth = _make_synthesizer(
        mock_inference_config, environment_config=_typed_tool_env_config()
    )
    dispatch = synth._build_tool_dispatch(_tool_multiturn_attr())
    messages = synth._execute_tool_calls(
        '<tool_call>{"name": "lookup", "arguments": {"limit": 5}}</tool_call>', dispatch
    )
    err = json.loads(_unwrap_tool_result(messages[0].content))["error"]
    assert "Invalid arguments for tool 'lookup'" in err
    assert "arguments.policy_id is required" in err


def test_execute_tool_calls_wrong_argument_type(mock_inference_config):
    synth = _make_synthesizer(
        mock_inference_config, environment_config=_typed_tool_env_config()
    )
    dispatch = synth._build_tool_dispatch(_tool_multiturn_attr())
    messages = synth._execute_tool_calls(
        '<tool_call>{"name": "lookup", '
        '"arguments": {"policy_id": "p1", "limit": "five"}}</tool_call>',
        dispatch,
    )
    err = json.loads(_unwrap_tool_result(messages[0].content))["error"]
    assert "Invalid arguments for tool 'lookup'" in err
    assert "arguments.limit must be an integer" in err


def test_execute_tool_calls_validation_runs_before_env_step(mock_inference_config):
    """Argument validation should short-circuit before hitting env.step()."""
    env_config = _typed_tool_env_config()
    synth = _make_synthesizer(mock_inference_config, environment_config=env_config)
    dispatch = synth._build_tool_dispatch(_tool_multiturn_attr())
    dispatch["lookup"].step = Mock(
        side_effect=AssertionError("step() must not be called")
    )

    messages = synth._execute_tool_calls(
        '<tool_call>{"name": "lookup", "arguments": {}}</tool_call>', dispatch
    )

    assert dispatch["lookup"].step.call_count == 0
    assert (
        "Invalid arguments"
        in json.loads(_unwrap_tool_result(messages[0].content))["error"]
    )


def test_execute_tool_calls_valid_arguments_hit_env_step(mock_inference_config):
    """Schema-valid arguments that miss the deterministic table still surface
    a structured ToolLookupError — not a silent null output.
    """
    synth = _make_synthesizer(
        mock_inference_config, environment_config=_typed_tool_env_config()
    )
    dispatch = synth._build_tool_dispatch(_tool_multiturn_attr())
    messages = synth._execute_tool_calls(
        '<tool_call>{"name": "lookup", '
        '"arguments": {"policy_id": "unknown", "limit": 5}}</tool_call>',
        dispatch,
    )
    err = json.loads(_unwrap_tool_result(messages[0].content))["error"]
    assert "No deterministic output matches" in err
    assert '"policy_id": "p1"' in err  # existing configured input is surfaced


def test_execute_tool_calls_env_raises(mock_inference_config):
    env_config = _tool_env_config()
    synth = _make_synthesizer(mock_inference_config, environment_config=env_config)
    dispatch = synth._build_tool_dispatch(_tool_multiturn_attr())
    dispatch["lookup"].step = Mock(side_effect=RuntimeError("boom"))
    messages = synth._execute_tool_calls(
        '<tool_call>{"name": "lookup", "arguments": {}}</tool_call>', dispatch
    )
    assert (
        "Tool 'lookup' raised"
        in json.loads(_unwrap_tool_result(messages[0].content))["error"]
    )
    assert "boom" in json.loads(_unwrap_tool_result(messages[0].content))["error"]


def test_execute_tool_calls_no_block_returns_empty(mock_inference_config):
    synth = _make_synthesizer(
        mock_inference_config, environment_config=_tool_env_config()
    )
    dispatch = synth._build_tool_dispatch(_tool_multiturn_attr())
    assert synth._execute_tool_calls("just plain prose", dispatch) == []


def test_is_final_response_detects_tool_call(mock_inference_config):
    synth = _make_synthesizer(
        mock_inference_config, environment_config=_tool_env_config()
    )
    assert synth._is_final_response("answer to user") is True
    assert (
        synth._is_final_response(
            'chatty <tool_call>{"name": "lookup", "arguments": {}}</tool_call>'
        )
        is False
    )


# --- Pre-result hallucination hardening ---


def testtruncate_after_last_tool_call_strips_trailing_prose():
    from oumi.environments.utils import truncate_after_last_tool_call

    text = (
        '<tool_call>{"name": "lookup", "arguments": {"id": "01"}}</tool_call>\n\n'
        "I found it! The book is 'Dune' by Frank Herbert. Due back 2024-01-15."
    )
    cleaned = truncate_after_last_tool_call(text)
    assert cleaned.endswith("</tool_call>")
    assert "Dune" not in cleaned
    assert "2024" not in cleaned


def testtruncate_after_last_tool_call_preserves_leading_prose():
    from oumi.environments.utils import truncate_after_last_tool_call

    text = (
        "Let me check that for you.\n"
        '<tool_call>{"name": "lookup", "arguments": {"id": "01"}}</tool_call>'
    )
    cleaned = truncate_after_last_tool_call(text)
    assert cleaned == text


def testtruncate_after_last_tool_call_keeps_plain_response():
    from oumi.environments.utils import truncate_after_last_tool_call

    text = "Here is a normal final response with no tool call."
    assert truncate_after_last_tool_call(text) == text


def testtruncate_after_last_tool_call_keeps_multiple_calls():
    from oumi.environments.utils import truncate_after_last_tool_call

    text = (
        '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
        '<tool_call>{"name": "b", "arguments": {}}</tool_call>'
        "Trailing hallucinated answer."
    )
    cleaned = truncate_after_last_tool_call(text)
    assert cleaned.endswith("</tool_call>")
    assert "hallucinated" not in cleaned
    assert cleaned.count("<tool_call>") == 2


def testclose_dangling_tool_call_appends_missing_close_tag():
    from oumi.environments.utils import close_dangling_tool_call

    truncated = '<tool_call>{"name": "lookup", "arguments": {"id": "01"}}'
    closed = close_dangling_tool_call(truncated)
    assert closed.endswith("</tool_call>")
    assert closed.count("<tool_call>") == closed.count("</tool_call>")


def testclose_dangling_tool_call_noop_when_balanced():
    from oumi.environments.utils import close_dangling_tool_call

    text = '<tool_call>{"name": "x", "arguments": {}}</tool_call>'
    assert close_dangling_tool_call(text) == text


def testclose_dangling_tool_call_noop_when_no_tool_call():
    from oumi.environments.utils import close_dangling_tool_call

    text = "Plain response, no tool tags here."
    assert close_dangling_tool_call(text) == text


def test_assistant_inference_config_adds_stop_sequence(mock_inference_config):
    synth = _make_synthesizer(mock_inference_config)
    cfg = synth._assistant_inference_config()
    assert "</tool_call>" in (cfg.generation.stop_strings or [])
    base_stops = list(mock_inference_config.generation.stop_strings or [])
    assert "</tool_call>" not in base_stops


def test_assistant_inference_config_preserves_existing_stops(
    mock_inference_config,
):
    mock_inference_config.generation.stop_strings = ["<|end|>", "STOP"]
    synth = _make_synthesizer(mock_inference_config)
    cfg = synth._assistant_inference_config()
    stops = cfg.generation.stop_strings
    assert stops is not None
    assert "<|end|>" in stops
    assert "STOP" in stops
    assert "</tool_call>" in stops


# --- _run_assistant_turn ---


def test_run_assistant_turn_strips_prose_after_tool_call_in_response(
    mock_inference_config,
):
    """Real models often emit <tool_call> AND a fabricated follow-up answer
    in the same message. Trailing prose must be stripped before storing the
    assistant message, otherwise the conversation contains contradictory
    data that poisons training.
    """
    env_config = _tool_env_config()
    engine = _scripted_inference_engine(
        [
            [
                '<tool_call>{"name": "lookup", "arguments": {"id": "01"}}'
                "</tool_call>\n\n"
                "I found it! The book is 'Dune' by Frank Herbert. Due 2024-01-15."
            ],
            ["The real status is ok."],
        ]
    )
    synth = _make_synthesizer(
        mock_inference_config,
        environment_config=env_config,
        inference_engine=engine,
    )
    dispatch = synth._build_tool_dispatch(_tool_multiturn_attr())

    msgs = synth._run_assistant_turn(
        samples=[{"target_turns": 2, "parsed_turn_plans": ["", ""]}],
        sample_indices=[0],
        histories=[[]],
        current_turn=2,
        multiturn_attribute=_tool_multiturn_attr(),
        tool_dispatch=dispatch,
    )

    assistant_msg = msgs[0][0]
    assert assistant_msg.role == Role.ASSISTANT
    assert isinstance(assistant_msg.content, str)
    assert "Dune" not in assistant_msg.content
    assert "2024" not in assistant_msg.content
    assert assistant_msg.content.rstrip().endswith("</tool_call>")
    assert msgs[0][-1].content == "The real status is ok."


def test_run_assistant_turn_rehydrates_stop_sequence_stripped_close_tag(
    mock_inference_config,
):
    """When a provider strips the matched stop_sequence (Anthropic drops
    </tool_call>), the pipeline must re-append it so downstream parsing
    round-trips.
    """
    env_config = _tool_env_config()
    engine = _scripted_inference_engine(
        [
            ['<tool_call>{"name": "lookup", "arguments": {"id": "01"}}'],
            ["done"],
        ]
    )
    synth = _make_synthesizer(
        mock_inference_config,
        environment_config=env_config,
        inference_engine=engine,
    )
    dispatch = synth._build_tool_dispatch(_tool_multiturn_attr())

    msgs = synth._run_assistant_turn(
        samples=[{"target_turns": 2, "parsed_turn_plans": ["", ""]}],
        sample_indices=[0],
        histories=[[]],
        current_turn=2,
        multiturn_attribute=_tool_multiturn_attr(),
        tool_dispatch=dispatch,
    )

    assistant_msg = msgs[0][0]
    assert isinstance(assistant_msg.content, str)
    assert assistant_msg.content.endswith("</tool_call>")
    assert msgs[0][1].role == Role.USER
    tool_payload = json.loads(_unwrap_tool_result(msgs[0][1].content))
    assert tool_payload == {"status": "ok"}


def test_run_assistant_turn_uses_assistant_inference_config(
    mock_inference_config,
):
    """Assistant turns must run through ``_assistant_inference_config()``
    (which adds </tool_call> to stop_strings) — not the base config.
    """
    env_config = _tool_env_config()
    engine = _scripted_inference_engine([["final answer"]])
    synth = _make_synthesizer(
        mock_inference_config,
        environment_config=env_config,
        inference_engine=engine,
    )
    dispatch = synth._build_tool_dispatch(_tool_multiturn_attr())

    synth._run_assistant_turn(
        samples=[{"target_turns": 2, "parsed_turn_plans": ["", ""]}],
        sample_indices=[0],
        histories=[[]],
        current_turn=2,
        multiturn_attribute=_tool_multiturn_attr(),
        tool_dispatch=dispatch,
    )

    call_kwargs = engine.infer.call_args_list[0].kwargs
    cfg = call_kwargs["inference_config"]
    assert "</tool_call>" in (cfg.generation.stop_strings or [])


def test_run_assistant_turn_lockstep_final_response(mock_inference_config):
    env_config = _tool_env_config()
    engine = _scripted_inference_engine([["final answer A", "final answer B"]])
    synth = _make_synthesizer(
        mock_inference_config, environment_config=env_config, inference_engine=engine
    )
    dispatch = synth._build_tool_dispatch(_tool_multiturn_attr())

    samples = [
        {"target_turns": 2, "parsed_turn_plans": ["", ""]},
        {"target_turns": 2, "parsed_turn_plans": ["", ""]},
    ]
    msgs = synth._run_assistant_turn(
        samples=samples,
        sample_indices=[0, 1],
        histories=[[], []],
        current_turn=2,
        multiturn_attribute=_tool_multiturn_attr(),
        tool_dispatch=dispatch,
    )

    assert engine.infer.call_count == 1
    assert len(msgs) == 2
    assert len(msgs[0]) == 1
    assert msgs[0][0].role == Role.ASSISTANT
    assert msgs[0][0].content == "final answer A"
    assert msgs[1][0].content == "final answer B"


def test_run_assistant_turn_asymmetric_batches(mock_inference_config):
    env_config = _tool_env_config()
    engine = _scripted_inference_engine(
        [
            [  # iteration 1: sample 0 finalizes, sample 1 calls tool
                "done!",
                '<tool_call>{"name": "lookup", "arguments": {"id": "01"}}</tool_call>',
            ],
            [  # iteration 2: only sample 1 re-enters the batch
                "final for sample 1",
            ],
        ]
    )
    synth = _make_synthesizer(
        mock_inference_config, environment_config=env_config, inference_engine=engine
    )
    dispatch = synth._build_tool_dispatch(_tool_multiturn_attr())
    msgs = synth._run_assistant_turn(
        samples=[
            {"target_turns": 2, "parsed_turn_plans": ["", ""]},
            {"target_turns": 2, "parsed_turn_plans": ["", ""]},
        ],
        sample_indices=[0, 1],
        histories=[[], []],
        current_turn=2,
        multiturn_attribute=_tool_multiturn_attr(),
        tool_dispatch=dispatch,
    )

    assert engine.infer.call_count == 2
    assert len(msgs[0]) == 1
    assert msgs[0][0].content == "done!"
    assert [m.role for m in msgs[1]] == [Role.ASSISTANT, Role.USER, Role.ASSISTANT]
    assert json.loads(_unwrap_tool_result(msgs[1][1].content)) == {"status": "ok"}
    assert msgs[1][2].content == "final for sample 1"


def test_run_assistant_turn_multiple_tool_calls_one_response(mock_inference_config):
    env_config = _tool_env_config()
    engine = _scripted_inference_engine(
        [
            [
                '<tool_call>{"name": "lookup", "arguments": {"id": "01"}}</tool_call>'
                '<tool_call>{"name": "lookup", "arguments": {"id": "01"}}</tool_call>'
            ],
            ["all done"],
        ]
    )
    synth = _make_synthesizer(
        mock_inference_config, environment_config=env_config, inference_engine=engine
    )
    dispatch = synth._build_tool_dispatch(_tool_multiturn_attr())
    msgs = synth._run_assistant_turn(
        samples=[{"target_turns": 2, "parsed_turn_plans": ["", ""]}],
        sample_indices=[0],
        histories=[[]],
        current_turn=2,
        multiturn_attribute=_tool_multiturn_attr(),
        tool_dispatch=dispatch,
    )
    roles = [m.role for m in msgs[0]]
    assert roles == [Role.ASSISTANT, Role.USER, Role.USER, Role.ASSISTANT]
    assert msgs[0][-1].content == "all done"


def test_run_assistant_turn_cap_hit_forces_finalize(mock_inference_config):
    env_config = _tool_env_config()
    tool_call = '<tool_call>{"name": "lookup", "arguments": {"id": "01"}}</tool_call>'
    engine = _scripted_inference_engine(
        [
            [tool_call],
            ["forced final answer"],
        ]
    )
    synth = _make_synthesizer(
        mock_inference_config, environment_config=env_config, inference_engine=engine
    )
    attr = _tool_multiturn_attr(cap=1)
    dispatch = synth._build_tool_dispatch(attr)
    msgs = synth._run_assistant_turn(
        samples=[{"target_turns": 2, "parsed_turn_plans": ["", ""]}],
        sample_indices=[0],
        histories=[[]],
        current_turn=2,
        multiturn_attribute=attr,
        tool_dispatch=dispatch,
    )

    assert engine.infer.call_count == 2
    second_call_conv = engine.infer.call_args_list[1].args[0][0]
    assert "tool-call limit" in second_call_conv.messages[-1].content
    assert [m.role for m in msgs[0]] == [Role.ASSISTANT, Role.USER, Role.ASSISTANT]
    assert msgs[0][-1].content == "forced final answer"


def test_run_assistant_turn_cap_hit_strips_residual_tool_calls(mock_inference_config):
    env_config = _tool_env_config()
    engine = _scripted_inference_engine(
        [
            ['<tool_call>{"name": "lookup", "arguments": {"id": "01"}}</tool_call>'],
            [
                "leftover prose "
                '<tool_call>{"name": "lookup", "arguments": {}}</tool_call>'
                " tail"
            ],
        ]
    )
    synth = _make_synthesizer(
        mock_inference_config, environment_config=env_config, inference_engine=engine
    )
    attr = _tool_multiturn_attr(cap=1)
    dispatch = synth._build_tool_dispatch(attr)
    msgs = synth._run_assistant_turn(
        samples=[{"target_turns": 2, "parsed_turn_plans": ["", ""]}],
        sample_indices=[0],
        histories=[[]],
        current_turn=2,
        multiturn_attribute=attr,
        tool_dispatch=dispatch,
    )
    final = msgs[0][-1]
    assert final.role == Role.ASSISTANT
    final_content = final.content
    assert isinstance(final_content, str)
    assert "<tool_call>" not in final_content
    assert "leftover prose" in final_content
    assert "tail" in final_content


def test_run_assistant_turn_cap_hit_only_tool_calls_yields_empty_final(
    mock_inference_config,
):
    env_config = _tool_env_config()
    engine = _scripted_inference_engine(
        [
            ['<tool_call>{"name": "lookup", "arguments": {"id": "01"}}</tool_call>'],
            ['<tool_call>{"name": "lookup", "arguments": {}}</tool_call>'],
        ]
    )
    synth = _make_synthesizer(
        mock_inference_config, environment_config=env_config, inference_engine=engine
    )
    attr = _tool_multiturn_attr(cap=1)
    dispatch = synth._build_tool_dispatch(attr)
    msgs = synth._run_assistant_turn(
        samples=[{"target_turns": 2, "parsed_turn_plans": ["", ""]}],
        sample_indices=[0],
        histories=[[]],
        current_turn=2,
        multiturn_attribute=attr,
        tool_dispatch=dispatch,
    )
    assert msgs[0][-1].content == ""


def test_run_assistant_turn_self_corrects_after_tool_error(mock_inference_config):
    """Model receives a structured tool error and recovers on next iteration."""
    env_config = _tool_env_config()
    engine = _scripted_inference_engine(
        [
            ['<tool_call>{"name": "lookup", "arguments": {"id": "99"}}</tool_call>'],
            ['<tool_call>{"name": "lookup", "arguments": {"id": "01"}}</tool_call>'],
            ["The lookup succeeded: status ok."],
        ]
    )
    synth = _make_synthesizer(
        mock_inference_config, environment_config=env_config, inference_engine=engine
    )
    dispatch = synth._build_tool_dispatch(_tool_multiturn_attr())

    msgs = synth._run_assistant_turn(
        samples=[{"target_turns": 2, "parsed_turn_plans": ["", ""]}],
        sample_indices=[0],
        histories=[[]],
        current_turn=2,
        multiturn_attribute=_tool_multiturn_attr(),
        tool_dispatch=dispatch,
    )

    assert engine.infer.call_count == 3
    sample_msgs = msgs[0]
    roles = [m.role for m in sample_msgs]
    assert roles == [
        Role.ASSISTANT,
        Role.USER,
        Role.ASSISTANT,
        Role.USER,
        Role.ASSISTANT,
    ]

    first_tool_payload = json.loads(_unwrap_tool_result(sample_msgs[1].content))
    assert "error" in first_tool_payload
    assert "No deterministic output matches" in first_tool_payload["error"]

    assert json.loads(_unwrap_tool_result(sample_msgs[3].content)) == {"status": "ok"}
    assert sample_msgs[-1].content == "The lookup succeeded: status ok."


def test_run_assistant_turn_self_corrects_after_invalid_arguments(
    mock_inference_config,
):
    """Schema-level validation error is surfaced, then model corrects."""
    env_config = _typed_tool_env_config()
    engine = _scripted_inference_engine(
        [
            ['<tool_call>{"name": "lookup", "arguments": {}}</tool_call>'],
            [
                '<tool_call>{"name": "lookup", '
                '"arguments": {"policy_id": "p1", "limit": 5}}</tool_call>'
            ],
            ["Policy looks good."],
        ]
    )
    synth = _make_synthesizer(
        mock_inference_config, environment_config=env_config, inference_engine=engine
    )
    dispatch = synth._build_tool_dispatch(_tool_multiturn_attr())

    msgs = synth._run_assistant_turn(
        samples=[{"target_turns": 2, "parsed_turn_plans": ["", ""]}],
        sample_indices=[0],
        histories=[[]],
        current_turn=2,
        multiturn_attribute=_tool_multiturn_attr(),
        tool_dispatch=dispatch,
    )

    assert engine.infer.call_count == 3
    sample_msgs = msgs[0]
    roles = [m.role for m in sample_msgs]
    assert roles == [
        Role.ASSISTANT,
        Role.USER,
        Role.ASSISTANT,
        Role.USER,
        Role.ASSISTANT,
    ]

    first_tool_payload = json.loads(_unwrap_tool_result(sample_msgs[1].content))
    assert "error" in first_tool_payload
    assert "Invalid arguments for tool 'lookup'" in first_tool_payload["error"]
    assert "arguments.policy_id is required" in first_tool_payload["error"]

    assert json.loads(_unwrap_tool_result(sample_msgs[3].content)) == {"policy": "ok"}
    assert sample_msgs[-1].content == "Policy looks good."


# --- end-to-end synthesize with tools ---


def test_synthesize_end_to_end_with_tool_use(mock_inference_config):
    env_config = _tool_env_config()
    plan_json = (
        '[{"turn": 1, "instruction": "ask"}, {"turn": 2, "instruction": "answer"}]'
    )
    engine = _scripted_inference_engine(
        [
            [plan_json],
            ["What is the status of order 01?"],
            ['<tool_call>{"name": "lookup", "arguments": {"id": "01"}}</tool_call>'],
            ["The order is ok."],
        ]
    )
    synth = _make_synthesizer(
        mock_inference_config, environment_config=env_config, inference_engine=engine
    )

    attr = MultiTurnAttribute(
        id="tool_conv",
        min_turns=2,
        max_turns=2,
        role_instruction_messages={
            Role.USER: "You are a user.",
            Role.ASSISTANT: "You are an assistant.",
        },
        available_environments=["env1"],
        available_tools=["lookup"],
    )

    records = synth.synthesize([{}], attr)

    assert len(records) == 1
    assert records[0] is not None
    conv = records[0]["tool_conv"]
    assert isinstance(conv, dict)
    roles = [m["role"] for m in conv["messages"]]
    assert roles == [Role.USER, Role.ASSISTANT, Role.USER, Role.ASSISTANT]
    assert conv["messages"][-1]["content"] == "The order is ok."


def test_synthesize_drops_sample_when_cap_hit_produces_empty_final(
    mock_inference_config,
):
    env_config = _tool_env_config()
    plan_json = (
        '[{"turn": 1, "instruction": "ask"}, {"turn": 2, "instruction": "answer"}]'
    )
    tool_call = '<tool_call>{"name": "lookup", "arguments": {"id": "01"}}</tool_call>'
    engine = _scripted_inference_engine(
        [
            [plan_json],
            ["ask"],
            [tool_call],
            [tool_call],
        ]
    )
    synth = _make_synthesizer(
        mock_inference_config, environment_config=env_config, inference_engine=engine
    )
    attr = MultiTurnAttribute(
        id="tool_conv",
        min_turns=2,
        max_turns=2,
        role_instruction_messages={
            Role.USER: "You are a user.",
            Role.ASSISTANT: "You are an assistant.",
        },
        available_environments=["env1"],
        available_tools=["lookup"],
        max_tool_calls_per_turn=1,
    )

    records = synth.synthesize([{}], attr)
    assert records == [None]


def test_synthesize_raises_when_tools_declared_without_env_config(
    mock_inference_config,
):
    """available_tools without environment_config must fail loudly."""
    synth = _make_synthesizer(mock_inference_config)
    attr = MultiTurnAttribute(
        id="tool_conv",
        min_turns=2,
        max_turns=2,
        role_instruction_messages={
            Role.USER: "You are a user.",
            Role.ASSISTANT: "You are an assistant.",
        },
        available_tools=["lookup"],
    )

    with pytest.raises(ValueError, match="available_tools"):
        synth.synthesize([{}], attr)


def test_synthesize_raises_when_environments_declared_without_env_config(
    mock_inference_config,
):
    """available_environments without environment_config must fail loudly."""
    synth = _make_synthesizer(mock_inference_config)
    attr = MultiTurnAttribute(
        id="tool_conv",
        min_turns=2,
        max_turns=2,
        role_instruction_messages={
            Role.USER: "You are a user.",
            Role.ASSISTANT: "You are an assistant.",
        },
        available_environments=["env1"],
    )

    with pytest.raises(ValueError, match="environment_config"):
        synth.synthesize([{}], attr)


# --- Regression + integration: planner prompt byte-equivalence when ungrounded ---


def test_create_planner_prompt_byte_identical_when_no_grounding(
    mock_inference_config,
):
    """Regression: when no env in scope has grounding, planner prompt is
    unchanged from the pre-grounding baseline.
    """
    env_config = _tool_env_config()
    synth = _make_synthesizer(mock_inference_config, environment_config=env_config)
    attr = _tool_multiturn_attr()
    sample = {
        "target_turns": 2,
        "conversation_plan": "",
        "parsed_turn_plans": [""] * 2,
    }

    conversation = synth._create_planner_prompt(attr, sample)
    planner_user_msg = conversation.messages[-1].content
    assert isinstance(planner_user_msg, str)

    # Real-text grounding markers must be absent. The strict-rules version of
    # the example_request still mentions "Ground this plan" once as part of
    # the few-shot demo, but the runtime base prompt must not include the
    # grounded section.
    assert (
        "USER turn instructions MAY inline concrete identifiers" not in planner_user_msg
    )
    assert "ASSISTANT turn instructions MUST NOT pre-resolve" not in planner_user_msg
    assert "user persona cannot see this list" not in planner_user_msg
    assert "{grounding_facts}" not in planner_user_msg


def test_end_to_end_grounded_conversation_uses_sampled_entity_ids(
    mock_inference_config,
):
    """Full synthesize() flow: grounded planner prompt drives the plan
    through a scripted inference engine, confirming the facts make it all
    the way into the planner call.
    """
    env_config = _grounded_env_config(n_entries=10, sample_size=3, seed=13)

    captured_planner_prompts: list[str] = []

    def infer_side_effect(conversations, **kwargs):
        results = []
        for conv in conversations:
            last = conv.messages[-1].content
            is_planner = isinstance(last, str) and "Plan" in last
            if is_planner:
                captured_planner_prompts.append(last)
                content = (
                    '[{"turn": 1, "instruction": "a"}, {"turn": 2, "instruction": "b"}]'
                )
            else:
                content = "turn content"
            results.append(
                Conversation(messages=[Message(role=Role.ASSISTANT, content=content)])
            )
        return results

    engine = Mock()
    engine.infer.side_effect = infer_side_effect

    synth = _make_synthesizer(
        mock_inference_config,
        environment_config=env_config,
        inference_engine=engine,
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
    synth.synthesize(samples, attr)

    assert captured_planner_prompts, "planner prompt was never captured"
    planner_prompt = captured_planner_prompts[0]
    assert "Ground this plan in these specific entities" in planner_prompt
    configured_ids = {str(i) for i in range(10)}
    facts = samples[0]["grounding_facts"]
    for fact in facts:
        fact_id = fact.data["id"]
        assert fact_id in configured_ids
        assert f'id="{fact_id}"' in planner_prompt


# --- JSON brace repair: tool-call recovery ---


def test_execute_tool_calls_recovers_from_trailing_extra_brace(mock_inference_config):
    """Tool-call bodies with an extra ``}}`` (seen in synth output) execute."""
    synth = _make_synthesizer(
        mock_inference_config, environment_config=_tool_env_config()
    )
    dispatch = synth._build_tool_dispatch(_tool_multiturn_attr())
    response = '<tool_call>{"name": "lookup", "arguments": {"id": "01"}}}</tool_call>'
    messages = synth._execute_tool_calls(response, dispatch)
    assert len(messages) == 1
    assert json.loads(_unwrap_tool_result(messages[0].content)) == {"status": "ok"}


def test_execute_tool_calls_recovers_from_missing_close_brace(mock_inference_config):
    """Stop-sequence truncation that drops a closing ``}`` is repaired."""
    synth = _make_synthesizer(
        mock_inference_config, environment_config=_tool_env_config()
    )
    dispatch = synth._build_tool_dispatch(_tool_multiturn_attr())
    response = '<tool_call>{"name": "lookup", "arguments": {"id": "01"}</tool_call>'
    messages = synth._execute_tool_calls(response, dispatch)
    assert len(messages) == 1
    assert json.loads(_unwrap_tool_result(messages[0].content)) == {"status": "ok"}


def testcanonicalize_tool_call_bodies_strips_extra_brace():
    """The observed ``}}}`` malformation is repaired into canonical JSON."""
    from oumi.environments.utils import canonicalize_tool_call_bodies

    text = (
        '<tool_call>{"name": "lookup_book_status", '
        '"arguments": {"book_id": "B008"}}}</tool_call>'
    )
    out = canonicalize_tool_call_bodies(text)
    assert "}}}" not in out
    assert out == (
        "<tool_call>"
        '{"name": "lookup_book_status", "arguments": {"book_id": "B008"}}'
        "</tool_call>"
    )


def testcanonicalize_tool_call_bodies_appends_missing_close():
    from oumi.environments.utils import canonicalize_tool_call_bodies

    text = '<tool_call>{"name": "lookup", "arguments": {"id": "01"}</tool_call>'
    out = canonicalize_tool_call_bodies(text)
    assert out == (
        '<tool_call>{"name": "lookup", "arguments": {"id": "01"}}</tool_call>'
    )


def testcanonicalize_tool_call_bodies_leaves_unfixable_body_intact():
    """Non-structural breakage (unquoted keys) is left for the executor."""
    from oumi.environments.utils import canonicalize_tool_call_bodies

    text = "<tool_call>{not valid at all}</tool_call>"
    assert canonicalize_tool_call_bodies(text) == text


def testcanonicalize_tool_call_bodies_noop_when_no_calls_present():
    from oumi.environments.utils import canonicalize_tool_call_bodies

    text = "Just a plain assistant reply, nothing to do here."
    assert canonicalize_tool_call_bodies(text) == text


def testcanonicalize_tool_call_bodies_handles_multiple_blocks():
    from oumi.environments.utils import canonicalize_tool_call_bodies

    text = (
        '<tool_call>{"name": "a", "arguments": {"id": "1"}}}</tool_call>'
        " some prose "
        '<tool_call>{"name": "b", "arguments": {"id": "2"}</tool_call>'
    )
    out = canonicalize_tool_call_bodies(text)
    assert "}}}" not in out
    assert out.count("<tool_call>") == 2
    assert out.count("</tool_call>") == 2
    assert '{"name": "a", "arguments": {"id": "1"}}' in out
    assert '{"name": "b", "arguments": {"id": "2"}}' in out



def test_parse_plan_unwraps_openai_object_form(mock_inference_config):
    """OpenAI structured-output returns ``{"turns": [...]}``; unwrap it."""
    with patch(
        "oumi.core.synthesis.conversation_synthesizer.build_inference_engine"
    ) as _:
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


def test_create_planner_prompt_example_models_role_aware_grounding(
    mock_inference_config,
):
    """Few-shot example: user turns inline IDs, assistant turns describe tool calls."""
    synth = _make_synthesizer(
        mock_inference_config, environment_config=_tool_env_config()
    )
    attr = _tool_multiturn_attr()
    sample = {
        "target_turns": 2,
        "conversation_plan": "",
        "parsed_turn_plans": [""] * 2,
    }
    conversation = synth._create_planner_prompt(attr, sample)

    example_request = conversation.messages[1].content
    example_response = conversation.messages[2].content
    assert isinstance(example_request, str)
    assert isinstance(example_response, str)
    assert "Ground this plan in these specific entities" in example_request
    assert 'order_id="ORD-4421"' in example_request

    plan = json.loads(example_response)
    turns_by_num = {t["turn"]: t["instruction"] for t in plan["turns"]}

    user_turns_with_id = sum(
        1 for n, inst in turns_by_num.items() if n % 2 == 1 and "ORD-4421" in inst
    )
    assert user_turns_with_id >= 1, "user turn should mention the ID verbatim"

    assistant_instructions = [inst for n, inst in turns_by_num.items() if n % 2 == 0]
    assert any(
        "lookup_order_status" in inst or "refund_order" in inst
        for inst in assistant_instructions
    ), "assistant turns should describe tool calls"

    assistant_turn_2 = turns_by_num[2]
    assert "ORD-4421" not in assistant_turn_2, (
        "assistant turn 2 must not pre-resolve the order_id — it should come "
        "from the user or the tool"
    )