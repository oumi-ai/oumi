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

"""Tests for ToolExecutor."""

import json

import pytest

from oumi.core.configs.params.tool_params import (
    DeterministicToolOutput,
    GeneratedToolOutput,
    ToolAttribute,
    ToolOutputStrategy,
)
from oumi.core.synthesis.tool_executor import (
    ToolCallError,
    ToolCallParsed,
    ToolExecutor,
)
from oumi.core.types.conversation import Message, Role


@pytest.fixture
def deterministic_tool():
    """Create a DETERMINISTIC tool with two possible outputs."""
    return ToolAttribute(
        id="tool_search_orders",
        name="SearchOrders",
        description="Search for customer orders by ID.",
        output_strategy=ToolOutputStrategy.DETERMINISTIC,
        parameters={
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order ID to look up",
                },
                "include_items": {
                    "type": "boolean",
                    "description": "Whether to include line items",
                },
            },
            "required": ["order_id"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "status": {"type": "string"},
            },
        },
        deterministic_outputs=[
            DeterministicToolOutput(
                values={"order_id": "ORD-001", "status": "delivered"},
                sample_rate=0.7,
            ),
            DeterministicToolOutput(
                values={"order_id": "ORD-002", "status": "pending"},
                sample_rate=0.3,
            ),
        ],
    )


@pytest.fixture
def generated_tool():
    """Create a GENERATED tool."""
    return ToolAttribute(
        id="tool_check_eligibility",
        name="CheckEligibility",
        description="Check return eligibility for an order.",
        output_strategy=ToolOutputStrategy.GENERATED,
        parameters={
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order ID",
                },
            },
            "required": ["order_id"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "eligible": {"type": "boolean"},
                "reason": {"type": "string"},
            },
        },
        generated_output=GeneratedToolOutput(
            instruction="Return eligibility based on order status and 30-day window."
        ),
    )


@pytest.fixture
def no_params_tool():
    """Create a tool with no parameters."""
    return ToolAttribute(
        id="tool_get_time",
        name="GetCurrentTime",
        description="Get the current server time.",
        output_strategy=ToolOutputStrategy.GENERATED,
        generated_output=GeneratedToolOutput(
            instruction="Return the current UTC time in ISO 8601 format."
        ),
    )


@pytest.fixture
def executor(deterministic_tool, generated_tool, no_params_tool):
    """Create a ToolExecutor with all fixture tools."""
    return ToolExecutor([deterministic_tool, generated_tool, no_params_tool])


def test_parse_tool_call_valid(executor):
    response = (
        'Some text <tool_call>{"name": "SearchOrders", '
        '"arguments": {"order_id": "ORD-123"}}</tool_call> more text'
    )
    result = executor.parse_and_validate_tool_call(response)

    assert isinstance(result, ToolCallParsed)
    assert result.tool_call["name"] == "SearchOrders"
    assert result.tool_call["arguments"] == {"order_id": "ORD-123"}


@pytest.mark.parametrize(
    "response",
    [
        "Just a regular response with no tool call.",
        "<tool_call>this is not json at all</tool_call>",
        '<tool_call>{"arguments": {"order_id": "ORD-123"}}</tool_call>',
        '<tool_call>{"name": "NonExistentTool", "arguments": {"x": 1}}</tool_call>',
        '<tool_call>["not", "a", "dict"]</tool_call>',
        '<tool_call>{"name": "SearchOrders", "arguments": "bad"}</tool_call>',
        '<tool_call>{"name": 123, "arguments": {}}</tool_call>',
    ],
    ids=[
        "no_tag",
        "malformed_json",
        "missing_name",
        "unknown_tool",
        "non_dict_json",
        "arguments_not_dict",
        "name_not_string",
    ],
)
def test_parse_tool_call_returns_none_for_invalid_input(executor, response):
    result = executor.parse_and_validate_tool_call(response)
    assert not isinstance(result, ToolCallParsed)


def test_parse_tool_call_extra_braces(executor):
    response = (
        '<tool_call>{"name": "SearchOrders", '
        '"arguments": {"order_id": "ORD-123"}}}</tool_call>'
    )
    result = executor.parse_and_validate_tool_call(response)

    assert isinstance(result, ToolCallParsed)
    assert result.tool_call["name"] == "SearchOrders"
    assert result.tool_call["arguments"] == {"order_id": "ORD-123"}


def test_parse_tool_call_empty_arguments(executor):
    response = '<tool_call>{"name": "GetCurrentTime"}</tool_call>'
    result = executor.parse_and_validate_tool_call(response)

    assert isinstance(result, ToolCallParsed)
    assert result.tool_call["name"] == "GetCurrentTime"
    assert result.tool_call["arguments"] == {}



def test_sample_deterministic_outputs(executor, deterministic_tool, generated_tool):
    selections = executor.sample_deterministic_outputs(
        [deterministic_tool, generated_tool]
    )

    assert deterministic_tool.id in selections
    assert generated_tool.id not in selections

    parsed = json.loads(selections[deterministic_tool.id])
    assert parsed["status"] in {"delivered", "pending"}


def test_resolve_output_deterministic_returns_preselected(executor, deterministic_tool):
    preselected = json.dumps({"order_id": "ORD-001", "status": "delivered"})
    selections = {deterministic_tool.id: preselected}

    tool_call = {"name": "SearchOrders", "arguments": {"order_id": "ORD-001"}}
    result = executor.resolve_output(tool_call, selections)

    assert result == preselected


def test_resolve_output_generated_returns_none(executor):
    tool_call = {"name": "CheckEligibility", "arguments": {"order_id": "ORD-001"}}
    result = executor.resolve_output(tool_call, {})
    assert result is None


def test_build_generated_simulator_prompt(executor):
    tool_call = {"name": "CheckEligibility", "arguments": {"order_id": "ORD-999"}}
    conversation = executor.build_generated_simulator_prompt(tool_call)

    # system + 1 few-shot pair (2 messages) + actual request = 4
    assert len(conversation.messages) == 4
    assert conversation.messages[0].role == Role.SYSTEM

    system_content = conversation.messages[0].content
    assert "CheckEligibility" in system_content
    assert "Check return eligibility" in system_content
    assert "Start with { or [" in system_content

    # Generic few-shot example to teach raw JSON format
    assert conversation.messages[1].role == Role.USER
    assert "CheckInventory" in conversation.messages[1].content
    assert conversation.messages[2].role == Role.ASSISTANT
    assert "SKU-1234" in conversation.messages[2].content

    # Actual request
    user_content = conversation.messages[3].content
    assert "ORD-999" in user_content


def test_build_generated_simulator_prompt_with_history(executor):
    tool_call = {"name": "CheckEligibility", "arguments": {"order_id": "ORD-999"}}
    history = [
        Message(role=Role.USER, content="I want to return my order."),
        Message(role=Role.ASSISTANT, content="Let me check that for you."),
    ]
    conversation = executor.build_generated_simulator_prompt(
        tool_call, conversation_history=history
    )

    # Actual request is the last message (index 3)
    user_content = conversation.messages[3].content
    assert "Conversation so far" in user_content
    assert "return my order" in user_content


def test_build_tool_catalog(deterministic_tool, generated_tool, no_params_tool):
    catalog = ToolExecutor.build_tool_catalog(
        [deterministic_tool, generated_tool, no_params_tool]
    )

    assert "SearchOrders" in catalog
    assert "CheckEligibility" in catalog
    assert "GetCurrentTime" in catalog
    assert '"type": "object"' in catalog
    assert '"order_id"' in catalog


def test_build_tool_catalog_includes_full_schema(
    deterministic_tool, generated_tool, no_params_tool
):
    """Catalog includes structured tool info with params, returns, and usage."""
    catalog = ToolExecutor.build_tool_catalog(
        [deterministic_tool, generated_tool, no_params_tool]
    )
    assert "### SearchOrders" in catalog
    assert "### CheckEligibility" in catalog
    assert "### GetCurrentTime" in catalog
    assert "(required)" in catalog
    assert "Parameters:" in catalog
    assert "<tool_call>" in catalog
    assert "Usage:" in catalog


def test_build_tool_definitions_format(deterministic_tool, generated_tool):
    definitions = ToolExecutor.build_tool_definitions(
        [deterministic_tool, generated_tool]
    )

    assert len(definitions) == 2
    for defn in definitions:
        assert defn["type"] == "function"
        assert "name" in defn["function"]
        assert "description" in defn["function"]
        assert "parameters" in defn["function"]

    params = definitions[0]["function"]["parameters"]
    assert "order_id" in params["properties"]
    assert params["required"] == ["order_id"]


def test_format_tool_call_message_structure():
    tool_call = {"name": "SearchOrders", "arguments": {"order_id": "ORD-001"}}
    result = ToolExecutor.format_tool_call_message(tool_call, call_id="call_abc123")

    assert result["role"] == "assistant"
    assert result["content"] is None
    assert len(result["tool_calls"]) == 1

    tc = result["tool_calls"][0]
    assert tc["id"] == "call_abc123"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "SearchOrders"
    assert json.loads(tc["function"]["arguments"]) == {"order_id": "ORD-001"}


def test_format_tool_result_message_structure():
    result = ToolExecutor.format_tool_result_message(
        call_id="call_abc123",
        content='{"order_id": "ORD-001", "status": "delivered"}',
        name="SearchOrders",
    )

    assert result["role"] == "tool"
    assert result["tool_call_id"] == "call_abc123"
    assert result["name"] == "SearchOrders"
    assert result["content"] == '{"order_id": "ORD-001", "status": "delivered"}'


def test_get_tool_by_name(executor, deterministic_tool):
    tool = executor.get_tool_by_name("SearchOrders")
    assert tool is not None
    assert tool.id == deterministic_tool.id

    assert executor.get_tool_by_name("NoSuchTool") is None


def test_build_capability_summary(deterministic_tool, generated_tool):
    summary = ToolExecutor.build_capability_summary(
        [deterministic_tool, generated_tool]
    )
    assert deterministic_tool.description in summary
    assert generated_tool.description in summary
    assert deterministic_tool.name not in summary
    assert generated_tool.name not in summary
    for line in summary.strip().split("\n"):
        assert line.startswith("- ")


def test_parse_tool_call_unclosed_tag(executor):
    """Fallback regex matches when LLM forgets closing tag."""
    response = '<tool_call>{"name": "SearchOrders", "arguments": {"order_id": "X"}}'
    result = executor.parse_and_validate_tool_call(response)
    assert isinstance(result, ToolCallParsed)
    assert result.tool_call["name"] == "SearchOrders"
    assert result.tool_call["arguments"] == {"order_id": "X"}


def test_parse_tool_call_trailing_comma(executor):
    """Trailing comma before } is cleaned up."""
    response = (
        '<tool_call>{"name": "SearchOrders", '
        '"arguments": {"order_id": "X",}}</tool_call>'
    )
    result = executor.parse_and_validate_tool_call(response)
    assert isinstance(result, ToolCallParsed)
    assert result.tool_call["name"] == "SearchOrders"


def test_parse_tool_call_markdown_fences_inside_tag(executor):
    """Markdown fences wrapping JSON inside the tag are handled."""
    response = (
        "<tool_call>```json\n"
        '{"name": "SearchOrders", "arguments": {"order_id": "X"}}\n'
        "```</tool_call>"
    )
    result = executor.parse_and_validate_tool_call(response)
    assert isinstance(result, ToolCallParsed)
    assert result.tool_call["name"] == "SearchOrders"


def test_parse_tool_call_open_tag_with_trailing_prose(executor):
    """Open-tag fallback doesn't consume trailing prose as JSON."""
    response = (
        '<tool_call>{"name": "SearchOrders", "arguments": {"order_id": "X"}}'
        "</tool_call> Here is some reasoning about the result."
    )
    result = executor.parse_and_validate_tool_call(response)
    assert isinstance(result, ToolCallParsed)
    assert result.tool_call["name"] == "SearchOrders"


def test_parse_tool_call_comma_in_string_value_preserved(executor):
    """Trailing comma fix doesn't corrupt commas inside string values."""
    response = (
        '<tool_call>{"name": "SearchOrders", '
        '"arguments": {"order_id": "items A, B]"}}</tool_call>'
    )
    result = executor.parse_and_validate_tool_call(response)
    assert isinstance(result, ToolCallParsed)
    assert result.tool_call["arguments"]["order_id"] == "items A, B]"


def test_strip_tool_tags_removes_both_tags():
    text = "<tool_call>some content</tool_call>"
    assert ToolExecutor.strip_tool_tags(text) == "some content"


def test_strip_tool_tags_no_tags():
    assert ToolExecutor.strip_tool_tags("plain text") == "plain text"


def test_strip_tool_tags_partial_open_only():
    text = "I tried to use <tool_call> but failed"
    assert ToolExecutor.strip_tool_tags(text) == "I tried to use  but failed"


class TestStripBareToolJson:
    def test_removes_bare_tool_call_json(self):
        text = (
            "Calling tool.\n\n"
            '{"name": "DescribeTable", "arguments": {"table": "users"}}\n\n'
            "Done."
        )
        result = ToolExecutor.strip_bare_tool_json(text)
        assert "DescribeTable" not in result
        assert "Calling tool." in result
        assert "Done." in result

    def test_preserves_non_tool_json(self):
        text = 'Here is the result: {"status": "ok", "count": 5}'
        result = ToolExecutor.strip_bare_tool_json(text)
        assert '{"status": "ok", "count": 5}' in result

    def test_removes_multiple_bare_tool_calls(self):
        text = (
            '{"name": "ToolA", "arguments": {"x": 1}}\n\n'
            "middle text\n\n"
            '{"name": "ToolB", "arguments": {"y": 2}}'
        )
        result = ToolExecutor.strip_bare_tool_json(text)
        assert "ToolA" not in result
        assert "ToolB" not in result
        assert "middle text" in result

    def test_handles_nested_json_in_arguments(self):
        text = (
            '{"name": "Insert", "arguments":'
            ' {"data": {"key": "val", "nested": {"a": 1}}}}'
        )
        result = ToolExecutor.strip_bare_tool_json(text)
        assert result == ""

    def test_no_change_when_no_bare_json(self):
        text = "Just plain text with no JSON at all."
        result = ToolExecutor.strip_bare_tool_json(text)
        assert result == text

    def test_empty_string(self):
        assert ToolExecutor.strip_bare_tool_json("") == ""


class TestSanitizeAssistantContent:
    def test_removes_complete_tagged_tool_call_block(self):
        text = (
            "Checking the data now.\n\n"
            '<tool_call>{"name": "SearchOrders", "arguments": {"order_id": "X"}}'
            "</tool_call>"
        )
        result = ToolExecutor.sanitize_assistant_content(text)
        assert result == "Checking the data now."

    def test_removes_dangling_open_tool_call(self):
        text = (
            "Perfect! Now let me create a summary analysis.\n\n"
            '<tool_call>{"name": "RunQuery", "arguments": {"sql": "SELECT *"'
        )
        result = ToolExecutor.sanitize_assistant_content(text)
        assert result == "Perfect! Now let me create a summary analysis."

    def test_removes_malformed_bare_tool_json_fragment(self):
        text = (
            "Excellent. Let me verify the changes:\n\n"
            '{"name": "RunQuery", "arguments": {"sql": "SELECT * FROM users"\n\n'
            "Done."
        )
        result = ToolExecutor.sanitize_assistant_content(text)
        assert '{"name": "RunQuery"' not in result
        assert "Excellent." in result
        assert "Done." in result

    def test_drops_artifact_only_content(self):
        assert ToolExecutor.sanitize_assistant_content("}") == ""
        assert ToolExecutor.sanitize_assistant_content(">") == ""


class TestParseAndValidateToolCall:
    def test_valid_tool_call_returns_parsed(self, executor):
        response = (
            '<tool_call>{"name": "SearchOrders",'
            ' "arguments": {"order_id": "ORD-123"}}</tool_call>'
        )
        result = executor.parse_and_validate_tool_call(response)
        assert isinstance(result, ToolCallParsed)
        assert result.tool_call["name"] == "SearchOrders"
        assert result.tool_call["arguments"] == {"order_id": "ORD-123"}

    def test_no_tool_call_returns_none(self, executor):
        result = executor.parse_and_validate_tool_call("Just a regular response.")
        assert result is None

    def test_malformed_json_returns_error(self, executor):
        response = "<tool_call>this is not json</tool_call>"
        result = executor.parse_and_validate_tool_call(response)
        assert isinstance(result, ToolCallError)
        assert "malformed_json" in result.error_json
        assert result.tool_name is None

    def test_unknown_tool_returns_error(self, executor):
        response = '<tool_call>{"name": "FakeToolXYZ", "arguments": {}}</tool_call>'
        result = executor.parse_and_validate_tool_call(response)
        assert isinstance(result, ToolCallError)
        assert "unknown_tool" in result.error_json
        assert "FakeToolXYZ" in result.error_json
        assert "SearchOrders" in result.error_json
        assert result.tool_name == "FakeToolXYZ"

    def test_missing_required_param_returns_error(self, executor):
        response = (
            '<tool_call>{"name": "SearchOrders",'
            ' "arguments": {"include_items": true}}</tool_call>'
        )
        result = executor.parse_and_validate_tool_call(response)
        assert isinstance(result, ToolCallError)
        assert "invalid_arguments" in result.error_json
        assert "order_id" in result.error_json
        assert result.tool_name == "SearchOrders"

    def test_wrong_type_returns_error(self, executor):
        response = (
            '<tool_call>{"name": "SearchOrders",'
            ' "arguments": {"order_id": 12345}}</tool_call>'
        )
        result = executor.parse_and_validate_tool_call(response)
        assert isinstance(result, ToolCallError)
        assert "invalid_arguments" in result.error_json
        assert result.tool_name == "SearchOrders"

    def test_unknown_param_returns_error(self, executor):
        """Requires 'additionalProperties: false' in the fixture schema."""
        response = (
            '<tool_call>{"name": "SearchOrders",'
            ' "arguments": {"order_id": "X",'
            ' "bogus": "val"}}</tool_call>'
        )
        result = executor.parse_and_validate_tool_call(response)
        assert isinstance(result, ToolCallError)
        assert "invalid_arguments" in result.error_json

    def test_tool_with_no_params_accepts_empty_args(self, executor):
        response = '<tool_call>{"name": "GetCurrentTime"}</tool_call>'
        result = executor.parse_and_validate_tool_call(response)
        assert isinstance(result, ToolCallParsed)
        assert result.tool_call["name"] == "GetCurrentTime"

    def test_missing_name_returns_error(self, executor):
        response = '<tool_call>{"arguments": {"order_id": "X"}}</tool_call>'
        result = executor.parse_and_validate_tool_call(response)
        assert isinstance(result, ToolCallError)
        assert "malformed" in result.error_json.lower()

    def test_arguments_not_dict_returns_error(self, executor):
        response = '<tool_call>{"name": "SearchOrders", "arguments": "bad"}</tool_call>'
        result = executor.parse_and_validate_tool_call(response)
        assert isinstance(result, ToolCallError)

    def test_unclosed_tag_still_works(self, executor):
        response = '<tool_call>{"name": "SearchOrders", "arguments": {"order_id": "X"}}'
        result = executor.parse_and_validate_tool_call(response)
        assert isinstance(result, ToolCallParsed)

    def test_extra_braces_still_works(self, executor):
        response = (
            '<tool_call>{"name": "SearchOrders",'
            ' "arguments": {"order_id": "X"}}}</tool_call>'
        )
        result = executor.parse_and_validate_tool_call(response)
        assert isinstance(result, ToolCallParsed)

    def test_error_json_is_valid_json(self, executor):
        import json

        cases = [
            "<tool_call>not json</tool_call>",
            '<tool_call>{"name": "Fake", "arguments": {}}</tool_call>',
            (
                '<tool_call>{"name": "SearchOrders",'
                ' "arguments": {"include_items": true}}'
                "</tool_call>"
            ),
        ]
        for response in cases:
            result = executor.parse_and_validate_tool_call(response)
            if isinstance(result, ToolCallError):
                parsed = json.loads(result.error_json)
                assert "error" in parsed
                assert "message" in parsed

    def test_bare_json_tool_call_parsed(self, executor):
        """Bare JSON without <tool_call> tags should be parsed as fallback."""
        response = (
            'Let me look that up.\n\n'
            '{"name": "SearchOrders", "arguments": {"order_id": "ORD-001"}}'
        )
        result = executor.parse_and_validate_tool_call(response)
        assert isinstance(result, ToolCallParsed)
        assert result.tool_call["name"] == "SearchOrders"
        assert result.tool_call["arguments"]["order_id"] == "ORD-001"

    def test_bare_json_unknown_tool_returns_error(self, executor):
        """Bare JSON referencing unknown tool should return ToolCallError."""
        response = '{"name": "UnknownTool", "arguments": {}}'
        result = executor.parse_and_validate_tool_call(response)
        assert isinstance(result, ToolCallError)

    def test_bare_json_invalid_args_returns_error(self, executor):
        """Bare JSON with schema-invalid arguments should return ToolCallError."""
        response = '{"name": "SearchOrders", "arguments": {"order_id": 123}}'
        result = executor.parse_and_validate_tool_call(response)
        assert isinstance(result, ToolCallError)

    def test_tagged_tool_call_preferred_over_bare(self, executor):
        """If both <tool_call> tag and bare JSON exist, tag takes precedence."""
        response = (
            '<tool_call>{"name": "SearchOrders",'
            ' "arguments": {"order_id": "A"}}</tool_call>\n'
            '{"name": "SearchOrders",'
            ' "arguments": {"order_id": "B"}}'
        )
        result = executor.parse_and_validate_tool_call(response)
        assert isinstance(result, ToolCallParsed)
        assert result.tool_call["arguments"]["order_id"] == "A"

    def test_bare_json_not_tool_shaped_returns_none(self, executor):
        """JSON without name/arguments fields should not be treated as tool call."""
        response = 'Here is the data: {"users": [1, 2, 3]}'
        result = executor.parse_and_validate_tool_call(response)
        assert result is None
