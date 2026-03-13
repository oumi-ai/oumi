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
from oumi.core.synthesis.tool_executor import ToolExecutor
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
    result = executor.parse_tool_call(response)

    assert result is not None
    assert result["name"] == "SearchOrders"
    assert result["arguments"] == {"order_id": "ORD-123"}


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
    assert executor.parse_tool_call(response) is None


def test_parse_tool_call_extra_braces(executor):
    response = (
        '<tool_call>{"name": "SearchOrders", '
        '"arguments": {"order_id": "ORD-123"}}}</tool_call>'
    )
    result = executor.parse_tool_call(response)

    assert result is not None
    assert result["name"] == "SearchOrders"
    assert result["arguments"] == {"order_id": "ORD-123"}


def test_parse_tool_call_empty_arguments(executor):
    response = '<tool_call>{"name": "GetCurrentTime"}</tool_call>'
    result = executor.parse_tool_call(response)

    assert result is not None
    assert result["name"] == "GetCurrentTime"
    assert result["arguments"] == {}


def test_validate_arguments_valid(executor):
    tool_call = {"name": "SearchOrders", "arguments": {"order_id": "ORD-001"}}
    assert executor.validate_arguments(tool_call) is None


def test_validate_arguments_missing_required(executor):
    tool_call = {"name": "SearchOrders", "arguments": {"include_items": True}}
    error = executor.validate_arguments(tool_call)

    assert error is not None
    assert error.error_type == "missing_required"
    assert "order_id" in error.message
    assert "order_id" in error.details["missing"]


def test_validate_arguments_unknown_params(executor):
    tool_call = {
        "name": "SearchOrders",
        "arguments": {"order_id": "ORD-001", "bogus_param": "value"},
    }
    error = executor.validate_arguments(tool_call)

    assert error is not None
    assert error.error_type == "unknown_parameters"
    assert "bogus_param" in error.message


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

    assert len(conversation.messages) == 2
    assert conversation.messages[0].role == Role.SYSTEM
    assert conversation.messages[1].role == Role.USER

    system_content = conversation.messages[0].content
    assert "CheckEligibility" in system_content
    assert "Check return eligibility" in system_content
    assert "valid JSON" in system_content

    user_content = conversation.messages[1].content
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

    user_content = conversation.messages[1].content
    assert "Conversation so far" in user_content
    assert "return my order" in user_content


def test_build_tool_catalog(deterministic_tool, generated_tool, no_params_tool):
    catalog = ToolExecutor.build_tool_catalog(
        [deterministic_tool, generated_tool, no_params_tool]
    )

    assert "SearchOrders" in catalog
    assert "CheckEligibility" in catalog
    assert "GetCurrentTime" in catalog
    assert "order_id (string, required)" in catalog
    assert "include_items (boolean)" in catalog
    assert "include_items (boolean, required)" not in catalog


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
