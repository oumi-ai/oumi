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

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# parse_tool_call
# ---------------------------------------------------------------------------


def test_parse_tool_call_valid(executor):
    """Test parsing a well-formed tool call."""
    response = (
        'Some text <tool_call>{"name": "SearchOrders", '
        '"arguments": {"order_id": "ORD-123"}}</tool_call> more text'
    )
    result = executor.parse_tool_call(response)

    assert result is not None
    assert result["name"] == "SearchOrders"
    assert result["arguments"] == {"order_id": "ORD-123"}


def test_parse_tool_call_no_tag(executor):
    """Test that parse_tool_call returns None when no tool_call tag is present."""
    result = executor.parse_tool_call("Just a regular response with no tool call.")
    assert result is None


def test_parse_tool_call_malformed_json(executor):
    """Test that parse_tool_call returns None for malformed JSON."""
    response = "<tool_call>this is not json at all</tool_call>"
    result = executor.parse_tool_call(response)
    assert result is None


def test_parse_tool_call_missing_name(executor):
    """Test that parse_tool_call returns None when name is missing."""
    response = '<tool_call>{"arguments": {"order_id": "ORD-123"}}</tool_call>'
    result = executor.parse_tool_call(response)
    assert result is None


def test_parse_tool_call_unknown_tool(executor):
    """Test that parse_tool_call returns None for an unknown tool name."""
    response = (
        '<tool_call>{"name": "NonExistentTool", "arguments": {"x": 1}}</tool_call>'
    )
    result = executor.parse_tool_call(response)
    assert result is None


def test_parse_tool_call_extra_braces(executor):
    """Test that parse_tool_call handles extra trailing braces from LLM artifacts."""
    response = (
        '<tool_call>{"name": "SearchOrders", '
        '"arguments": {"order_id": "ORD-123"}}}</tool_call>'
    )
    result = executor.parse_tool_call(response)

    assert result is not None
    assert result["name"] == "SearchOrders"
    assert result["arguments"] == {"order_id": "ORD-123"}


def test_parse_tool_call_non_dict_json(executor):
    """Test that parse_tool_call returns None when JSON is not a dict."""
    response = '<tool_call>["not", "a", "dict"]</tool_call>'
    result = executor.parse_tool_call(response)
    assert result is None


def test_parse_tool_call_arguments_not_dict(executor):
    """Test that parse_tool_call returns None when arguments is not a dict."""
    response = '<tool_call>{"name": "SearchOrders", "arguments": "bad"}</tool_call>'
    result = executor.parse_tool_call(response)
    assert result is None


def test_parse_tool_call_empty_arguments(executor):
    """Test parsing a tool call with no arguments field defaults to empty dict."""
    response = '<tool_call>{"name": "GetCurrentTime"}</tool_call>'
    result = executor.parse_tool_call(response)

    assert result is not None
    assert result["name"] == "GetCurrentTime"
    assert result["arguments"] == {}


def test_parse_tool_call_name_not_string(executor):
    """Test that parse_tool_call returns None when name is not a string."""
    response = '<tool_call>{"name": 123, "arguments": {}}</tool_call>'
    result = executor.parse_tool_call(response)
    assert result is None


# ---------------------------------------------------------------------------
# validate_arguments
# ---------------------------------------------------------------------------


def test_validate_arguments_valid(executor):
    """Test that valid arguments pass validation."""
    tool_call = {"name": "SearchOrders", "arguments": {"order_id": "ORD-001"}}
    error = executor.validate_arguments(tool_call)
    assert error is None


def test_validate_arguments_valid_with_optional(executor):
    """Test that providing both required and optional arguments passes."""
    tool_call = {
        "name": "SearchOrders",
        "arguments": {"order_id": "ORD-001", "include_items": True},
    }
    error = executor.validate_arguments(tool_call)
    assert error is None


def test_validate_arguments_missing_required(executor):
    """Test that missing required arguments are detected."""
    tool_call = {"name": "SearchOrders", "arguments": {"include_items": True}}
    error = executor.validate_arguments(tool_call)

    assert error is not None
    assert error.error_type == "missing_required"
    assert "order_id" in error.message
    assert error.tool_name == "SearchOrders"
    assert "order_id" in error.details["missing"]


def test_validate_arguments_unknown_params(executor):
    """Test that unknown parameters are detected."""
    tool_call = {
        "name": "SearchOrders",
        "arguments": {"order_id": "ORD-001", "bogus_param": "value"},
    }
    error = executor.validate_arguments(tool_call)

    assert error is not None
    assert error.error_type == "unknown_parameters"
    assert "bogus_param" in error.message
    assert "bogus_param" in error.details["unknown"]


def test_validate_arguments_unknown_tool(executor):
    """Test validation for an unknown tool name."""
    tool_call = {"name": "FakeTool", "arguments": {"x": 1}}
    error = executor.validate_arguments(tool_call)

    assert error is not None
    assert error.error_type == "unknown_tool"
    assert error.tool_name == "FakeTool"


def test_validate_arguments_no_parameters_tool(executor):
    """Test that a tool with no parameters passes validation with empty args."""
    tool_call = {"name": "GetCurrentTime", "arguments": {}}
    error = executor.validate_arguments(tool_call)
    assert error is None


# ---------------------------------------------------------------------------
# sample_deterministic_outputs
# ---------------------------------------------------------------------------


def test_sample_deterministic_outputs_returns_for_deterministic_tools(
    executor, deterministic_tool, generated_tool
):
    """Test that sample_deterministic_outputs returns entries for DETERMINISTIC tools."""
    selections = executor.sample_deterministic_outputs(
        [deterministic_tool, generated_tool]
    )

    assert deterministic_tool.id in selections
    assert generated_tool.id not in selections

    # The selection should be valid JSON matching one of the canned outputs
    parsed = json.loads(selections[deterministic_tool.id])
    valid_statuses = {"delivered", "pending"}
    assert parsed["status"] in valid_statuses


def test_sample_deterministic_outputs_skips_generated_tools(executor, generated_tool):
    """Test that GENERATED tools are skipped entirely."""
    selections = executor.sample_deterministic_outputs([generated_tool])
    assert len(selections) == 0


def test_sample_deterministic_outputs_respects_weights(executor, deterministic_tool):
    """Test that sampling respects sample_rate weights over many iterations."""
    counts = {"delivered": 0, "pending": 0}
    n = 1000

    for _ in range(n):
        selections = executor.sample_deterministic_outputs([deterministic_tool])
        parsed = json.loads(selections[deterministic_tool.id])
        counts[parsed["status"]] += 1

    # With 0.7/0.3 weights over 1000 samples, delivered should dominate
    assert counts["delivered"] > counts["pending"]
    # Rough sanity: delivered should be at least 50% (very conservative bound)
    assert counts["delivered"] > n * 0.5


def test_sample_deterministic_outputs_empty_list(executor):
    """Test with an empty tools list."""
    selections = executor.sample_deterministic_outputs([])
    assert selections == {}


# ---------------------------------------------------------------------------
# resolve_output
# ---------------------------------------------------------------------------


def test_resolve_output_deterministic_returns_preselected(executor, deterministic_tool):
    """Test that DETERMINISTIC tools return the pre-selected output."""
    preselected = json.dumps({"order_id": "ORD-001", "status": "delivered"})
    selections = {deterministic_tool.id: preselected}

    tool_call = {"name": "SearchOrders", "arguments": {"order_id": "ORD-001"}}
    result = executor.resolve_output(tool_call, selections)

    assert result == preselected


def test_resolve_output_deterministic_missing_selection(executor, deterministic_tool):
    """Test that a missing deterministic selection returns an error JSON."""
    tool_call = {"name": "SearchOrders", "arguments": {"order_id": "ORD-001"}}
    result = executor.resolve_output(tool_call, {})

    assert result is not None
    parsed = json.loads(result)
    assert "error" in parsed


def test_resolve_output_generated_returns_none(executor):
    """Test that GENERATED tools return None (caller must use simulator)."""
    tool_call = {"name": "CheckEligibility", "arguments": {"order_id": "ORD-001"}}
    result = executor.resolve_output(tool_call, {})
    assert result is None


def test_resolve_output_unknown_tool(executor):
    """Test that an unknown tool returns error JSON."""
    tool_call = {"name": "NoSuchTool", "arguments": {}}
    result = executor.resolve_output(tool_call, {})

    assert result is not None
    parsed = json.loads(result)
    assert "error" in parsed
    assert "NoSuchTool" in parsed["error"]


# ---------------------------------------------------------------------------
# build_generated_simulator_prompt
# ---------------------------------------------------------------------------


def test_build_generated_simulator_prompt_structure(executor):
    """Test that the simulator prompt has correct message structure."""
    tool_call = {"name": "CheckEligibility", "arguments": {"order_id": "ORD-999"}}
    conversation = executor.build_generated_simulator_prompt(tool_call)

    assert len(conversation.messages) == 2
    assert conversation.messages[0].role == Role.SYSTEM
    assert conversation.messages[1].role == Role.USER


def test_build_generated_simulator_prompt_system_content(executor):
    """Test that the system message contains tool metadata."""
    tool_call = {"name": "CheckEligibility", "arguments": {"order_id": "ORD-999"}}
    conversation = executor.build_generated_simulator_prompt(tool_call)

    system_content = conversation.messages[0].content
    assert "CheckEligibility" in system_content
    assert "Check return eligibility" in system_content
    assert "Parameter schema" in system_content
    assert "output schema" in system_content.lower()
    assert "Return eligibility based on order status" in system_content
    assert "valid JSON" in system_content


def test_build_generated_simulator_prompt_user_content(executor):
    """Test that the user message contains the tool call arguments."""
    tool_call = {"name": "CheckEligibility", "arguments": {"order_id": "ORD-999"}}
    conversation = executor.build_generated_simulator_prompt(tool_call)

    user_content = conversation.messages[1].content
    assert "CheckEligibility" in user_content
    assert "ORD-999" in user_content
    assert "Generate the tool's JSON output" in user_content


def test_build_generated_simulator_prompt_with_history(executor):
    """Test that conversation history is included when provided."""
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
    assert "check that for you" in user_content


def test_build_generated_simulator_prompt_without_history(executor):
    """Test that the prompt works without conversation history."""
    tool_call = {"name": "CheckEligibility", "arguments": {"order_id": "ORD-999"}}
    conversation = executor.build_generated_simulator_prompt(tool_call)

    user_content = conversation.messages[1].content
    assert "Conversation so far" not in user_content


def test_build_generated_simulator_prompt_no_params_tool(executor):
    """Test simulator prompt for a tool with no parameters."""
    tool_call = {"name": "GetCurrentTime", "arguments": {}}
    conversation = executor.build_generated_simulator_prompt(tool_call)

    system_content = conversation.messages[0].content
    assert "GetCurrentTime" in system_content
    # Should not include "Parameter schema" since the tool has no parameters
    assert "Parameter schema" not in system_content


# ---------------------------------------------------------------------------
# build_tool_catalog
# ---------------------------------------------------------------------------


def test_build_tool_catalog_contains_all_tools(
    deterministic_tool, generated_tool, no_params_tool
):
    """Test that the catalog lists all tools with descriptions."""
    catalog = ToolExecutor.build_tool_catalog(
        [deterministic_tool, generated_tool, no_params_tool]
    )

    assert "SearchOrders" in catalog
    assert "CheckEligibility" in catalog
    assert "GetCurrentTime" in catalog
    assert "Search for customer orders" in catalog


def test_build_tool_catalog_shows_required_params(deterministic_tool):
    """Test that required parameters are marked in the catalog."""
    catalog = ToolExecutor.build_tool_catalog([deterministic_tool])

    assert "order_id (string, required)" in catalog
    assert "include_items (boolean)" in catalog
    # include_items should NOT have ", required"
    assert "include_items (boolean, required)" not in catalog


def test_build_tool_catalog_shows_param_descriptions(deterministic_tool):
    """Test that parameter descriptions are shown in the catalog."""
    catalog = ToolExecutor.build_tool_catalog([deterministic_tool])

    assert "The order ID to look up" in catalog
    assert "Whether to include line items" in catalog


def test_build_tool_catalog_no_params_tool(no_params_tool):
    """Test catalog output for a tool with no parameters."""
    catalog = ToolExecutor.build_tool_catalog([no_params_tool])

    assert "GetCurrentTime" in catalog
    assert "Get the current server time" in catalog
    # Should only have the one line for the tool itself
    lines = [line for line in catalog.split("\n") if line.strip()]
    assert len(lines) == 1


def test_build_tool_catalog_empty_list():
    """Test catalog with empty tools list."""
    catalog = ToolExecutor.build_tool_catalog([])
    assert catalog == ""


# ---------------------------------------------------------------------------
# build_tool_definitions
# ---------------------------------------------------------------------------


def test_build_tool_definitions_format(deterministic_tool, generated_tool):
    """Test that tool definitions follow the OpenAI function format."""
    definitions = ToolExecutor.build_tool_definitions(
        [deterministic_tool, generated_tool]
    )

    assert len(definitions) == 2
    for defn in definitions:
        assert defn["type"] == "function"
        assert "name" in defn["function"]
        assert "description" in defn["function"]
        assert "parameters" in defn["function"]


def test_build_tool_definitions_includes_parameters(deterministic_tool):
    """Test that parameters are included in the definition."""
    definitions = ToolExecutor.build_tool_definitions([deterministic_tool])
    params = definitions[0]["function"]["parameters"]

    assert "properties" in params
    assert "order_id" in params["properties"]
    assert params["required"] == ["order_id"]


def test_build_tool_definitions_no_params_gets_empty_object(no_params_tool):
    """Test that a tool with no parameters gets an empty object schema."""
    definitions = ToolExecutor.build_tool_definitions([no_params_tool])
    params = definitions[0]["function"]["parameters"]

    assert params == {"type": "object", "properties": {}}


# ---------------------------------------------------------------------------
# format_tool_call_message
# ---------------------------------------------------------------------------


def test_format_tool_call_message_structure():
    """Test that format_tool_call_message produces correct OpenAI format."""
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


def test_format_tool_call_message_empty_arguments():
    """Test formatting a tool call with empty arguments."""
    tool_call = {"name": "GetCurrentTime", "arguments": {}}
    result = ToolExecutor.format_tool_call_message(tool_call, call_id="call_xyz")

    tc = result["tool_calls"][0]
    assert json.loads(tc["function"]["arguments"]) == {}


# ---------------------------------------------------------------------------
# format_tool_result_message
# ---------------------------------------------------------------------------


def test_format_tool_result_message_structure():
    """Test that format_tool_result_message produces correct OpenAI format."""
    result = ToolExecutor.format_tool_result_message(
        call_id="call_abc123",
        content='{"order_id": "ORD-001", "status": "delivered"}',
        name="SearchOrders",
    )

    assert result["role"] == "tool"
    assert result["tool_call_id"] == "call_abc123"
    assert result["name"] == "SearchOrders"
    assert result["content"] == '{"order_id": "ORD-001", "status": "delivered"}'


def test_format_tool_result_message_error_content():
    """Test formatting a tool result with error content."""
    error_content = json.dumps({"error": "Order not found"})
    result = ToolExecutor.format_tool_result_message(
        call_id="call_err",
        content=error_content,
        name="SearchOrders",
    )

    assert result["role"] == "tool"
    assert result["tool_call_id"] == "call_err"
    assert result["name"] == "SearchOrders"
    parsed = json.loads(result["content"])
    assert parsed["error"] == "Order not found"


# ---------------------------------------------------------------------------
# get_tool_by_name
# ---------------------------------------------------------------------------


def test_get_tool_by_name_found(executor, deterministic_tool):
    """Test looking up a tool by name."""
    tool = executor.get_tool_by_name("SearchOrders")
    assert tool is not None
    assert tool.id == deterministic_tool.id


def test_get_tool_by_name_not_found(executor):
    """Test that looking up a nonexistent tool returns None."""
    tool = executor.get_tool_by_name("NoSuchTool")
    assert tool is None
