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

import pytest

from oumi.core.configs.params.synthesis_params import (
    GeneralSynthesisParams,
    MultiTurnAttribute,
)
from oumi.core.configs.params.tool_params import (
    DeterministicToolOutput,
    GeneratedToolOutput,
    ToolAttribute,
    ToolOutputStrategy,
)
from oumi.core.types.conversation import Role


def test_deterministic_tool_output_empty_values_raises():
    with pytest.raises(ValueError, match="values cannot be empty"):
        DeterministicToolOutput(values={})


@pytest.mark.parametrize("rate", [-0.1, 1.1])
def test_deterministic_tool_output_invalid_sample_rate_raises(rate):
    with pytest.raises(ValueError, match="sample_rate must be between 0 and 1"):
        DeterministicToolOutput(values={"x": 1}, sample_rate=rate)


def _make_deterministic_tool(**overrides) -> ToolAttribute:
    defaults = dict(
        id="tool1",
        name="MyTool",
        description="A tool",
        output_strategy=ToolOutputStrategy.DETERMINISTIC,
        deterministic_outputs=[
            DeterministicToolOutput(values={"a": 1}),
        ],
    )
    defaults.update(overrides)
    return ToolAttribute(**defaults)  # type: ignore[arg-type]


def _make_generated_tool(**overrides) -> ToolAttribute:
    defaults = dict(
        id="tool2",
        name="GenTool",
        description="A generated tool",
        output_strategy=ToolOutputStrategy.GENERATED,
        generated_output=GeneratedToolOutput(instruction="Do something."),
    )
    defaults.update(overrides)
    return ToolAttribute(**defaults)  # type: ignore[arg-type]


def test_tool_attribute_deterministic_without_outputs_raises():
    with pytest.raises(ValueError, match="deterministic_outputs cannot be empty"):
        ToolAttribute(
            id="t",
            name="T",
            description="d",
            output_strategy=ToolOutputStrategy.DETERMINISTIC,
            deterministic_outputs=[],
        )


def test_tool_attribute_generated_without_output_raises():
    with pytest.raises(ValueError, match="generated_output must be provided"):
        ToolAttribute(
            id="t",
            name="T",
            description="d",
            output_strategy=ToolOutputStrategy.GENERATED,
            generated_output=None,
        )


@pytest.mark.parametrize(
    "field,value",
    [("id", ""), ("name", ""), ("description", "")],
)
def test_tool_attribute_empty_field_raises(field, value):
    with pytest.raises(ValueError, match=f"{field} cannot be empty"):
        _make_generated_tool(**{field: value})


def test_tool_attribute_normalizes_undefined_sample_rates():
    outputs = [
        DeterministicToolOutput(values={"a": 1}),
        DeterministicToolOutput(values={"b": 2}),
    ]
    tool = _make_deterministic_tool(deterministic_outputs=outputs)
    assert tool.deterministic_outputs[0].sample_rate == pytest.approx(0.5)
    assert tool.deterministic_outputs[1].sample_rate == pytest.approx(0.5)


def test_tool_attribute_normalizes_mixed_sample_rates():
    outputs = [
        DeterministicToolOutput(values={"a": 1}, sample_rate=0.7),
        DeterministicToolOutput(values={"b": 2}),
    ]
    tool = _make_deterministic_tool(deterministic_outputs=outputs)
    assert tool.deterministic_outputs[0].sample_rate == pytest.approx(0.7)
    assert tool.deterministic_outputs[1].sample_rate == pytest.approx(0.3)


def test_tool_attribute_sample_rates_exceeding_one_raises():
    outputs = [
        DeterministicToolOutput(values={"a": 1}, sample_rate=0.6),
        DeterministicToolOutput(values={"b": 2}, sample_rate=0.6),
    ]
    with pytest.raises(ValueError, match="sample rates must sum to at most 1.0"):
        _make_deterministic_tool(deterministic_outputs=outputs)


def _make_multiturn_attr(**overrides) -> MultiTurnAttribute:
    defaults = dict(
        id="chat",
        min_turns=1,
        max_turns=3,
        role_instruction_messages={
            Role.USER: "You are a user.",
            Role.ASSISTANT: "You are an assistant.",
        },
        available_tools=[],
    )
    defaults.update(overrides)
    return MultiTurnAttribute(**defaults)  # type: ignore[arg-type]


def test_synthesis_params_valid_tool_references():
    tool = _make_generated_tool(id="search")
    mt = _make_multiturn_attr(available_tools=["search"])
    params = GeneralSynthesisParams(
        tools=[tool],
        multiturn_attributes=[mt],
    )
    assert params.tools is not None
    assert len(params.tools) == 1


def test_synthesis_params_undefined_tool_reference_raises():
    tool = _make_generated_tool(id="search")
    mt = _make_multiturn_attr(available_tools=["nonexistent"])
    with pytest.raises(ValueError, match="references unknown tool 'nonexistent'"):
        GeneralSynthesisParams(
            tools=[tool],
            multiturn_attributes=[mt],
        )


def test_synthesis_params_available_tools_without_tools_defined_raises():
    mt = _make_multiturn_attr(available_tools=["search"])
    with pytest.raises(ValueError, match="tools must be defined"):
        GeneralSynthesisParams(
            tools=None,
            multiturn_attributes=[mt],
        )


def test_synthesis_params_duplicate_tool_ids_raises():
    t1 = _make_generated_tool(id="dup")
    t2 = _make_generated_tool(id="dup")
    mt = _make_multiturn_attr(available_tools=["dup"])
    with pytest.raises(ValueError, match="duplicate tool ids"):
        GeneralSynthesisParams(
            tools=[t1, t2],
            multiturn_attributes=[mt],
        )
