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

# ---------------------------------------------------------------------------
# ToolOutputStrategy
# ---------------------------------------------------------------------------


def test_tool_output_strategy_values():
    assert ToolOutputStrategy.DETERMINISTIC == "deterministic"
    assert ToolOutputStrategy.GENERATED == "generated"


def test_tool_output_strategy_is_str_enum():
    assert isinstance(ToolOutputStrategy.DETERMINISTIC, str)
    assert isinstance(ToolOutputStrategy.GENERATED, str)


# ---------------------------------------------------------------------------
# DeterministicToolOutput
# ---------------------------------------------------------------------------


def test_deterministic_tool_output_valid():
    output = DeterministicToolOutput(values={"status": "ok"})
    assert output.values == {"status": "ok"}
    assert output.sample_rate is None


def test_deterministic_tool_output_valid_with_sample_rate():
    output = DeterministicToolOutput(values={"x": 1}, sample_rate=0.5)
    assert output.sample_rate == 0.5


def test_deterministic_tool_output_boundary_sample_rates():
    DeterministicToolOutput(values={"x": 1}, sample_rate=0.0)
    DeterministicToolOutput(values={"x": 1}, sample_rate=1.0)


def test_deterministic_tool_output_empty_values_raises():
    with pytest.raises(ValueError, match="values cannot be empty"):
        DeterministicToolOutput(values={})


def test_deterministic_tool_output_negative_sample_rate_raises():
    with pytest.raises(ValueError, match="sample_rate must be between 0 and 1"):
        DeterministicToolOutput(values={"x": 1}, sample_rate=-0.1)


def test_deterministic_tool_output_sample_rate_above_one_raises():
    with pytest.raises(ValueError, match="sample_rate must be between 0 and 1"):
        DeterministicToolOutput(values={"x": 1}, sample_rate=1.1)


# ---------------------------------------------------------------------------
# GeneratedToolOutput
# ---------------------------------------------------------------------------


def test_generated_tool_output_valid():
    output = GeneratedToolOutput(instruction="Return a result.")
    assert output.instruction == "Return a result."


def test_generated_tool_output_empty_instruction_raises():
    with pytest.raises(ValueError, match="instruction cannot be empty"):
        GeneratedToolOutput(instruction="")


# ---------------------------------------------------------------------------
# ToolAttribute - DETERMINISTIC strategy
# ---------------------------------------------------------------------------


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


def test_tool_attribute_deterministic_valid():
    tool = _make_deterministic_tool()
    assert tool.id == "tool1"
    assert tool.output_strategy == ToolOutputStrategy.DETERMINISTIC
    # sample_rate should be normalized (only one output -> 1.0)
    assert tool.deterministic_outputs[0].sample_rate == 1.0


def test_tool_attribute_deterministic_without_outputs_raises():
    with pytest.raises(ValueError, match="deterministic_outputs cannot be empty"):
        ToolAttribute(
            id="t",
            name="T",
            description="d",
            output_strategy=ToolOutputStrategy.DETERMINISTIC,
            deterministic_outputs=[],
        )


# ---------------------------------------------------------------------------
# ToolAttribute - GENERATED strategy
# ---------------------------------------------------------------------------


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


def test_tool_attribute_generated_valid():
    tool = _make_generated_tool()
    assert tool.output_strategy == ToolOutputStrategy.GENERATED
    assert tool.generated_output is not None


def test_tool_attribute_generated_without_output_raises():
    with pytest.raises(ValueError, match="generated_output must be provided"):
        ToolAttribute(
            id="t",
            name="T",
            description="d",
            output_strategy=ToolOutputStrategy.GENERATED,
            generated_output=None,
        )


# ---------------------------------------------------------------------------
# ToolAttribute - empty fields
# ---------------------------------------------------------------------------


def test_tool_attribute_empty_id_raises():
    with pytest.raises(ValueError, match="id cannot be empty"):
        _make_generated_tool(id="")


def test_tool_attribute_empty_name_raises():
    with pytest.raises(ValueError, match="name cannot be empty"):
        _make_generated_tool(name="")


def test_tool_attribute_empty_description_raises():
    with pytest.raises(ValueError, match="description cannot be empty"):
        _make_generated_tool(description="")


# ---------------------------------------------------------------------------
# ToolAttribute - sample rate normalization
# ---------------------------------------------------------------------------


def test_tool_attribute_normalizes_undefined_sample_rates():
    outputs = [
        DeterministicToolOutput(values={"a": 1}),
        DeterministicToolOutput(values={"b": 2}),
    ]
    tool = _make_deterministic_tool(deterministic_outputs=outputs)
    # Both undefined -> uniform -> 0.5 each
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


def test_tool_attribute_sample_rates_exactly_one():
    outputs = [
        DeterministicToolOutput(values={"a": 1}, sample_rate=0.4),
        DeterministicToolOutput(values={"b": 2}, sample_rate=0.6),
    ]
    tool = _make_deterministic_tool(deterministic_outputs=outputs)
    assert tool.deterministic_outputs[0].sample_rate == pytest.approx(0.4)
    assert tool.deterministic_outputs[1].sample_rate == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# GeneralSynthesisParams - tool cross-validation
# ---------------------------------------------------------------------------


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


def test_synthesis_params_no_tools_no_references_passes():
    mt = _make_multiturn_attr(available_tools=[])
    params = GeneralSynthesisParams(
        tools=None,
        multiturn_attributes=[mt],
    )
    # Both should be normalized to None when empty/unused
    assert params.tools is None


def test_synthesis_params_no_multiturn_no_tools_passes():
    params = GeneralSynthesisParams()
    assert params.tools is None
    assert params.multiturn_attributes is None
