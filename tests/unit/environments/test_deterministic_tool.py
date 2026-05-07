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

from oumi.environments.deterministic_tool import (
    DeterministicTool,
    DeterministicToolOutput,
)


def test_deterministic_tool_output_allows_empty_input():
    entry = DeterministicToolOutput(input={}, output={"msg": "ok"})
    assert entry.input == {}


def test_deterministic_tool_output_allows_empty_output():
    entry = DeterministicToolOutput(input={"id": "1"}, output={})
    assert entry.output == {}


def test_deterministic_tool_output_matches_exact():
    entry = DeterministicToolOutput(
        input={"id": "01", "status": "pending"},
        output={"message": "Order is pending"},
    )
    assert entry.matches({"id": "01", "status": "pending"}) is True
    assert entry.matches({"status": "pending", "id": "01"}) is True


def test_deterministic_tool_output_no_match():
    entry = DeterministicToolOutput(
        input={"id": "01"},
        output={"message": "ok"},
    )
    assert entry.matches({"id": "02"}) is False
    assert entry.matches({"id": "01", "extra": "arg"}) is False


def test_deterministic_tool_create_coerces_outputs():
    tool = DeterministicTool.create(
        {
            "id": "lookup",
            "name": "Lookup",
            "description": "Lookup.",
            "deterministic_outputs": [
                {"input": {"id": "1"}, "output": {"msg": "ok"}},
            ],
        }
    )
    assert isinstance(tool, DeterministicTool)
    assert isinstance(tool.deterministic_outputs[0], DeterministicToolOutput)
    assert tool.deterministic_outputs[0].input == {"id": "1"}


def test_deterministic_tool_create_passthrough():
    tool = DeterministicTool(
        id="lookup",
        name="Lookup",
        description="Lookup.",
        deterministic_outputs=[
            DeterministicToolOutput(input={"id": "1"}, output={"msg": "ok"}),
        ],
    )
    assert DeterministicTool.create(tool) is tool
