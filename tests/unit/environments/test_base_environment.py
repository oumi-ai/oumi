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

import dataclasses
import random
from typing import Any

import pytest

from oumi.core.configs.params.grounding_params import GroundingFact
from oumi.core.types.tool_call import ToolResult
from oumi.environments.base_environment import BaseEnvironment


def test_base_environment_is_not_a_dataclass():
    assert not dataclasses.is_dataclass(BaseEnvironment)


def test_base_environment_cannot_be_instantiated_directly():
    with pytest.raises(TypeError, match=r"abstract|instantiate"):
        BaseEnvironment()  # type: ignore[abstract]


def test_subclass_without_step_cannot_instantiate():
    class Incomplete(BaseEnvironment):
        pass

    with pytest.raises(TypeError, match=r"abstract|instantiate"):
        Incomplete()  # type: ignore[abstract]


class _MinimalEnv(BaseEnvironment):
    """Concrete BaseEnvironment subclass that doesn't override grounding hooks."""

    def step(self, calls: list[tuple[str, dict[str, Any]]]) -> list[ToolResult]:
        return [ToolResult(output={}) for _ in calls]


def test_default_sample_grounding_returns_empty():
    env = _MinimalEnv()
    assert env.sample_grounding(n=5, rng=random.Random(0)) == []


def test_default_describe_grounding_empty_list_returns_empty_string():
    env = _MinimalEnv()
    assert env.describe_grounding([]) == ""


def test_default_describe_grounding_delegates_to_helper():
    env = _MinimalEnv()
    facts = [GroundingFact(data={"id": "42", "title": "Dune"})]
    assert env.describe_grounding(facts) == '- id="42", title="Dune"'


def test_default_describe_grounding_skips_empty_data_facts():
    env = _MinimalEnv()
    facts = [
        GroundingFact(data={}),
        GroundingFact(data={"id": "1"}),
        GroundingFact(data={}),
    ]
    assert env.describe_grounding(facts) == '- id="1"'


def test_default_describe_grounding_escapes_embedded_quotes():
    env = _MinimalEnv()
    facts = [GroundingFact(data={"title": 'Foo "the" Bar'})]
    assert env.describe_grounding(facts) == r'- title="Foo \"the\" Bar"'
