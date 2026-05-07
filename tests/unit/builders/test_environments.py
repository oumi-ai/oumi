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

from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.configs.params.tool_params import ToolParams
from oumi.core.types.tool_call import ToolResult


def test_build_environment_imports_without_explicit_env_package():
    from oumi.builders.environments import build_environment

    params = EnvironmentParams(
        id="lookup",
        name="Lookup",
        description="A deterministic lookup environment",
        env_type="deterministic",
        tools=[ToolParams(id="get", name="Get", description="Lookup something.")],
        env_kwargs={
            "lookup_table": {
                "get": [{"input": {"q": "hi"}, "output": {"a": "hi"}}],
            }
        },
    )
    env = build_environment(params)
    assert env.step("get", {"q": "hi"}) == ToolResult(output={"a": "hi"})


def test_build_environment_unknown_env_type_raises():
    from oumi.builders.environments import build_environment

    params = EnvironmentParams(
        id="x",
        name="x",
        description="x",
        env_type="not_a_real_env_type",
    )
    with pytest.raises(ValueError, match="Unknown env_type 'not_a_real_env_type'"):
        build_environment(params)


def test_build_environment_dispatches_synthetic():
    from oumi.builders.environments import build_environment
    from oumi.environments.synthetic_environment import SyntheticEnvironment

    params = EnvironmentParams(
        id="faq",
        name="FAQ",
        description="FAQ tools",
        env_type="synthetic",
        tools=[ToolParams(id="answer", name="Answer", description="Answer.")],
        env_kwargs={"system_prompt": "Answer FAQs."},
    )
    env = build_environment(params)
    assert isinstance(env, SyntheticEnvironment)


def test_build_environment_dispatches_deterministic():
    from oumi.builders.environments import build_environment
    from oumi.environments.deterministic_environment import DeterministicEnvironment

    params = EnvironmentParams(
        id="lookup",
        name="Lookup",
        description="d",
        env_type="deterministic",
        tools=[ToolParams(id="t", name="t", description="t")],
        env_kwargs={"lookup_table": {"t": [{"input": {}, "output": {}}]}},
    )
    env = build_environment(params)
    assert isinstance(env, DeterministicEnvironment)
