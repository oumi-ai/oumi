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

from unittest.mock import Mock

import pytest

from oumi.core.configs.inference_config import InferenceConfig
from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.configs.params.generation_params import GenerationParams
from oumi.core.configs.params.model_params import ModelParams
from oumi.core.configs.params.tool_params import ToolError, ToolParams
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.core.types.tool_call import JSONSchema, ToolResult
from oumi.environments.deterministic_tool import (
    DeterministicTool,
    DeterministicToolOutput,
)
from oumi.environments.synthetic_environment import (
    SyntheticEnvironment,
    SyntheticEnvironmentKwargs,
    SyntheticStateParams,
)


def _make_state_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "files": {
                "type": "object",
                "properties": {"count": {"type": "integer"}},
                "required": ["count"],
            }
        },
        "required": ["files"],
    }


def _make_tool(**overrides) -> ToolParams:
    defaults: dict = dict(id="answer", name="Answer", description="Answer.")
    defaults.update(overrides)
    return ToolParams(**defaults)


def _make_params(**overrides) -> EnvironmentParams:
    defaults: dict = dict(
        id="faq",
        name="FAQ",
        description="FAQ tools",
        env_type="synthetic",
        tools=[_make_tool()],
        env_kwargs={"system_prompt": "Answer FAQs."},
    )
    defaults.update(overrides)
    return EnvironmentParams(**defaults)


def test_from_params_constructs_stateless():
    env = SyntheticEnvironment.from_params(_make_params())
    assert isinstance(env, SyntheticEnvironment)
    assert env.current_state is None
    assert isinstance(env._kwargs, SyntheticEnvironmentKwargs)


def test_from_params_constructs_stateful():
    params = _make_params(
        env_kwargs={
            "system_prompt": "You manage a filesystem.",
            "state_params": SyntheticStateParams(
                state_schema=_make_state_schema(),
                initial_state={"files": {"count": 1}},
            ),
            "cache_by_input": False,
        }
    )
    env = SyntheticEnvironment.from_params(params)
    assert env.current_state == {"files": {"count": 1}}


def test_empty_system_prompt_raises():
    params = _make_params(env_kwargs={"system_prompt": ""})
    with pytest.raises(ValueError, match="system_prompt cannot be empty"):
        SyntheticEnvironment.from_params(params)


def test_rejects_deterministic_outputs_on_tool():
    bad_tool = DeterministicTool(
        id="answer",
        name="Answer",
        description="Answer.",
        deterministic_outputs=[
            DeterministicToolOutput(input={"id": "01"}, output={"msg": "ok"}),
        ],
    )
    params = _make_params(tools=[bad_tool])
    with pytest.raises(ValueError, match="cannot define deterministic_outputs"):
        SyntheticEnvironment.from_params(params)


def test_rejects_cache_when_stateful():
    params = _make_params(
        env_kwargs={
            "system_prompt": "p",
            "state_params": SyntheticStateParams(),
            "cache_by_input": True,
        }
    )
    with pytest.raises(ValueError, match="cache_by_input must be False"):
        SyntheticEnvironment.from_params(params)


def test_cache_round_trip_stateless_caching():
    env = SyntheticEnvironment.from_params(_make_params())
    result = ToolResult(output={"temp": 72})
    env._cache_result("answer", {"city": "SF"}, result)
    cached = env._resolve_cached("answer", {"city": "SF"})
    assert cached == result
    assert cached is not result


def test_step_unknown_tool_raises():
    env = SyntheticEnvironment.from_params(_make_params())
    with pytest.raises(ValueError, match="Tool 'missing' not found"):
        env.step([("missing", {})])


def test_step_without_attach_inference_raises():
    env = SyntheticEnvironment.from_params(_make_params())
    with pytest.raises(RuntimeError, match="attach_inference"):
        env.step([("answer", {})])


# ---------- step() implementation tests ----------


def _typed_tool(**overrides) -> ToolParams:
    defaults: dict = dict(
        id="answer",
        name="Answer",
        description="Answer FAQs.",
        parameters=JSONSchema(
            type="object",
            properties={"q": JSONSchema(type="string")},
            required=["q"],
        ),
        output_schema=JSONSchema(
            type="object",
            properties={"a": JSONSchema(type="string")},
            required=["a"],
        ),
    )
    defaults.update(overrides)
    return ToolParams(**defaults)


def _typed_params() -> EnvironmentParams:
    return EnvironmentParams(
        id="faq",
        name="FAQ",
        description="FAQ env",
        env_type="synthetic",
        tools=[_typed_tool()],
        env_kwargs={"system_prompt": "Answer FAQs as a JSON tool."},
    )


def _attached_env(infer_returns: list[str]) -> tuple[SyntheticEnvironment, Mock]:
    """Build an env with a mock engine that drains ``infer_returns`` in order.

    Each call consumed by the simulator pops the next element from the queue,
    so a single ``infer_returns`` list spans multiple ``env.step`` invocations
    and multiple within-batch calls.
    """
    env = SyntheticEnvironment.from_params(_typed_params())
    mock_engine = Mock()
    queue = list(infer_returns)

    def fake_infer(convs, inference_config):
        out = []
        for conv in convs:
            text = queue.pop(0)
            out.append(
                Conversation(
                    messages=conv.messages
                    + [Message(role=Role.ASSISTANT, content=text)]
                )
            )
        return out

    mock_engine.infer = Mock(side_effect=fake_infer)
    base_config = InferenceConfig(
        model=ModelParams(model_name="stub"),
        generation=GenerationParams(max_new_tokens=32),
    )
    env.attach_inference(engine=mock_engine, base_config=base_config)
    return env, mock_engine


def test_step_zero_calls_returns_empty():
    env, mock = _attached_env([])
    assert env.step([]) == []
    assert mock.infer.call_count == 0


def test_step_unknown_tool_raises_before_engine_check():
    """Unknown tool id raises ValueError even when no engine is attached."""
    env = SyntheticEnvironment.from_params(_typed_params())
    with pytest.raises(ValueError, match="Tool 'missing' not found"):
        env.step([("missing", {})])


def test_step_happy_path_single_call():
    env, mock = _attached_env(['{"a": "hello"}'])
    results = env.step([("answer", {"q": "hi"})])
    assert results == [ToolResult(output={"a": "hello"})]
    assert mock.infer.call_count == 1


def test_step_batches_same_tool_in_one_infer():
    env, mock = _attached_env(['{"a": "one"}', '{"a": "two"}'])
    results = env.step([("answer", {"q": "first"}), ("answer", {"q": "second"})])
    assert results == [
        ToolResult(output={"a": "one"}),
        ToolResult(output={"a": "two"}),
    ]
    assert mock.infer.call_count == 1
    convs_arg = mock.infer.call_args[0][0]
    assert len(convs_arg) == 2


def test_step_cache_hit_short_circuits_inference():
    env, mock = _attached_env(['{"a": "fresh"}'])
    first = env.step([("answer", {"q": "hi"})])
    second = env.step([("answer", {"q": "hi"})])
    assert first == second
    assert mock.infer.call_count == 1  # second call hit the cache


def test_step_mixed_cache_hit_and_miss():
    env, mock = _attached_env(['{"a": "first"}', '{"a": "second"}'])
    env.step([("answer", {"q": "first"})])  # populates cache
    results = env.step([("answer", {"q": "first"}), ("answer", {"q": "second"})])
    assert results == [
        ToolResult(output={"a": "first"}),
        ToolResult(output={"a": "second"}),
    ]
    # 2 infers total: one for first call's miss, one for second call's miss.
    assert mock.infer.call_count == 2
    second_batch = mock.infer.call_args_list[1][0][0]
    assert len(second_batch) == 1  # only the unique miss got inferred


def test_step_repairs_truncated_braces():
    # Missing closing brace — repair_json_braces re-adds it.
    env, _ = _attached_env(['{"a": "needs-repair"'])
    results = env.step([("answer", {"q": "x"})])
    assert results == [ToolResult(output={"a": "needs-repair"})]


def test_step_schema_validation_failure_raises_tool_error():
    env, _ = _attached_env(['{"wrong_field": "oops"}'])  # missing required "a"
    with pytest.raises(ToolError, match="failed schema validation"):
        env.step([("answer", {"q": "x"})])


def test_step_unparseable_output_raises_tool_error():
    env, _ = _attached_env(["totally not json at all"])
    with pytest.raises(ToolError, match="not valid JSON"):
        env.step([("answer", {"q": "x"})])


def test_step_empty_response_raises_tool_error():
    env, _ = _attached_env([""])
    with pytest.raises(ToolError, match="empty response"):
        env.step([("answer", {"q": "x"})])


def test_step_no_output_schema_accepts_any_object():
    """When tool has no output_schema, any JSON object is accepted."""
    tool = _typed_tool(output_schema=None)
    params = EnvironmentParams(
        id="faq",
        name="FAQ",
        description="FAQ env",
        env_type="synthetic",
        tools=[tool],
        env_kwargs={"system_prompt": "Answer FAQs."},
    )
    env = SyntheticEnvironment.from_params(params)
    mock_engine = Mock()
    mock_engine.infer = Mock(
        return_value=[
            Conversation(
                messages=[
                    Message(role=Role.SYSTEM, content="x"),
                    Message(role=Role.USER, content="y"),
                    Message(role=Role.ASSISTANT, content='{"anything": [1, 2]}'),
                ]
            )
        ]
    )
    env.attach_inference(
        engine=mock_engine,
        base_config=InferenceConfig(model=ModelParams(model_name="stub")),
    )
    results = env.step([("answer", {"q": "x"})])
    assert results == [ToolResult(output={"anything": [1, 2]})]


def test_simulator_inference_config_overlays_guided_decoding():
    env, _ = _attached_env(['{"a": "x"}'])
    tool = env._lookup_tool("answer")
    cfg = env._simulator_inference_config(tool)
    assert cfg.generation is not None
    assert cfg.generation.guided_decoding is not None
    expected_schema = tool.output_schema.model_dump(mode="json", exclude_none=True)  # type: ignore[union-attr]
    assert cfg.generation.guided_decoding.json == expected_schema


def test_build_call_conv_has_system_and_user_messages():
    env, _ = _attached_env(['{"a": "x"}'])
    tool = env._lookup_tool("answer")
    conv = env._build_call_conv(tool, {"q": "hi"})
    assert len(conv.messages) == 2
    assert conv.messages[0].role == Role.SYSTEM
    assert conv.messages[1].role == Role.USER
    assert "answer" in str(conv.messages[0].content)
    user_payload = str(conv.messages[1].content)
    assert '"tool"' in user_payload and '"answer"' in user_payload
