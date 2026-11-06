import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

_MODULE_PATH = (
    Path(__file__).parents[4]
    / "src"
    / "oumi"
    / "core"
    / "trainers"
    / "verl_agent_loops.py"
)


class _RecordingSingleTurnAgentLoop:
    """Stands in for verl's SingleTurnAgentLoop.

    Mirrors the one behavior under test: ``run`` applies the chat template with
    the vision inputs and no tools, which is why the schemas have to be injected
    on the way through.
    """

    def __init__(self, *args, **kwargs):
        self.template_calls: list[dict] = []

    async def run(self, sampling_params, **kwargs):
        messages = list(kwargs["raw_prompt"])
        await self.apply_chat_template(messages, images=None, videos=None)
        return "rollout-output"

    async def apply_chat_template(self, messages, tools=None, **kwargs):
        self.template_calls.append({"messages": messages, "tools": tools})
        return [1, 2, 3]


@pytest.fixture
def agent_loops():
    """Loads the agent loop module against a stub verl, under a throwaway name."""
    agent_loop_mod: Any = types.ModuleType("verl.experimental.agent_loop.agent_loop")
    agent_loop_mod.AgentLoopOutput = object
    single_turn_mod: Any = types.ModuleType(
        "verl.experimental.agent_loop.single_turn_agent_loop"
    )
    single_turn_mod.SingleTurnAgentLoop = _RecordingSingleTurnAgentLoop

    stubs = {
        "verl": types.ModuleType("verl"),
        "verl.experimental": types.ModuleType("verl.experimental"),
        "verl.experimental.agent_loop": types.ModuleType(
            "verl.experimental.agent_loop"
        ),
        "verl.experimental.agent_loop.agent_loop": agent_loop_mod,
        "verl.experimental.agent_loop.single_turn_agent_loop": single_turn_mod,
    }
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(
            "_oumi_verl_agent_loops_under_test", _MODULE_PATH
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}
_PROMPT = [{"role": "user", "content": "Weather in Athens?"}]


def _run(loop, **kwargs):
    return asyncio.run(loop.run({}, raw_prompt=_PROMPT, **kwargs))


def test_row_tools_reach_the_chat_template(agent_loops):
    loop = agent_loops.ToolAwareSingleTurnAgentLoop()

    _run(loop, tools_json=json.dumps([_WEATHER_TOOL]))

    assert loop.template_calls[0]["tools"] == [_WEATHER_TOOL]


def test_a_row_without_tools_renders_as_before(agent_loops):
    loop = agent_loops.ToolAwareSingleTurnAgentLoop()

    _run(loop)

    assert loop.template_calls[0]["tools"] is None


def test_an_empty_tool_list_renders_as_no_tools(agent_loops):
    loop = agent_loops.ToolAwareSingleTurnAgentLoop()

    _run(loop, tools_json=json.dumps([]))

    assert loop.template_calls[0]["tools"] is None


def test_an_explicit_tools_argument_wins(agent_loops):
    """A caller passing tools is not overridden by the row's."""
    loop = agent_loops.ToolAwareSingleTurnAgentLoop()
    loop._row_tools = [_WEATHER_TOOL]

    asyncio.run(loop.apply_chat_template(_PROMPT, tools=[]))

    assert loop.template_calls[0]["tools"] == []


@pytest.mark.parametrize(
    "tools_json",
    ["not-json", json.dumps({"not": "a list"}), json.dumps(["not an object"])],
)
def test_an_unusable_tools_column_stops_the_rollout(agent_loops, tools_json):
    """Rendering without the tools would silently train on the wrong prompt."""
    loop = agent_loops.ToolAwareSingleTurnAgentLoop()

    with pytest.raises(ValueError, match="tools_json"):
        _run(loop, tools_json=tools_json)


def test_tools_do_not_leak_between_rollouts(agent_loops):
    """verl builds one loop per sample; a reused instance must not carry over."""
    loop = agent_loops.ToolAwareSingleTurnAgentLoop()

    _run(loop, tools_json=json.dumps([_WEATHER_TOOL]))
    _run(loop)

    assert loop.template_calls[0]["tools"] == [_WEATHER_TOOL]
    assert loop.template_calls[1]["tools"] is None


def test_the_column_name_matches_the_one_the_exporter_writes(agent_loops):
    """A drift here renders every prompt without its tools, silently."""
    from oumi.core.trainers import verl_grpo_trainer

    assert agent_loops._TOOLS_JSON_KEY == verl_grpo_trainer._TOOLS_JSON_KEY


async def _tokenizes_without_the_template(*args, **kwargs):
    """Stands in for verl 0.9+, whose run() never calls apply_chat_template."""
    return [1, 2, 3]


def test_a_verl_that_tokenizes_elsewhere_fails_loudly(agent_loops):
    """A renamed template hook must not silently drop the row's tools."""
    loop = agent_loops.ToolAwareSingleTurnAgentLoop()
    loop.apply_chat_template = _tokenizes_without_the_template

    with pytest.raises(RuntimeError, match="ct_build_initial_tokens"):
        _run(loop, tools_json=json.dumps([_WEATHER_TOOL]))


def test_a_row_without_tools_does_not_trip_the_guard(agent_loops):
    loop = agent_loops.ToolAwareSingleTurnAgentLoop()
    loop.apply_chat_template = _tokenizes_without_the_template

    _run(loop)  # nothing to lose, so nothing to complain about
