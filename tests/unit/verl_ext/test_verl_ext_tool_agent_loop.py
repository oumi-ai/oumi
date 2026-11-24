import ast
import asyncio
import enum
import importlib
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

import oumi_verl_ext
from oumi_verl_ext.tool_episode import TOOL_AGENT_LOOP_NAME, judge_transcript

_EXT_DIR = Path(oumi_verl_ext.__path__[0])


def _tool(name: str) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name}}


@pytest.mark.parametrize("trailing_tool_messages", [0, 1, 2])
def test_transcript_ends_at_the_last_assistant_message(trailing_tool_messages):
    messages = [
        {"role": "user", "content": "Stock of pens?"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "3"},
        {"role": "assistant", "content": "Checking.<|im_end|>"},
        *[{"role": "tool", "content": "4"}] * trailing_tool_messages,
    ]

    transcript = judge_transcript(messages, None, ["<|im_end|>"])

    assert transcript == [*messages[:3], {"role": "assistant", "content": "Checking."}]


@pytest.mark.parametrize("module", ["tool_agent_loop.py", "tool_episode.py"])
def test_loop_modules_import_only_verl_and_the_stdlib(module):
    imported = set()
    for node in ast.walk(ast.parse((_EXT_DIR / module).read_text())):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])

    assert imported <= set(sys.stdlib_module_names) | {"verl", "oumi_verl_ext"}


class _AgentState(enum.Enum):
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"


class _Tokenizer:
    all_special_tokens = ["<|im_end|>"]

    def decode(self, ids: list[str]) -> str:
        return "".join(ids)


class _ScriptedToolAgentLoop:
    """Stands in for verl 0.9.1's ToolAgentLoop, with its termination checks.

    Each scripted turn is a list of token strings, and calls a tool when it holds
    "<tool_call>". A turn that reaches response_length or a turn cap ends before it
    is parsed, as verl's does.
    """

    def __init__(
        self,
        turns,
        *,
        max_assistant_turns=None,
        max_user_turns=None,
        response_length=100,
    ):
        self.turns = list(turns)
        self.max_assistant_turns = max_assistant_turns
        self.max_user_turns = max_user_turns
        self.response_length = response_length
        self.tokenizer = _Tokenizer()
        self.tool_schemas = [_tool("lookup")]
        self.run_kwargs: dict[str, Any] = {}

    async def run(self, sampling_params, **kwargs):
        self.run_kwargs = kwargs
        agent_data = SimpleNamespace(
            messages=[{"role": "user", "content": "Stock of pens?"}],
            response_ids=[],
            response_mask=[],
            user_turns=0,
            assistant_turns=0,
        )
        state = _AgentState.GENERATING
        while state != _AgentState.TERMINATED:
            if state == _AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            else:
                state = await self._handle_processing_tools_state(agent_data)
        return SimpleNamespace(extra_fields={})

    async def _handle_generating_state(
        self, agent_data, sampling_params, ignore_termination=False
    ):
        ids = self.turns.pop(0)
        agent_data.assistant_turns += 1
        agent_data.response_ids = ids
        agent_data.response_mask = agent_data.response_mask + [1] * len(ids)
        if len(agent_data.response_mask) >= self.response_length:
            return _AgentState.TERMINATED
        if self.max_assistant_turns and (
            agent_data.assistant_turns >= self.max_assistant_turns
        ):
            return _AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return _AgentState.TERMINATED
        agent_data.messages.append({"role": "assistant", "content": "".join(ids)})
        if "<tool_call>" in ids:
            return _AgentState.PROCESSING_TOOLS
        return _AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data):
        agent_data.messages.append({"role": "tool", "content": "3"})
        agent_data.response_mask = agent_data.response_mask + [0]
        agent_data.user_turns += 1
        return _AgentState.GENERATING


@pytest.fixture
def stub_tool_agent_loop():
    """Loads the loop module against stub verl modules, under a throwaway name."""

    def module(name: str, **attrs: Any) -> types.ModuleType:
        stub = types.ModuleType(name)
        stub.__dict__.update(attrs)
        return stub

    agent_loop = "verl.experimental.agent_loop"
    stubs = {
        "verl": module("verl"),
        "verl.experimental": module("verl.experimental"),
        agent_loop: module(agent_loop),
        f"{agent_loop}.agent_loop": module(
            f"{agent_loop}.agent_loop", AgentLoopOutput=object
        ),
        f"{agent_loop}.tool_agent_loop": module(
            f"{agent_loop}.tool_agent_loop",
            AgentData=object,
            AgentState=_AgentState,
            ToolAgentLoop=_ScriptedToolAgentLoop,
        ),
    }
    saved = {
        name: sys.modules.get(name)
        for name in [*stubs, "oumi_verl_ext.tool_agent_loop"]
    }
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(
            "_oumi_tool_agent_loop_under_test", _EXT_DIR / "tool_agent_loop.py"
        )
        assert spec is not None and spec.loader is not None
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        yield loaded
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def test_agent_loop_yaml_registers_the_loop_class(stub_tool_agent_loop):
    (entry,) = yaml.safe_load((_EXT_DIR / "agent_loops.yaml").read_text())
    module_name, _, class_name = entry["_target_"].rpartition(".")

    loop_class = getattr(importlib.import_module(module_name), class_name)

    assert entry["name"] == TOOL_AGENT_LOOP_NAME
    assert loop_class.__qualname__ == "OumiToolAgentLoop"
    assert issubclass(loop_class, _ScriptedToolAgentLoop)


def _lookup_turn(text: str = "") -> list[str]:
    return [text, "<tool_call>", "lookup", "</tool_call>"]


def _run(loop, **kwargs) -> dict[str, Any]:
    output = asyncio.run(loop.run({}, raw_prompt=[], **kwargs))
    return output.extra_fields


def _last_message(fields: dict[str, Any]) -> dict[str, Any]:
    return json.loads(fields["oumi_tool_transcript_json"])[-1]


def test_extra_fields_carry_the_transcript_and_tool_schemas_as_json(
    stub_tool_agent_loop,
):
    loop = stub_tool_agent_loop.OumiToolAgentLoop(
        [_lookup_turn(), ["Answer: 3", "<|im_end|>"]]
    )

    fields = _run(loop)

    # The platform's judge reads these names.
    assert json.loads(fields["oumi_tool_transcript_json"]) == [
        {"role": "user", "content": "Stock of pens?"},
        {"role": "assistant", "content": "<tool_call>lookup</tool_call>"},
        {"role": "tool", "content": "3"},
        {"role": "assistant", "content": "Answer: 3"},
    ]
    assert json.loads(fields["oumi_tool_schemas_json"]) == [_tool("lookup")]


def test_a_budget_cut_turn_keeps_only_its_tokens_inside_the_response_budget(
    stub_tool_agent_loop,
):
    # 4 tokens and a tool token leave 5 of response_length 10 for the cut turn.
    cut_turn = ["a", "b", "<tool_call>", "c", "d", "e"]
    loop = stub_tool_agent_loop.OumiToolAgentLoop(
        [_lookup_turn(), cut_turn], response_length=10
    )

    fields = _run(loop)

    assert _last_message(fields) == {"role": "assistant", "content": "ab<tool_call>cd"}


@pytest.mark.parametrize("cap", [{"max_assistant_turns": 2}, {"max_user_turns": 1}])
def test_a_turn_verl_s_turn_cap_stopped_is_appended_as_text(stub_tool_agent_loop, cap):
    loop = stub_tool_agent_loop.OumiToolAgentLoop(
        [_lookup_turn("1"), _lookup_turn("2")], **cap
    )

    fields = _run(loop)

    assert _last_message(fields) == {
        "role": "assistant",
        "content": "2<tool_call>lookup</tool_call>",
    }


@pytest.mark.parametrize("tools_kwargs", [None, {"lookup": {"create_kwargs": {}}}])
def test_a_null_tools_kwargs_reaches_verl_as_an_empty_map(
    stub_tool_agent_loop, tools_kwargs
):
    loop = stub_tool_agent_loop.OumiToolAgentLoop([["answer"]])

    _run(loop, tools_kwargs=tools_kwargs)

    assert loop.run_kwargs["tools_kwargs"] == (tools_kwargs or {})
