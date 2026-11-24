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
from pydantic import BaseModel, Field

import oumi_verl_ext
from oumi_verl_ext.tool_episode import (
    RENDERED_TOOLS_KEY,
    TOOL_AGENT_LOOP_NAME,
    TRANSCRIPT_KEY,
    CallCounts,
    choose_episode_tools,
    judge_transcript,
    parse_row_tools,
    verl_parser_schema,
)

_EXT_DIR = Path(oumi_verl_ext.__path__[0])


# Mirrors verl v0.9.1 verl/tools/schemas.py:21-56, which CI can't import.
class _VerlPropertySchema(BaseModel):
    type: str | list[str]
    description: str | None = None
    enum: list[Any] | None = None


class _VerlParametersSchema(BaseModel):
    type: str
    properties: dict[str, _VerlPropertySchema]
    required: list[str] = Field(default_factory=list)


class _VerlFunctionSchema(BaseModel):
    name: str
    description: str
    parameters: _VerlParametersSchema
    strict: bool = False


class _VerlToolSchema(BaseModel):
    type: str
    function: _VerlFunctionSchema


def _tool(name: str, **function: Any) -> dict[str, Any]:
    return {"type": "function", "function": {"name": name, **function}}


GET_STOCK = _tool(
    "get_stock",
    description="Units in stock.",
    parameters={"type": "object", "properties": {"item": {"type": "string"}}},
)


@pytest.mark.parametrize(
    "run_tools", ["nothing", "get_stock", "get_weather", "get_stock_ends_episode"]
)
def test_episode_tools_follow_evaluation_s_toolset_rule(run_tools):
    originals = [
        _tool(
            "get_weather",
            description="Weather for a city.",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "default": "Athens"},
                    "days": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["city"],
            },
        ),
        _tool(
            "no_type",
            parameters={
                "properties": {"q": {"anyOf": [{"type": "string"}, {"type": "null"}]}}
            },
        ),
        _tool(
            "list_type",
            description="",
            parameters={"type": ["object"], "properties": {}, "required": None},
        ),
        _tool("no_parameters", description="Takes nothing."),
    ]
    row_tools = parse_row_tools(json.dumps(originals))
    attached, ends_episode = {
        "nothing": ({}, set()),
        "get_stock": ({"get_stock": GET_STOCK}, set()),
        # verl's dump of a run tool drops `default` and `items`.
        "get_weather": (
            {"get_weather": _tool("get_weather", description="Weather.")},
            set(),
        ),
        "get_stock_ends_episode": ({"get_stock": GET_STOCK}, {"get_stock"}),
    }[run_tools]

    episode = choose_episode_tools(row_tools, attached, frozenset(ends_episode))

    expected = {
        "nothing": (originals, set(), {tool["function"]["name"] for tool in originals}),
        "get_stock": ([GET_STOCK], {"get_stock"}, set()),
        "get_weather": ([originals[0]], {"get_weather"}, set()),
        "get_stock_ends_episode": ([GET_STOCK], set(), {"get_stock"}),
    }[run_tools]
    assert (episode.schemas, episode.executable, episode.ending) == expected
    for tool in episode.schemas:
        _VerlToolSchema.model_validate(verl_parser_schema(tool))
    assert row_tools == originals


@pytest.mark.parametrize(
    "tools_json", ["not-json", json.dumps({"a": 1}), json.dumps([{"type": "x"}])]
)
def test_malformed_row_tools_json_fails_with_a_named_error(tools_json):
    with pytest.raises(ValueError, match="tools_json") as error:
        parse_row_tools(tools_json)
    assert tools_json not in str(error.value)


@pytest.mark.parametrize(
    "parameters", ["SECRET", {"properties": "SECRET"}, {"properties": {"q": "SECRET"}}]
)
def test_a_row_tool_verl_cannot_parse_fails_with_a_named_error(parameters):
    tool = _tool("bad", parameters=parameters)

    with pytest.raises(ValueError, match="'bad'") as error:
        verl_parser_schema(tool)
    assert "SECRET" not in str(error.value)


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


def test_transcript_keeps_a_budget_cut_final_turn():
    messages = [{"role": "user", "content": "Hi"}, {"role": "tool", "content": "3"}]

    transcript = judge_transcript(messages, "The answer is", [])

    assert transcript[-1] == {"role": "assistant", "content": "The answer is"}


def test_call_counters_count_calls_drops_and_malformed_calls():
    counts = CallCounts()
    assert not counts.add_turn(
        attempted=4, call_names=["a", "b", "c"], max_parallel_calls=1
    )
    assert not counts.add_turn(attempted=1, call_names=["a"], max_parallel_calls=1)
    assert (counts.calls, counts.dropped, counts.malformed) == (4, 2, 1)
    assert not counts.add_turn(
        attempted=2, call_names=[], max_parallel_calls=1, cut=True
    )
    # qwen3_coder can parse two calls from one marker.
    assert not counts.add_turn(attempted=1, call_names=["a", "b"], max_parallel_calls=2)
    assert (counts.calls, counts.dropped, counts.malformed) == (6, 4, 1)

    def turn(names, **kwargs):
        counts = CallCounts()
        ends = counts.add_turn(
            attempted=len(names),
            call_names=names,
            max_parallel_calls=2,
            ending=frozenset({"end"}),
            **kwargs,
        )
        return ends, counts.dropped

    assert turn(["lookup", "end", "lookup"]) == (True, 2)
    assert turn(["lookup", "lookup", "end"]) == (False, 1)
    assert turn(["lookup", "lookup"], capped=True) == (True, 2)
    assert turn(["end", "lookup"], capped=True) == (True, 1)
    assert turn([], capped=True) == (True, 0)


def test_extra_field_names_are_the_ones_the_platform_judge_reads():
    assert (TRANSCRIPT_KEY, RENDERED_TOOLS_KEY) == (
        "oumi_tool_transcript_json",
        "oumi_tool_schemas_json",
    )


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


class _ToolSchema:
    def __init__(self, **schema: Any):
        self.schema = schema

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        return self.schema


class _BaseTool:
    def __init__(self, config: dict, tool_schema: _ToolSchema):
        self.config = config
        self.tool_schema = tool_schema


class _Tokenizer:
    all_special_tokens = ["<|im_end|>"]

    def decode(self, ids: list[str]) -> str:
        return "".join(ids)


class _ScriptedToolAgentLoop:
    """Stands in for verl 0.9.1's ToolAgentLoop, with its turn-cap checks.

    Each scripted turn is (token strings, call names); a turn that runs past
    response_length ends before it is parsed, as verl's does.
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
        self.max_parallel_calls = 1
        self.response_length = response_length
        self.tokenizer = _Tokenizer()
        self.tool_parser = SimpleNamespace(tool_call_start_token="<tool_call>")
        self.tools = {"lookup": _BaseTool({}, _ToolSchema(**_tool("lookup")))}
        self.run_kwargs: dict[str, Any] = {}
        self.ran: list[str] = []

    async def run(self, sampling_params, **kwargs):
        self.run_kwargs = kwargs
        agent_data = SimpleNamespace(
            messages=[{"role": "user", "content": "Stock of pens?"}],
            response_ids=[],
            response_mask=[],
            tool_calls=[],
            user_turns=0,
            assistant_turns=0,
            extra_fields={},
            _active_tools=self.tools,
        )
        state = await self._handle_pending_state(agent_data, sampling_params)
        while state != _AgentState.TERMINATED:
            if state == _AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            else:
                state = await self._handle_processing_tools_state(agent_data)
        return SimpleNamespace(extra_fields=agent_data.extra_fields)

    async def _handle_pending_state(self, agent_data, sampling_params):
        return _AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data, sampling_params, ignore_termination=False
    ):
        ids, calls = self.turns.pop(0)
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
        agent_data.tool_calls = [SimpleNamespace(name=name) for name in calls]
        agent_data.messages.append({"role": "assistant", "content": "".join(ids)})
        if agent_data.tool_calls:
            return _AgentState.PROCESSING_TOOLS
        return _AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data):
        for call in agent_data.tool_calls[: self.max_parallel_calls]:
            self.ran.append(call.name)
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
        "verl.tools": module("verl.tools"),
        "verl.tools.base_tool": module("verl.tools.base_tool", BaseTool=_BaseTool),
        "verl.tools.schemas": module(
            "verl.tools.schemas", OpenAIFunctionToolSchema=_ToolSchema
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


def _lookup_turn(text: str = "") -> tuple[list[str], list[str]]:
    return [text, "<tool_call>", "lookup", "</tool_call>"], ["lookup"]


def _run(loop, **kwargs) -> dict[str, Any]:
    output = asyncio.run(loop.run({}, raw_prompt=[], **kwargs))
    return output.extra_fields


@pytest.mark.parametrize(
    ("caps", "ran", "last_turn", "dropped"),
    [
        ({"max_assistant_turns": 1}, ["lookup"], "1", 0),
        ({"max_user_turns": 1}, ["lookup"], "2", 1),
        ({}, ["lookup", "lookup"], "answer", 0),
    ],
)
def test_the_loop_enforces_both_turn_caps_itself(
    stub_tool_agent_loop, caps, ran, last_turn, dropped
):
    turns = [_lookup_turn("1"), _lookup_turn("2"), (["answer"], [])]
    loop = stub_tool_agent_loop.OumiToolAgentLoop(turns, **caps)

    assert (loop.max_assistant_turns, loop.max_user_turns) == (None, None)
    fields = _run(loop)

    assert loop.ran == ran
    last = json.loads(fields[TRANSCRIPT_KEY])[-1]
    assert last["role"] == "assistant" and last["content"].startswith(last_turn)
    assert fields["oumi_tool_calls_dropped"] == dropped


def test_a_budget_cut_turn_keeps_only_its_tokens_inside_the_response_budget(
    stub_tool_agent_loop,
):
    # 4 tokens and a tool token leave 5 of response_length 10 for the cut turn.
    cut_turn = (["a", "b", "<tool_call>", "c", "d", "e"], ["lookup"])
    loop = stub_tool_agent_loop.OumiToolAgentLoop(
        [_lookup_turn(), cut_turn], response_length=10
    )

    fields = _run(loop)

    assert json.loads(fields[TRANSCRIPT_KEY])[-1] == {
        "role": "assistant",
        "content": "ab<tool_call>cd",
    }
    assert (fields["oumi_tool_calls"], fields["oumi_tool_calls_dropped"]) == (1, 1)
    assert fields["oumi_tool_calls_malformed"] == 0


@pytest.mark.parametrize("tools_kwargs", [None, {"lookup": {"create_kwargs": {}}}])
def test_a_null_tools_kwargs_reaches_verl_as_an_empty_map(
    stub_tool_agent_loop, tools_kwargs
):
    loop = stub_tool_agent_loop.OumiToolAgentLoop([(["answer"], [])])

    _run(loop, tools_kwargs=tools_kwargs)

    assert loop.run_kwargs["tools_kwargs"] == (tools_kwargs or {})
