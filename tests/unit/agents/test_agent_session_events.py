# Copyright 2025 - Oumi
# Licensed under the Apache License, Version 2.0
"""Contract tests for the ``on_event`` hook on :class:`AgentSession`.

The hook is the seam between the session loop and any TUI / transcript
writer. These tests pin the event vocabulary and the relative ordering
the renderer depends on. We use the same ``ScriptedEngine`` / ``FakeEnv``
helpers as ``test_agent_session.py`` to keep the surface consistent.
"""

from __future__ import annotations

import copy
from typing import Any

import oumi.environments  # noqa: F401  populates env registry
from oumi.agents.agent_session import AgentSession
from oumi.agents.tool_router import ToolRouter
from oumi.core.configs.environment_config import EnvironmentConfig
from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.configs.params.tool_params import ToolParams
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.core.types.tool_call import FunctionCall, ToolCall, ToolResult
from oumi.environments.base_environment import BaseEnvironment


def _tool(tool_id: str) -> ToolParams:
    return ToolParams(
        id=tool_id,
        name=tool_id,
        description=f"Tool {tool_id}.",
        parameters={"type": "object", "properties": {}},
    )


def _env_params(env_id: str, tools: list[ToolParams]) -> EnvironmentParams:
    return EnvironmentParams(
        id=env_id,
        name=env_id,
        description=f"Env {env_id}.",
        env_type="synthetic",
        tools=tools,
        env_kwargs={"system_prompt": "."},
    )


def _make_tool_call(call_id: str, name: str, arguments: str) -> ToolCall:
    return ToolCall(id=call_id, function=FunctionCall(name=name, arguments=arguments))


class FakeEnv(BaseEnvironment):
    def __init__(self) -> None:
        self.responses: dict[str, ToolResult] = {}
        self.raise_on: dict[str, Exception] = {}

    def step(self, tool_id: str, arguments: dict[str, Any]) -> ToolResult:
        if tool_id in self.raise_on:
            raise self.raise_on[tool_id]
        return self.responses.get(tool_id, ToolResult(output={"status": "ok"}))


class ScriptedEngine:
    def __init__(self, scripted: list[Message]) -> None:
        self._scripted = list(scripted)
        self.calls = 0

    def infer(self, conversations: list[Conversation]) -> list[Conversation]:
        msg = self._scripted[self.calls]
        self.calls += 1
        out = []
        for conv in conversations:
            new = copy.deepcopy(conv)
            new.messages.append(msg)
            out.append(new)
        return out


def _build_session(
    engine: ScriptedEngine,
    env: BaseEnvironment,
    tools: list[ToolParams],
) -> AgentSession:
    env_config = EnvironmentConfig(environments=[_env_params("e", tools)])
    return AgentSession(
        engine=engine,  # type: ignore[arg-type]
        envs={"e": env},
        router=ToolRouter.from_environment_config(env_config),
    )


class _Recorder:
    """Capture ``(kind, data)`` tuples in order."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def __call__(self, kind: str, data: dict) -> None:
        self.events.append((kind, dict(data)))


def test_no_tool_call_emits_user_inference_assistant_turnend():
    """Plain text reply: user → inference_start → inference_end →
    assistant_text → turn_end (no tool events)."""
    engine = ScriptedEngine([Message(role=Role.ASSISTANT, content="hi there")])
    session = _build_session(engine, FakeEnv(), [_tool("noop")])
    rec = _Recorder()
    session.set_event_handler(rec)

    session.send("hello")

    kinds = [k for k, _ in rec.events]
    assert kinds == [
        "user",
        "inference_start",
        "inference_end",
        "assistant_text",
        "turn_end",
    ]
    assert rec.events[0][1]["text"] == "hello"
    assert rec.events[3][1]["text"] == "hi there"
    assert rec.events[4][1]["tool_calls"] == 0


def test_tool_call_emits_tool_call_then_tool_result_in_order():
    engine = ScriptedEngine(
        [
            Message(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[_make_tool_call("c1", "ping", "{}")],
            ),
            Message(role=Role.ASSISTANT, content="done"),
        ]
    )
    env = FakeEnv()
    env.responses["ping"] = ToolResult(output={"status": "ok", "echo": "pong"})
    session = _build_session(engine, env, [_tool("ping")])
    rec = _Recorder()
    session.set_event_handler(rec)

    session.send("ping")

    kinds = [k for k, _ in rec.events]
    # Loop is: user, inference x, tool_call, tool_result, inference x, text, end
    assert kinds == [
        "user",
        "inference_start",
        "inference_end",
        "tool_call",
        "tool_result",
        "inference_start",
        "inference_end",
        "assistant_text",
        "turn_end",
    ]
    tool_call = next(d for k, d in rec.events if k == "tool_call")
    assert tool_call["name"] == "ping"
    assert tool_call["arguments"] == "{}"
    tool_result = next(d for k, d in rec.events if k == "tool_result")
    assert tool_result["name"] == "ping"
    assert tool_result["payload"]["echo"] == "pong"
    assert tool_result["is_error"] is False
    assert rec.events[-1][1]["tool_calls"] == 1


def test_env_exception_marks_tool_result_as_error():
    """``status == "error"`` payload must surface ``is_error=True`` so the
    TUI renders the panel red."""
    engine = ScriptedEngine(
        [
            Message(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[_make_tool_call("c1", "boom", "{}")],
            ),
            Message(role=Role.ASSISTANT, content="recovered"),
        ]
    )
    env = FakeEnv()
    env.raise_on["boom"] = RuntimeError("kaboom")
    session = _build_session(engine, env, [_tool("boom")])
    rec = _Recorder()
    session.set_event_handler(rec)

    session.send("crash please")

    tool_result = next(d for k, d in rec.events if k == "tool_result")
    assert tool_result["is_error"] is True
    assert tool_result["payload"]["error"] == "RuntimeError"


def test_unknown_tool_short_circuits_with_error_event():
    """The router-level pre-gate also flows through the event stream so
    the TUI shows the model's hallucinated call and the rejection."""
    engine = ScriptedEngine(
        [
            Message(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[_make_tool_call("c1", "ghost", "{}")],
            ),
            Message(role=Role.ASSISTANT, content="ok"),
        ]
    )
    session = _build_session(engine, FakeEnv(), [_tool("real")])
    rec = _Recorder()
    session.set_event_handler(rec)

    session.send("call ghost")

    # tool_call still fires (the model emitted it); tool_result carries error.
    tool_result = next(d for k, d in rec.events if k == "tool_result")
    assert tool_result["is_error"] is True
    assert tool_result["payload"]["error"] == "unknown_tool"


def test_handler_can_be_detached_with_none():
    engine = ScriptedEngine([Message(role=Role.ASSISTANT, content="hi")])
    session = _build_session(engine, FakeEnv(), [_tool("noop")])
    rec = _Recorder()
    session.set_event_handler(rec)
    session.set_event_handler(None)

    session.send("hello")

    assert rec.events == []  # detached before send → nothing recorded
