# Copyright 2025 - Oumi
# Licensed under the Apache License, Version 2.0
"""Contract tests for the agent chat loop."""

from __future__ import annotations

import copy
from typing import Any

import pytest

import oumi.environments  # noqa: F401  populates env registry
from oumi.agents.agent_session import AgentSession
from oumi.agents.tool_router import ToolRouter
from oumi.core.configs.environment_config import EnvironmentConfig
from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.configs.params.tool_params import ToolParams
from oumi.core.types.conversation import Conversation, Message, Role
from oumi.core.types.tool_call import FunctionCall, ToolCall, ToolResult
from oumi.environments.base_environment import BaseEnvironment


def _tool(tool_id: str, *, parameters: dict | None = None) -> ToolParams:
    return ToolParams(
        id=tool_id,
        name=tool_id,
        description=f"Tool {tool_id}.",
        parameters=parameters or {"type": "object", "properties": {}},
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
    """Records every step call. Returns whatever the test plants in ``responses``."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.responses: dict[str, ToolResult] = {}
        self.raise_on: dict[str, Exception] = {}
        self.closed = False

    def step(self, tool_id: str, arguments: dict[str, Any]) -> ToolResult:
        self.calls.append((tool_id, arguments))
        if tool_id in self.raise_on:
            raise self.raise_on[tool_id]
        return self.responses.get(tool_id, ToolResult(output={"status": "ok"}))

    def close(self) -> None:
        self.closed = True


class ScriptedEngine:
    """Replays a fixed list of assistant messages, one per ``infer`` call."""

    def __init__(
        self,
        scripted_messages: list[Message],
        usage_per_call: list[dict[str, int] | None] | None = None,
    ) -> None:
        self._scripted = list(scripted_messages)
        self._usage_per_call = usage_per_call or [None] * len(scripted_messages)
        self.calls = 0

    def infer(self, conversations: list[Conversation]) -> list[Conversation]:
        if self.calls >= len(self._scripted):
            raise AssertionError(
                f"ScriptedEngine ran out of scripted messages at call "
                f"{self.calls + 1}; conversation history was:\n"
                f"{conversations[0].messages}"
            )
        msg = self._scripted[self.calls]
        usage = self._usage_per_call[self.calls]
        self.calls += 1
        out = []
        for conv in conversations:
            new = copy.deepcopy(conv)
            new.messages.append(msg)
            if usage is not None:
                new.metadata["usage"] = usage
            out.append(new)
        return out


def _build_session(
    engine: ScriptedEngine,
    envs: dict[str, BaseEnvironment],
    tools_per_env: dict[str, list[ToolParams]],
    *,
    system_prompt: str = "",
) -> AgentSession:
    env_config = EnvironmentConfig(
        environments=[
            _env_params(env_id, tools) for env_id, tools in tools_per_env.items()
        ]
    )
    router = ToolRouter.from_environment_config(env_config)
    return AgentSession(
        engine=engine,  # type: ignore[arg-type]  fake quacks the same shape
        envs=envs,
        router=router,
        system_prompt=system_prompt,
    )


def test_assistant_with_no_tool_calls_returns_text_immediately():
    engine = ScriptedEngine([Message(role=Role.ASSISTANT, content="Hello!")])
    env = FakeEnv()
    session = _build_session(engine, {"e": env}, {"e": [_tool("noop")]})

    reply = session.send("hi")

    assert reply == "Hello!"
    assert engine.calls == 1
    assert env.calls == []


def test_single_tool_call_routes_to_env_and_loops_back():
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
    session = _build_session(engine, {"e": env}, {"e": [_tool("ping")]})

    reply = session.send("ping the env")

    assert reply == "done"
    assert env.calls == [("ping", {})]
    # Conversation history: system? user, assistant(toolcall), tool, assistant(final)
    msgs = session.conversation.messages
    assert msgs[0].role == Role.USER
    assert msgs[1].role == Role.ASSISTANT and msgs[1].tool_calls
    assert msgs[2].role == Role.TOOL
    assert msgs[3].role == Role.ASSISTANT and msgs[3].content == "done"


def test_unknown_tool_name_does_not_reach_env():
    """SLM hallucinates a tool name. The harness must reject it in the
    pre-gate and surface a structured error to the model — never call
    ``env.step`` with an unknown id."""
    engine = ScriptedEngine(
        [
            Message(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[_make_tool_call("c1", "hallucinated", "{}")],
            ),
            Message(role=Role.ASSISTANT, content="ok I will not retry"),
        ]
    )
    env = FakeEnv()
    session = _build_session(engine, {"e": env}, {"e": [_tool("real_tool")]})

    reply = session.send("call something that doesn't exist")

    assert env.calls == []  # pre-gate kept the call out of the env
    assert reply == "ok I will not retry"
    tool_msg = next(m for m in session.conversation.messages if m.role == Role.TOOL)
    assert "unknown_tool" in str(tool_msg.content)


def test_invalid_arguments_become_tool_error_message():
    """Schema mismatch must not crash the loop — wrap as tool error."""
    schema = {
        "type": "object",
        "properties": {"x": {"type": "integer"}},
        "required": ["x"],
    }
    engine = ScriptedEngine(
        [
            Message(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[_make_tool_call("c1", "echo", '{"x": "not-int"}')],
            ),
            Message(role=Role.ASSISTANT, content="recovered"),
        ]
    )
    env = FakeEnv()
    session = _build_session(
        engine, {"e": env}, {"e": [_tool("echo", parameters=schema)]}
    )

    session.send("bad call")

    assert env.calls == []
    tool_msg = next(m for m in session.conversation.messages if m.role == Role.TOOL)
    assert "invalid_arguments" in str(tool_msg.content)


def test_env_exception_becomes_tool_error_message():
    """If env.step raises, the loop must continue with a structured error
    (not crash the chat session)."""
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
    session = _build_session(engine, {"e": env}, {"e": [_tool("boom")]})

    reply = session.send("trigger an error")

    assert reply == "recovered"
    tool_msg = next(m for m in session.conversation.messages if m.role == Role.TOOL)
    assert "RuntimeError" in str(tool_msg.content)
    assert "kaboom" in str(tool_msg.content)


def test_duplicate_call_in_same_turn_is_short_circuited():
    """SLM loop-breaker: same (name, raw_args) twice in one user turn
    returns a duplicate_call error instead of re-executing."""
    engine = ScriptedEngine(
        [
            Message(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[_make_tool_call("c1", "ping", "{}")],
            ),
            Message(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[_make_tool_call("c2", "ping", "{}")],
            ),
            Message(role=Role.ASSISTANT, content="ok stopping"),
        ]
    )
    env = FakeEnv()
    session = _build_session(engine, {"e": env}, {"e": [_tool("ping")]})

    session.send("call twice")

    # Env was called exactly once — second identical call short-circuited.
    assert env.calls == [("ping", {})]
    tool_msgs = [m for m in session.conversation.messages if m.role == Role.TOOL]
    assert len(tool_msgs) == 2
    assert "duplicate_call" in str(tool_msgs[1].content)


def test_dedup_window_resets_on_next_user_turn():
    """The dedup gate is per-user-turn. If the user sends a new message,
    the same tool+args is allowed again."""
    engine = ScriptedEngine(
        [
            Message(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[_make_tool_call("c1", "ping", "{}")],
            ),
            Message(role=Role.ASSISTANT, content="first done"),
            Message(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[_make_tool_call("c2", "ping", "{}")],
            ),
            Message(role=Role.ASSISTANT, content="second done"),
        ]
    )
    env = FakeEnv()
    session = _build_session(engine, {"e": env}, {"e": [_tool("ping")]})

    session.send("first")
    session.send("second")

    # Both turns executed; the dedup window did not leak across turns.
    assert env.calls == [("ping", {}), ("ping", {})]


def test_close_tears_down_every_environment():
    engine = ScriptedEngine([Message(role=Role.ASSISTANT, content="hi")])
    env_a = FakeEnv()
    env_b = FakeEnv()
    session = _build_session(
        engine,
        {"a": env_a, "b": env_b},
        {"a": [_tool("a_tool")], "b": [_tool("b_tool")]},
    )

    with session:
        session.send("hello")

    assert env_a.closed and env_b.closed


def test_session_routes_calls_to_correct_env_when_multiple():
    """Multi-env routing: tool 'a_tool' must hit env_a, not env_b."""
    engine = ScriptedEngine(
        [
            Message(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[_make_tool_call("c1", "a_tool", "{}")],
            ),
            Message(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[_make_tool_call("c2", "b_tool", "{}")],
            ),
            Message(role=Role.ASSISTANT, content="done"),
        ]
    )
    env_a = FakeEnv()
    env_b = FakeEnv()
    session = _build_session(
        engine,
        {"a": env_a, "b": env_b},
        {"a": [_tool("a_tool")], "b": [_tool("b_tool")]},
    )

    session.send("hit both")

    assert env_a.calls == [("a_tool", {})]
    assert env_b.calls == [("b_tool", {})]


def test_engine_returning_no_new_messages_raises():
    """Defensive: if the engine returns an empty / unchanged conversation,
    we cannot continue safely. Surface as a session error."""

    class BrokenEngine:
        def infer(self, conversations: list[Conversation]) -> list[Conversation]:
            return [copy.deepcopy(c) for c in conversations]  # no append

    env = FakeEnv()
    env_config = EnvironmentConfig(environments=[_env_params("e", [_tool("ping")])])
    session = AgentSession(
        engine=BrokenEngine(),  # type: ignore[arg-type]
        envs={"e": env},
        router=ToolRouter.from_environment_config(env_config),
    )
    from oumi.agents.exceptions import AgentSessionError

    with pytest.raises(AgentSessionError, match="no new messages"):
        session.send("anything")


def test_system_prompt_is_first_message_when_provided():
    engine = ScriptedEngine([Message(role=Role.ASSISTANT, content="hi")])
    env = FakeEnv()
    session = _build_session(
        engine,
        {"e": env},
        {"e": [_tool("ping")]},
        system_prompt="Be helpful.",
    )

    session.send("hello")

    msgs = session.conversation.messages
    assert msgs[0].role == Role.SYSTEM
    assert msgs[0].compute_flattened_text_content() == "Be helpful."


def _capture_events(session: AgentSession, message: str) -> list[tuple[str, dict]]:
    """Run ``session.send`` while collecting every event the loop emits."""
    events: list[tuple[str, dict]] = []
    session.set_event_handler(lambda kind, data: events.append((kind, dict(data))))
    session.send(message)
    return events


def test_inference_end_event_forwards_usage_when_present():
    """The session forwards ``conversation.metadata['usage']`` to the TUI."""
    engine = ScriptedEngine(
        [Message(role=Role.ASSISTANT, content="hi")],
        usage_per_call=[
            {"prompt_tokens": 50, "completion_tokens": 12, "total_tokens": 62}
        ],
    )
    env = FakeEnv()
    session = _build_session(engine, {"e": env}, {"e": [_tool("ping")]})

    events = _capture_events(session, "hello")

    inference_end = next(e for e in events if e[0] == "inference_end")
    assert inference_end[1]["usage"]["total_tokens"] == 62


def test_inference_end_event_carries_none_when_engine_omits_usage():
    """Engines that don't report usage produce ``usage=None``, not fake numbers."""
    engine = ScriptedEngine([Message(role=Role.ASSISTANT, content="hi")])
    env = FakeEnv()
    session = _build_session(engine, {"e": env}, {"e": [_tool("ping")]})

    events = _capture_events(session, "hello")

    inference_end = next(e for e in events if e[0] == "inference_end")
    assert inference_end[1]["usage"] is None


def test_tool_result_event_includes_latency():
    """``tool_result`` carries a non-negative ``duration_ms``."""
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
    session = _build_session(engine, {"e": env}, {"e": [_tool("ping")]})

    events = _capture_events(session, "ping")

    tool_result = next(e for e in events if e[0] == "tool_result")
    assert "duration_ms" in tool_result[1]
    assert isinstance(tool_result[1]["duration_ms"], float)
    assert tool_result[1]["duration_ms"] >= 0


def test_turn_end_event_accumulates_session_tokens_across_turns():
    """``session_total_tokens`` accumulates across multiple ``send()`` calls."""
    engine = ScriptedEngine(
        [
            Message(role=Role.ASSISTANT, content="first"),
            Message(role=Role.ASSISTANT, content="second"),
        ],
        usage_per_call=[
            {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60},
            {"prompt_tokens": 60, "completion_tokens": 15, "total_tokens": 75},
        ],
    )
    env = FakeEnv()
    session = _build_session(engine, {"e": env}, {"e": [_tool("ping")]})

    first_events = _capture_events(session, "hi")
    first_turn_end = next(e for e in first_events if e[0] == "turn_end")
    assert first_turn_end[1]["session_total_tokens"] == 60

    # Reuse the same session — totals must accumulate, not reset.
    second_events: list[tuple[str, dict]] = []
    session.set_event_handler(
        lambda kind, data: second_events.append((kind, dict(data)))
    )
    session.send("again")
    second_turn_end = next(e for e in second_events if e[0] == "turn_end")
    assert second_turn_end[1]["session_total_tokens"] == 135
