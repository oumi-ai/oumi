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

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from oumi.core.types.tool_call import ToolResult
from oumi.environments.executable_tool import ExecutableTool
from scripts.examples import browser_agent_harness


@dataclass(frozen=True)
class _FakeFunction:
    name: str
    arguments: str


@dataclass(frozen=True)
class _FakeToolCall:
    id: str
    function: _FakeFunction

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments,
            },
        }


@dataclass(frozen=True)
class _FakeMessage:
    content: str | None = None
    tool_calls: tuple[_FakeToolCall, ...] = ()
    refusal: str | None = None

    def model_dump(
        self,
        *,
        mode: str,
        include: set[str],
        exclude_none: bool,
    ) -> dict[str, Any]:
        assert mode == "json"
        value = {
            "role": "assistant",
            "content": self.content,
            "tool_calls": [call.as_dict() for call in self.tool_calls],
        }
        return {
            key: item
            for key, item in value.items()
            if key in include and (item is not None or not exclude_none)
        }


class _FakeCompletions:
    def __init__(self, messages: list[_FakeMessage]) -> None:
        self._messages = iter(messages)
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append(deepcopy(kwargs))
        return SimpleNamespace(choices=[SimpleNamespace(message=next(self._messages))])


class _FakeClient:
    def __init__(self, messages: list[_FakeMessage]) -> None:
        self.completions = _FakeCompletions(messages)
        self.chat = SimpleNamespace(completions=self.completions)


class _FakeEnvironment:
    def __init__(self) -> None:
        self.calls: list[list[tuple[str, dict[str, Any]]]] = []

    def step(self, calls: list[tuple[str, dict[str, Any]]]) -> list[ToolResult]:
        self.calls.append(calls)
        return [ToolResult(output={"text": "Oumi builds an AI platform."})]


def _tool_call(call_id: str = "call-1") -> _FakeToolCall:
    return _FakeToolCall(
        id=call_id,
        function=_FakeFunction(name="read_text", arguments='{"selector": "main"}'),
    )


def _tools() -> list[ExecutableTool]:
    return [
        ExecutableTool(
            id="read_text",
            name="read_text",
            description="Read visible text.",
            parameters={"type": "object", "properties": {}},
            executor="oumi.environments.playwright_executors.read_text",
        )
    ]


def test_run_agent_executes_tool_call_then_returns_final_answer() -> None:
    client: Any = _FakeClient(
        [
            _FakeMessage(tool_calls=(_tool_call(),)),
            _FakeMessage(content="Oumi builds an AI platform."),
        ]
    )
    env = _FakeEnvironment()

    answer = browser_agent_harness.run_agent(client, env, _tools(), "What is Oumi?")

    assert answer == "Oumi builds an AI platform."
    assert env.calls == [[("read_text", {"selector": "main"})]]
    assert len(client.completions.requests) == 2


def test_run_agent_serializes_assistant_call_before_tool_result() -> None:
    tool_call = _tool_call()
    client: Any = _FakeClient(
        [
            _FakeMessage(tool_calls=(tool_call,)),
            _FakeMessage(content="Done."),
        ]
    )

    browser_agent_harness.run_agent(
        client, _FakeEnvironment(), _tools(), "Read the page."
    )

    second_request_messages = client.completions.requests[1]["messages"]
    assert [message["role"] for message in second_request_messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert second_request_messages[2] == {
        "role": "assistant",
        "tool_calls": [tool_call.as_dict()],
    }
    assert second_request_messages[3] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": '{"text": "Oumi builds an AI platform."}',
    }


@pytest.mark.parametrize(
    ("message", "expected_error"),
    [
        (
            _FakeMessage(refusal="I cannot help with that."),
            "Model refused the task: I cannot help with that.",
        ),
        (_FakeMessage(content="  "), "Model returned an empty response."),
    ],
)
def test_run_agent_rejects_refusal_or_empty_response(
    message: _FakeMessage, expected_error: str
) -> None:
    client: Any = _FakeClient([message])
    with pytest.raises(RuntimeError, match=expected_error):
        browser_agent_harness.run_agent(
            client, _FakeEnvironment(), _tools(), "Do the task."
        )


def test_run_agent_raises_when_max_turns_are_exhausted() -> None:
    client: Any = _FakeClient(
        [
            _FakeMessage(tool_calls=(_tool_call("call-1"),)),
            _FakeMessage(tool_calls=(_tool_call("call-2"),)),
        ]
    )
    env = _FakeEnvironment()

    with pytest.raises(RuntimeError, match="Agent did not finish within 2 turns."):
        browser_agent_harness.run_agent(
            client, env, _tools(), "Keep browsing.", max_turns=2
        )

    assert len(env.calls) == 2
    assert len(client.completions.requests) == 2
