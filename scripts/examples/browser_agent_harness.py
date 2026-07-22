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

"""Minimal OpenAI agent loop driving a Kernel browser via Oumi's env."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Protocol

from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.configs.params.tool_params import ToolError
from oumi.core.types.tool_call import ToolResult
from oumi.environments.browser_executable_environment import (
    BrowserExecutableEnvironment,
)
from oumi.environments.executable_tool import ExecutableTool

if TYPE_CHECKING:
    from openai import OpenAI  # pyright: ignore[reportMissingImports]

_SYSTEM_PROMPT = (
    "You are a web-browsing agent. Use the browser tools to complete the task. "
    "When you have the answer, reply in plain text without calling a tool."
)
_PLAYWRIGHT = "oumi.environments.playwright_executors"


class _ToolFunction(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def arguments(self) -> str: ...


class _ToolCall(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def function(self) -> _ToolFunction: ...


class _ToolEnvironment(Protocol):
    def step(self, calls: list[tuple[str, dict[str, Any]]]) -> list[ToolResult]: ...


def _is_playwright_error(error: Exception) -> bool:
    try:
        from playwright.sync_api import (  # pyright: ignore[reportMissingImports]
            Error as PlaywrightError,
        )
    except ModuleNotFoundError:
        return False
    return isinstance(error, PlaywrightError)


def browser_tools() -> list[ExecutableTool]:
    """Build the browser toolset."""
    return [
        ExecutableTool(
            id="navigate",
            name="navigate",
            description="Navigate to a URL.",
            parameters={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
            executor=f"{_PLAYWRIGHT}.navigate",
        ),
        ExecutableTool(
            id="click",
            name="click",
            description="Click the element matching a CSS selector.",
            parameters={
                "type": "object",
                "properties": {"selector": {"type": "string"}},
                "required": ["selector"],
            },
            executor=f"{_PLAYWRIGHT}.click",
        ),
        ExecutableTool(
            id="type_text",
            name="type_text",
            description="Type text into the element matching a CSS selector.",
            parameters={
                "type": "object",
                "properties": {
                    "selector": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["selector", "text"],
            },
            executor=f"{_PLAYWRIGHT}.type_text",
        ),
        ExecutableTool(
            id="read_text",
            name="read_text",
            description="Read the visible text of the page, or of a CSS selector.",
            parameters={
                "type": "object",
                "properties": {
                    "selector": {"type": "string"},
                    "max_chars": {"type": "integer"},
                },
            },
            executor=f"{_PLAYWRIGHT}.read_text",
        ),
    ]


def build_openai_tools(tools: list[ExecutableTool]) -> list[dict[str, Any]]:
    """Convert environment tools to the OpenAI tools format."""
    return [
        tool.to_tool_definition().model_dump(mode="json", exclude_none=True)
        for tool in tools
    ]


def run_tool_calls(
    env: _ToolEnvironment, tool_calls: Iterable[_ToolCall]
) -> list[dict[str, Any]]:
    """Execute model tool calls and return tool messages."""
    messages: list[dict[str, Any]] = []
    for call in tool_calls:
        print(f"  -> {call.function.name} {call.function.arguments}", flush=True)
        try:
            arguments = json.loads(call.function.arguments or "{}")
            [result] = env.step([(call.function.name, arguments)])
            output = result.output
        except (json.JSONDecodeError, ToolError) as e:
            output = {"error": str(e)}
        except Exception as e:
            if not _is_playwright_error(e):
                raise
            output = {"error": str(e)}
        content = output if isinstance(output, str) else json.dumps(output)
        messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
    return messages


def run_agent(
    client: OpenAI,
    env: _ToolEnvironment,
    tools: list[ExecutableTool],
    task: str,
    *,
    model: str = "gpt-4o-mini",
    max_turns: int = 10,
) -> str:
    """Drive the tool-call loop until the model answers without calling a tool."""
    openai_tools = build_openai_tools(tools)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    for _ in range(max_turns):
        response = client.chat.completions.create(
            model=model, messages=messages, tools=openai_tools
        )
        message = response.choices[0].message
        if not message.tool_calls:
            if message.refusal:
                raise RuntimeError(f"Model refused the task: {message.refusal}")
            if not message.content or not message.content.strip():
                raise RuntimeError("Model returned an empty response.")
            return message.content
        messages.append(
            message.model_dump(
                mode="json",
                include={"role", "content", "tool_calls"},
                exclude_none=True,
            )
        )
        messages.extend(run_tool_calls(env, message.tool_calls))
    raise RuntimeError(f"Agent did not finish within {max_turns} turns.")


def main() -> None:
    """Open a Kernel browser session and run the agent loop, then tear it down."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "task",
        nargs="?",
        default=(
            "Go to oumi.ai, from the homepage tell me what they do, and then go "
            "to Company page and read all the names of the team members."
        ),
    )
    parser.add_argument("--start-url", default=None, help="Initial Kernel page URL.")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--keep-open",
        type=int,
        default=0,
        help="Seconds to hold the session open after finishing, so you can watch.",
    )
    parser.add_argument(
        "--selfcheck", action="store_true", help="Run the offline self-check and exit."
    )
    args = parser.parse_args()

    if args.selfcheck:
        _selfcheck()
        print("selfcheck OK")
        return

    # Lazy import keeps this module import-safe (and self-checkable) without openai.
    from openai import OpenAI  # pyright: ignore[reportMissingImports]

    tools = browser_tools()
    params = EnvironmentParams(
        id="browser-demo",
        name="browser-demo",
        description="Kernel browser agent demo.",
        env_type="browser",
        env_kwargs={"start_url": args.start_url} if args.start_url else {},
        tools=tools,
    )
    env = BrowserExecutableEnvironment.from_params(params)
    print(
        f"\n>>> Live View (open to watch): {env._session.live_view_url}\n", flush=True
    )
    try:
        print(run_agent(OpenAI(), env, tools, args.task, model=args.model))
        if args.keep_open > 0:
            print(f"\nHolding session open {args.keep_open}s — watch the Live View...")
            time.sleep(args.keep_open)
    finally:
        env.close()


def _selfcheck() -> None:
    """Run the offline self-check."""
    openai_tools = build_openai_tools(browser_tools())
    assert [t["function"]["name"] for t in openai_tools] == [
        "navigate",
        "click",
        "type_text",
        "read_text",
    ]
    assert openai_tools[0]["type"] == "function"
    assert openai_tools[0]["function"]["parameters"]["required"] == ["url"]

    class _FakeEnv:
        def step(self, calls):
            ((tool_id, arguments),) = calls
            if tool_id == "boom":
                raise ToolError("bad arguments")
            return [ToolResult(output={"echo": tool_id, "args": arguments})]

    class _FakeFunction:
        def __init__(self, name: str, arguments: str) -> None:
            self.name = name
            self.arguments = arguments

    class _FakeCall:
        def __init__(self, id: str, name: str, arguments: str) -> None:
            self.id = id
            self.function = _FakeFunction(name, arguments)

    messages = run_tool_calls(
        _FakeEnv(),
        [
            _FakeCall("c1", "navigate", '{"url": "https://example.com"}'),
            _FakeCall("c2", "boom", "{}"),
        ],
    )
    assert messages[0]["tool_call_id"] == "c1"
    assert json.loads(messages[0]["content"]) == {
        "echo": "navigate",
        "args": {"url": "https://example.com"},
    }
    assert json.loads(messages[1]["content"]) == {"error": "bad arguments"}


if __name__ == "__main__":
    main()
