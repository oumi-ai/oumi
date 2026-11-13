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

"""Tests for BrowserExecutableEnvironment (no live Kernel session required)."""

from __future__ import annotations

import contextlib
import threading
from typing import Any

import pytest

from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.types.tool_call import ToolResult
from oumi.environments.browser_executable_environment import (
    BrowserExecutableEnvironment,
)
from oumi.environments.executable_tool import ExecutableTool

_SESSION = "oumi.environments.browser_executable_environment.KernelBrowserSession"


class _FakeSession:
    def __init__(self, create_kwargs: dict[str, Any] | None = None) -> None:
        self.create_kwargs = create_kwargs
        self.closed = False

    @contextlib.contextmanager
    def page(self):
        yield "fake-page"

    def close(self) -> None:
        self.closed = True


def echo(arguments: dict[str, Any], context: Any) -> ToolResult:
    return ToolResult(
        output={
            "args": arguments,
            "page": context,
            "thread": threading.current_thread().name,
        }
    )


def _params(**kwargs: Any) -> EnvironmentParams:
    return EnvironmentParams(
        id="b", name="b", description="d", env_type="browser", **kwargs
    )


def test_step_runs_the_executor_with_the_page_on_the_playwright_thread():
    tool = ExecutableTool(
        id="t", name="t", description="d", executor=f"{__name__}.echo"
    )
    session: Any = _FakeSession()
    env = BrowserExecutableEnvironment(_params(tools=[tool]), session)
    assert env.requires_isolation() is True
    assert env.is_replayable() is False

    [result] = env.step([("t", {"x": 1})])
    output: Any = result.output
    assert output["args"] == {"x": 1}
    assert output["page"] == "fake-page"
    assert output["thread"].startswith("oumi-playwright")

    env.close()
    assert session.closed is True


def test_from_params_forwards_set_kwargs_to_browser_create(monkeypatch):
    monkeypatch.setattr(_SESSION, _FakeSession)
    env = BrowserExecutableEnvironment.from_params(
        _params(env_kwargs={"start_url": "https://example.com", "headless": True})
    )
    session: Any = env._session
    assert session.create_kwargs == {
        "start_url": "https://example.com",
        "headless": True,
    }


def test_from_params_rejects_a_profile_that_saves_changes():
    with pytest.raises(ValueError, match="save_changes"):
        BrowserExecutableEnvironment.from_params(
            _params(env_kwargs={"profile": {"name": "p", "save_changes": True}})
        )
