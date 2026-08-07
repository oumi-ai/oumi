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
from typing import Any

import pytest

from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.types.tool_call import ToolResult
from oumi.environments.browser_executable_environment import (
    BrowserExecutableEnvironment,
)
from oumi.environments.executable_tool import ExecutableTool

_SESSION = "oumi.environments.browser_executable_environment.KernelBrowserSession"


class _FakePage:
    marker = "fake-page"


class _FakeSession:
    def __init__(self, **kwargs: Any) -> None:
        self.closed = False
        self.page_entries = 0

    @contextlib.contextmanager
    def page(self):
        self.page_entries += 1
        yield _FakePage()

    def close(self) -> None:
        self.closed = True


def echo(arguments: dict[str, Any], context: Any) -> ToolResult:
    return ToolResult(output={"args": arguments, "page": context.marker})


def _params(**kwargs: Any) -> EnvironmentParams:
    return EnvironmentParams(
        id="b", name="b", description="d", env_type="browser", **kwargs
    )


def _tool(executor: str) -> ExecutableTool:
    return ExecutableTool(id="t", name="t", description="d", executor=executor)


def _env(session: Any, **params_kwargs: Any) -> BrowserExecutableEnvironment:
    return BrowserExecutableEnvironment(_params(**params_kwargs), session)


def _capturing_factory(captured: dict[str, Any], created: list[Any] | None = None):
    """Stand in for KernelBrowserSession, recording the kwargs it was built with."""

    def _factory(**kwargs: Any) -> _FakeSession:
        captured.update(kwargs)
        session = _FakeSession(**kwargs)
        if created is not None:
            created.append(session)
        return session

    return _factory


def test_requires_isolation_is_true():
    env = _env(_FakeSession())
    assert env.requires_isolation() is True
    assert env.is_replayable() is False


def test_step_passes_live_page_as_context():
    session = _FakeSession()
    env = _env(session, tools=[_tool(f"{__name__}.echo")])
    [result] = env.step([("t", {"x": 1})])
    assert result.output == {"args": {"x": 1}, "page": "fake-page"}
    # One CDP context per call — a cached connection would change this.
    assert session.page_entries == 1
    env.step([("t", {"x": 2})])
    assert session.page_entries == 2


def test_close_delegates_to_session():
    session = _FakeSession()
    _env(session).close()
    assert session.closed is True


def test_from_params_create_mode_forwards_create_fields(monkeypatch):
    captured: dict[str, Any] = {}
    monkeypatch.setattr(_SESSION, _capturing_factory(captured))
    BrowserExecutableEnvironment.from_params(
        _params(
            env_kwargs={
                "start_url": "https://example.com",
                "headless": True,
                "stealth": True,
            }
        )
    )
    assert "pool" not in captured
    assert captured["create_kwargs"] == {
        "start_url": "https://example.com",
        "headless": True,
        "stealth": True,
    }


def test_from_params_pool_mode_acquires(monkeypatch):
    captured: dict[str, Any] = {}
    monkeypatch.setattr(_SESSION, _capturing_factory(captured))
    BrowserExecutableEnvironment.from_params(
        _params(
            env_kwargs={
                "pool": "rl-browser",
                "acquire_timeout_seconds": 30,
                "start_url": "https://example.com",
            }
        )
    )
    assert captured["pool"] == "rl-browser"
    assert captured["acquire_timeout_seconds"] == 30
    assert captured["start_url"] == "https://example.com"
    assert "create_kwargs" not in captured


def test_from_params_rejects_unknown_env_kwargs():
    with pytest.raises(ValueError):
        BrowserExecutableEnvironment.from_params(_params(env_kwargs={"api_key": "x"}))
    with pytest.raises(ValueError):
        BrowserExecutableEnvironment.from_params(_params(env_kwargs={"headles": True}))


@pytest.mark.parametrize(
    "viewport",
    [
        {"width": 1280, "height": 720},
        # Kernel's own viewport type allows refresh_rate.
        {"width": 1280, "height": 720, "refresh_rate": 25},
    ],
)
def test_from_params_forwards_viewport_to_create(monkeypatch, viewport):
    captured: dict[str, Any] = {}
    monkeypatch.setattr(_SESSION, _capturing_factory(captured))
    BrowserExecutableEnvironment.from_params(_params(env_kwargs={"viewport": viewport}))
    assert captured["create_kwargs"] == {"viewport": viewport}


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("headless", True),
        ("stealth", True),
        ("profile", {"name": "test-profile"}),
        ("proxy_id", "proxy-1"),
        ("viewport", {"width": 1280, "height": 720}),
        ("timeout_seconds", 30),
    ],
)
def test_from_params_rejects_create_only_options_with_pool(option, value):
    with pytest.raises(ValueError, match="Browser pool mode does not support"):
        BrowserExecutableEnvironment.from_params(
            _params(env_kwargs={"pool": "rl-browser", option: value})
        )


def test_from_params_rejects_pool_only_options_without_pool():
    with pytest.raises(ValueError, match="Create mode does not support"):
        BrowserExecutableEnvironment.from_params(
            _params(env_kwargs={"acquire_timeout_seconds": 30})
        )


def test_from_params_closes_session_if_executor_wiring_fails(monkeypatch):
    created: list[_FakeSession] = []
    monkeypatch.setattr(_SESSION, _capturing_factory({}, created))
    with pytest.raises(ValueError):
        BrowserExecutableEnvironment.from_params(_params(tools=[_tool("not_dotted")]))
    assert created and created[0].closed is True
