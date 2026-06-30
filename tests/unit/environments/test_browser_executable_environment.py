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

from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.environments.browser_executable_environment import (
    BrowserExecutableEnvironment,
)
from oumi.environments.browser_session import current_page
from oumi.environments.executable_tool import ExecutableTool


class _FakePage:
    marker = "fake-page"


class _FakeSession:
    """Stand-in for KernelBrowserSession that needs no Kernel API call."""

    def __init__(self, create_kwargs: dict[str, Any] | None = None) -> None:
        self.create_kwargs = create_kwargs
        self.session_id = "fake-123"
        self.closed = False

    @contextlib.contextmanager
    def page(self):
        yield _FakePage()

    def close(self) -> None:
        self.closed = True


def echo(**kwargs: Any) -> dict:
    """Executor that echoes its unpacked args and the ambient page marker."""
    return {"args": kwargs, "page": current_page().marker}


def _params(**kwargs: Any) -> EnvironmentParams:
    return EnvironmentParams(
        id="b", name="b", description="d", env_type="browser", **kwargs
    )


def _env(session: Any, **params_kwargs: Any) -> BrowserExecutableEnvironment:
    return BrowserExecutableEnvironment(_params(**params_kwargs), session)


def test_from_params_forwards_only_create_keys(monkeypatch):
    """env_kwargs is allow-listed to create params; api_key and junk are dropped."""
    captured: dict[str, Any] = {}

    def _factory(create_kwargs: dict[str, Any] | None = None) -> _FakeSession:
        captured["create_kwargs"] = create_kwargs
        return _FakeSession(create_kwargs)

    monkeypatch.setattr(
        "oumi.environments.browser_executable_environment.KernelBrowserSession",
        _factory,
    )
    BrowserExecutableEnvironment.from_params(
        _params(
            env_kwargs={
                "start_url": "https://example.com",
                "stealth": True,
                "profile": {"name": "auth"},
                "api_key": "SECRET",  # must never reach the SDK
                "bogus": 1,
            }
        )
    )
    assert captured["create_kwargs"] == {
        "start_url": "https://example.com",
        "stealth": True,
        "profile": {"name": "auth"},
    }


def test_requires_isolation_is_true():
    """Each rollout must get its own session."""
    assert _env(_FakeSession()).requires_isolation() is True


def test_step_binds_page_and_unpacks_args():
    """Executors get tool params unpacked and read the bound page via current_page()."""
    tool = ExecutableTool(
        id="t", name="t", description="d", executor=f"{__name__}.echo"
    )
    env = _env(_FakeSession(), tools=[tool])
    [result] = env.step([("t", {"x": 1})])
    assert result.output == {"args": {"x": 1}, "page": "fake-page"}


def test_close_tears_down_session():
    """Env close() delegates to session teardown."""
    session = _FakeSession()
    _env(session).close()
    assert session.closed is True
