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

"""Tests for KernelBrowserSession (no live Kernel session required)."""

from __future__ import annotations

import asyncio
import sys
import threading
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from oumi.core.configs.params.tool_params import ToolEnvironmentError
from oumi.environments import browser_session
from oumi.environments.browser_session import (
    KernelBrowserSession,
    run_on_playwright_thread,
)


class _APIError(Exception):
    pass


class _PlaywrightError(Exception):
    pass


@pytest.fixture
def fakes(monkeypatch):
    """Install fake kernel/playwright modules that record what the session does."""
    events: list[Any] = []
    state: dict[str, Any] = {
        "pages": [SimpleNamespace(marker="cdp-page")],
        "connect_error": None,
        "delete_error": None,
    }

    def create(**kwargs: Any):
        events.append(("create", kwargs))
        return SimpleNamespace(session_id="s1", cdp_ws_url="ws://x")

    def delete_by_id(session_id: str) -> None:
        events.append("delete")
        if state["delete_error"]:
            raise state["delete_error"]

    def connect_over_cdp(url: str):
        events.append("connect")
        if state["connect_error"]:
            raise _PlaywrightError(state["connect_error"])
        context = SimpleNamespace(pages=state["pages"])
        return SimpleNamespace(
            contexts=[context], close=lambda: events.append("disconnect")
        )

    def start():
        events.append("playwright_start")
        return SimpleNamespace(
            chromium=SimpleNamespace(connect_over_cdp=connect_over_cdp)
        )

    kernel = ModuleType("kernel")
    kernel.__dict__["APIError"] = _APIError
    kernel.__dict__["Kernel"] = lambda: SimpleNamespace(
        browsers=SimpleNamespace(create=create, delete_by_id=delete_by_id),
        close=lambda: events.append("client_close"),
    )
    sync_api = ModuleType("playwright.sync_api")
    sync_api.__dict__["sync_playwright"] = lambda: SimpleNamespace(start=start)
    sync_api.__dict__["Error"] = _PlaywrightError
    monkeypatch.setitem(sys.modules, "kernel", kernel)
    playwright = ModuleType("playwright")
    playwright.__dict__["sync_api"] = sync_api
    monkeypatch.setitem(sys.modules, "playwright", playwright)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", sync_api)
    monkeypatch.setattr(browser_session, "_playwright", None)
    return SimpleNamespace(events=events, state=state)


def _use_page(session: KernelBrowserSession) -> Any:
    def body():
        with session.page() as page:
            return page

    return run_on_playwright_thread(body)


def test_browser_is_created_on_first_use_and_deleted_on_close(fakes):
    session = KernelBrowserSession(create_kwargs={"headless": True})
    assert fakes.events == []
    assert getattr(_use_page(session), "marker") == "cdp-page"
    _use_page(session)
    session.close()
    session.close()
    # One browser and one Playwright driver, but a CDP connection per use.
    assert fakes.events == [
        "playwright_start",
        ("create", {"headless": True}),
        "connect",
        "disconnect",
        "connect",
        "disconnect",
        "delete",
        "client_close",
    ]


def test_close_logs_but_does_not_raise_kernel_api_errors(fakes, caplog):
    fakes.state["delete_error"] = _APIError("404")
    session = KernelBrowserSession()
    _use_page(session)
    with caplog.at_level("WARNING"):
        session.close()
    assert "leak" in caplog.text
    assert fakes.events[-1] == "client_close"


def test_page_after_close_raises(fakes):
    session = KernelBrowserSession()
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        _use_page(session)


def test_page_closes_tabs_the_call_opened(fakes):
    closed: list[str] = []
    pages = [SimpleNamespace(close=lambda: closed.append("original"))]
    fakes.state["pages"] = pages
    session = KernelBrowserSession()

    def open_popup():
        with session.page():
            pages.insert(0, SimpleNamespace(close=lambda: closed.append("popup")))

    run_on_playwright_thread(open_popup)
    assert closed == ["popup"]


def test_page_connect_failure_explains_the_standby_timeout(fakes):
    fakes.state["connect_error"] = "ws closed"
    with pytest.raises(ToolEnvironmentError, match="timeout_seconds"):
        _use_page(KernelBrowserSession())


def test_playwright_thread_runs_calls_made_inside_a_running_event_loop():
    """verl's agent loop calls tools on its event-loop thread."""

    def running_loop() -> bool:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return False
        return True

    async def from_the_loop() -> tuple[bool, int]:
        return run_on_playwright_thread(lambda: (running_loop(), threading.get_ident()))

    in_loop, thread_id = asyncio.run(from_the_loop())
    assert in_loop is False
    assert thread_id != threading.get_ident()
