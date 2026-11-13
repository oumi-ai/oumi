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

"""Tests for KernelBrowserSession lifecycle (no live Kernel session required)."""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from oumi.core.configs.params.tool_params import ToolEnvironmentError
from oumi.environments import browser_session
from oumi.environments.browser_session import (
    KernelBrowserSession,
    run_on_playwright_thread,
)
from oumi.utils import packaging


class _APIError(Exception):
    pass


class _PlaywrightError(Exception):
    pass


class _FakeKernel:
    """Records every Kernel API call made by the session under test."""

    def __init__(
        self,
        events: list[tuple],
        *,
        acquire_result: Any = "browser",
        reset_success: bool = True,
        reset_error: str | None = None,
        free_error: Exception | None = None,
    ) -> None:
        browser = SimpleNamespace(session_id="s1", cdp_ws_url="ws://x")
        if acquire_result == "browser":
            acquire_result = browser

        def create(**kwargs: Any):
            events.append(("create", kwargs))
            return browser

        def delete_by_id(session_id: str) -> None:
            events.append(("delete", session_id))
            if free_error is not None:
                raise free_error

        def acquire(name: str, *, acquire_timeout_seconds: int, timeout: int):
            events.append(("acquire", name, acquire_timeout_seconds, timeout))
            return acquire_result

        def release(name: str, *, session_id: str, reuse: bool) -> None:
            events.append(("release", name, session_id, reuse))
            if free_error is not None:
                raise free_error

        def execute(*, id: str, code: str, timeout_sec: int):
            events.append(("reset", id, code))
            if reset_error:
                raise RuntimeError(reset_error)
            return SimpleNamespace(success=reset_success, error="goto timeout")

        self.browsers = SimpleNamespace(
            create=create,
            delete_by_id=delete_by_id,
            playwright=SimpleNamespace(execute=execute),
        )
        self.browser_pools = SimpleNamespace(acquire=acquire, release=release)
        self._events = events

    def close(self) -> None:
        self._events.append(("client_close",))


@pytest.fixture
def fakes(monkeypatch):
    """Install fake kernel/playwright modules and a fresh Playwright thread."""
    events: list[tuple] = []
    kernel = ModuleType("kernel")
    kernel.__dict__["APIError"] = _APIError
    kernel.__dict__["Kernel"] = lambda: _FakeKernel(events)

    cdp_closes: list[int] = []
    cdp_state: dict[str, Any] = {
        "pages": [SimpleNamespace(marker="cdp-page")],
        "connect_error": None,
        "close_error": None,
    }

    class _CDP:
        @property
        def contexts(self):
            return [
                SimpleNamespace(
                    pages=cdp_state["pages"],
                    new_page=lambda: SimpleNamespace(marker="new-page"),
                )
            ]

        def close(self) -> None:
            cdp_closes.append(1)
            if cdp_state["close_error"]:
                raise _PlaywrightError(cdp_state["close_error"])

    def connect_over_cdp(url: str) -> _CDP:
        events.append(("connect", url))
        if cdp_state["connect_error"]:
            raise _PlaywrightError(cdp_state["connect_error"])
        return _CDP()

    starts: list[int] = []

    def sync_playwright():
        def start():
            starts.append(1)
            return SimpleNamespace(
                chromium=SimpleNamespace(connect_over_cdp=connect_over_cdp)
            )

        return SimpleNamespace(start=start)

    playwright = ModuleType("playwright")
    sync_api = ModuleType("playwright.sync_api")
    sync_api.__dict__["sync_playwright"] = sync_playwright
    sync_api.__dict__["Error"] = _PlaywrightError
    playwright.__dict__["sync_api"] = sync_api
    monkeypatch.setitem(sys.modules, "kernel", kernel)
    monkeypatch.setitem(sys.modules, "playwright", playwright)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", sync_api)
    monkeypatch.setattr(
        browser_session, "_PLAYWRIGHT_THREAD", browser_session._PlaywrightThread()
    )
    packaging._is_module_available.cache_clear()

    def use_kernel(**kwargs: Any) -> None:
        kernel.__dict__["Kernel"] = lambda: _FakeKernel(events, **kwargs)

    yield SimpleNamespace(
        events=events,
        cdp=cdp_state,
        cdp_closes=cdp_closes,
        playwright_starts=starts,
        use_kernel=use_kernel,
    )
    packaging._is_module_available.cache_clear()


def _use_page(session: KernelBrowserSession) -> Any:
    def body():
        with session.page() as page:
            return page

    return run_on_playwright_thread(body)


def _kernel_calls(events: list[tuple]) -> list[str]:
    return [e[0] for e in events if e[0] != "connect"]


def test_construction_opens_no_browser(fakes):
    KernelBrowserSession(create_kwargs={"headless": True}).close()
    KernelBrowserSession(pool="rl").close()
    assert _kernel_calls(fakes.events) == ["client_close", "client_close"]


def test_create_mode_creates_once_and_reuses_the_browser(fakes):
    session = KernelBrowserSession(create_kwargs={"headless": True})
    assert getattr(_use_page(session), "marker") == "cdp-page"
    _use_page(session)
    session.close()
    session.close()
    assert fakes.events == [
        ("create", {"headless": True}),
        ("connect", "ws://x"),
        ("connect", "ws://x"),
        ("delete", "s1"),
        ("client_close",),
    ]
    # One CDP connection per use, closed each time; one Playwright driver total.
    assert fakes.cdp_closes == [1, 1]
    assert fakes.playwright_starts == [1]


def test_close_logs_but_does_not_raise_kernel_api_errors(fakes, caplog):
    fakes.use_kernel(free_error=_APIError("404"))
    session = KernelBrowserSession(create_kwargs={})
    _use_page(session)
    with caplog.at_level("WARNING"):
        session.close()
    assert "leak" in caplog.text
    assert fakes.events[-1] == ("client_close",)


def test_close_still_raises_unexpected_errors_and_closes_the_client(fakes):
    fakes.use_kernel(free_error=ValueError("bug"))
    session = KernelBrowserSession(create_kwargs={})
    _use_page(session)
    with pytest.raises(ValueError, match="bug"):
        session.close()
    assert fakes.events[-1] == ("client_close",)


def test_page_after_close_raises(fakes):
    session = KernelBrowserSession(create_kwargs={})
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        _use_page(session)


def test_direct_construction_rejects_pool_with_create_kwargs(fakes):
    with pytest.raises(ValueError, match="create_kwargs"):
        KernelBrowserSession(pool="rl", create_kwargs={"headless": True})


def test_construction_without_kernel_installed_points_at_the_extra(monkeypatch):
    """Missing-dependency errors must direct users to the browser extra."""
    import importlib

    def raise_import_error(name: str) -> None:
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", raise_import_error)
    packaging._is_module_available.cache_clear()
    try:
        with pytest.raises(ImportError, match=r"oumi\[browser\]"):
            KernelBrowserSession(create_kwargs={})
    finally:
        packaging._is_module_available.cache_clear()


def test_pool_acquire_gives_the_http_request_more_time_than_the_poll(fakes):
    session = KernelBrowserSession(pool="rl", acquire_timeout_seconds=90)
    _use_page(session)
    session.close()
    [acquire] = [e for e in fakes.events if e[0] == "acquire"]
    assert acquire == (
        "acquire",
        "rl",
        90,
        90 + browser_session._ACQUIRE_HTTP_MARGIN_SECONDS,
    )
    assert ("release", "rl", "s1", True) in fakes.events


def test_pool_acquire_timeout_raises_a_clear_error(fakes):
    fakes.use_kernel(acquire_result="")
    session = KernelBrowserSession(pool="rl")
    with pytest.raises(ToolEnvironmentError, match="num_samples"):
        _use_page(session)
    session.close()
    assert _kernel_calls(fakes.events) == ["acquire", "client_close"]


@pytest.mark.parametrize(
    "kernel_kwargs",
    [{"reset_error": "reset boom"}, {"reset_success": False}],
    ids=["reset-raised", "reset-reported-failure"],
)
def test_failed_reset_quarantines_the_browser_and_retries_next_use(
    fakes, kernel_kwargs
):
    fakes.use_kernel(**kernel_kwargs)
    session = KernelBrowserSession(pool="rl")
    with pytest.raises(RuntimeError):
        _use_page(session)
    assert ("release", "rl", "s1", False) in fakes.events
    # The quarantined browser is dropped, so the next use acquires a fresh one.
    with pytest.raises(RuntimeError):
        _use_page(session)
    assert [e[0] for e in fakes.events].count("acquire") == 2


@pytest.mark.parametrize(
    ("start_url", "expected"),
    [("https://ex.com", 'page.goto("https://ex.com"'), (None, 'page.goto("about:')],
)
def test_pool_reset_navigates_to_start_url(fakes, start_url, expected):
    session = KernelBrowserSession(pool="rl", start_url=start_url)
    _use_page(session)
    [reset] = [e for e in fakes.events if e[0] == "reset"]
    assert expected in reset[2]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_reset_js_is_valid_javascript_in_kernels_scope():
    """Kernel pre-declares page/context/browser; redeclaring one breaks every reset."""
    wrapped = (
        "async function main(page, context, browser) {\n"
        f"{browser_session._RESET_JS}\n"
        'await page.goto("about:blank");\n'
        "}\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "reset.js"
        script.write_text(wrapped)
        result = subprocess.run(
            ["node", "--check", str(script)], capture_output=True, text=True
        )
    assert result.returncode == 0, result.stderr


def test_page_creates_new_page_when_context_is_empty(fakes):
    fakes.cdp["pages"] = []
    session = KernelBrowserSession(create_kwargs={})
    assert getattr(_use_page(session), "marker") == "new-page"


def test_page_closes_tabs_the_call_opened(fakes):
    closed: list[str] = []
    original = SimpleNamespace(marker="original", close=lambda: closed.append("o"))
    pages = [original]
    fakes.cdp["pages"] = pages
    session = KernelBrowserSession(create_kwargs={})

    def open_popup():
        with session.page():
            pages.insert(0, SimpleNamespace(close=lambda: closed.append("popup")))

    run_on_playwright_thread(open_popup)
    assert closed == ["popup"]


def test_page_close_error_does_not_mask_body_exception(fakes):
    fakes.cdp["close_error"] = "cdp teardown boom"
    session = KernelBrowserSession(create_kwargs={})

    def body():
        with session.page():
            raise ValueError("body boom")

    with pytest.raises(ValueError, match="body boom"):
        run_on_playwright_thread(body)
    assert fakes.cdp_closes == [1]


def test_page_connect_failure_explains_the_standby_timeout(fakes):
    fakes.cdp["connect_error"] = "ws closed"
    session = KernelBrowserSession(create_kwargs={})
    with pytest.raises(ToolEnvironmentError, match="timeout_seconds"):
        _use_page(session)


def test_page_outside_the_playwright_thread_raises(fakes):
    session = KernelBrowserSession(create_kwargs={})
    with pytest.raises(RuntimeError, match="Playwright thread"), session.page():
        pass


def test_playwright_thread_runs_calls_made_inside_a_running_event_loop(fakes):
    """verl's agent loop calls tools on its event-loop thread."""

    def no_running_loop() -> bool:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return True
        return False

    async def from_the_loop() -> tuple[bool, int]:
        return run_on_playwright_thread(
            lambda: (no_running_loop(), threading.get_ident())
        )

    ok, thread_id = asyncio.run(from_the_loop())
    assert ok is True
    assert thread_id != threading.get_ident()


def test_playwright_thread_runs_nested_calls_inline():
    """A run() issued from the Playwright thread itself runs inline, not deadlocked."""
    thread = browser_session._PlaywrightThread()
    outer = thread.run(lambda: thread.run(threading.get_ident))
    assert outer != threading.get_ident()
