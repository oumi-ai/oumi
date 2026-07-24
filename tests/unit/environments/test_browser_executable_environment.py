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
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.configs.params.tool_params import ToolError
from oumi.core.types.tool_call import ToolResult
from oumi.environments import browser_session
from oumi.environments.browser_executable_environment import (
    BrowserExecutableEnvironment,
)
from oumi.environments.browser_session import KernelBrowserSession
from oumi.environments.executable_tool import ExecutableTool
from oumi.utils import packaging

_SESSION = "oumi.environments.browser_executable_environment.KernelBrowserSession"


@pytest.fixture
def browser_modules(monkeypatch):
    kernel = ModuleType("kernel")
    kernel.__dict__["Kernel"] = None
    playwright = ModuleType("playwright")
    playwright_sync_api = ModuleType("playwright.sync_api")
    playwright_sync_api.__dict__["sync_playwright"] = None
    playwright.__dict__["sync_api"] = playwright_sync_api
    monkeypatch.setitem(sys.modules, "kernel", kernel)
    monkeypatch.setitem(sys.modules, "playwright", playwright)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", playwright_sync_api)
    packaging._is_module_available.cache_clear()
    yield SimpleNamespace(kernel=kernel, playwright=playwright_sync_api)
    packaging._is_module_available.cache_clear()


class _FakePage:
    marker = "fake-page"


class _FakeSession:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.session_id = "fake-123"
        self.closed = False

    @contextlib.contextmanager
    def page(self):
        yield _FakePage()

    def close(self) -> None:
        self.closed = True


def echo(arguments: dict[str, Any], context: Any) -> ToolResult:
    return ToolResult(output={"args": arguments, "page": context.marker})


def returns_dict(arguments: dict[str, Any], context: Any) -> dict:
    return {"not": "a ToolResult"}


def _params(**kwargs: Any) -> EnvironmentParams:
    return EnvironmentParams(
        id="b", name="b", description="d", env_type="browser", **kwargs
    )


def _tool(executor: str) -> ExecutableTool:
    return ExecutableTool(id="t", name="t", description="d", executor=executor)


def _env(session: Any, **params_kwargs: Any) -> BrowserExecutableEnvironment:
    return BrowserExecutableEnvironment(_params(**params_kwargs), session)


def _patch_pool_kernel(
    monkeypatch,
    kernel: Any,
    events: list[tuple],
    *,
    reset_success: bool = True,
    reset_error: str | None = None,
) -> list[str]:
    reset_codes: list[str] = []

    class _Pools:
        def acquire(self, name: str, *, acquire_timeout_seconds: int):
            events.append(("acquire", name, acquire_timeout_seconds))
            return SimpleNamespace(
                session_id="s1", cdp_ws_url="ws://x", browser_live_view_url="lv"
            )

        def release(self, name: str, *, session_id: str, reuse: bool) -> None:
            events.append(("release", name, session_id, reuse))

    class _Playwright:
        def execute(self, *, id: str, code: str, timeout_sec: int):
            events.append(("reset", id))
            reset_codes.append(code)
            if reset_error:
                raise RuntimeError(reset_error)
            return SimpleNamespace(success=reset_success, error="goto timeout")

    monkeypatch.setattr(
        kernel,
        "Kernel",
        lambda: SimpleNamespace(
            browser_pools=_Pools(),
            browsers=SimpleNamespace(playwright=_Playwright()),
            close=lambda: events.append(("close",)),
        ),
    )
    return reset_codes


def _patch_page_dependencies(
    monkeypatch,
    kernel: Any,
    playwright: Any,
    *,
    pages: list[Any],
    new_page: Any = None,
    close_error: str | None = None,
) -> None:
    context = SimpleNamespace(pages=pages)
    if new_page is not None:
        context.new_page = new_page

    class _CDP:
        contexts = [context]

        def close(self) -> None:
            if close_error:
                raise RuntimeError(close_error)

    cdp = _CDP()

    class _Playwright:
        chromium = SimpleNamespace(connect_over_cdp=lambda _: cdp)

        def __enter__(self):
            return self

        def __exit__(self, *args: Any) -> bool:
            return False

    monkeypatch.setattr(playwright, "sync_playwright", lambda: _Playwright())
    monkeypatch.setattr(
        kernel,
        "Kernel",
        lambda: SimpleNamespace(
            browsers=SimpleNamespace(
                create=lambda **_: SimpleNamespace(
                    session_id="s1", cdp_ws_url="ws://x", browser_live_view_url=None
                )
            ),
            close=lambda: None,
        ),
    )


def test_requires_isolation_is_true():
    env = _env(_FakeSession())
    assert env.requires_isolation() is True
    assert env.is_replayable() is False


def test_step_passes_live_page_as_context():
    env = _env(_FakeSession(), tools=[_tool(f"{__name__}.echo")])
    [result] = env.step([("t", {"x": 1})])
    assert result.output == {"args": {"x": 1}, "page": "fake-page"}


def test_executor_must_return_toolresult():
    env = _env(_FakeSession(), tools=[_tool(f"{__name__}.returns_dict")])
    with pytest.raises(ToolError):
        env.step([("t", {})])


def test_close_delegates_to_session():
    session = _FakeSession()
    _env(session).close()
    assert session.closed is True


def test_from_params_create_mode_forwards_create_fields(monkeypatch):
    captured: dict[str, Any] = {}

    def _factory(**kwargs: Any) -> _FakeSession:
        captured.update(kwargs)
        return _FakeSession(**kwargs)

    monkeypatch.setattr(_SESSION, _factory)
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

    def _factory(**kwargs: Any) -> _FakeSession:
        captured.update(kwargs)
        return _FakeSession(**kwargs)

    monkeypatch.setattr(_SESSION, _factory)
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
        {},
        {"width": 1280},
        {"height": 720},
        {"width": 1280, "height": 720, "device_scale_factor": 2},
        {"width": "1280", "height": 720},
        {"width": 1280, "height": 720.0},
        {"width": True, "height": 720},
    ],
)
def test_from_params_rejects_malformed_viewport(viewport):
    with pytest.raises(ValueError, match="viewport"):
        BrowserExecutableEnvironment.from_params(
            _params(env_kwargs={"viewport": viewport})
        )


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
    with pytest.raises(ValueError, match=option):
        BrowserExecutableEnvironment.from_params(
            _params(env_kwargs={"pool": "rl-browser", option: value})
        )


def test_from_params_closes_session_if_executor_wiring_fails(monkeypatch):
    created: list[_FakeSession] = []

    def _factory(**kwargs: Any) -> _FakeSession:
        session = _FakeSession(**kwargs)
        created.append(session)
        return session

    monkeypatch.setattr(_SESSION, _factory)
    with pytest.raises(ValueError):
        BrowserExecutableEnvironment.from_params(_params(tools=[_tool("not_dotted")]))
    assert created and created[0].closed is True


def test_create_mode_close_deletes_once_and_always_closes_client(
    monkeypatch, browser_modules
):
    kernel = browser_modules.kernel

    class _Browsers:
        def __init__(self) -> None:
            self.deletes = 0
            self.raise_once = False

        def create(self, **kwargs: Any):
            return SimpleNamespace(
                session_id="s1", cdp_ws_url="ws://x", browser_live_view_url=None
            )

        def delete_by_id(self, session_id: str) -> None:
            self.deletes += 1
            if self.raise_once:
                self.raise_once = False
                raise RuntimeError("transient")

    browsers = _Browsers()
    kernel_closes: list[bool] = []
    monkeypatch.setattr(
        kernel,
        "Kernel",
        lambda: SimpleNamespace(
            browsers=browsers, close=lambda: kernel_closes.append(True)
        ),
    )
    session = KernelBrowserSession(create_kwargs={})

    browsers.raise_once = True
    with pytest.raises(RuntimeError):
        session.close()
    # A failed delete still closes the HTTP client, and close stays idempotent.
    assert kernel_closes == [True]
    session.close()
    session.close()
    assert browsers.deletes == 1


def test_pool_mode_comprehensively_resets_and_releases_with_reuse(
    monkeypatch, browser_modules
):
    kernel = browser_modules.kernel

    events: list[tuple] = []
    reset_codes = _patch_pool_kernel(monkeypatch, kernel, events)
    session = KernelBrowserSession(
        pool="rl", acquire_timeout_seconds=30, start_url="https://ex.com"
    )
    session.close()
    assert len(reset_codes) == 1
    reset_code = reset_codes[0]
    assert "context.clearCookies()" in reset_code
    assert "localStorage.clear()" in reset_code
    assert "existingPage.close()" in reset_code
    assert 'page.goto("https://ex.com"' in reset_code
    assert events == [
        ("acquire", "rl", 30),
        ("reset", "s1"),
        ("release", "rl", "s1", True),
        ("close",),
    ]


@pytest.mark.parametrize("identifier", ["page", "context", "browser"])
def test_reset_js_does_not_redeclare_kernel_globals(identifier):
    # Kernel's playwright.execute pre-declares these; redeclaring one makes the
    # whole reset fail with "Identifier ... has already been declared".
    for keyword in ("const", "let", "var"):
        assert f"{keyword} {identifier} " not in browser_session._RESET_JS
        assert f"{keyword} {identifier}=" not in browser_session._RESET_JS


def test_pool_mode_reset_failure_releases_without_reuse(monkeypatch, browser_modules):
    kernel = browser_modules.kernel

    events: list[tuple] = []
    _patch_pool_kernel(monkeypatch, kernel, events, reset_error="reset boom")
    with pytest.raises(RuntimeError, match="reset boom"):
        KernelBrowserSession(pool="rl")
    assert events == [
        ("acquire", "rl", 60),
        ("reset", "s1"),
        ("release", "rl", "s1", False),
        ("close",),
    ]


def test_page_yields_default_cdp_page_and_swallows_close_error(
    monkeypatch, browser_modules
):
    kernel = browser_modules.kernel
    pw = browser_modules.playwright

    _patch_page_dependencies(
        monkeypatch,
        kernel,
        pw,
        pages=[SimpleNamespace(marker="cdp-page")],
        close_error="cdp teardown boom",
    )
    session = KernelBrowserSession(create_kwargs={})
    with session.page() as page:
        assert page.marker == "cdp-page"


def test_pool_mode_reset_reported_failure_releases_without_reuse(
    monkeypatch, browser_modules
):
    kernel = browser_modules.kernel

    events: list[tuple] = []
    _patch_pool_kernel(monkeypatch, kernel, events, reset_success=False)
    with pytest.raises(RuntimeError, match="Kernel browser reset failed"):
        KernelBrowserSession(pool="rl")
    assert events == [
        ("acquire", "rl", 60),
        ("reset", "s1"),
        ("release", "rl", "s1", False),
        ("close",),
    ]


def test_page_creates_new_page_when_context_is_empty(monkeypatch, browser_modules):
    kernel = browser_modules.kernel
    pw = browser_modules.playwright

    made: list[bool] = []

    def _new_page():
        made.append(True)
        return SimpleNamespace(marker="new-page")

    _patch_page_dependencies(monkeypatch, kernel, pw, pages=[], new_page=_new_page)
    session = KernelBrowserSession(create_kwargs={})
    with session.page() as page:
        assert page.marker == "new-page"
    assert made == [True]


def test_page_close_error_does_not_mask_body_exception(monkeypatch, browser_modules):
    kernel = browser_modules.kernel
    pw = browser_modules.playwright

    _patch_page_dependencies(
        monkeypatch,
        kernel,
        pw,
        pages=[SimpleNamespace()],
        close_error="cdp teardown boom",
    )
    session = KernelBrowserSession(create_kwargs={})
    with pytest.raises(ValueError, match="body boom"), session.page():
        raise ValueError("body boom")
