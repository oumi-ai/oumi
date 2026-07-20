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
from types import SimpleNamespace
from typing import Any

import pytest

from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.configs.params.tool_params import ToolError
from oumi.core.types.tool_call import ToolResult
from oumi.environments.browser_executable_environment import (
    BrowserExecutableEnvironment,
)
from oumi.environments.browser_session import KernelBrowserSession
from oumi.environments.executable_tool import ExecutableTool

_SESSION = "oumi.environments.browser_executable_environment.KernelBrowserSession"


class _FakePage:
    marker = "fake-page"


class _FakeSession:
    """Stand-in for KernelBrowserSession that needs no Kernel API call."""

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
    """Executor that echoes its arguments and the bound page marker."""
    return ToolResult(output={"args": arguments, "page": context.marker})


def returns_dict(arguments: dict[str, Any], context: Any) -> dict:
    """Executor that (wrongly) returns a plain dict instead of a ToolResult."""
    return {"not": "a ToolResult"}


def _params(**kwargs: Any) -> EnvironmentParams:
    return EnvironmentParams(
        id="b", name="b", description="d", env_type="browser", **kwargs
    )


def _tool(executor: str) -> ExecutableTool:
    return ExecutableTool(id="t", name="t", description="d", executor=executor)


def _env(session: Any, **params_kwargs: Any) -> BrowserExecutableEnvironment:
    return BrowserExecutableEnvironment(_params(**params_kwargs), session)


# --- env dispatch (direct construction with a fake session) ------------------


def test_requires_isolation_is_true():
    """Each rollout must get its own browser."""
    assert _env(_FakeSession()).requires_isolation() is True


def test_step_passes_live_page_as_context():
    """Executors receive the bound page as `context` and return a ToolResult."""
    env = _env(_FakeSession(), tools=[_tool(f"{__name__}.echo")])
    [result] = env.step([("t", {"x": 1})])
    assert result.output == {"args": {"x": 1}, "page": "fake-page"}


def test_executor_must_return_toolresult():
    """A non-ToolResult executor return is rejected by the base contract."""
    env = _env(_FakeSession(), tools=[_tool(f"{__name__}.returns_dict")])
    with pytest.raises(ToolError):
        env.step([("t", {})])


def test_close_delegates_to_session():
    """Env close() tears down the session."""
    session = _FakeSession()
    _env(session).close()
    assert session.closed is True


# --- from_params: create vs pool routing, kwargs validation ------------------


def test_from_params_create_mode_forwards_create_fields(monkeypatch):
    """No pool -> create mode; only recognized create fields are forwarded."""
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
    """A `pool` name routes to acquire/release with its acquire params."""
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
    """Unknown env_kwargs (e.g. a misplaced api_key or a typo) are rejected."""
    with pytest.raises(ValueError):
        BrowserExecutableEnvironment.from_params(_params(env_kwargs={"api_key": "x"}))
    with pytest.raises(ValueError):
        BrowserExecutableEnvironment.from_params(_params(env_kwargs={"headles": True}))


def test_from_params_closes_session_if_executor_wiring_fails(monkeypatch):
    """A bad executor path must close (not leak) the freshly-opened session."""
    created: list[_FakeSession] = []

    def _factory(**kwargs: Any) -> _FakeSession:
        session = _FakeSession(**kwargs)
        created.append(session)
        return session

    monkeypatch.setattr(_SESSION, _factory)
    with pytest.raises(ValueError):
        BrowserExecutableEnvironment.from_params(_params(tools=[_tool("not_dotted")]))
    assert created and created[0].closed is True


# --- KernelBrowserSession lifecycle (fake Kernel client) ---------------------


def test_create_mode_close_deletes_once_and_retries(monkeypatch):
    """Create mode: close() deletes once, stays retryable after a failed delete."""
    pytest.importorskip("kernel")
    import kernel  # pyright: ignore[reportMissingImports]

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
    monkeypatch.setattr(kernel, "Kernel", lambda: SimpleNamespace(browsers=browsers))
    session = KernelBrowserSession(create_kwargs={})

    browsers.raise_once = True
    with pytest.raises(RuntimeError):
        session.close()
    session.close()
    session.close()
    assert browsers.deletes == 2


def test_pool_mode_acquires_resets_and_releases_with_reuse(monkeypatch):
    """Pool mode: acquire -> reset -> release(reuse=True) back to the pool."""
    pytest.importorskip("kernel")
    import kernel  # pyright: ignore[reportMissingImports]

    events: list[tuple] = []

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
            return SimpleNamespace(success=True)

    monkeypatch.setattr(
        kernel,
        "Kernel",
        lambda: SimpleNamespace(
            browser_pools=_Pools(),
            browsers=SimpleNamespace(playwright=_Playwright()),
        ),
    )
    session = KernelBrowserSession(
        pool="rl", acquire_timeout_seconds=30, start_url="https://ex.com"
    )
    session.close()
    assert events == [
        ("acquire", "rl", 30),
        ("reset", "s1"),
        ("release", "rl", "s1", True),
    ]


def test_pool_mode_reset_failure_releases_without_reuse(monkeypatch):
    """A failed reset on acquire releases the browser with reuse=False."""
    pytest.importorskip("kernel")
    import kernel  # pyright: ignore[reportMissingImports]

    events: list[tuple] = []

    class _Pools:
        def acquire(self, name: str, *, acquire_timeout_seconds: int):
            return SimpleNamespace(
                session_id="s1", cdp_ws_url="ws://x", browser_live_view_url=None
            )

        def release(self, name: str, *, session_id: str, reuse: bool) -> None:
            events.append(("release", reuse))

    class _Playwright:
        def execute(self, *, id: str, code: str, timeout_sec: int):
            raise RuntimeError("reset boom")

    monkeypatch.setattr(
        kernel,
        "Kernel",
        lambda: SimpleNamespace(
            browser_pools=_Pools(),
            browsers=SimpleNamespace(playwright=_Playwright()),
        ),
    )
    KernelBrowserSession(pool="rl").close()
    assert events == [("release", False)]


def test_page_yields_default_cdp_page_and_swallows_close_error(monkeypatch):
    """page() attaches over CDP, yields the default page, and swallows close errors."""
    pytest.importorskip("kernel")
    pytest.importorskip("playwright")
    import kernel  # pyright: ignore[reportMissingImports]
    import playwright.sync_api as pw  # pyright: ignore[reportMissingImports]

    class _Page:
        marker = "cdp-page"

    class _Ctx:
        pages = [_Page()]

    class _CDP:
        contexts = [_Ctx()]

        def close(self) -> None:
            raise RuntimeError("cdp teardown boom")  # must be swallowed

    class _PW:
        chromium = SimpleNamespace(connect_over_cdp=lambda url: _CDP())

        def __enter__(self):
            return self

        def __exit__(self, *a) -> bool:
            return False

    monkeypatch.setattr(pw, "sync_playwright", lambda: _PW())
    monkeypatch.setattr(
        kernel,
        "Kernel",
        lambda: SimpleNamespace(
            browsers=SimpleNamespace(
                create=lambda **k: SimpleNamespace(
                    session_id="s1", cdp_ws_url="ws://x", browser_live_view_url=None
                )
            )
        ),
    )
    session = KernelBrowserSession(create_kwargs={})
    with session.page() as page:  # exiting runs cdp.close(), which raises
        assert page.marker == "cdp-page"


def test_pool_mode_reset_reported_failure_releases_without_reuse(monkeypatch):
    """A reset reporting success=False (no raise) still releases with reuse=False."""
    pytest.importorskip("kernel")
    import kernel  # pyright: ignore[reportMissingImports]

    events: list[tuple] = []

    class _Pools:
        def acquire(self, name: str, *, acquire_timeout_seconds: int):
            return SimpleNamespace(
                session_id="s1", cdp_ws_url="ws://x", browser_live_view_url=None
            )

        def release(self, name: str, *, session_id: str, reuse: bool) -> None:
            events.append(("release", reuse))

    class _Playwright:
        def execute(self, *, id: str, code: str, timeout_sec: int):
            return SimpleNamespace(success=False, error="goto timeout")

    monkeypatch.setattr(
        kernel,
        "Kernel",
        lambda: SimpleNamespace(
            browser_pools=_Pools(),
            browsers=SimpleNamespace(playwright=_Playwright()),
        ),
    )
    KernelBrowserSession(pool="rl").close()
    assert events == [("release", False)]


def test_page_creates_new_page_when_context_is_empty(monkeypatch):
    """page() calls new_page() when the default context has no pages."""
    pytest.importorskip("kernel")
    pytest.importorskip("playwright")
    import kernel  # pyright: ignore[reportMissingImports]
    import playwright.sync_api as pw  # pyright: ignore[reportMissingImports]

    made: list[bool] = []

    class _Ctx:
        pages: list = []

        def new_page(self):
            made.append(True)
            return SimpleNamespace(marker="new-page")

    class _CDP:
        contexts = [_Ctx()]

        def close(self) -> None:
            pass

    class _PW:
        chromium = SimpleNamespace(connect_over_cdp=lambda url: _CDP())

        def __enter__(self):
            return self

        def __exit__(self, *a) -> bool:
            return False

    monkeypatch.setattr(pw, "sync_playwright", lambda: _PW())
    monkeypatch.setattr(
        kernel,
        "Kernel",
        lambda: SimpleNamespace(
            browsers=SimpleNamespace(
                create=lambda **k: SimpleNamespace(
                    session_id="s1", cdp_ws_url="ws://x", browser_live_view_url=None
                )
            )
        ),
    )
    session = KernelBrowserSession(create_kwargs={})
    with session.page() as page:
        assert page.marker == "new-page"
    assert made == [True]


def test_page_close_error_does_not_mask_body_exception(monkeypatch):
    """If the body raises and cdp.close also raises, the body error propagates."""
    pytest.importorskip("kernel")
    pytest.importorskip("playwright")
    import kernel  # pyright: ignore[reportMissingImports]
    import playwright.sync_api as pw  # pyright: ignore[reportMissingImports]

    class _Ctx:
        pages = [SimpleNamespace()]

    class _CDP:
        contexts = [_Ctx()]

        def close(self) -> None:
            raise RuntimeError("cdp teardown boom")

    class _PW:
        chromium = SimpleNamespace(connect_over_cdp=lambda url: _CDP())

        def __enter__(self):
            return self

        def __exit__(self, *a) -> bool:
            return False

    monkeypatch.setattr(pw, "sync_playwright", lambda: _PW())
    monkeypatch.setattr(
        kernel,
        "Kernel",
        lambda: SimpleNamespace(
            browsers=SimpleNamespace(
                create=lambda **k: SimpleNamespace(
                    session_id="s1", cdp_ws_url="ws://x", browser_live_view_url=None
                )
            )
        ),
    )
    session = KernelBrowserSession(create_kwargs={})
    with pytest.raises(ValueError, match="body boom"):
        with session.page():
            raise ValueError("body boom")
