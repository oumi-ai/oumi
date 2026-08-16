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

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from oumi.environments import browser_session
from oumi.environments.browser_session import KernelBrowserSession
from oumi.utils import packaging


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
            return SimpleNamespace(session_id="s1", cdp_ws_url="ws://x")

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
) -> list[int]:
    """Patch the CDP seam; returns a list appended to on every cdp.close()."""
    context = SimpleNamespace(pages=pages)
    if new_page is not None:
        context.new_page = new_page
    closes: list[int] = []

    class _CDP:
        contexts = [context]

        def close(self) -> None:
            closes.append(1)
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
                create=lambda **_: SimpleNamespace(session_id="s1", cdp_ws_url="ws://x")
            ),
            close=lambda: None,
        ),
    )
    return closes


def test_create_mode_close_deletes_once_and_always_closes_client(
    monkeypatch, browser_modules
):
    kernel = browser_modules.kernel

    class _Browsers:
        def __init__(self) -> None:
            self.deletes = 0
            self.raise_once = False

        def create(self, **kwargs: Any):
            return SimpleNamespace(session_id="s1", cdp_ws_url="ws://x")

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


def test_close_logs_when_the_session_leaks(monkeypatch, browser_modules, caplog):
    kernel = browser_modules.kernel

    def raise_delete_error(_: str) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(
        kernel,
        "Kernel",
        lambda: SimpleNamespace(
            browsers=SimpleNamespace(
                create=lambda **_: SimpleNamespace(
                    session_id="s1", cdp_ws_url="ws://x"
                ),
                delete_by_id=raise_delete_error,
            ),
            close=lambda: None,
        ),
    )
    session = KernelBrowserSession(create_kwargs={})
    with pytest.raises(RuntimeError), caplog.at_level("WARNING"):
        session.close()
    assert "leak" in caplog.text


def test_direct_construction_rejects_pool_with_create_kwargs(browser_modules):
    with pytest.raises(ValueError, match="create_kwargs"):
        KernelBrowserSession(pool="rl", create_kwargs={"headless": True})


def test_construction_without_kernel_installed_points_at_the_extra(monkeypatch):
    """Missing-dependency errors must direct users to the browser extra."""
    import importlib

    def raise_import_error(name: str) -> None:
        raise ImportError(name)

    monkeypatch.setattr(
        importlib,
        "import_module",
        raise_import_error,
    )
    packaging._is_module_available.cache_clear()
    try:
        with pytest.raises(ImportError, match=r"oumi\[browser\]"):
            KernelBrowserSession(create_kwargs={})
    finally:
        packaging._is_module_available.cache_clear()


@pytest.mark.parametrize(
    ("kwargs", "expected_reuse"),
    [
        ({}, True),
        ({"reset_error": "reset boom"}, False),
        ({"reset_success": False}, False),
    ],
    ids=["clean-reset-reuses", "reset-raised", "reset-reported-failure"],
)
def test_pool_mode_releases_with_reuse_matching_reset_outcome(
    monkeypatch, browser_modules, kwargs, expected_reuse
):
    kernel = browser_modules.kernel
    events: list[tuple] = []
    _patch_pool_kernel(monkeypatch, kernel, events, **kwargs)

    if expected_reuse:
        KernelBrowserSession(pool="rl", acquire_timeout_seconds=30).close()
        acquire_timeout = 30
    else:
        with pytest.raises(RuntimeError):
            KernelBrowserSession(pool="rl")
        acquire_timeout = 60

    assert events == [
        ("acquire", "rl", acquire_timeout),
        ("reset", "s1"),
        ("release", "rl", "s1", expected_reuse),
        ("close",),
    ]


def test_pool_mode_reset_navigates_to_start_url(monkeypatch, browser_modules):
    kernel = browser_modules.kernel
    reset_codes = _patch_pool_kernel(monkeypatch, kernel, [])
    KernelBrowserSession(pool="rl", start_url="https://ex.com").close()
    assert 'page.goto("https://ex.com"' in reset_codes[0]


def test_pool_mode_reset_defaults_to_about_blank(monkeypatch, browser_modules):
    kernel = browser_modules.kernel
    reset_codes = _patch_pool_kernel(monkeypatch, kernel, [])
    KernelBrowserSession(pool="rl").close()
    assert 'page.goto("about:blank"' in reset_codes[0]


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


def test_page_yields_existing_cdp_page_and_closes_the_client(
    monkeypatch, browser_modules
):
    kernel = browser_modules.kernel
    pw = browser_modules.playwright
    closes = _patch_page_dependencies(
        monkeypatch, kernel, pw, pages=[SimpleNamespace(marker="cdp-page")]
    )
    session = KernelBrowserSession(create_kwargs={})
    with session.page() as page:
        assert getattr(page, "marker") == "cdp-page"
    assert closes == [1]


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
        assert getattr(page, "marker") == "new-page"
    assert made == [True]


def test_page_close_error_does_not_mask_body_exception(monkeypatch, browser_modules):
    kernel = browser_modules.kernel
    pw = browser_modules.playwright
    closes = _patch_page_dependencies(
        monkeypatch,
        kernel,
        pw,
        pages=[SimpleNamespace()],
        close_error="cdp teardown boom",
    )
    session = KernelBrowserSession(create_kwargs={})
    with pytest.raises(ValueError, match="body boom"), session.page():
        raise ValueError("body boom")
    assert closes == [1]
