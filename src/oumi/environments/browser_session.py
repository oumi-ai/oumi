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

"""Per-rollout Kernel browser session — the browser-env isolation primitive.

Each session is a single-tenant Kernel cloud browser (a microVM); one rollout
owns one session and tears it down on close. ``page()`` yields a live Playwright
page connected over CDP for the duration of one tool call.

Executors read the bound page via the module-level ``current_page()`` rather than
receiving it as an argument — the ambient-handle pattern mirrors the database
env's ``current_connection()``. The environment binds it with ``using_page`` for
each tool call.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from oumi.utils.packaging import require_kernel, require_playwright

if TYPE_CHECKING:
    from kernel import Kernel  # pyright: ignore[reportMissingImports]


#: The live Playwright page bound for the current tool call. Typed ``Any`` — the
#: concrete type comes from the optional ``playwright`` dependency and is faked in
#: tests; executor authors can annotate locally (``page: Page = current_page()``).
_page_var: ContextVar[Any] = ContextVar("oumi_browser_page")


@contextmanager
def using_page(page: Any) -> Iterator[None]:
    """Bind ``page`` as the active page for the duration of one tool call."""
    token = _page_var.set(page)
    try:
        yield
    finally:
        _page_var.reset(token)


def current_page() -> Any:
    """Return the page bound for the current tool call.

    Raises ``LookupError`` if called outside a ``using_page`` block (i.e. an
    executor invoked outside the environment's execution context).
    """
    return _page_var.get()


class KernelBrowserSession:
    """A single Kernel cloud browser session, owned by one rollout.

    ``__init__`` allocates a fresh session via ``browsers.create``; ``close``
    deletes it. ``page()`` yields a live Playwright page connected over CDP.
    """

    def __init__(self, create_kwargs: dict[str, Any] | None = None) -> None:
        """Open a fresh Kernel browser session.

        Args:
            create_kwargs: Keyword arguments forwarded verbatim to
                ``kernel.browsers.create`` (e.g. ``start_url``, ``headless``,
                ``stealth``, ``profile``, ``proxy_id``, ``viewport``,
                ``timeout_seconds``). The API key is read from ``KERNEL_API_KEY``
                by the SDK and must never be passed here.
        """
        require_kernel("BrowserExecutableEnvironment")
        from kernel import Kernel  # pyright: ignore[reportMissingImports]

        self._kernel: Kernel = Kernel()
        self._browser = self._kernel.browsers.create(**(create_kwargs or {}))
        self._closed = False

    @property
    def session_id(self) -> str:
        """Kernel session id, also used for teardown."""
        return self._browser.session_id

    @property
    def cdp_ws_url(self) -> str:
        """CDP websocket URL; ``page()`` attaches a Playwright driver here."""
        return self._browser.cdp_ws_url

    @property
    def live_view_url(self) -> str | None:
        """Human Live View URL for watching/taking over the session (headful only)."""
        return self._browser.browser_live_view_url

    @contextmanager
    def page(self) -> Iterator[Any]:
        """Yield a live Playwright page, connected over CDP for the call.

        Uses the browser's existing default context/page (Kernel starts every
        session with one). Closing only disconnects the CDP client — the remote
        Kernel session and its page state persist for the next tool call.
        """
        require_playwright("Playwright browser executors")
        from playwright.sync_api import (  # pyright: ignore[reportMissingImports]
            sync_playwright,
        )

        with sync_playwright() as p:
            cdp = p.chromium.connect_over_cdp(self.cdp_ws_url)
            try:
                context = cdp.contexts[0]
                page = context.pages[0] if context.pages else context.new_page()
                yield page
            finally:
                cdp.close()

    def close(self) -> None:
        """Delete the Kernel session. Idempotent — safe to call from ``finally``."""
        if self._closed:
            return
        self._closed = True
        self._kernel.browsers.delete_by_id(self.session_id)
