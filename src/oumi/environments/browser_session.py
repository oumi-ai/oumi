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

A session is either created on demand (``browsers.create``, deleted on close) or
acquired from a warm **browser pool** (``browser_pools.acquire``, released back on
close). Each session is a single-tenant Kernel cloud browser (a microVM); one
rollout owns one. ``page()`` yields a live Playwright page connected over CDP for
the duration of one tool call — executors receive that page as their ``context``.

Pool mode (set ``pool``) is the throughput path for RL/synthesis: acquisition is
served from pre-warmed browsers, and on release the browser goes *back* to the
pool for reuse. Because pooled browsers are reused, ``_reset`` returns the browser
to a clean slate on acquire (close stray tabs, land on ``start_url``); if that
reset fails the browser is released with ``reuse=False`` so the pool refills a
fresh one.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from oumi.utils.logging import logger
from oumi.utils.packaging import require_kernel, require_playwright

if TYPE_CHECKING:
    from kernel import Kernel  # pyright: ignore[reportMissingImports]


class KernelBrowserSession:
    """A single Kernel cloud browser session, owned by one rollout.

    Create mode (``pool`` is None): ``browsers.create`` on open, ``delete_by_id``
    on close. Pool mode (``pool`` set): ``browser_pools.acquire`` on open,
    ``browser_pools.release`` back to the pool on close.
    """

    def __init__(
        self,
        *,
        create_kwargs: dict[str, Any] | None = None,
        pool: str | None = None,
        acquire_timeout_seconds: int = 60,
        start_url: str | None = None,
    ) -> None:
        """Open a session by creating one or acquiring from a pool.

        Args:
            create_kwargs: Forwarded to ``browsers.create`` in create mode (e.g.
                ``start_url``, ``headless``, ``stealth``, ``profile``,
                ``proxy_id``, ``viewport``, ``timeout_seconds``). Ignored in pool
                mode. The API key is read from ``KERNEL_API_KEY`` by the SDK.
            pool: Name of a warm browser pool to acquire from. When set, the
                session acquires/releases instead of creating/deleting.
            acquire_timeout_seconds: Pool-mode acquire timeout.
            start_url: Pool-mode landing URL used by the post-acquire reset.
        """
        require_kernel("BrowserExecutableEnvironment")
        from kernel import Kernel  # pyright: ignore[reportMissingImports]

        self._kernel: Kernel = Kernel()
        self._pool = pool
        self._reuse = True
        self._closed = False
        if pool is not None:
            self._browser = self._kernel.browser_pools.acquire(
                pool, acquire_timeout_seconds=acquire_timeout_seconds
            )
            self._reset(start_url)
        else:
            self._browser = self._kernel.browsers.create(**(create_kwargs or {}))

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

    def _reset(self, start_url: str | None) -> None:
        """Return a reused pooled browser to a clean slate.

        Closes stray tabs left by a prior rollout and lands the main page on
        ``start_url`` (or ``about:blank``). On failure the browser is marked for
        release with ``reuse=False`` so the pool refills a fresh one.
        """
        target = start_url or "about:blank"
        code = (
            "const pages = context.pages();\n"
            "for (let i = 1; i < pages.length; i++) { await pages[i].close(); }\n"
            "if (pages.length > 0) {\n"
            f"  await pages[0].goto({json.dumps(target)}, {{ waitUntil: 'load' }});\n"
            "}"
        )
        try:
            self._kernel.browsers.playwright.execute(
                id=self.session_id, code=code, timeout_sec=15
            )
        except Exception:
            logger.warning(
                "Kernel browser: reset on pool acquire failed for session %s; "
                "releasing without reuse.",
                self.session_id,
            )
            self._reuse = False

    @contextmanager
    def page(self) -> Iterator[Any]:
        """Yield a live Playwright page, connected over CDP for the call.

        Uses the browser's existing default context/page. Closing only
        disconnects the CDP client — the remote session and its page state
        persist for the next tool call.
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
                # Swallow teardown errors so a CDP close failure neither masks an
                # executor exception nor fails an otherwise-successful call.
                try:
                    cdp.close()
                except Exception:
                    logger.warning(
                        "Kernel browser: CDP client close failed during teardown."
                    )

    def close(self) -> None:
        """Release the session back to its pool, or delete it. Idempotent.

        ``_closed`` is set only after a successful teardown, so a transient Kernel
        API failure raises (loud) and leaves the session retryable rather than
        leaking the remote microVM behind the guard.
        """
        if self._closed:
            return
        if self._pool is not None:
            self._kernel.browser_pools.release(
                self._pool, session_id=self.session_id, reuse=self._reuse
            )
        else:
            self._kernel.browsers.delete_by_id(self.session_id)
        self._closed = True
