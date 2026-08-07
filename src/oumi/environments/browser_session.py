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

"""Lifecycle management for a per-rollout Kernel browser session."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from oumi.utils.logging import logger
from oumi.utils.packaging import require_kernel, require_playwright

if TYPE_CHECKING:
    from playwright.sync_api import Page  # pyright: ignore[reportMissingImports]

_DEFAULT_ACQUIRE_TIMEOUT_SECONDS = 60

# Clears cookies/permissions context-wide, but storage is origin-scoped, so only
# origins with a page still open get purged. Kernel's playwright.execute
# pre-declares `page`, `context` and `browser` — redeclaring one is a hard error.
_RESET_JS = """
await context.clearCookies();
await context.clearPermissions();
for (const existingPage of context.pages()) {
  await existingPage.evaluate(async () => {
    try { localStorage.clear(); } catch {}
    try { sessionStorage.clear(); } catch {}
    try {
      const dbs = await indexedDB.databases();
      await Promise.all(dbs.map(({ name }) => new Promise((resolve) => {
        if (!name) { resolve(); return; }
        const request = indexedDB.deleteDatabase(name);
        request.onsuccess = request.onerror = request.onblocked = resolve;
      })));
    } catch {}
    try {
      const keys = await caches.keys();
      await Promise.all(keys.map((key) => caches.delete(key)));
    } catch {}
    try {
      const regs = await navigator.serviceWorker.getRegistrations();
      await Promise.all(regs.map((reg) => reg.unregister()));
    } catch {}
  });
}
for (const existingPage of context.pages()) {
  if (existingPage !== page) { await existingPage.close(); }
}
"""


class KernelBrowserSession:
    """A single Kernel cloud browser session, owned by one rollout."""

    def __init__(
        self,
        *,
        create_kwargs: dict[str, Any] | None = None,
        pool: str | None = None,
        acquire_timeout_seconds: int | None = None,
        start_url: str | None = None,
    ) -> None:
        """Create a browser session or acquire one from a pool.

        Args:
            create_kwargs: Arguments forwarded to ``browsers.create``.
            pool: Name of a browser pool to acquire from.
            acquire_timeout_seconds: Pool-mode acquire timeout; defaults to
                ``_DEFAULT_ACQUIRE_TIMEOUT_SECONDS``.
            start_url: Landing URL used when resetting a pooled browser.
        """
        if pool is not None and create_kwargs is not None:
            raise ValueError("create_kwargs cannot be used with a browser pool")

        require_kernel("BrowserExecutableEnvironment")
        from kernel import Kernel  # pyright: ignore[reportMissingImports]

        self._kernel = Kernel()
        self._pool = pool
        self._closed = False
        try:
            if pool is not None:
                self._browser = self._kernel.browser_pools.acquire(
                    pool,
                    acquire_timeout_seconds=(
                        acquire_timeout_seconds
                        if acquire_timeout_seconds is not None
                        else _DEFAULT_ACQUIRE_TIMEOUT_SECONDS
                    ),
                )
                try:
                    self._reset(start_url)
                except BaseException:
                    self._kernel.browser_pools.release(
                        pool, session_id=self._browser.session_id, reuse=False
                    )
                    raise
            else:
                self._browser = self._kernel.browsers.create(**(create_kwargs or {}))
        except BaseException:
            self._kernel.close()
            raise

    @property
    def session_id(self) -> str:
        """Return the Kernel session ID."""
        return self._browser.session_id

    @property
    def cdp_ws_url(self) -> str:
        """Return the CDP websocket URL."""
        return self._browser.cdp_ws_url

    def _reset(self, start_url: str | None) -> None:
        """Reset a pooled browser before use."""
        target = json.dumps(start_url or "about:blank")
        # Navigate after the clears, not via acquire(start_url=...), so the landing
        # page isn't the one we wipe storage out from under.
        code = _RESET_JS + f"await page.goto({target}, {{ waitUntil: 'load' }});"
        response = self._kernel.browsers.playwright.execute(
            id=self.session_id, code=code, timeout_sec=15
        )
        if not response.success:
            raise RuntimeError(
                "Kernel browser reset failed for session "
                f"{self.session_id}: {response.error}"
            )

    @contextmanager
    def page(self) -> Iterator[Page]:
        """Yield a Playwright page connected to the browser over CDP.

        Binds the context's first page, so a tab the model opens (a target=_blank
        click) is never driven.
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
                try:
                    cdp.close()
                except Exception:
                    logger.warning(
                        "Kernel browser: CDP client close failed during teardown.",
                        exc_info=True,
                    )

    def close(self) -> None:
        """Release a pooled session or delete a one-off session."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._pool is not None:
                self._kernel.browser_pools.release(
                    self._pool, session_id=self.session_id, reuse=True
                )
            else:
                self._kernel.browsers.delete_by_id(self.session_id)
        except BaseException:
            # Callers close in a finally and upstream may suppress, so log here or a
            # leaked billable session is invisible. Not retried: the client is gone.
            logger.warning(
                "Kernel browser: failed to release session %s; it will leak until "
                "Kernel's inactivity timeout reaps it.",
                self.session_id,
                exc_info=True,
            )
            raise
        finally:
            self._kernel.close()
