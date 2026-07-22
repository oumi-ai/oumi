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
    from kernel import Kernel  # pyright: ignore[reportMissingImports]


class KernelBrowserSession:
    """A single Kernel cloud browser session, owned by one rollout."""

    def __init__(
        self,
        *,
        create_kwargs: dict[str, Any] | None = None,
        pool: str | None = None,
        acquire_timeout_seconds: int = 60,
        start_url: str | None = None,
    ) -> None:
        """Create a browser session or acquire one from a pool.

        Args:
            create_kwargs: Arguments forwarded to ``browsers.create``.
            pool: Name of a browser pool to acquire from.
            acquire_timeout_seconds: Pool-mode acquire timeout.
            start_url: Landing URL used when resetting a pooled browser.
        """
        if pool is not None and create_kwargs is not None:
            raise ValueError("create_kwargs cannot be used with a browser pool")

        require_kernel("BrowserExecutableEnvironment")
        from kernel import Kernel  # pyright: ignore[reportMissingImports]

        self._kernel: Kernel = Kernel()
        self._pool = pool
        self._browser_closed = False
        self._closed = False
        try:
            if pool is not None:
                self._browser = self._kernel.browser_pools.acquire(
                    pool, acquire_timeout_seconds=acquire_timeout_seconds
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

    @property
    def live_view_url(self) -> str | None:
        """Return the live-view URL, if available."""
        return self._browser.browser_live_view_url

    def _reset(self, start_url: str | None) -> None:
        """Reset a pooled browser before use."""
        target = start_url or "about:blank"
        code = (
            "await context.clearCookies();\n"
            "await context.clearPermissions();\n"
            "const pages = context.pages();\n"
            "const page = pages.length > 0 ? pages[0] : await context.newPage();\n"
            "for (const existingPage of pages) {\n"
            "  await existingPage.evaluate(async () => {\n"
            "    try { localStorage.clear(); } catch {}\n"
            "    try { sessionStorage.clear(); } catch {}\n"
            "    try {\n"
            "      const databases = await indexedDB.databases();\n"
            "      await Promise.all(databases.map(({ name }) => "
            "new Promise((resolve) => {\n"
            "        if (!name) { resolve(); return; }\n"
            "        const request = indexedDB.deleteDatabase(name);\n"
            "        request.onsuccess = request.onerror = "
            "request.onblocked = resolve;\n"
            "      })));\n"
            "    } catch {}\n"
            "    try {\n"
            "      const cacheNames = await caches.keys();\n"
            "      await Promise.all(cacheNames.map((name) => caches.delete(name)));\n"
            "    } catch {}\n"
            "    try {\n"
            "      const registrations = await "
            "navigator.serviceWorker.getRegistrations();\n"
            "      await Promise.all(registrations.map((registration) => "
            "registration.unregister()));\n"
            "    } catch {}\n"
            "  });\n"
            "}\n"
            "for (let i = 1; i < pages.length; i++) { await pages[i].close(); }\n"
            f"await page.goto({json.dumps(target)}, {{ waitUntil: 'load' }});"
        )
        response = self._kernel.browsers.playwright.execute(
            id=self.session_id, code=code, timeout_sec=15
        )
        if not response.success:
            raise RuntimeError(
                "Kernel browser reset failed for session "
                f"{self.session_id}: {response.error}"
            )

    @contextmanager
    def page(self) -> Iterator[Any]:
        """Yield a Playwright page connected to the browser over CDP."""
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
        if not self._browser_closed:
            if self._pool is not None:
                self._kernel.browser_pools.release(
                    self._pool, session_id=self.session_id, reuse=True
                )
            else:
                self._kernel.browsers.delete_by_id(self.session_id)
            self._browser_closed = True
        self._kernel.close()
        self._closed = True
