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
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, TypeVar

from oumi.utils.logging import logger
from oumi.utils.packaging import require_kernel, require_playwright

if TYPE_CHECKING:
    from playwright.sync_api import (  # pyright: ignore[reportMissingImports]
        Page,
        Playwright,
    )

_T = TypeVar("_T")

_DEFAULT_ACQUIRE_TIMEOUT_SECONDS = 60
# Extra HTTP read time on top of the acquire long-poll, so the client doesn't give
# up (and retry the POST) while the server is still waiting for a free browser.
_ACQUIRE_HTTP_MARGIN_SECONDS = 15
_RESET_TIMEOUT_SECONDS = 15

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


class _PlaywrightThread:
    """The one thread in this process that drives Playwright.

    Playwright's sync API is bound to the thread that started it, and it refuses to
    run on a thread with a running asyncio loop, which is where verl's agent loop
    calls tools. Sending every browser call to this thread makes the env usable
    from any caller and keeps one Playwright driver alive for the process instead
    of starting a new one for every tool call.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._executor: ThreadPoolExecutor | None = None
        self._thread_id: int | None = None
        self._playwright: Playwright | None = None

    def run(self, fn: Callable[[], _T]) -> _T:
        """Run ``fn`` on the Playwright thread and return its result."""
        if threading.get_ident() == self._thread_id:
            # Already on the thread (e.g. a GC finalizer closing a session); waiting
            # on our own queue would deadlock.
            return fn()
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="oumi-playwright"
                )
            executor = self._executor
        return executor.submit(self._call, fn).result()

    def _call(self, fn: Callable[[], _T]) -> _T:
        self._thread_id = threading.get_ident()
        return fn()

    def playwright(self) -> Playwright:
        """Return the process-wide Playwright instance, starting it on first use."""
        if threading.get_ident() != self._thread_id:
            raise RuntimeError(
                "Playwright can only be used on the Playwright thread; wrap the call "
                "in run_on_playwright_thread()."
            )
        if self._playwright is None:
            from playwright.sync_api import (  # pyright: ignore[reportMissingImports]
                sync_playwright,
            )

            self._playwright = sync_playwright().start()
        return self._playwright


_PLAYWRIGHT_THREAD = _PlaywrightThread()


def run_on_playwright_thread(fn: Callable[[], _T]) -> _T:
    """Run ``fn`` on the process-wide Playwright thread and return its result.

    ``KernelBrowserSession.page()`` only works inside such a call.
    """
    return _PLAYWRIGHT_THREAD.run(fn)


class KernelBrowserSession:
    """A Kernel cloud browser owned by one rollout, created on first use.

    The browser is created, or acquired from a pool, by the first ``page()`` call
    rather than at construction. ``ToolRouter`` builds a throwaway env just to read
    ``requires_isolation()``, and the synthesizer builds every sample's env before
    planning, so opening the browser eagerly would bill one browser for nothing
    and leave the rest idle long enough for Kernel to delete them.
    """

    def __init__(
        self,
        *,
        create_kwargs: dict[str, Any] | None = None,
        pool: str | None = None,
        acquire_timeout_seconds: int | None = None,
        start_url: str | None = None,
    ) -> None:
        """Configure the session. No browser exists until the first ``page()``.

        Args:
            create_kwargs: Arguments forwarded to ``browsers.create``.
            pool: Name of a browser pool to acquire from.
            acquire_timeout_seconds: Pool-mode acquire timeout; defaults to
                ``_DEFAULT_ACQUIRE_TIMEOUT_SECONDS``.
            start_url: Landing URL used when resetting a pooled browser.

        Raises:
            ValueError: If both ``create_kwargs`` and ``pool`` are given.
            ImportError: If the ``oumi[browser]`` extra is not installed.
        """
        if pool is not None and create_kwargs is not None:
            raise ValueError("create_kwargs cannot be used with a browser pool")

        require_kernel("BrowserExecutableEnvironment")
        require_playwright("BrowserExecutableEnvironment")
        import kernel  # pyright: ignore[reportMissingImports]

        # Constructing the client checks KERNEL_API_KEY, so a missing key fails
        # when the env is built rather than at the first tool call.
        self._kernel = kernel.Kernel()
        self._api_error: type[Exception] = kernel.APIError
        self._create_kwargs = dict(create_kwargs or {})
        self._pool = pool
        self._acquire_timeout_seconds = (
            acquire_timeout_seconds
            if acquire_timeout_seconds is not None
            else _DEFAULT_ACQUIRE_TIMEOUT_SECONDS
        )
        self._start_url = start_url
        self._browser: Any = None
        self._closed = False

    def _open(self) -> Any:
        """Create a browser, or acquire and reset one from the pool."""
        if self._pool is None:
            return self._kernel.browsers.create(**self._create_kwargs)

        browser = self._kernel.browser_pools.acquire(
            self._pool,
            acquire_timeout_seconds=self._acquire_timeout_seconds,
            timeout=self._acquire_timeout_seconds + _ACQUIRE_HTTP_MARGIN_SECONDS,
        )
        # The API answers 204 No Content when no browser frees up in time, which
        # the SDK hands back as an empty string.
        if not browser:
            raise TimeoutError(
                f"No browser in Kernel pool {self._pool!r} became available within "
                f"{self._acquire_timeout_seconds}s. Each sample holds its browser "
                "until the run ends, so size the pool to at least num_samples or "
                "raise acquire_timeout_seconds."
            )
        try:
            self._reset(browser.session_id)
        except BaseException:
            # Don't hand a half-reset browser to the next rollout.
            self._free(browser.session_id, reuse=False)
            raise
        return browser

    def _reset(self, session_id: str) -> None:
        """Return a pooled browser to a clean slate on ``start_url``."""
        target = json.dumps(self._start_url or "about:blank")
        # Navigate after the clears, not via acquire(start_url=...), so the landing
        # page isn't the one we wipe storage out from under.
        code = _RESET_JS + f"await page.goto({target}, {{ waitUntil: 'load' }});"
        response = self._kernel.browsers.playwright.execute(
            id=session_id, code=code, timeout_sec=_RESET_TIMEOUT_SECONDS
        )
        if not response.success:
            raise RuntimeError(
                f"Kernel browser reset failed for session {session_id}: "
                f"{response.error}"
            )

    def _free(self, session_id: str, *, reuse: bool) -> None:
        """Release or delete a browser, logging rather than raising API errors.

        Freeing is best effort: Kernel deletes a leaked browser once its timeout
        runs out, while raising here would make the synthesizer drop a finished
        run, since it raises cleanup errors before writing any records.
        """
        try:
            if self._pool is not None:
                self._kernel.browser_pools.release(
                    self._pool, session_id=session_id, reuse=reuse
                )
            else:
                self._kernel.browsers.delete_by_id(session_id)
        except self._api_error:
            logger.warning(
                "Kernel browser: failed to free session %s; it will leak until "
                "Kernel's timeout deletes it.",
                session_id,
                exc_info=True,
            )

    @contextmanager
    def page(self) -> Iterator[Page]:
        """Yield the browser's first page over a new CDP connection.

        Must run inside ``run_on_playwright_thread``. The CDP connection is closed
        after each use on purpose: Kernel puts a browser with no CDP client into
        standby, which is not billed, so the browser costs nothing while the model
        works between tool calls. Binds the context's first page and closes any
        tab the model opens (a target=_blank click), so popups are never driven.

        Raises:
            RuntimeError: If the session is closed, or the browser can no longer
                be reached.
        """
        if self._closed:
            raise RuntimeError("Kernel browser session is already closed.")
        from playwright.sync_api import (  # pyright: ignore[reportMissingImports]
            Error as PlaywrightError,
        )

        playwright = _PLAYWRIGHT_THREAD.playwright()
        if self._browser is None:
            self._browser = self._open()
        try:
            cdp = playwright.chromium.connect_over_cdp(self._browser.cdp_ws_url)
        except PlaywrightError as e:
            raise RuntimeError(
                f"Could not connect to Kernel browser {self._browser.session_id}. "
                "Kernel deletes a browser once timeout_seconds pass in standby, so "
                "it may have been deleted between tool calls; raise timeout_seconds "
                "(or the pool's timeout) if this repeats."
            ) from e
        try:
            context = cdp.contexts[0]
            page = context.pages[0] if context.pages else context.new_page()
            yield page
            # Tab order isn't stable across connections, so a tab the model opened
            # (target=_blank) could be pages[0] next call; close it now.
            for extra in context.pages:
                if extra != page:
                    extra.close()
        finally:
            try:
                cdp.close()
            except PlaywrightError:
                logger.warning(
                    "Kernel browser: CDP client close failed during teardown.",
                    exc_info=True,
                )

    def close(self) -> None:
        """Release a pooled browser or delete a created one, if one was opened."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._browser is not None:
                self._free(self._browser.session_id, reuse=True)
        finally:
            self._kernel.close()
