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

"""A per-rollout Kernel browser, driven over Playwright."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, TypeVar

from oumi.core.configs.params.tool_params import ToolEnvironmentError
from oumi.utils.logging import logger
from oumi.utils.packaging import require_browser

if TYPE_CHECKING:
    from kernel.types import (  # pyright: ignore[reportMissingImports]
        BrowserCreateResponse,
    )
    from playwright.sync_api import (  # pyright: ignore[reportMissingImports]
        Page,
        Playwright,
    )

_T = TypeVar("_T")

_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="oumi-playwright")
_playwright: Playwright | None = None


def run_on_playwright_thread(fn: Callable[[], _T]) -> _T:
    """Run ``fn`` on the one thread in this process that drives Playwright.

    Playwright's sync API is bound to the thread that started it, and it refuses to
    run on a thread with a running asyncio loop, which is where verl's agent loop
    calls tools. ``KernelBrowserSession.page()`` only works inside such a call.
    """
    return _EXECUTOR.submit(fn).result()


class KernelBrowserSession:
    """A Kernel cloud browser owned by one rollout, created on first use.

    The browser is created by the first ``page()`` call rather than at
    construction. ``ToolRouter`` builds a throwaway env just to read
    ``requires_isolation()``, and the synthesizer builds every sample's env before
    planning, so creating it eagerly would bill one browser for nothing and leave
    the rest idle long enough for Kernel to delete them.
    """

    def __init__(self, create_kwargs: dict[str, Any] | None = None) -> None:
        """Configure the session with arguments for ``browsers.create``.

        Raises:
            ImportError: If the ``oumi[browser]`` extra is not installed.
        """
        require_browser("BrowserExecutableEnvironment")
        import kernel  # pyright: ignore[reportMissingImports]

        # Constructing the client checks KERNEL_API_KEY, so a missing key fails
        # when the env is built rather than at the first tool call.
        self._kernel = kernel.Kernel()
        self._api_error: type[Exception] = kernel.APIError
        self._create_kwargs = dict(create_kwargs or {})
        self._browser: BrowserCreateResponse | None = None
        self._closed = False

    @contextmanager
    def page(self) -> Iterator[Page]:
        """Yield the browser's first page over a new CDP connection.

        Must run inside ``run_on_playwright_thread``. The connection is closed
        after each use on purpose: Kernel puts a browser with no CDP client into
        standby, which is not billed, so the browser costs nothing while the model
        works between tool calls.

        Raises:
            RuntimeError: If the session is closed.
            ToolEnvironmentError: If the browser can't be created or reached.
        """
        if self._closed:
            raise RuntimeError("Kernel browser session is already closed.")
        from playwright import sync_api  # pyright: ignore[reportMissingImports]

        global _playwright
        if _playwright is None:
            _playwright = sync_api.sync_playwright().start()
        if self._browser is None:
            try:
                self._browser = self._kernel.browsers.create(**self._create_kwargs)
            except self._api_error as e:
                raise ToolEnvironmentError(
                    f"Kernel could not create a browser: {e}"
                ) from e
        try:
            cdp = _playwright.chromium.connect_over_cdp(self._browser.cdp_ws_url)
        except sync_api.Error as e:
            raise ToolEnvironmentError(
                f"Could not connect to Kernel browser {self._browser.session_id}. "
                "Kernel deletes a browser once timeout_seconds pass in standby, so "
                "it may have been deleted between tool calls; raise timeout_seconds "
                "if this repeats."
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
            cdp.close()

    def close(self) -> None:
        """Delete the browser, if one was created.

        A Kernel API error is logged, not raised: Kernel deletes a leaked browser
        once its timeout runs out, while raising would make the synthesizer drop
        a finished run, since it raises cleanup errors before writing any records.
        """
        if self._closed:
            return
        self._closed = True
        try:
            if self._browser is not None:
                self._kernel.browsers.delete_by_id(self._browser.session_id)
        except self._api_error:
            logger.warning(
                "Kernel browser: failed to delete session %s; it will leak until "
                "Kernel's timeout deletes it.",
                self._browser.session_id if self._browser else None,
                exc_info=True,
            )
        finally:
            self._kernel.close()
