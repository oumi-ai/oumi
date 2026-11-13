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

"""Executable environment backed by a per-rollout Kernel browser session."""

from __future__ import annotations

import functools
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from oumi.core.configs.params.base_params import BaseParams
from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.registry import register_environment
from oumi.core.types.tool_call import ToolResult
from oumi.environments.browser_session import (
    KernelBrowserSession,
    run_on_playwright_thread,
)
from oumi.environments.executable_environment import ExecutableEnvironment
from oumi.environments.executable_tool import ExecutableTool
from oumi.environments.utils import parse_env_kwargs

if TYPE_CHECKING:
    from playwright.sync_api import Page  # pyright: ignore[reportMissingImports]

_POOL_FIELDS = ("pool", "acquire_timeout_seconds")


@dataclass
class BrowserExecutableEnvironmentKwargs(BaseParams):
    """Type-specific ``env_kwargs`` for :class:`BrowserExecutableEnvironment`."""

    pool: str | None = None
    acquire_timeout_seconds: int | None = None
    start_url: str | None = None
    headless: bool | None = None
    stealth: bool | None = None
    profile: dict[str, Any] | None = None
    proxy_id: str | None = None
    viewport: dict[str, int] | None = None
    timeout_seconds: int | None = None

    def __finalize_and_validate__(self) -> None:
        """Reject kwargs that don't apply to the selected mode."""
        if self.profile is not None and self.profile.get("save_changes") is True:
            raise ValueError(
                "Browser profiles with save_changes=True cannot be used in an "
                "isolated rollout environment."
            )

        # Each mode ignores the other's kwargs, so reject misplaced ones rather
        # than silently dropping them.
        if self.pool is None:
            if self.acquire_timeout_seconds is not None:
                raise ValueError("acquire_timeout_seconds requires pool.")
            return
        misplaced = [
            name
            for name, value in self
            if value is not None and name not in (*_POOL_FIELDS, "start_url")
        ]
        if misplaced:
            raise ValueError(
                f"Browser pool mode does not support {misplaced}; configure them "
                "on the pool instead."
            )


@register_environment("browser")
class BrowserExecutableEnvironment(ExecutableEnvironment):
    """Runs browser-action tools against an isolated Kernel browser.

    ``tool.read_only`` is not enforced (a browser has no read-only mode, unlike
    DatabaseExecutableEnvironment), so set ``read_only: false`` on tools that
    change page state.
    """

    def __init__(
        self, params: EnvironmentParams, session: KernelBrowserSession
    ) -> None:
        """Wrap ``session``; its browser opens on the first tool call."""
        super().__init__(params)
        self._session = session

    @classmethod
    def from_params(cls, params: EnvironmentParams) -> BrowserExecutableEnvironment:
        """Build a KernelBrowserSession from ``env_kwargs``."""
        kwargs = parse_env_kwargs(
            BrowserExecutableEnvironmentKwargs,
            params,
            env_label="BrowserExecutableEnvironment",
        )
        if kwargs.pool is not None:
            session = KernelBrowserSession(
                pool=kwargs.pool,
                acquire_timeout_seconds=kwargs.acquire_timeout_seconds,
                start_url=kwargs.start_url,
            )
        else:
            create_kwargs = {
                name: value
                for name, value in kwargs
                if name not in _POOL_FIELDS and value is not None
            }
            session = KernelBrowserSession(create_kwargs=create_kwargs)
        try:
            return cls(params, session)
        except BaseException:
            session.close()
            raise

    def requires_isolation(self) -> bool:
        """Each sample needs its own browser, held until the run ends."""
        return True

    def is_replayable(self) -> bool:
        """Browser actions mutate live remote page state, so never replay a batch."""
        return False

    def step(self, calls: list[tuple[str, dict[str, Any]]]) -> list[ToolResult]:
        """Execute the batch on the Playwright thread (see ``_PlaywrightThread``)."""
        return run_on_playwright_thread(functools.partial(super().step, calls))

    def _build_execution_context(
        self, tool: ExecutableTool, arguments: dict[str, Any]
    ) -> AbstractContextManager[Page]:
        """Return ``session.page()``, entered once per tool call."""
        return self._session.page()

    def close(self) -> None:
        """Close the browser session."""
        self._session.close()
