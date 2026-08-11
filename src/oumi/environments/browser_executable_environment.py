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

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from oumi.core.configs.params.base_params import BaseParams
from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.registry import register_environment
from oumi.environments.browser_session import KernelBrowserSession
from oumi.environments.executable_environment import ExecutableEnvironment
from oumi.environments.executable_tool import ExecutableTool
from oumi.environments.utils import parse_env_kwargs
from oumi.utils.logging import logger

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

        # Each mode ignores the other's kwargs, so a misplaced one would silently
        # do nothing. Create-only kwargs are inferred by excluding _POOL_FIELDS;
        # add new pool-only options there for validation and create-mode forwarding,
        # then explicitly forward them when constructing KernelBrowserSession.
        if self.pool is None:
            misplaced = [
                name
                for name, value in self
                if name in _POOL_FIELDS and name != "pool" and value is not None
            ]
            mode, hint = "Create", "set pool to use them"
        else:
            misplaced = [
                name
                for name, value in self
                if name not in (*_POOL_FIELDS, "start_url") and value is not None
            ]
            mode, hint = "Browser pool", "configure them on the pool instead"
        if misplaced:
            raise ValueError(f"{mode} mode does not support {misplaced}; {hint}.")


@register_environment("browser")
class BrowserExecutableEnvironment(ExecutableEnvironment):
    """Runs browser-action tools against an isolated Kernel browser.

    Unlike the database sibling, ``tool.read_only`` is not enforced here — a
    browser has no read-only mode — so mark mutating tools ``read_only: false``
    for the flag to describe them honestly.
    """

    def __init__(
        self, params: EnvironmentParams, session: KernelBrowserSession
    ) -> None:
        """Bind the env to its params and an already-open Kernel browser session."""
        super().__init__(params)
        self._session = session

    @classmethod
    def from_params(cls, params: EnvironmentParams) -> BrowserExecutableEnvironment:
        """Open a Kernel browser session from ``env_kwargs``."""
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
            try:
                session.close()
            except BaseException:
                logger.warning(
                    "Kernel browser: failed to close session after environment "
                    "initialization failed.",
                    exc_info=True,
                )
            raise

    def requires_isolation(self) -> bool:
        """Require a distinct Kernel browser session for each sample."""
        return True

    def is_replayable(self) -> bool:
        """Browser actions mutate live remote page state, so never replay a batch."""
        return False

    def _build_execution_context(
        self, tool: ExecutableTool, arguments: dict[str, Any]
    ) -> AbstractContextManager[Page]:
        """Return the Playwright page used as the executor context.

        Entered once per tool call, so each call opens a fresh Playwright driver
        and CDP connection; the state that matters lives on the remote browser.
        """
        return self._session.page()

    def close(self) -> None:
        """Close the browser session."""
        self._session.close()
