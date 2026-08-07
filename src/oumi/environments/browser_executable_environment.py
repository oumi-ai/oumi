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

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

from oumi.core.configs.params.base_params import BaseParams
from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.registry import register_environment
from oumi.environments.browser_session import KernelBrowserSession
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
        # Each mode ignores the other's kwargs, so a misplaced one would silently
        # do nothing. Buckets derive from the field list to stay exhaustive.
        if self.pool is None:
            misplaced = [
                f for f in _POOL_FIELDS if f != "pool" and getattr(self, f) is not None
            ]
            mode, hint = "Create", "set pool to use them"
        else:
            misplaced = [
                f.name
                for f in fields(self)
                if f.name not in (*_POOL_FIELDS, "start_url")
                and getattr(self, f.name) is not None
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
                field.name: getattr(kwargs, field.name)
                for field in fields(kwargs)
                if field.name not in _POOL_FIELDS
                and getattr(kwargs, field.name) is not None
            }
            session = KernelBrowserSession(create_kwargs=create_kwargs)
        try:
            return cls(params, session)
        except BaseException:
            session.close()
            raise

    def requires_isolation(self) -> bool:
        """A Kernel session is single-tenant, so never share one across samples.

        Consumers build every per-sample env up front, so one live (billable)
        Kernel browser per sample is open for the whole run.
        """
        return True

    def is_replayable(self) -> bool:
        """Browser actions mutate live remote page state, so never replay a batch."""
        return False

    @contextmanager
    def _build_execution_context(
        self, tool: ExecutableTool, arguments: dict[str, Any]
    ) -> Iterator[Page]:
        """Yield the Playwright page used as the executor context.

        Entered once per tool call, so each call opens a fresh Playwright driver
        and CDP connection; the state that matters lives on the remote browser.
        """
        with self._session.page() as page:
            yield page

    def close(self) -> None:
        """Close the browser session."""
        self._session.close()
