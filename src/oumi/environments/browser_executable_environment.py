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
from typing import Any

from oumi.core.configs.params.base_params import BaseParams
from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.registry import register_environment
from oumi.environments.browser_session import KernelBrowserSession
from oumi.environments.executable_environment import ExecutableEnvironment
from oumi.environments.executable_tool import ExecutableTool
from oumi.environments.utils import parse_env_kwargs

#: ``env_kwargs`` fields that select/parameterize the browser pool. Every other
#: dataclass field is a ``browsers.create`` kwarg forwarded in create mode, so a
#: new create field added to the kwargs class needs no change here.
_POOL_FIELDS = ("pool", "acquire_timeout_seconds")


@dataclass
class BrowserExecutableEnvironmentKwargs(BaseParams):
    """Type-specific ``env_kwargs`` for :class:`BrowserExecutableEnvironment`.

    Set ``pool`` to acquire from a warm Kernel browser pool (created out of band,
    e.g. ``kernel browser-pool create --name rl-browser --size 50``) and release
    back to it; leave it ``None`` to create a one-off session per rollout. The
    Kernel API key is read from ``KERNEL_API_KEY`` — never from config.
    """

    pool: str | None = None
    acquire_timeout_seconds: int = 60
    start_url: str | None = None
    headless: bool | None = None
    stealth: bool | None = None
    profile: dict[str, Any] | None = None
    proxy_id: str | None = None
    viewport: dict[str, int] | None = None
    timeout_seconds: int | None = None


@register_environment("browser")
class BrowserExecutableEnvironment(ExecutableEnvironment):
    """Runs browser-action tools against a per-rollout Kernel cloud browser.

    Each rollout gets its own browser — acquired from a warm pool (``pool`` set)
    or created on demand — so :meth:`requires_isolation` is ``True``. For each
    tool call the env opens a live Playwright page on the session; executors
    receive it as ``context`` and return a ``ToolResult``.
    """

    def __init__(
        self, params: EnvironmentParams, session: KernelBrowserSession
    ) -> None:
        """Bind the env to its params and an already-open Kernel browser session."""
        super().__init__(params)
        self._session = session

    @classmethod
    def from_params(cls, params: EnvironmentParams) -> BrowserExecutableEnvironment:
        """Open a Kernel browser session from ``env_kwargs`` and bind the env.

        Acquires from ``pool`` when set, otherwise creates a session from the
        recognized create fields. A failure wiring the env (e.g. a bad executor
        path) closes the freshly-opened session so it can't leak.
        """
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
        """Each rollout needs its own browser; never share across samples."""
        return True

    @contextmanager
    def _build_execution_context(
        self, tool: ExecutableTool, arguments: dict[str, Any]
    ) -> Iterator[Any]:
        """Yield the live Playwright page passed to the executor as ``context``."""
        with self._session.page() as page:
            yield page

    def close(self) -> None:
        """Tear down the rollout's Kernel browser session."""
        self._session.close()
