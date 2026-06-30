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

import importlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from oumi.core.configs.params.environment_params import EnvironmentParams
from oumi.core.registry import register_environment
from oumi.core.types.tool_call import ToolResult
from oumi.environments.browser_session import KernelBrowserSession, using_page
from oumi.environments.executable_environment import ExecutableEnvironment
from oumi.environments.executable_tool import ExecutableTool

#: ``env_kwargs`` keys forwarded to ``kernel.browsers.create``. The API key is
#: deliberately excluded — credentials come from ``KERNEL_API_KEY``, never config.
_BROWSER_CREATE_KEYS = (
    "start_url",
    "headless",
    "stealth",
    "profile",
    "proxy_id",
    "viewport",
    "timeout_seconds",
    "extensions",
    "name",
    "gpu",
    "kiosk_mode",
    "tags",
    "chrome_policy",
)


def _import_executor(dotted: str, tool_id: str) -> Callable[..., Any]:
    """Resolve a dotted import path to a callable."""
    module_path, _, attr = dotted.rpartition(".")
    if not module_path or not attr:
        raise ValueError(
            f"Tool '{tool_id}': executor '{dotted}' must be a dotted import path."
        )
    module = importlib.import_module(module_path)
    executor = getattr(module, attr, None)
    if not callable(executor):
        raise ValueError(
            f"Tool '{tool_id}': executor '{dotted}' did not resolve to a callable."
        )
    return executor


@register_environment("browser")
class BrowserExecutableEnvironment(ExecutableEnvironment):
    """Runs browser-action tools against a per-rollout Kernel cloud browser.

    Each rollout gets its own Kernel session (a single-tenant microVM); see
    :meth:`requires_isolation`. For each tool call the env opens a live page on
    the session and binds it via ``using_page``; executors take their declared
    tool params directly and read the page through ``current_page()``.
    """

    def __init__(
        self, params: EnvironmentParams, session: KernelBrowserSession
    ) -> None:
        """Bind the env to its params and an already-open Kernel browser session."""
        self._params = params
        self._session = session
        self._executors = {
            tool.id: _import_executor(tool.executor, tool.id) for tool in params.tools
        }

    @classmethod
    def from_params(cls, params: EnvironmentParams) -> BrowserExecutableEnvironment:
        """Open a fresh Kernel browser session from ``env_kwargs`` and bind the env."""
        kwargs = dict(params.env_kwargs or {})
        create_kwargs = {k: kwargs[k] for k in _BROWSER_CREATE_KEYS if k in kwargs}
        session = KernelBrowserSession(create_kwargs)
        return cls(params, session)

    def requires_isolation(self) -> bool:
        """Each rollout needs its own microVM session; never share across samples."""
        return True

    @contextmanager
    def _build_execution_context(
        self, tool: ExecutableTool, arguments: dict[str, Any]
    ) -> Iterator[None]:
        """Open a live page on the session and bind it for the executor."""
        with self._session.page() as page, using_page(page):
            yield None

    def _invoke_executor(
        self, executor: Callable[..., Any], arguments: dict[str, Any], ctx: Any
    ) -> Any:
        """Call the executor with unpacked tool params; the page is ambient.

        Executors read the bound page via ``current_page()`` and return a dict
        or ``ToolResult``; a plain return value is wrapped.
        """
        result = executor(**arguments)
        return result if isinstance(result, ToolResult) else ToolResult(output=result)

    def close(self) -> None:
        """Tear down the rollout's Kernel browser session."""
        self._session.close()
