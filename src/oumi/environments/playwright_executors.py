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

"""Example browser tool executors for ``BrowserExecutableEnvironment``.

Reference these by dotted path from a tool's ``executor`` field (see
``configs/examples/synthesis/browser_oumi_synth.yaml`` for the full wiring). Each
executor takes ``(arguments, context)`` — where ``context`` is the live Playwright
page the environment bound for this tool call — and returns a ``ToolResult``. Page
state persists on the remote Kernel session across calls, so executors stay
stateless. Needs ``oumi[browser]``.
"""

from __future__ import annotations

from typing import Any

from oumi.core.types.tool_call import ToolResult

#: Cap returned page text so a single observation can't blow the context window.
_DEFAULT_MAX_CHARS = 8000
_DEFAULT_TIMEOUT_MS = 5000


def navigate(arguments: dict[str, Any], context: Any) -> ToolResult:
    """Navigate to ``arguments['url']``; returns the resolved url and page title."""
    page = context
    page.goto(arguments["url"], wait_until=arguments.get("wait_until", "load"))
    return ToolResult(output={"url": page.url, "title": page.title()})


def click(arguments: dict[str, Any], context: Any) -> ToolResult:
    """Click the element matching ``arguments['selector']``."""
    page = context
    timeout_ms = arguments.get("timeout_ms", _DEFAULT_TIMEOUT_MS)
    page.click(arguments["selector"], timeout=timeout_ms)
    return ToolResult(output={"clicked": arguments["selector"], "url": page.url})


def type_text(arguments: dict[str, Any], context: Any) -> ToolResult:
    """Fill ``arguments['selector']`` with ``arguments['text']``."""
    page = context
    timeout_ms = arguments.get("timeout_ms", _DEFAULT_TIMEOUT_MS)
    page.fill(arguments["selector"], arguments["text"], timeout=timeout_ms)
    return ToolResult(output={"typed_into": arguments["selector"]})


def read_text(arguments: dict[str, Any], context: Any) -> ToolResult:
    """Read inner text of a selector (defaults to ``body``), truncated.

    The selector wait uses a short timeout so a missing selector fails fast and
    the agent can retry, rather than blocking on Playwright's 30s default.
    """
    page = context
    selector = arguments.get("selector", "body")
    max_chars = arguments.get("max_chars", _DEFAULT_MAX_CHARS)
    text = page.inner_text(selector, timeout=_DEFAULT_TIMEOUT_MS)
    return ToolResult(output={"text": text[: max(0, max_chars)], "url": page.url})
