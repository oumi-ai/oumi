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

Reference these by dotted path from a tool's ``executor`` field, e.g.::

    tools:
      - id: navigate
        name: navigate
        description: Load a URL.
        executor: oumi.environments.playwright_executors.navigate
        parameters:
          type: object
          properties: {url: {type: string}}
          required: [url]

The environment binds the live page for each tool call; executors read it via
``current_page()``, take their declared tool parameters directly, and return a
dict (auto-wrapped into a ``ToolResult``). Requires ``pip install 'oumi[browser]'``.

Page/navigation state persists on the remote Kernel session across tool calls,
so executors are stateless: each call reads the page, acts, and returns.
"""

from __future__ import annotations

from oumi.environments.browser_session import current_page

#: Cap returned page text so a single observation can't blow the context window.
_DEFAULT_MAX_CHARS = 8000
_DEFAULT_TIMEOUT_MS = 5000


def navigate(url: str, wait_until: str = "load") -> dict:
    """Navigate to ``url``; returns the resolved url and page title."""
    page = current_page()
    page.goto(url, wait_until=wait_until)
    return {"url": page.url, "title": page.title()}


def click(selector: str, timeout_ms: int = _DEFAULT_TIMEOUT_MS) -> dict:
    """Click the element matching ``selector``."""
    page = current_page()
    page.click(selector, timeout=timeout_ms)
    return {"clicked": selector, "url": page.url}


def type_text(selector: str, text: str, timeout_ms: int = _DEFAULT_TIMEOUT_MS) -> dict:
    """Fill ``selector`` with ``text``."""
    page = current_page()
    page.fill(selector, text, timeout=timeout_ms)
    return {"typed_into": selector}


def read_text(selector: str = "body", max_chars: int = _DEFAULT_MAX_CHARS) -> dict:
    """Read inner text of ``selector`` (defaults to ``body``), truncated.

    The selector wait uses a short timeout so a missing selector fails fast and
    the agent can retry, rather than blocking on Playwright's 30s default.
    """
    page = current_page()
    text = page.inner_text(selector, timeout=_DEFAULT_TIMEOUT_MS)
    return {"text": text[:max_chars], "url": page.url}
