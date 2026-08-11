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

"""Browser tool executors for ``BrowserExecutableEnvironment``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from typing_extensions import NotRequired, TypedDict

from oumi.core.types.tool_call import ToolResult

if TYPE_CHECKING:
    from playwright.sync_api import Page  # pyright: ignore[reportMissingImports]

_DEFAULT_MAX_CHARS = 8000
_DEFAULT_TIMEOUT_MS = 5000


class NavigateArguments(TypedDict):
    url: str
    wait_until: NotRequired[
        Literal["load", "domcontentloaded", "networkidle", "commit"]
    ]


class ClickArguments(TypedDict):
    selector: str


class TypeTextArguments(TypedDict):
    selector: str
    text: str


class ReadTextArguments(TypedDict):
    selector: NotRequired[str]
    max_chars: NotRequired[int]


def navigate(arguments: NavigateArguments, context: Page) -> ToolResult:
    """Navigate to ``arguments['url']``; returns the resolved url and page title."""
    context.goto(arguments["url"], wait_until=arguments.get("wait_until", "load"))
    return ToolResult(output={"url": context.url, "title": context.title()})


def click(arguments: ClickArguments, context: Page) -> ToolResult:
    """Click the element matching ``arguments['selector']``."""
    context.click(arguments["selector"], timeout=_DEFAULT_TIMEOUT_MS)
    return ToolResult(output={"clicked": arguments["selector"], "url": context.url})


def type_text(arguments: TypeTextArguments, context: Page) -> ToolResult:
    """Fill ``arguments['selector']`` with ``arguments['text']``."""
    context.fill(arguments["selector"], arguments["text"], timeout=_DEFAULT_TIMEOUT_MS)
    return ToolResult(output={"typed_into": arguments["selector"]})


def read_text(arguments: ReadTextArguments, context: Page) -> ToolResult:
    """Read truncated inner text of a selector, defaulting to ``body``."""
    selector = arguments.get("selector", "body")
    # JSON Schema "integer" admits 1.0, which would fail as a slice index.
    max_chars = int(arguments.get("max_chars", _DEFAULT_MAX_CHARS))
    if max_chars < 0:
        raise ValueError("max_chars must be non-negative")
    text = context.inner_text(selector, timeout=_DEFAULT_TIMEOUT_MS)
    return ToolResult(
        output={
            "text": text[:max_chars],
            "truncated": len(text) > max_chars,
            "url": context.url,
        }
    )
