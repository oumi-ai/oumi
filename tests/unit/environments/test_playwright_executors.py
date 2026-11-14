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

"""Tests for the built-in Playwright browser executors."""

from __future__ import annotations

from typing import Any

import pytest

from oumi.core.types.tool_call import ToolResult
from oumi.environments import playwright_executors as pe


class _FakePage:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.url = "https://example.com/landing"
        self.history: list[str] = []

    def goto(
        self, url: str, wait_until: str = "load", timeout: int | None = None
    ) -> None:
        self.calls.append(("goto", url, wait_until, timeout))

    def title(self) -> str:
        return "Example Domain"

    def click(self, selector: str, timeout: int | None = None) -> None:
        self.calls.append(("click", selector, timeout))

    def fill(self, selector: str, text: str, timeout: int | None = None) -> None:
        self.calls.append(("fill", selector, text, timeout))

    def inner_text(self, selector: str, timeout: int | None = None) -> str:
        self.calls.append(("inner_text", selector, timeout))
        return "x" * 20_000

    def go_back(
        self, wait_until: str = "load", timeout: int | None = None
    ) -> object | None:
        self.calls.append(("go_back", wait_until, timeout))
        if not self.history:
            return None
        self.url = self.history.pop()
        return object()


def test_navigate_returns_url_and_title():
    page: Any = _FakePage()
    result = pe.navigate({"url": "https://example.com"}, page)
    assert isinstance(result, ToolResult)
    assert result.output == {"url": page.url, "title": "Example Domain"}
    assert page.calls == [
        ("goto", "https://example.com", "load", pe._NAVIGATION_TIMEOUT_MS)
    ]


def test_go_back_returns_the_previous_page():
    page: Any = _FakePage()
    page.history.append("https://example.com/blog")
    result = pe.go_back({}, page)
    assert result.output == {"url": "https://example.com/blog", "navigated": True}
    # Waits for commit only: a back/forward-cache restore never fires load.
    assert page.calls == [("go_back", "commit", pe._NAVIGATION_TIMEOUT_MS)]


def test_go_back_reports_no_navigation_at_the_start_of_history():
    page: Any = _FakePage()
    result = pe.go_back({}, page)
    assert isinstance(result.output, dict)
    assert result.output["navigated"] is False
    assert result.output["url"] == "https://example.com/landing"


def test_click_passes_default_timeout():
    page: Any = _FakePage()
    result = pe.click({"selector": "#submit"}, page)
    assert result.output == {"clicked": "#submit", "url": page.url}
    assert page.calls == [("click", "#submit", pe._DEFAULT_TIMEOUT_MS)]


def test_type_text_fills_selector():
    page: Any = _FakePage()
    result = pe.type_text({"selector": "#q", "text": "hello"}, page)
    assert result.output == {"typed_into": "#q"}
    assert page.calls == [("fill", "#q", "hello", pe._DEFAULT_TIMEOUT_MS)]


def test_read_text_defaults_to_body_and_flags_truncation():
    page: Any = _FakePage()
    result = pe.read_text({"max_chars": 50}, page)
    assert page.calls == [("inner_text", "body", pe._DEFAULT_TIMEOUT_MS)]
    assert result.output == {"text": "x" * 50, "truncated": True, "url": page.url}


def test_read_text_rejects_negative_max_chars():
    page: Any = _FakePage()
    with pytest.raises(ValueError, match="max_chars"):
        pe.read_text({"max_chars": -1}, page)
