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

from typing import TYPE_CHECKING, Any

from oumi.core.types.tool_call import ToolResult

if TYPE_CHECKING:
    from playwright.sync_api import Page  # pyright: ignore[reportMissingImports]

_DEFAULT_MAX_CHARS = 8000
_DEFAULT_TIMEOUT_MS = 5000
# Playwright's own default, made explicit so every navigation names its timeout.
_NAVIGATION_TIMEOUT_MS = 30_000


def navigate(arguments: dict[str, Any], context: Page) -> ToolResult:
    """Navigate to ``arguments['url']``; returns the resolved url and page title."""
    context.goto(
        arguments["url"],
        wait_until=arguments.get("wait_until", "load"),
        timeout=_NAVIGATION_TIMEOUT_MS,
    )
    return ToolResult(output={"url": context.url, "title": context.title()})


def go_back(arguments: dict[str, Any], context: Page) -> ToolResult:
    """Return to the previous page in history.

    ``navigated`` is False when there is nothing to go back to, so the model can
    tell a no-op from a real navigation. It compares URLs because ``go_back``
    returns ``None`` for client-routed and cache-restored pages that did change.

    Waits only for the commit and returns no title: on a page restored from
    Chromium's back/forward cache, this CDP connection never gets a JS context, so
    ``title()`` hangs and load events never fire. The next tool call's fresh
    connection reads the page normally.
    """
    # shortcut: the next call may see a still-loading page on slow sites; add a
    # bounded wait there if that shows up in rollouts.
    before = context.url
    context.go_back(wait_until="commit", timeout=_NAVIGATION_TIMEOUT_MS)
    return ToolResult(output={"url": context.url, "navigated": context.url != before})


def click(arguments: dict[str, Any], context: Page) -> ToolResult:
    """Click the element matching ``arguments['selector']``."""
    context.click(arguments["selector"], timeout=_DEFAULT_TIMEOUT_MS)
    return ToolResult(output={"clicked": arguments["selector"], "url": context.url})


def type_text(arguments: dict[str, Any], context: Page) -> ToolResult:
    """Fill ``arguments['selector']`` with ``arguments['text']``."""
    context.fill(arguments["selector"], arguments["text"], timeout=_DEFAULT_TIMEOUT_MS)
    return ToolResult(output={"typed_into": arguments["selector"]})


def read_text(arguments: dict[str, Any], context: Page) -> ToolResult:
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


# Runs in the page. Gives each field a selector the other tools accept: #id,
# else tag[name] (plus [value] for radios and checkboxes, which share a name),
# else Playwright's :has-text() for buttons.
_READ_FORM_JS = """
(root) => Array.from(root.querySelectorAll("input, select, textarea, button"))
  .filter((el) => el.type !== "hidden")
  .map((el) => {
    const tag = el.tagName.toLowerCase();
    const type = tag === "input" ? el.type : tag;
    const checkable = type === "radio" || type === "checkbox";
    const text = (el.innerText || "").trim();
    let selector = null;
    if (el.id) {
      selector = "#" + CSS.escape(el.id);
    } else if (el.name) {
      selector = `${tag}[name=${JSON.stringify(el.name)}]`;
      if (checkable) selector += `[value=${JSON.stringify(el.value)}]`;
    } else if (tag === "button" && text) {
      selector = `button:has-text(${JSON.stringify(text)})`;
    }
    // A label that wraps its control also contains the control's own text.
    const labelEl = el.labels && el.labels[0];
    const labelText = labelEl ? labelEl.innerText.replace(text, "") : "";
    const field = {
      selector,
      type,
      label: (labelText || el.getAttribute("aria-label") || el.placeholder || text)
        .trim(),
      value: checkable ? el.checked : el.value,
      required: Boolean(el.required),
    };
    if (tag === "select") {
      field.options = Array.from(el.options)
        .map((o) => ({ value: o.value, label: o.text.trim() }));
    }
    return field;
  })
"""


def read_form(arguments: dict[str, Any], context: Page) -> ToolResult:
    """List the fields under ``arguments['selector']`` (default ``body``).

    Each field has a ``selector`` to pass to ``type_text``, ``click`` or
    ``select_option``, its label, type and current value (``checked`` for radios
    and checkboxes), and a select's options as ``{value, label}``.
    """
    selector = arguments.get("selector", "body")
    fields = context.locator(selector).first.evaluate(
        _READ_FORM_JS, timeout=_DEFAULT_TIMEOUT_MS
    )
    return ToolResult(output={"fields": fields, "url": context.url})


def select_option(arguments: dict[str, Any], context: Page) -> ToolResult:
    """Choose ``arguments['value']`` in a select, matching an option value or label."""
    selected = context.select_option(
        arguments["selector"], arguments["value"], timeout=_DEFAULT_TIMEOUT_MS
    )
    return ToolResult(output={"selected": ", ".join(selected)})
