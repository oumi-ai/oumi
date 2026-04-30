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

"""Tests for BaseAnalyzer.get_text_content tool-call handling."""

from oumi.analyze.base import BaseAnalyzer
from oumi.core.types.conversation import ContentItem, Message, Role, Type
from oumi.core.types.tool_call import FunctionCall, ToolCall


def _tool_call() -> ToolCall:
    return ToolCall(
        id="call_abc",
        function=FunctionCall(name="get_weather", arguments='{"city": "SF"}'),
    )


def test_get_text_content_string_content_unchanged():
    """Plain string content returns as-is (no tool_calls path)."""
    msg = Message(role=Role.ASSISTANT, content="Hello there")
    assert BaseAnalyzer.get_text_content(msg) == "Hello there"


def test_get_text_content_multimodal_text_items_joined():
    """List content with text items returns the joined text."""
    msg = Message(
        role=Role.USER,
        content=[
            ContentItem(type=Type.TEXT, content="part one"),
            ContentItem(type=Type.TEXT, content="part two"),
        ],
    )
    assert BaseAnalyzer.get_text_content(msg) == "part one part two"


def test_get_text_content_stringifies_tool_calls_when_content_none():
    """Tool-only assistant turns contribute their tool_calls to text analyses."""
    msg = Message(role=Role.ASSISTANT, content=None, tool_calls=[_tool_call()])
    text = BaseAnalyzer.get_text_content(msg)
    # Function name and argument substring both survive into the proxy form
    # so token-count and regex analyses see them.
    assert "get_weather" in text
    assert "SF" in text


def test_get_text_content_concats_text_and_tool_calls():
    """Mixed `content` (str) + `tool_calls` turns include both in the output."""
    msg = Message(
        role=Role.ASSISTANT,
        content="Let me check that.",
        tool_calls=[_tool_call()],
    )
    text = BaseAnalyzer.get_text_content(msg)
    assert "Let me check that." in text
    assert "get_weather" in text


def test_get_text_content_concats_multimodal_content_and_tool_calls():
    """List-content + tool_calls also concatenates."""
    msg = Message(
        role=Role.ASSISTANT,
        content=[ContentItem(type=Type.TEXT, content="Checking…")],
        tool_calls=[_tool_call()],
    )
    text = BaseAnalyzer.get_text_content(msg)
    assert "Checking…" in text
    assert "get_weather" in text
