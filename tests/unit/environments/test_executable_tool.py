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

"""Skeleton-shape tests for ExecutableTool."""

from __future__ import annotations

import pytest

from oumi.environments.executable_tool import ExecutableTool


def test_rejects_empty_executor():
    """An ExecutableTool with no executor dotted path is invalid at construction."""
    with pytest.raises(ValueError, match="executor"):
        ExecutableTool(id="t", name="t", description="d")


def test_accepts_dotted_path():
    """Dotted-path executor is stored as-is."""
    tool = ExecutableTool(
        id="t", name="t", description="d", executor="oumi.examples.foo.bar"
    )
    assert tool.executor == "oumi.examples.foo.bar"
