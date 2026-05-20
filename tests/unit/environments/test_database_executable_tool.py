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

"""Skeleton-shape tests for ``DatabaseExecutableTool``."""

from __future__ import annotations

import pytest

from oumi.environments.database_executable_tool import DatabaseExecutableTool


def test_inherits_executor_requirement():
    """DatabaseExecutableTool keeps ExecutableTool's non-empty executor check."""
    with pytest.raises(ValueError, match="executor"):
        DatabaseExecutableTool(id="t", name="t", description="d")


def test_carries_optional_statement_timeout():
    """Per-tool statement_timeout_ms is stored as-is for env-side validation."""
    tool = DatabaseExecutableTool(
        id="t",
        name="t",
        description="d",
        executor="oumi.examples.foo.bar",
        statement_timeout_ms=5000,
    )
    assert tool.statement_timeout_ms == 5000


def test_create_from_mapping():
    """``create`` builds an instance from a raw mapping (YAML round-trip)."""
    tool = DatabaseExecutableTool.create(
        {
            "id": "t",
            "name": "t",
            "description": "d",
            "executor": "x.y",
            "statement_timeout_ms": 1000,
        }
    )
    assert tool.statement_timeout_ms == 1000
    assert tool.executor == "x.y"


def test_create_rejects_non_mapping():
    """Anything that isn't a mapping or a DatabaseExecutableTool is rejected."""
    with pytest.raises(TypeError, match="mappings"):
        DatabaseExecutableTool.create("not a mapping")  # type: ignore[arg-type]
