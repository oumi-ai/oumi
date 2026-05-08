# Copyright 2025 - Oumi
# Licensed under the Apache License, Version 2.0
"""Unit tests for DatabaseExecutableTool."""

from __future__ import annotations

import pytest

from oumi.environments.database_executable_tool import DatabaseExecutableTool


def test_default_no_per_tool_overrides():
    tool = DatabaseExecutableTool(
        id="t1",
        name="T1",
        description="A tool.",
        executor="some.module.func",
    )
    assert tool.statement_timeout_ms is None


def test_per_tool_timeout_override():
    tool = DatabaseExecutableTool(
        id="t1",
        name="T1",
        description="A tool.",
        executor="some.module.func",
        statement_timeout_ms=500,
    )
    assert tool.statement_timeout_ms == 500


def test_create_from_mapping():
    raw = {
        "id": "t1",
        "name": "T1",
        "description": "A tool.",
        "parameters": {"type": "object"},
        "executor": "some.module.func",
        "statement_timeout_ms": 250,
    }
    tool = DatabaseExecutableTool.create(raw)
    assert isinstance(tool, DatabaseExecutableTool)
    assert tool.executor == "some.module.func"
    assert tool.statement_timeout_ms == 250


def test_create_passes_through_existing_tool():
    tool = DatabaseExecutableTool(
        id="t1",
        name="T1",
        description="A tool.",
        executor="some.module.func",
    )
    assert DatabaseExecutableTool.create(tool) is tool


def test_create_rejects_non_mapping():
    with pytest.raises(TypeError, match="mappings"):
        DatabaseExecutableTool.create(["not", "a", "mapping"])


def test_empty_executor_raises():
    with pytest.raises(ValueError, match="executor"):
        DatabaseExecutableTool(
            id="t1",
            name="T1",
            description="A tool.",
            executor="",
        )
