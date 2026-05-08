# Copyright 2025 - Oumi
# Licensed under the Apache License, Version 2.0
"""Unit tests for ExecutableTool base class."""

from __future__ import annotations

import pytest

from oumi.environments.executable_tool import ExecutableTool


def test_valid_tool_constructs():
    tool = ExecutableTool(
        id="t1",
        name="T1",
        description="A tool.",
        executor="some.module.func",
    )
    assert tool.executor == "some.module.func"


def test_empty_executor_raises():
    with pytest.raises(ValueError, match="executor"):
        ExecutableTool(
            id="t1",
            name="T1",
            description="A tool.",
            executor="",
        )


def test_inherits_toolparams_validation():
    # Empty id should still fail via parent ToolParams.__post_init__
    with pytest.raises(ValueError, match="id"):
        ExecutableTool(
            id="",
            name="T1",
            description="A tool.",
            executor="some.module.func",
        )
