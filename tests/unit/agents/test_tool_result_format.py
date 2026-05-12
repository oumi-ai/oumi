# Copyright 2025 - Oumi
# Licensed under the Apache License, Version 2.0
"""Contract tests for tool-output compaction."""

import json

from oumi.agents.tool_result_format import compact_tool_output


def test_small_output_passes_through_unchanged():
    output = {"status": "ok", "patient_id": "P1"}
    assert compact_tool_output(output, max_chars=1024) == output


def test_long_string_is_clipped():
    output = {"status": "ok", "report": "x" * 5000}
    compacted = compact_tool_output(output, max_chars=200, keep_string_chars=80)
    assert isinstance(compacted, dict)
    assert "more chars" in compacted["report"]
    assert compacted["__truncated"] is True


def test_long_list_is_clipped_with_count_marker():
    output = {"items": list(range(1000))}
    compacted = compact_tool_output(output, max_chars=200, keep_list_items=3)
    assert isinstance(compacted, dict)
    items = compacted["items"]
    assert items[:3] == [0, 1, 2]
    assert "more items" in str(items[-1])
    assert compacted["__truncated"] is True


def test_compacted_output_is_json_serializable():
    """The session json.dumps the compacted output into a tool message —
    if compaction itself produces unserializable output, the loop crashes."""
    output = {"rows": [{"col": "x" * 100} for _ in range(50)]}
    compacted = compact_tool_output(output, max_chars=200)
    json.dumps(compacted)  # must not raise


def test_compaction_handles_non_native_json_values_via_default_str():
    """``json.dumps(default=str)`` covers DB result values like datetimes —
    the loop must not crash on those."""
    import datetime

    output = {"timestamp": datetime.datetime(2026, 5, 9, 12, 0)}
    compacted = compact_tool_output(output, max_chars=200)
    json.dumps(compacted, default=str)  # must not raise
