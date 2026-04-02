# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for oumi.utils.json_patch."""

import pytest

from oumi.utils.json_patch import (
    JsonPatchError,
    JsonPatchValidationError,
    apply_json_patch,
    parse_patch_response,
)


class TestApplyJsonPatch:
    def test_replace_op(self):
        doc = {"users": {"1": {"name": "Alice", "role": "admin"}}}
        patch = [{"op": "replace", "path": "/users/1/role", "value": "viewer"}]
        result = apply_json_patch(doc, patch)
        assert result["users"]["1"]["role"] == "viewer"

    def test_add_op(self):
        doc = {"users": {"1": {"name": "Alice"}}}
        patch = [{"op": "add", "path": "/users/2", "value": {"name": "Bob"}}]
        result = apply_json_patch(doc, patch)
        assert result["users"]["2"] == {"name": "Bob"}
        assert result["users"]["1"] == {"name": "Alice"}

    def test_remove_op(self):
        doc = {"users": {"1": {"name": "Alice"}, "2": {"name": "Bob"}}}
        patch = [{"op": "remove", "path": "/users/2"}]
        result = apply_json_patch(doc, patch)
        assert "2" not in result["users"]
        assert "1" in result["users"]

    def test_multi_op_patch(self):
        doc = {"items": {"a": {"count": 1}, "b": {"count": 2}}}
        patch = [
            {"op": "replace", "path": "/items/a/count", "value": 10},
            {"op": "replace", "path": "/items/b/count", "value": 20},
        ]
        result = apply_json_patch(doc, patch)
        assert result["items"]["a"]["count"] == 10
        assert result["items"]["b"]["count"] == 20

    def test_does_not_mutate_original(self):
        doc = {"x": {"1": {"v": "old"}}}
        patch = [{"op": "replace", "path": "/x/1/v", "value": "new"}]
        apply_json_patch(doc, patch)
        assert doc["x"]["1"]["v"] == "old"

    def test_empty_patch_returns_copy(self):
        doc = {"a": 1}
        result = apply_json_patch(doc, [])
        assert result == doc
        assert result is not doc

    def test_invalid_path_raises_error(self):
        doc = {"users": {"1": {"name": "Alice"}}}
        patch = [{"op": "replace", "path": "/nonexistent/field", "value": "x"}]
        with pytest.raises(JsonPatchError):
            apply_json_patch(doc, patch)

    def test_malformed_op_raises_error(self):
        doc = {"a": 1}
        patch = [{"bad_key": "replace"}]
        with pytest.raises(JsonPatchError):
            apply_json_patch(doc, patch)

    def test_atomic_failure(self):
        """If second op fails, original document must be unchanged."""
        doc = {"a": {"1": {"v": "orig"}}, "b": {"1": {"v": "orig"}}}
        patch = [
            {"op": "replace", "path": "/a/1/v", "value": "changed"},
            {"op": "replace", "path": "/nonexistent/path", "value": "bad"},
        ]
        with pytest.raises(JsonPatchError):
            apply_json_patch(doc, patch)
        assert doc["a"]["1"]["v"] == "orig"

    def test_schema_validation_pass(self):
        doc = {"count": 0}
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        }
        patch = [{"op": "replace", "path": "/count", "value": 5}]
        result = apply_json_patch(doc, patch, schema=schema)
        assert result["count"] == 5

    def test_schema_validation_fail(self):
        doc = {"count": 0}
        schema = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        }
        patch = [{"op": "replace", "path": "/count", "value": "not_an_int"}]
        with pytest.raises(JsonPatchValidationError):
            apply_json_patch(doc, patch, schema=schema)


class TestParsePatchResponse:
    def test_bare_json_array(self):
        text = '[{"op": "replace", "path": "/a", "value": 1}]'
        result = parse_patch_response(text)
        assert result == [{"op": "replace", "path": "/a", "value": 1}]

    def test_markdown_fenced(self):
        text = '```json\n[{"op": "add", "path": "/b", "value": 2}]\n```'
        result = parse_patch_response(text)
        assert result == [{"op": "add", "path": "/b", "value": 2}]

    def test_surrounding_prose(self):
        text = (
            'Here is the patch:\n[{"op": "remove", "path": "/c"}]\nThis removes key c.'
        )
        result = parse_patch_response(text)
        assert result == [{"op": "remove", "path": "/c"}]

    def test_empty_array(self):
        text = "[]"
        result = parse_patch_response(text)
        assert result == []

    def test_invalid_text(self):
        result = parse_patch_response("this is not json at all")
        assert result is None

    def test_dict_instead_of_list(self):
        text = '{"op": "replace", "path": "/a", "value": 1}'
        result = parse_patch_response(text)
        assert result is None
