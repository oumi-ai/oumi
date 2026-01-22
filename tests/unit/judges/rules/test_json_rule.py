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

import pytest

from oumi.judges.rules.json_rule import JsonRule


class TestJsonRule:
    """Test cases for the JsonRule class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.rule = JsonRule()

    # Basic JSON validation tests

    def test_valid_json_object(self):
        """Test that valid JSON object passes."""
        input_data = {"response": '{"name": "test", "value": 123}'}
        rule_config = {"input_field": "response"}

        result, score = self.rule.apply(input_data, rule_config)

        assert result is True
        assert score == 1.0

    def test_valid_json_array(self):
        """Test that valid JSON array passes."""
        input_data = {"response": '[1, 2, 3, "four"]'}
        rule_config = {"input_field": "response"}

        result, score = self.rule.apply(input_data, rule_config)

        assert result is True
        assert score == 1.0

    def test_valid_json_string(self):
        """Test that valid JSON string passes."""
        input_data = {"response": '"hello world"'}
        rule_config = {"input_field": "response"}

        result, score = self.rule.apply(input_data, rule_config)

        assert result is True
        assert score == 1.0

    def test_valid_json_number(self):
        """Test that valid JSON number passes."""
        input_data = {"response": "42.5"}
        rule_config = {"input_field": "response"}

        result, score = self.rule.apply(input_data, rule_config)

        assert result is True
        assert score == 1.0

    def test_valid_json_boolean(self):
        """Test that valid JSON boolean passes."""
        input_data = {"response": "true"}
        rule_config = {"input_field": "response"}

        result, score = self.rule.apply(input_data, rule_config)

        assert result is True
        assert score == 1.0

    def test_valid_json_null(self):
        """Test that valid JSON null passes."""
        input_data = {"response": "null"}
        rule_config = {"input_field": "response"}

        result, score = self.rule.apply(input_data, rule_config)

        assert result is True
        assert score == 1.0

    def test_invalid_json(self):
        """Test that invalid JSON fails."""
        input_data = {"response": "not valid json {"}
        rule_config = {"input_field": "response"}

        result, score = self.rule.apply(input_data, rule_config)

        assert result is False
        assert score == 0.0

    def test_invalid_json_trailing_comma(self):
        """Test that JSON with trailing comma fails."""
        input_data = {"response": '{"name": "test",}'}
        rule_config = {"input_field": "response"}

        result, score = self.rule.apply(input_data, rule_config)

        assert result is False
        assert score == 0.0

    def test_invalid_json_single_quotes(self):
        """Test that JSON with single quotes fails."""
        input_data = {"response": "{'name': 'test'}"}
        rule_config = {"input_field": "response"}

        result, score = self.rule.apply(input_data, rule_config)

        assert result is False
        assert score == 0.0

    # Required keys tests

    def test_required_keys_present(self):
        """Test that required keys validation passes when all keys present."""
        input_data = {"response": '{"name": "John", "age": 30, "city": "NYC"}'}
        rule_config = {"input_field": "response", "required_keys": ["name", "age"]}

        result, score = self.rule.apply(input_data, rule_config)

        assert result is True
        assert score == 1.0

    def test_required_keys_missing(self):
        """Test that required keys validation fails when key is missing."""
        input_data = {"response": '{"name": "John"}'}
        rule_config = {"input_field": "response", "required_keys": ["name", "age"]}

        result, score = self.rule.apply(input_data, rule_config)

        assert result is False
        assert score == 0.0

    def test_required_keys_on_non_object(self):
        """Test that required keys validation fails on non-object JSON."""
        input_data = {"response": "[1, 2, 3]"}
        rule_config = {"input_field": "response", "required_keys": ["name"]}

        result, score = self.rule.apply(input_data, rule_config)

        assert result is False
        assert score == 0.0

    # Schema validation tests

    def test_schema_type_object(self):
        """Test schema validation with object type."""
        input_data = {"response": '{"name": "John"}'}
        rule_config = {
            "input_field": "response",
            "schema": {"type": "object"},
        }

        result, score = self.rule.apply(input_data, rule_config)

        assert result is True
        assert score == 1.0

    def test_schema_type_object_fails_for_array(self):
        """Test schema validation fails when expecting object but got array."""
        input_data = {"response": "[1, 2, 3]"}
        rule_config = {
            "input_field": "response",
            "schema": {"type": "object"},
        }

        result, score = self.rule.apply(input_data, rule_config)

        assert result is False
        assert score == 0.0

    def test_schema_type_array(self):
        """Test schema validation with array type."""
        input_data = {"response": "[1, 2, 3]"}
        rule_config = {
            "input_field": "response",
            "schema": {"type": "array"},
        }

        result, score = self.rule.apply(input_data, rule_config)

        assert result is True
        assert score == 1.0

    def test_schema_type_string(self):
        """Test schema validation with string type."""
        input_data = {"response": '"hello"'}
        rule_config = {
            "input_field": "response",
            "schema": {"type": "string"},
        }

        result, score = self.rule.apply(input_data, rule_config)

        assert result is True
        assert score == 1.0

    def test_schema_type_integer(self):
        """Test schema validation with integer type."""
        input_data = {"response": "42"}
        rule_config = {
            "input_field": "response",
            "schema": {"type": "integer"},
        }

        result, score = self.rule.apply(input_data, rule_config)

        assert result is True
        assert score == 1.0

    def test_schema_type_integer_fails_for_float(self):
        """Test schema validation fails when expecting integer but got float."""
        input_data = {"response": "42.5"}
        rule_config = {
            "input_field": "response",
            "schema": {"type": "integer"},
        }

        result, score = self.rule.apply(input_data, rule_config)

        assert result is False
        assert score == 0.0

    def test_schema_type_number(self):
        """Test schema validation with number type accepts float."""
        input_data = {"response": "42.5"}
        rule_config = {
            "input_field": "response",
            "schema": {"type": "number"},
        }

        result, score = self.rule.apply(input_data, rule_config)

        assert result is True
        assert score == 1.0

    def test_schema_type_number_accepts_integer(self):
        """Test schema validation with number type accepts integer."""
        input_data = {"response": "42"}
        rule_config = {
            "input_field": "response",
            "schema": {"type": "number"},
        }

        result, score = self.rule.apply(input_data, rule_config)

        assert result is True
        assert score == 1.0

    def test_schema_type_boolean(self):
        """Test schema validation with boolean type."""
        input_data = {"response": "true"}
        rule_config = {
            "input_field": "response",
            "schema": {"type": "boolean"},
        }

        result, score = self.rule.apply(input_data, rule_config)

        assert result is True
        assert score == 1.0

    def test_schema_required_fields(self):
        """Test schema validation with required fields."""
        input_data = {"response": '{"name": "John", "age": 30}'}
        rule_config = {
            "input_field": "response",
            "schema": {
                "type": "object",
                "required": ["name", "age"],
            },
        }

        result, score = self.rule.apply(input_data, rule_config)

        assert result is True
        assert score == 1.0

    def test_schema_required_fields_missing(self):
        """Test schema validation fails when required field is missing."""
        input_data = {"response": '{"name": "John"}'}
        rule_config = {
            "input_field": "response",
            "schema": {
                "type": "object",
                "required": ["name", "age"],
            },
        }

        result, score = self.rule.apply(input_data, rule_config)

        assert result is False
        assert score == 0.0

    def test_schema_nested_properties(self):
        """Test schema validation with nested property types."""
        input_data = {"response": '{"name": "John", "age": 30}'}
        rule_config = {
            "input_field": "response",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
            },
        }

        result, score = self.rule.apply(input_data, rule_config)

        assert result is True
        assert score == 1.0

    def test_schema_nested_properties_wrong_type(self):
        """Test schema validation fails with wrong nested property type."""
        input_data = {"response": '{"name": "John", "age": "thirty"}'}
        rule_config = {
            "input_field": "response",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
            },
        }

        result, score = self.rule.apply(input_data, rule_config)

        assert result is False
        assert score == 0.0

    def test_schema_array_items(self):
        """Test schema validation with array items type."""
        input_data = {"response": "[1, 2, 3]"}
        rule_config = {
            "input_field": "response",
            "schema": {
                "type": "array",
                "items": {"type": "integer"},
            },
        }

        result, score = self.rule.apply(input_data, rule_config)

        assert result is True
        assert score == 1.0

    def test_schema_array_items_wrong_type(self):
        """Test schema validation fails with wrong array item type."""
        input_data = {"response": '[1, "two", 3]'}
        rule_config = {
            "input_field": "response",
            "schema": {
                "type": "array",
                "items": {"type": "integer"},
            },
        }

        result, score = self.rule.apply(input_data, rule_config)

        assert result is False
        assert score == 0.0

    # Edge cases

    def test_default_input_field(self):
        """Test that default input field is 'text'."""
        input_data = {"text": '{"valid": true}'}
        rule_config = {}

        result, score = self.rule.apply(input_data, rule_config)

        assert result is True
        assert score == 1.0

    def test_missing_input_field_raises_error(self):
        """Test that missing input field raises ValueError."""
        input_data = {"other_field": '{"valid": true}'}
        rule_config = {"input_field": "response"}

        with pytest.raises(ValueError) as exc_info:
            self.rule.apply(input_data, rule_config)

        assert "input_field 'response' not found" in str(exc_info.value)

    def test_empty_json_object(self):
        """Test that empty JSON object is valid."""
        input_data = {"response": "{}"}
        rule_config = {"input_field": "response"}

        result, score = self.rule.apply(input_data, rule_config)

        assert result is True
        assert score == 1.0

    def test_empty_json_array(self):
        """Test that empty JSON array is valid."""
        input_data = {"response": "[]"}
        rule_config = {"input_field": "response"}

        result, score = self.rule.apply(input_data, rule_config)

        assert result is True
        assert score == 1.0

    def test_complex_nested_json(self):
        """Test validation of complex nested JSON structure."""
        complex_json = """{
            "users": [
                {"name": "John", "age": 30},
                {"name": "Jane", "age": 25}
            ],
            "metadata": {
                "total": 2,
                "page": 1
            }
        }"""
        input_data = {"response": complex_json}
        rule_config = {"input_field": "response"}

        result, score = self.rule.apply(input_data, rule_config)

        assert result is True
        assert score == 1.0

    def test_boolean_not_valid_number(self):
        """Test that boolean is not accepted as number type."""
        input_data = {"response": "true"}
        rule_config = {
            "input_field": "response",
            "schema": {"type": "number"},
        }

        result, score = self.rule.apply(input_data, rule_config)

        assert result is False
        assert score == 0.0

    def test_boolean_not_valid_integer(self):
        """Test that boolean is not accepted as integer type."""
        input_data = {"response": "true"}
        rule_config = {
            "input_field": "response",
            "schema": {"type": "integer"},
        }

        result, score = self.rule.apply(input_data, rule_config)

        assert result is False
        assert score == 0.0

    # JSON extraction tests

    def test_extract_json_with_prefix(self):
        """Test extracting JSON from text with prefix."""
        input_data = {"response": 'Here is the generated JSON: {"name": "test"}'}
        rule_config = {"input_field": "response", "extract_json": True}

        result, score = self.rule.apply(input_data, rule_config)

        assert result is True
        assert score == 1.0

    def test_extract_json_with_suffix(self):
        """Test extracting JSON from text with suffix."""
        input_data = {
            "response": '{"name": "test"}\n\nLet me know if you need anything else.'
        }
        rule_config = {"input_field": "response", "extract_json": True}

        result, score = self.rule.apply(input_data, rule_config)

        assert result is True
        assert score == 1.0

    def test_extract_json_with_prefix_and_suffix(self):
        """Test extracting JSON from text with both prefix and suffix."""
        input_data = {
            "response": "Here is the result:\n\n"
            '{"name": "John", "age": 30}\n\nHope this helps!'
        }
        rule_config = {"input_field": "response", "extract_json": True}

        result, score = self.rule.apply(input_data, rule_config)

        assert result is True
        assert score == 1.0

    def test_extract_json_code_block(self):
        """Test extracting JSON from markdown code block."""
        input_data = {"response": '```json\n{"name": "test"}\n```'}
        rule_config = {"input_field": "response", "extract_json": True}

        result, score = self.rule.apply(input_data, rule_config)

        assert result is True
        assert score == 1.0

    def test_extract_json_array(self):
        """Test extracting JSON array from text with prefix."""
        input_data = {"response": "The list is: [1, 2, 3]"}
        rule_config = {"input_field": "response", "extract_json": True}

        result, score = self.rule.apply(input_data, rule_config)

        assert result is True
        assert score == 1.0

    def test_extract_json_nested_braces(self):
        """Test extracting JSON with nested objects."""
        input_data = {
            "response": 'Result: {"outer": {"inner": {"value": 1}}}'
        }
        rule_config = {"input_field": "response", "extract_json": True}

        result, score = self.rule.apply(input_data, rule_config)

        assert result is True
        assert score == 1.0

    def test_extract_json_with_string_containing_braces(self):
        """Test extracting JSON where string values contain braces."""
        input_data = {
            "response": 'Output: {"message": "Use {name} as placeholder"}'
        }
        rule_config = {"input_field": "response", "extract_json": True}

        result, score = self.rule.apply(input_data, rule_config)

        assert result is True
        assert score == 1.0

    def test_extract_json_disabled_by_default(self):
        """Test that JSON extraction is disabled by default."""
        input_data = {"response": 'Here is the JSON: {"name": "test"}'}
        rule_config = {"input_field": "response"}

        result, score = self.rule.apply(input_data, rule_config)

        assert result is False
        assert score == 0.0

    def test_extract_json_no_json_found(self):
        """Test extraction fails when no JSON is present."""
        input_data = {"response": "No JSON here, just plain text."}
        rule_config = {"input_field": "response", "extract_json": True}

        result, score = self.rule.apply(input_data, rule_config)

        assert result is False
        assert score == 0.0

    def test_extract_json_with_required_keys(self):
        """Test extraction combined with required keys validation."""
        input_data = {"response": 'Result: {"name": "John", "age": 30}'}
        rule_config = {
            "input_field": "response",
            "extract_json": True,
            "required_keys": ["name", "age"],
        }

        result, score = self.rule.apply(input_data, rule_config)

        assert result is True
        assert score == 1.0

    def test_extract_json_with_schema(self):
        """Test extraction combined with schema validation."""
        input_data = {"response": 'Generated: {"name": "John", "age": 30}'}
        rule_config = {
            "input_field": "response",
            "extract_json": True,
            "schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                "required": ["name"],
            },
        }

        result, score = self.rule.apply(input_data, rule_config)

        assert result is True
        assert score == 1.0
