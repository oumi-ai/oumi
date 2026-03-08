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

from oumi.utils.placeholders import IndexableValue, resolve_placeholders


def test_placeholder_bracket_notation_with_indexable_value():
    """Test that bracket notation works with IndexableValue in placeholders."""
    items = [
        {"prompt": "Example 1", "response": "Response 1"},
        {"prompt": "Example 2", "response": "Response 2"},
    ]

    indexable = IndexableValue(items)

    # Test direct access - returns _DictWrapper
    item0 = indexable[0]
    assert item0["prompt"] == "Example 1"
    assert item0["response"] == "Response 1"

    item1 = indexable[1]
    assert item1["prompt"] == "Example 2"
    assert item1["response"] == "Response 2"

    assert len(indexable) == 2

    # Test in placeholder resolution
    template = "First: {examples[0].prompt}, Second: {examples[1].response}"
    result = resolve_placeholders(template, {"examples": items})

    assert result == "First: Example 1, Second: Response 2"


def test_indexable_value_error_handling():
    """Test IndexableValue error handling for invalid access."""
    items = [{"prompt": "Example 1"}, {"prompt": "Example 2"}]
    indexable = IndexableValue(items)

    # Test out of range (positive)
    with pytest.raises(IndexError, match="out of range"):
        indexable[5]

    # Test negative index - should work like Python lists
    assert indexable[-1]["prompt"] == "Example 2"
    assert indexable[-2]["prompt"] == "Example 1"

    # Test out of range (negative)
    with pytest.raises(IndexError, match="out of range"):
        indexable[-3]


def test_indexable_value_negative_indices():
    """Test that negative indices work like Python lists."""
    items = [
        {"prompt": "Example 1", "response": "Response 1"},
        {"prompt": "Example 2", "response": "Response 2"},
        {"prompt": "Example 3", "response": "Response 3"},
    ]

    # Test negative indices in placeholder resolution
    template = "Last: {examples[-1].prompt}, Second to last: {examples[-2].response}"
    result = resolve_placeholders(template, {"examples": items})
    assert result == "Last: Example 3, Second to last: Response 2"


def test_indexable_value_nested_dict_access():
    """Test that IndexableValue supports nested dictionary access."""
    items = [
        {"user": {"name": "Alice", "age": 30}, "score": 95},
        {"user": {"name": "Bob", "age": 25}, "score": 87},
    ]

    indexable = IndexableValue(items)

    # Test accessing nested dicts through bracket notation
    item0 = indexable[0]
    assert item0["user"]["name"] == "Alice"
    assert item0["user"]["age"] == 30
    assert item0["score"] == 95

    # Test in template with nested access
    # Note: Python's format_map doesn't support chained attribute access
    # like {items[0].user.name}. But it does support dict-style access
    template = "User: {items[0].user}, Score: {items[0].score}"
    result = resolve_placeholders(template, {"items": items})
    assert "Alice" in str(result)


def test_resolve_placeholders_basic():
    """Test basic placeholder resolution."""
    template = "Hello {name}, you are {age} years old."
    values = {"name": "Alice", "age": 30}

    result = resolve_placeholders(template, values)
    assert result == "Hello Alice, you are 30 years old."


def test_resolve_placeholders_missing_values_allowed():
    """Test placeholder resolution with missing values allowed."""
    template = "Hello {name}, you live in {city}."
    values = {"name": "Alice"}

    # With missing_values_allowed=True, missing placeholders are kept
    result = resolve_placeholders(template, values, missing_values_allowed=True)
    assert result == "Hello Alice, you live in {city}."


def test_resolve_placeholders_missing_values_not_allowed():
    """Test placeholder resolution raises error when missing values not allowed."""
    template = "Hello {name}, you live in {city}."
    values = {"name": "Alice"}

    # With missing_values_allowed=False (default), should raise ValueError
    with pytest.raises(ValueError, match="Missing value for placeholder: city"):
        resolve_placeholders(template, values, missing_values_allowed=False)


def test_indexable_value_with_empty_list():
    """Test IndexableValue with empty list."""
    items = []
    indexable = IndexableValue(items)

    assert len(indexable) == 0

    # Accessing any index should raise IndexError
    with pytest.raises(IndexError, match="out of range"):
        indexable[0]
