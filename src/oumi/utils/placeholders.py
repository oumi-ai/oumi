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

from collections.abc import Mapping


class _DictWrapper:
    """Wrapper that allows dict keys to be accessed as attributes in format strings.

    Enables {item.field} syntax where item is a dictionary with a 'field' key.
    """

    def __init__(self, data: dict):
        """Initialize with a dictionary.

        Args:
            data: Dictionary to wrap for attribute-style access.
        """
        self._data = data

    def __getattr__(self, key: str):
        """Support attribute-style access: item.field.

        Args:
            key: Dictionary key to access.

        Returns:
            Value at the specified key.

        Raises:
            AttributeError: If key is not in dictionary.
        """
        try:
            return self._data[key]
        except KeyError as e:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{key}'"
            ) from e

    def __getitem__(self, key):
        """Support dict-style access: item['field'].

        Args:
            key: Dictionary key to access.

        Returns:
            Value at the specified key.
        """
        return self._data[key]


class IndexableValue:
    """Wrapper for list values that supports bracket notation in format strings.

    Enables {examples[0].field} syntax in templates by implementing __getitem__.
    """

    def __init__(self, items: list[dict]):
        """Initialize with a list of dictionaries.

        Args:
            items: List of dictionaries to wrap for indexed access.
        """
        self._items = items

    def __getitem__(self, index: int | str):
        """Support bracket notation: examples[0].

        Args:
            index: Integer index to access (supports negative indices).
                   Can be passed as an int or a string representation of an int.

        Returns:
            Dictionary at the specified index, wrapped to support attribute access.

        Raises:
            TypeError: If index is not an integer or string representation of one.
            IndexError: If index is out of range.
        """
        # Convert string indices to integers (needed for format_map)
        if isinstance(index, str):
            try:
                index = int(index)
            except ValueError as e:
                raise TypeError(
                    "Index must be integer or string representation of integer, "
                    f"got string '{index}'"
                ) from e
        elif not isinstance(index, int):
            raise TypeError(f"Index must be integer, got {type(index).__name__}")

        # Handle negative indices like Python lists
        if index < 0:
            index = len(self._items) + index

        if index < 0 or index >= len(self._items):
            raise IndexError("Index out of range")
        return _DictWrapper(self._items[index])

    def __len__(self) -> int:
        """Return the number of items."""
        return len(self._items)


class SafeDict(dict):
    def __init__(self, missing_values_allowed: bool, *args, **kwargs):
        """Initialize the SafeDict with the missing_values_allowed flag."""
        self.missing_values_allowed = missing_values_allowed
        self.placeholder_names = set()
        super().__init__(*args, **kwargs)

    def __missing__(self, key: str) -> str:
        """Handle missing keys in the dictionary."""
        self.placeholder_names.add(key)
        if self.missing_values_allowed:
            return "{" + key + "}"
        else:
            raise ValueError(f"Missing value for placeholder: {key}")

    def __getitem__(self, key):
        """Override to wrap list values with IndexableValue for bracket support.

        Args:
            key: Dictionary key to access.

        Returns:
            Value at the key, with lists of dicts wrapped in IndexableValue.
        """
        value = super().__getitem__(key)

        # Wrap lists of dicts to support bracket notation like {examples[0].field}
        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
            return IndexableValue(value)

        return value


def resolve_placeholders(
    text: str,
    values_dict: Mapping[str, object],
    missing_values_allowed: bool = False,
) -> str:
    """Resolve placeholder {variables} in the provided text from the values_dict."""
    return text.format_map(SafeDict(missing_values_allowed, values_dict))


def get_placeholders(text: str) -> set[str]:
    """Extract placeholder variable names from text with {variable} syntax."""
    safe_dict = SafeDict(missing_values_allowed=True)
    text.format_map(safe_dict)
    return safe_dict.placeholder_names
