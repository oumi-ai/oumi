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

import json
from typing import Any

from oumi.core.registry import RegistryType, register
from oumi.judges.rules.base_rule import BaseRule


@register("json", RegistryType.RULE)
class JsonRule(BaseRule):
    """Rule that validates if input text is valid JSON with optional schema validation.

    Config Parameters:
        input_field (str): The field name to extract text from input_data (default: "text")
        required_keys (list[str]): Optional list of keys that must be present in the JSON
        schema (dict): Optional JSON Schema to validate against
        extract_json (bool): If True, attempt to extract JSON from text with prefix/suffix
            (default: False). Useful when output contains "Here is the JSON: {...}"

    Examples:
        Basic JSON validation:
        >>> rule_config = {
        ...     "input_field": "response"
        ... }
        >>> rule = JsonRule()
        >>> result, score = rule.apply({"response": '{"name": "test"}'}, rule_config)
        >>> print(result, score)
        True 1.0

        Validate with required keys:
        >>> rule_config = {
        ...     "input_field": "response",
        ...     "required_keys": ["name", "age"]
        ... }
        >>> result, score = rule.apply(
        ...     {"response": '{"name": "John", "age": 30}'},
        ...     rule_config
        ... )
        >>> print(result, score)
        True 1.0

        Validate with JSON Schema:
        >>> rule_config = {
        ...     "input_field": "response",
        ...     "schema": {
        ...         "type": "object",
        ...         "properties": {
        ...             "name": {"type": "string"},
        ...             "age": {"type": "integer"}
        ...         },
        ...         "required": ["name"]
        ...     }
        ... }
        >>> result, score = rule.apply(
        ...     {"response": '{"name": "John", "age": 30}'},
        ...     rule_config
        ... )
        >>> print(result, score)
        True 1.0

        Extract JSON from text with prefix/suffix:
        >>> rule_config = {
        ...     "input_field": "response",
        ...     "extract_json": True
        ... }
        >>> result, score = rule.apply(
        ...     {"response": 'Here is the JSON: {"name": "test"}'},
        ...     rule_config
        ... )
        >>> print(result, score)
        True 1.0
    """

    def apply(
        self, input_data: dict[str, str], rule_config: dict
    ) -> tuple[bool, float]:
        """Apply JSON validation to input data.

        Args:
            input_data: Dictionary containing input fields
            rule_config: Configuration with 'input_field', 'required_keys', 'schema',
                'extract_json'

        Returns:
            Tuple of (judgment: bool, score: float)
            - judgment: True if JSON is valid and passes all validations
            - score: 1.0 if judgment is True, 0.0 otherwise
        """
        input_field = rule_config.get("input_field", "text")
        if input_field not in input_data:
            raise ValueError(
                f"input_field '{input_field}' not found in input_data. "
                f"Available fields: {list(input_data.keys())}"
            )

        text = input_data[input_field]
        extract_json = rule_config.get("extract_json", False)

        # Step 1: Try to parse as JSON (with optional extraction)
        success, parsed = self._parse_json(text, extract_json)
        if not success:
            return (False, 0.0)

        # Step 2: Check required keys if specified
        required_keys = rule_config.get("required_keys")
        if required_keys:
            if not isinstance(parsed, dict):
                return (False, 0.0)
            missing_keys = set(required_keys) - set(parsed.keys())
            if missing_keys:
                return (False, 0.0)

        # Step 3: Validate against schema if specified
        schema = rule_config.get("schema")
        if schema:
            if not self._validate_schema(parsed, schema):
                return (False, 0.0)

        return (True, 1.0)

    def _parse_json(self, text: str, extract: bool) -> tuple[bool, Any]:
        """Parse JSON from text, optionally extracting it from surrounding text.

        Args:
            text: The text to parse
            extract: If True, attempt to extract JSON from text with prefix/suffix

        Returns:
            Tuple of (success: bool, parsed_data: Any)
            - success: True if parsing succeeded
            - parsed_data: The parsed JSON data (may be None for valid JSON null)
        """
        # First, try direct parsing
        try:
            return (True, json.loads(text))
        except json.JSONDecodeError:
            pass

        # If extraction is enabled, try to find JSON in the text
        if extract:
            extracted = self._extract_json_from_text(text)
            if extracted is not None:
                try:
                    return (True, json.loads(extracted))
                except json.JSONDecodeError:
                    pass

        return (False, None)

    def _extract_json_from_text(self, text: str) -> str | None:
        """Extract JSON object or array from text with prefix/suffix.

        Handles common patterns like:
        - "Here is the JSON: {...}"
        - "```json\\n{...}\\n```"
        - "{...}\\n\\nLet me know if you need anything else."

        Args:
            text: Text potentially containing JSON with prefix/suffix

        Returns:
            Extracted JSON string, or None if no valid JSON structure found
        """
        # Try to find JSON object {...} or array [...]
        for start_char, end_char in [('{', '}'), ('[', ']')]:
            start_idx = text.find(start_char)
            if start_idx == -1:
                continue

            # Find matching closing bracket by counting nesting
            depth = 0
            in_string = False
            escape_next = False

            for i in range(start_idx, len(text)):
                char = text[i]

                if escape_next:
                    escape_next = False
                    continue

                if char == '\\' and in_string:
                    escape_next = True
                    continue

                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue

                if in_string:
                    continue

                if char == start_char:
                    depth += 1
                elif char == end_char:
                    depth -= 1
                    if depth == 0:
                        return text[start_idx : i + 1]

        return None

    def _validate_schema(self, data: Any, schema: dict) -> bool:
        """Validate data against a JSON Schema.

        Uses jsonschema library if available, otherwise falls back to basic validation.

        Args:
            data: The parsed JSON data to validate
            schema: JSON Schema to validate against

        Returns:
            True if data validates against schema, False otherwise
        """
        try:
            import jsonschema

            jsonschema.validate(instance=data, schema=schema)
            return True
        except ImportError:
            # Fallback to basic validation without jsonschema library
            return self._basic_schema_validation(data, schema)
        except jsonschema.ValidationError:
            return False

    def _basic_schema_validation(self, data: Any, schema: dict) -> bool:
        """Basic schema validation without jsonschema library.

        Supports a subset of JSON Schema:
        - type validation (object, array, string, number, integer, boolean, null)
        - required fields for objects
        - properties for objects

        Args:
            data: The data to validate
            schema: JSON Schema dict

        Returns:
            True if basic validation passes
        """
        schema_type = schema.get("type")

        # Type validation
        if schema_type:
            if not self._check_type(data, schema_type):
                return False

        # Object-specific validations
        if schema_type == "object" and isinstance(data, dict):
            # Check required fields
            required = schema.get("required", [])
            if not all(key in data for key in required):
                return False

            # Validate properties
            properties = schema.get("properties", {})
            for key, prop_schema in properties.items():
                if key in data:
                    if not self._basic_schema_validation(data[key], prop_schema):
                        return False

        # Array-specific validations
        if schema_type == "array" and isinstance(data, list):
            items_schema = schema.get("items")
            if items_schema:
                for item in data:
                    if not self._basic_schema_validation(item, items_schema):
                        return False

        return True

    def _check_type(self, data: Any, schema_type: str) -> bool:
        """Check if data matches the expected JSON Schema type."""
        type_map = {
            "object": dict,
            "array": list,
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "null": type(None),
        }

        expected_type = type_map.get(schema_type)
        if expected_type is None:
            return True  # Unknown type, allow it

        # Special case: integers are also valid numbers
        if schema_type == "number" and isinstance(data, bool):
            return False  # bool is a subclass of int, but not a valid number

        if schema_type == "integer" and isinstance(data, bool):
            return False  # bool is a subclass of int, but not a valid integer

        return isinstance(data, expected_type)
