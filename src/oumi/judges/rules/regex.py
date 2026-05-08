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

import re

from oumi.core.registry import RegistryType, register
from oumi.judges.rules.base_rule import BaseRule


@register("regex", RegistryType.RULE)
class RegexRule(BaseRule):
    r"""Rule that checks if input text matches a regex pattern.

    Config Parameters:
        pattern (str): The regex pattern to match against
        input_field (str): The field name to extract text from input_data
        match_mode (str): How to match - "search", "match", "fullmatch"
        inverse (bool): If True, pass when pattern does NOT match (default: False)
        flags (int): Optional regex flags (e.g., re.IGNORECASE) (default: 0)

    Examples:
        Match a phone number pattern:
        >>> rule_config = {
        ...     "pattern": r"\\d{3}-\\d{4}",
        ...     "input_field": "text",
        ...     "match_mode": "search"
        ... }
        >>> rule = RegexRule()
        >>> result, score = rule.apply({"text": "Call 555-1234"}, rule_config)
        >>> print(result, score)
        True 1.0

        Inverse matching (expect NOT to match):
        >>> rule_config = {
        ...     "pattern": r"error|fail",
        ...     "input_field": "output",
        ...     "inverse": True
        ... }
        >>> result, score = rule.apply({"output": "Success!"}, rule_config)
        >>> print(result, score)
        True 1.0
    """

    def apply(
        self, input_data: dict[str, str], rule_config: dict
    ) -> tuple[bool, float]:
        """Apply regex pattern matching to input data.

        Args:
            input_data: Dictionary containing input fields
            (e.g., {"text": "...", "expected": "..."})
            rule_config: Configuration with 'pattern', 'input_field', 'inverse', etc.

        Returns:
            Tuple of (judgment: bool, score: float)
            - judgment: True if test passes (matches, or doesn't match if inverse=True)
            - score: 1.0 if judgment is True, 0.0 otherwise

        Raises:
            ValueError: If required config parameters are missing or invalid
        """
        pattern = rule_config.get("pattern")
        if not pattern:
            raise ValueError("rule_config must contain 'pattern' for regex rule")

        input_field = rule_config.get("input_field", "text")
        if input_field not in input_data:
            raise ValueError(
                f"input_field '{input_field}' not found in input_data. "
                f"Available fields: {list(input_data.keys())}"
            )

        text = input_data[input_field]
        match_mode = rule_config.get("match_mode", "search")
        inverse = rule_config.get("inverse", False)
        flags = rule_config.get("flags", 0)

        try:
            compiled_pattern = re.compile(pattern, flags)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{pattern}': {e}") from e

        if match_mode == "search":
            match = compiled_pattern.search(text)
        elif match_mode == "match":
            match = compiled_pattern.match(text)
        elif match_mode == "fullmatch":
            match = compiled_pattern.fullmatch(text)
        else:
            raise ValueError(
                f"Invalid match_mode '{match_mode}'. "
                "Must be one of: 'search', 'match', 'fullmatch'"
            )

        matched = match is not None
        judgment = matched != inverse
        score = 1.0 if judgment else 0.0

        return (judgment, score)
