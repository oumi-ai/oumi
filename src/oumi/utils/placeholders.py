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


class SafeDict(dict):
    def __init__(self, missing_values_allowed: bool, *args, **kwargs):
        """Initialize the SafeDict with the missing_values_allowed flag."""
        self.missing_values_allowed = missing_values_allowed
        super().__init__(*args, **kwargs)

    def __missing__(self, key: str) -> str:
        """Handle missing keys in the dictionary."""
        if self.missing_values_allowed:
            return "{" + key + "}"
        else:
            raise ValueError(f"Missing value for placeholder: {key}")


def resolve_placeholders(
    text: str,
    values_dict: dict[str, str],
    missing_values_allowed: bool = False,
) -> str:
    """Resolve placeholder {variables} in the provided text from the values_dict."""
    return text.format_map(SafeDict(missing_values_allowed, values_dict))
