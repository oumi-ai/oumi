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

"""Input sanitization utilities for command parsing."""


def make_safe(input_text: str) -> str:
    r"""Sanitize input text to prevent false positive command detection.

    This function ensures that only the first line of input is considered
    for command parsing, preventing complex multi-line content (like file paths,
    code snippets, or pasted text) from being misinterpreted as commands.

    Args:
        input_text: Raw user input that may contain newlines and complex content.

    Returns:
        The first line of input, stripped of leading/trailing whitespace.

    Examples:
        >>> make_safe("/help()")
        '/help()'

        >>> make_safe("/Users/path/file.txt\\nMore content here")
        '/Users/path/file.txt'

        >>> make_safe("Regular text\\n/attach(something)\\nMore text")
        'Regular text'
    """
    if not input_text or not isinstance(input_text, str):
        return ""

    # Extract only the first line to prevent multi-line content
    # from interfering with command detection
    first_line = input_text.split("\n", 1)[0].strip()

    return first_line
