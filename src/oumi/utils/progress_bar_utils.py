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
from typing import Optional


class ProgressBarHandler:
    """Class that provides progress bar detection and consolidation functionality.

    This class can be used by any log stream class that needs to consolidate
    progress bar updates into single lines.
    """

    def __init__(self):
        """Initialize progress bar handling attributes."""
        self._progress_buffer = ""
        self._last_progress_line = ""
        self._progress_patterns = [
            r"\d+%",  # Percentage patterns like "50%"
            r"\d+/\d+",  # Fraction patterns like "100/200"
            r"\[.*\]",  # Bracket patterns like "[████████░░]"
        ]

    def _is_progress_line(self, line: str) -> bool:
        """Check if a line looks like a progress bar update.

        Args:
            line: The line to check.

        Returns:
            True if the line appears to be a progress bar update.
        """
        # Look for multiple progress indicators in the same line
        pattern_matches = 0
        for pattern in self._progress_patterns:
            if re.search(pattern, line):
                pattern_matches += 1

        # If we have multiple progress indicators, it's likely a progress bar
        if pattern_matches >= 2:
            return True

        return False

    def _handle_progress_update(self, line: str) -> Optional[str]:
        """Consolidate progress bar updates into a single line.

        - Ignores empty/whitespace lines (often from progress redraws).
        - If not a progress bar line, flushes any accumulated progress and
          returns the current line.
        - If a progress bar line, updates the buffer and waits to output
          until the progress is complete.

        Args:
            line: The new line that might be a progress update.

        Returns:
            The line to return
        """
        # This is needed since we sometimes get empty lines from the stream for
        # progress bar updates.
        if not line.strip():
            return None

        if not self._is_progress_line(line):
            # Not a progress line, return any accumulated progress and the new line
            result = ""
            if self._progress_buffer:
                result = self._progress_buffer + "\n"
                self._progress_buffer = ""
                self._last_progress_line = ""
            return result + line if line else result

        # Update the progress buffer (overwrite previous progress)
        self._progress_buffer = line.rstrip()
        self._last_progress_line = line.rstrip()
        return None  # Still accumulating

    def _get_remaining_progress(self) -> str:
        """Get any remaining accumulated progress buffer.

        Returns:
            The remaining progress buffer as a string, or empty string if none.
        """
        if self._progress_buffer:
            result = self._progress_buffer + "\n"
            self._progress_buffer = ""
            self._last_progress_line = ""
            return result
        return ""
