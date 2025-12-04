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

"""Pytest configuration and fixtures for oumi_chat tests."""

import tempfile
from pathlib import Path

import pytest

# Import fixtures from chat_test_utils to make them available
# pytest_plugins is the standard way to share fixtures across test modules
pytest_plugins = ["tests.oumi_chat.utils.chat_test_utils"]


@pytest.fixture(autouse=True)
def cleanup_test_files(request):
    """Automatically clean up test-generated files after each test."""
    # Get the current working directory at test start
    test_cwd = Path.cwd()

    def _cleanup_test_files():
        """Clean up test files in the current working directory."""
        test_file_patterns = [
            "test_output*.json",
            "test_*.json",
            "test_*.txt",
            "test_*.pdf",
            "test_*.csv",
            "test_*.md",
            "test_*.cast",
            "test_*.bin",
            "*_test_*.json",
            "*_test_*.txt",
            "*_test_*.pdf",
            "*_test_*.csv",
            "*_test_*.md",
            "*_test_*.cast",
            "*_test_*.bin",
            "stress_test_output*.json",
            "analysis_report*.md",
            "project_analysis*.md",
            "*_attachment*.txt",
            "*_cleanup_test_*.txt",
            "deeply_nested*.json",
            "sales_data*.json",
            "config*.json",
            "requirements*.txt",
            "readme*.md",
            "*_report*.md",
            # Command router test files
            "file1.json",
            "file2.json",
            "output.json",
            "file.txt",
            "test.json",
            "refinement_*.md",
            "demo.cast",
            # Malformed command test artifacts (these shouldn't be created!)
            "'mixed\"",
            '"unclosed',
        ]

        for pattern in test_file_patterns:
            for file_path in test_cwd.glob(pattern):
                try:
                    if file_path.is_file():
                        file_path.unlink()
                except Exception:
                    pass  # Ignore cleanup errors

        # Also clean up temp files from the system temp directory
        temp_dir = Path(tempfile.gettempdir())
        for pattern in ["tmp*test*", "*_test_*", "stress_test_*"]:
            for file_path in temp_dir.glob(pattern):
                try:
                    if file_path.is_file():
                        # Clean up files that are recent (from the current test session)
                        import time

                        current_time = time.time()
                        file_age = current_time - file_path.stat().st_mtime
                        if file_age < 3600:  # Files created in the last hour
                            file_path.unlink()
                except Exception:
                    pass  # Ignore cleanup errors

    # Use finalizer to ensure cleanup runs even on test failure
    request.addfinalizer(_cleanup_test_files)

    yield  # Let the test run
