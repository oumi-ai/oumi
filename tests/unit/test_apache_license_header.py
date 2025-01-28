import glob
import os

import pytest

_APACHE_LICENSE = """\
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

"""


def _backtrack_on_path(path, n):
    """Goes up n directories in the current path."""
    output_path = path
    for _ in range(n):
        output_path = os.path.dirname(output_path)
    return output_path


def _get_all_source_file_paths() -> list[str]:
    """Recursively returns all configs in the /src/oumi/ dir of the repo."""
    path_to_current_file = os.path.realpath(__file__)
    repo_root = _backtrack_on_path(path_to_current_file, 3)
    py_source_pattern = os.path.join(repo_root, "src", "oumi", "**", "*.py")
    all_py_source_files = glob.glob(py_source_pattern, recursive=True)
    assert len(all_py_source_files) > 0, "No Python source files found to parse."
    return all_py_source_files


@pytest.mark.parametrize(
    "py_source_path",
    _get_all_source_file_paths(),
)
def test_python_source_files_start_with_apache_header(py_source_path: str):
    with open(py_source_path) as f:
        file_contents = f.read()
    # Skip empty files
    if not file_contents.strip():
        return
    assert file_contents.startswith(
        _APACHE_LICENSE
    ), f"File {py_source_path} does not start with Apache license header."
