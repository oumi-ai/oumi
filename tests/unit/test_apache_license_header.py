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


def _get_all_source_file_paths(exclude_prefixes: list[str] = []) -> list[str]:
    """Recursively returns all configs in the src/oumi/ dir of the repo.

    Args:
        exclude_prefixes (list[str]): List of prefixes to exclude from the search.
            These prefixes should be specified relative to the repo root.

    Returns:
        list[str]: List of all Python source files in the repo minus the exclusions.
    """
    path_to_current_file = os.path.realpath(__file__)
    repo_root = _backtrack_on_path(path_to_current_file, 3)
    py_source_pattern = os.path.join(repo_root, "src", "oumi", "**", "*.py")
    all_py_source_files = glob.glob(py_source_pattern, recursive=True)
    if exclude_prefixes:
        # Get absolute paths for the directories to exclude
        full_exclude_paths = []
        for exclude_path in exclude_prefixes:
            full_exclude_paths.append(os.path.join(repo_root, exclude_path))
        print(f"Excluding {full_exclude_paths} from the search.")

        exclude_files = []
        for file in all_py_source_files:
            for full_exclude_path in full_exclude_paths:
                if file.startswith(full_exclude_path):
                    exclude_files.append(file)
                    break
        print(f"Excluded {len(exclude_files)} files.")
        all_py_source_files = [
            file for file in all_py_source_files if file not in exclude_files
        ]
    assert len(all_py_source_files) > 0, "No Python source files found to parse."
    return all_py_source_files


@pytest.mark.parametrize(
    "py_source_path",
    _get_all_source_file_paths(
        exclude_prefixes=[
            "src/oumi/datasets/grpo/countdown.py",
            "src/oumi/datasets/grpo/rewards/countdown_rewards.py",
            "src/oumi/models/experimental/cambrian",
            "src/oumi/core/types/proto/generated/",
            "src/oumi/utils/verl_model_merger.py",
        ]
    ),
)
def test_python_source_files_start_with_apache_header(py_source_path: str):
    with open(py_source_path) as f:
        file_contents = f.read()
    assert file_contents.startswith(_APACHE_LICENSE), (
        f"File {py_source_path} does not start with Apache license header."
    )
