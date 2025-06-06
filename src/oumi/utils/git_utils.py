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

import subprocess
from pathlib import Path
from typing import Optional


def get_git_revision_hash() -> Optional[str]:
    """Get the current git revision hash.

    Returns:
        Optional[str]: The current git revision hash, or None if it cannot be retrieved.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_tag() -> Optional[str]:
    """Get the current git tag.

    Returns:
        Optional[str]: The current git tag, or None if it cannot be retrieved.
    """
    try:
        return subprocess.check_output(
            ["git", "describe", "--tags", "--abbrev=0", "--exact-match"],
            cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_root_dir() -> Optional[Path]:
    """Get the root directory of the current git repository.

    Returns:
        Optional[str]: The root directory of the current git repository, or None if it
        cannot be retrieved.
    """
    try:
        dir_path = Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
        return dir_path if dir_path.is_dir() else None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
