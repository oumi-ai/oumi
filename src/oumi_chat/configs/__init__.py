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

"""Configuration files and parameters for oumi_chat."""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oumi_chat.configs.style_params import StyleParams

__all__ = ["StyleParams", "get_configs_dir", "get_macros_dir"]


def get_configs_dir() -> Path:
    """Get the path to the oumi_chat configs directory."""
    return Path(__file__).parent


def get_macros_dir() -> Path:
    """Get the path to the oumi_chat macros directory."""
    return Path(__file__).parent / "macros"


def __getattr__(name: str):
    """Lazy import to avoid circular import issues."""
    if name == "StyleParams":
        from oumi_chat.configs.style_params import StyleParams

        return StyleParams
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
