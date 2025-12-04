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

"""Oumi Chat - Interactive chat functionality for Oumi."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from oumi_chat.infer import infer_interactive
    from oumi_chat.style_params import StyleParams

__all__ = [
    "infer_interactive",
    "StyleParams",
]


def __getattr__(name: str):
    """Lazy import to avoid circular import issues."""
    if name == "infer_interactive":
        from oumi_chat.infer import infer_interactive

        return infer_interactive
    if name == "StyleParams":
        from oumi_chat.style_params import StyleParams

        return StyleParams
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
