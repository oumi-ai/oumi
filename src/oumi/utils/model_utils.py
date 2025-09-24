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

"""Utilities for reasoning about model names and capabilities."""

from __future__ import annotations

from typing import Optional


def is_qwen_omni_model(model_name: Optional[str]) -> bool:
    """Returns True if the supplied model name corresponds to a Qwen Omni model."""

    if not model_name:
        return False
    lowered = model_name.lower()
    return "qwen" in lowered and "omni" in lowered


__all__ = ["is_qwen_omni_model"]
