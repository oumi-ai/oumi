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

import pytest

from oumi.core.configs.params.grounding_params import ToolGroundingConfig


def test_tool_grounding_config_basic():
    cfg = ToolGroundingConfig(key="book_id", fields=["book_id", "title", "status"])
    assert cfg.key == "book_id"
    assert cfg.fields == ["book_id", "title", "status"]


def test_tool_grounding_config_requires_key_in_fields():
    with pytest.raises(ValueError, match="must include 'key'"):
        ToolGroundingConfig(key="book_id", fields=["title", "status"])


def test_tool_grounding_config_requires_non_empty_fields():
    with pytest.raises(ValueError, match="fields must be non-empty"):
        ToolGroundingConfig(key="book_id", fields=[])


def test_tool_grounding_config_requires_non_empty_key():
    with pytest.raises(ValueError, match="key cannot be empty"):
        ToolGroundingConfig(key="", fields=["foo"])


def test_tool_grounding_config_rejects_duplicate_fields():
    with pytest.raises(ValueError, match="duplicate"):
        ToolGroundingConfig(key="book_id", fields=["book_id", "title", "title"])
