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

from oumi.core.configs.params.grounding_params import (
    GroundingConfig,
    GroundingFact,
    ToolGroundingConfig,
)

# --- ToolGroundingConfig ---


def test_tool_grounding_config_basic():
    cfg = ToolGroundingConfig(fields=["book_id", "title", "status"])
    assert cfg.fields == ["book_id", "title", "status"]


def test_tool_grounding_config_rejects_empty_fields():
    with pytest.raises(ValueError, match="fields must be non-empty"):
        ToolGroundingConfig(fields=[])


def test_tool_grounding_config_rejects_duplicate_fields():
    with pytest.raises(ValueError, match="duplicate"):
        ToolGroundingConfig(fields=["book_id", "title", "title"])


# --- GroundingConfig ---


def test_grounding_config_defaults():
    cfg = GroundingConfig()
    assert cfg.sample_size == 3
    assert cfg.seed is None
    assert cfg.tools == {}


def test_grounding_config_rejects_zero_sample_size():
    with pytest.raises(ValueError, match="sample_size must be >= 1"):
        GroundingConfig(sample_size=0)


def test_grounding_config_coerces_tools_dict():
    """Raw tool entries are coerced to ToolGroundingConfig instances."""
    cfg = GroundingConfig(
        tools={"lookup_book_status": {"fields": ["book_id", "title"]}}  # type: ignore[dict-item]
    )
    assert isinstance(cfg.tools["lookup_book_status"], ToolGroundingConfig)
    assert cfg.tools["lookup_book_status"].fields == ["book_id", "title"]


def test_grounding_config_passthrough_for_already_typed_entries():
    tg = ToolGroundingConfig(fields=["a", "b"])
    cfg = GroundingConfig(tools={"t": tg})
    assert cfg.tools["t"] is tg


# --- GroundingFact ---


def test_grounding_fact_default_empty():
    fact = GroundingFact()
    assert fact.data == {}


def test_grounding_fact_holds_data():
    fact = GroundingFact(data={"id": "B001", "title": "Dune"})
    assert fact.data == {"id": "B001", "title": "Dune"}
