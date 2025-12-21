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

"""Tests for the comment parser module."""

import tempfile
from pathlib import Path

import pytest

from oumi.core.configs.metadata.comment_parser import (
    parse_metadata_comments,
    parse_tags,
)


class TestParseMetadataComments:
    """Tests for parse_metadata_comments function."""

    def test_empty_file(self, tmp_path: Path):
        """Test parsing an empty file."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        result = parse_metadata_comments(config_file)
        assert result == {}

    def test_no_metadata(self, tmp_path: Path):
        """Test parsing a file with no @meta comments."""
        config_file = tmp_path / "no_meta.yaml"
        config_file.write_text("""# Regular comment
# Another comment
model:
  model_name: "test"
""")

        result = parse_metadata_comments(config_file)
        assert result == {}

    def test_single_metadata(self, tmp_path: Path):
        """Test parsing a file with a single @meta comment."""
        config_file = tmp_path / "single_meta.yaml"
        config_file.write_text("""# @meta training_method: sft

model:
  model_name: "test"
""")

        result = parse_metadata_comments(config_file)
        assert result == {"training_method": "sft"}

    def test_multiple_metadata(self, tmp_path: Path):
        """Test parsing a file with multiple @meta comments."""
        config_file = tmp_path / "multi_meta.yaml"
        config_file.write_text("""# @meta training_method: sft
# @meta finetuning_type: lora
# @meta min_vram_gb: 20
# @meta tags: beginner-friendly, single-gpu
# @meta description: Test config for LoRA fine-tuning

model:
  model_name: "test"
""")

        result = parse_metadata_comments(config_file)
        assert result == {
            "training_method": "sft",
            "finetuning_type": "lora",
            "min_vram_gb": "20",
            "tags": "beginner-friendly, single-gpu",
            "description": "Test config for LoRA fine-tuning",
        }

    def test_mixed_comments(self, tmp_path: Path):
        """Test parsing with mixed regular and @meta comments."""
        config_file = tmp_path / "mixed.yaml"
        config_file.write_text("""# Regular comment at top
# @meta training_method: dpo
# Another regular comment
# @meta finetuning_type: full

model:
  model_name: "test"
""")

        result = parse_metadata_comments(config_file)
        assert result == {
            "training_method": "dpo",
            "finetuning_type": "full",
        }

    def test_stops_at_non_comment(self, tmp_path: Path):
        """Test that parsing stops at the first non-comment line."""
        config_file = tmp_path / "stops.yaml"
        config_file.write_text("""# @meta training_method: sft

model:
  model_name: "test"
# @meta should_not_be_parsed: true
""")

        result = parse_metadata_comments(config_file)
        # Should only have the first @meta, not the one after "model:"
        assert result == {"training_method": "sft"}

    def test_whitespace_handling(self, tmp_path: Path):
        """Test that whitespace is properly handled."""
        config_file = tmp_path / "whitespace.yaml"
        config_file.write_text("""#   @meta   training_method:   sft
# @meta finetuning_type:lora

model:
  model_name: "test"
""")

        result = parse_metadata_comments(config_file)
        assert result == {
            "training_method": "sft",
            "finetuning_type": "lora",
        }

    def test_nonexistent_file(self, tmp_path: Path):
        """Test parsing a nonexistent file returns empty dict."""
        config_file = tmp_path / "nonexistent.yaml"

        result = parse_metadata_comments(config_file)
        assert result == {}


class TestParseTags:
    """Tests for parse_tags function."""

    def test_empty_string(self):
        """Test parsing an empty string."""
        assert parse_tags("") == []

    def test_single_tag(self):
        """Test parsing a single tag."""
        assert parse_tags("vision") == ["vision"]

    def test_multiple_tags(self):
        """Test parsing multiple comma-separated tags."""
        assert parse_tags("beginner-friendly, single-gpu, vision") == [
            "beginner-friendly",
            "single-gpu",
            "vision",
        ]

    def test_whitespace_handling(self):
        """Test that whitespace is properly trimmed."""
        assert parse_tags("  tag1  ,  tag2  ,  tag3  ") == ["tag1", "tag2", "tag3"]

    def test_empty_elements(self):
        """Test that empty elements are filtered out."""
        assert parse_tags("tag1,,tag2,  ,tag3") == ["tag1", "tag2", "tag3"]
