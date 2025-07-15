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

import json
import tempfile
from pathlib import Path

from oumi.utils.analysis_utils import save_results


def test_save_results_jsonl_format():
    """Test that save_results creates proper JSONL format."""
    # Sample analysis results
    results = {
        "dataset_name": "test_dataset",
        "total_conversations": 100,
        "conversations_analyzed": 5,
        "total_messages": 15,
        "messages": [
            {
                "conversation_id": "conv_0",
                "message_index": 0,
                "role": "user",
                "text_content": "Hello",
                "length_char_count": 5,
            },
            {
                "conversation_id": "conv_0",
                "message_index": 1,
                "role": "assistant",
                "text_content": "Hi there!",
                "length_char_count": 9,
            },
        ],
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "test_results.jsonl"

        # Save results
        save_results(results, str(output_path))

        # Verify file exists
        assert output_path.exists()

        # Read and verify JSONL format
        with open(output_path, encoding="utf-8") as f:
            lines = f.readlines()

        # Should have 3 lines: metadata + 2 messages
        assert len(lines) == 3

        # Parse first line (metadata)
        metadata = json.loads(lines[0].strip())
        assert metadata["dataset_name"] == "test_dataset"
        assert metadata["total_conversations"] == 100
        assert metadata["conversations_analyzed"] == 5
        assert metadata["total_messages"] == 15
        assert "messages" not in metadata  # Messages should be on separate lines

        # Parse message lines
        message1 = json.loads(lines[1].strip())
        assert message1["conversation_id"] == "conv_0"
        assert message1["message_index"] == 0
        assert message1["role"] == "user"
        assert message1["text_content"] == "Hello"
        assert message1["length_char_count"] == 5

        message2 = json.loads(lines[2].strip())
        assert message2["conversation_id"] == "conv_0"
        assert message2["message_index"] == 1
        assert message2["role"] == "assistant"
        assert message2["text_content"] == "Hi there!"
        assert message2["length_char_count"] == 9


def test_save_results_empty_messages():
    """Test save_results with empty messages list."""
    results = {
        "dataset_name": "empty_dataset",
        "total_conversations": 0,
        "conversations_analyzed": 0,
        "total_messages": 0,
        "messages": [],
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "empty_results.jsonl"

        # Save results
        save_results(results, str(output_path))

        # Read and verify
        with open(output_path, encoding="utf-8") as f:
            lines = f.readlines()

        # Should have only 1 line (metadata)
        assert len(lines) == 1

        metadata = json.loads(lines[0].strip())
        assert metadata["dataset_name"] == "empty_dataset"
        assert metadata["total_messages"] == 0
        assert "messages" not in metadata


def test_save_results_unicode_support():
    """Test that save_results handles Unicode content correctly."""
    results = {
        "dataset_name": "unicode_dataset",
        "total_conversations": 1,
        "conversations_analyzed": 1,
        "total_messages": 1,
        "messages": [
            {
                "conversation_id": "conv_0",
                "message_index": 0,
                "role": "user",
                "text_content": "Hello 世界! 🌍",
                "length_char_count": 12,
            },
        ],
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "unicode_results.jsonl"

        # Save results
        save_results(results, str(output_path))

        # Read and verify Unicode is preserved
        with open(output_path, encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 2  # metadata + 1 message

        message = json.loads(lines[1].strip())
        assert message["text_content"] == "Hello 世界! 🌍"
