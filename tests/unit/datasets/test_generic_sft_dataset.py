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

"""Tests for GenericSftDataset."""

import tempfile
from pathlib import Path

import jsonlines
import pytest

from oumi.core.types.conversation import Conversation, Role
from oumi.datasets import GenericSftDataset


@pytest.fixture
def sample_oumi_data():
    """Sample data in Oumi format."""
    return [
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm doing well!"},
            ]
        },
    ]


@pytest.fixture
def sample_alpaca_data():
    """Sample data in Alpaca format."""
    return [
        {
            "instruction": "Translate to French",
            "input": "Hello",
            "output": "Bonjour",
        },
        {
            "instruction": "Translate to Spanish",
            "input": "Goodbye",
            "output": "Adiós",
        },
    ]


@pytest.fixture
def sample_sharegpt_data():
    """Sample data in ShareGPT format."""
    return [
        {
            "conversations": [
                {"from": "human", "value": "What is AI?"},
                {"from": "gpt", "value": "AI stands for Artificial Intelligence."},
            ]
        },
    ]


class TestGenericSftDatasetWithOumiFormat:
    """Tests for GenericSftDataset with Oumi format."""

    def test_explicit_converter(self, sample_oumi_data):
        """Test loading with explicit oumi converter."""
        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, mode="w"
        ) as f:
            writer = jsonlines.Writer(f)
            writer.write_all(sample_oumi_data)
            writer.close()
            temp_path = f.name

        try:
            dataset = GenericSftDataset(
                dataset_path=temp_path,
                converter="oumi",
            )

            assert len(dataset) == 2
            assert dataset.converter_name == "oumi"

            # Get first item and check it's a Conversation
            item = dataset[0]
            conv = dataset.conversation(item)
            assert isinstance(conv, Conversation)
            assert len(conv.messages) == 2
            assert conv.messages[0].role == Role.USER
        finally:
            Path(temp_path).unlink()

    def test_auto_detection_oumi(self, sample_oumi_data):
        """Test auto-detection of Oumi format."""
        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, mode="w"
        ) as f:
            writer = jsonlines.Writer(f)
            writer.write_all(sample_oumi_data)
            writer.close()
            temp_path = f.name

        try:
            dataset = GenericSftDataset(
                dataset_path=temp_path,
                converter="auto",
            )

            assert dataset.converter_name == "oumi"
        finally:
            Path(temp_path).unlink()


class TestGenericSftDatasetWithAlpacaFormat:
    """Tests for GenericSftDataset with Alpaca format."""

    def test_explicit_converter(self, sample_alpaca_data):
        """Test loading with explicit alpaca converter."""
        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, mode="w"
        ) as f:
            writer = jsonlines.Writer(f)
            writer.write_all(sample_alpaca_data)
            writer.close()
            temp_path = f.name

        try:
            dataset = GenericSftDataset(
                dataset_path=temp_path,
                converter="alpaca",
            )

            assert len(dataset) == 2
            assert dataset.converter_name == "alpaca"

            item = dataset[0]
            conv = dataset.conversation(item)
            assert len(conv.messages) == 2
            assert conv.messages[0].role == Role.USER
            assert "Translate to French" in conv.messages[0].content
        finally:
            Path(temp_path).unlink()

    def test_auto_detection_alpaca(self, sample_alpaca_data):
        """Test auto-detection of Alpaca format."""
        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, mode="w"
        ) as f:
            writer = jsonlines.Writer(f)
            writer.write_all(sample_alpaca_data)
            writer.close()
            temp_path = f.name

        try:
            dataset = GenericSftDataset(
                dataset_path=temp_path,
            )

            assert dataset.converter_name == "alpaca"
        finally:
            Path(temp_path).unlink()

    def test_with_converter_kwargs(self, sample_alpaca_data):
        """Test converter with factory kwargs for system prompt."""
        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, mode="w"
        ) as f:
            writer = jsonlines.Writer(f)
            writer.write_all(sample_alpaca_data)
            writer.close()
            temp_path = f.name

        try:
            dataset = GenericSftDataset(
                dataset_path=temp_path,
                converter="alpaca",
                converter_kwargs={"include_system_prompt": True},
            )

            item = dataset[0]
            conv = dataset.conversation(item)
            # With system prompt, should have 3 messages
            assert len(conv.messages) == 3
            assert conv.messages[0].role == Role.SYSTEM
        finally:
            Path(temp_path).unlink()


class TestGenericSftDatasetWithShareGPTFormat:
    """Tests for GenericSftDataset with ShareGPT format."""

    def test_explicit_converter(self, sample_sharegpt_data):
        """Test loading with explicit sharegpt converter."""
        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, mode="w"
        ) as f:
            writer = jsonlines.Writer(f)
            writer.write_all(sample_sharegpt_data)
            writer.close()
            temp_path = f.name

        try:
            dataset = GenericSftDataset(
                dataset_path=temp_path,
                converter="sharegpt",
            )

            assert len(dataset) == 1
            assert dataset.converter_name == "sharegpt"

            item = dataset[0]
            conv = dataset.conversation(item)
            assert len(conv.messages) == 2
            assert conv.messages[0].role == Role.USER
            assert conv.messages[0].content == "What is AI?"
        finally:
            Path(temp_path).unlink()


class TestGenericSftDatasetErrors:
    """Tests for GenericSftDataset error handling."""

    def test_unknown_converter_raises(self, sample_oumi_data):
        """Test that unknown converter raises ValueError."""
        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, mode="w"
        ) as f:
            writer = jsonlines.Writer(f)
            writer.write_all(sample_oumi_data)
            writer.close()
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Unknown converter"):
                GenericSftDataset(
                    dataset_path=temp_path,
                    converter="nonexistent_converter",
                )
        finally:
            Path(temp_path).unlink()

    def test_empty_dataset_auto_detect_raises(self):
        """Test that auto-detection with empty dataset raises ValueError."""
        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, mode="w"
        ) as f:
            # Write empty file
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Cannot auto-detect"):
                GenericSftDataset(
                    dataset_path=temp_path,
                    converter="auto",
                )
        finally:
            Path(temp_path).unlink()
