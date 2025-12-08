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

"""Tests for the oumi view command."""

import json
import tempfile
from pathlib import Path

import pytest


class TestExtractConversationData:
    """Tests for the _extract_conversation_data method."""

    @pytest.fixture
    def extractor(self):
        """Create a minimal instance to test _extract_conversation_data."""
        # Import here to avoid issues if textual is not installed
        try:
            from oumi.cli.view_app import ConversationViewerApp

            # Create a dummy app just to access the method
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False
            ) as f:
                f.write('{"messages": []}\n')
                temp_path = f.name

            app = ConversationViewerApp(file_path=temp_path)
            yield app._extract_conversation_data

            # Cleanup
            Path(temp_path).unlink(missing_ok=True)
        except ImportError:
            pytest.skip("textual not installed")

    def test_direct_format_with_messages(self, extractor):
        """Test direct format with messages at top level."""
        data = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            "metadata": {"source": "test"},
        }
        result = extractor(data)
        assert result == data
        assert "messages" in result
        assert len(result["messages"]) == 2

    def test_synth_format(self, extractor):
        """Test synth output format with synth_conversation field."""
        data = {
            "synth_conversation": {
                "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "4"},
                ]
            },
            "synth_question": "What is 2+2?",
            "synth_answer": "4",
        }
        result = extractor(data)
        assert result is not None
        assert "messages" in result
        assert len(result["messages"]) == 2
        # Check that synth_question/synth_answer are added to metadata
        assert "metadata" in result
        assert result["metadata"]["synth_question"] == "What is 2+2?"
        assert result["metadata"]["synth_answer"] == "4"

    def test_synth_format_with_existing_metadata(self, extractor):
        """Test synth format preserves existing metadata."""
        data = {
            "synth_conversation": {
                "messages": [{"role": "user", "content": "Hi"}],
                "metadata": {"existing_key": "existing_value"},
            },
            "synth_question": "Hi",
        }
        result = extractor(data)
        assert result is not None
        assert result["metadata"]["existing_key"] == "existing_value"
        assert result["metadata"]["synth_question"] == "Hi"

    def test_nested_conversation_field(self, extractor):
        """Test nested conversation field format."""
        data = {
            "conversation": {
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Help me"},
                ]
            },
            "other_field": "ignored",
        }
        result = extractor(data)
        assert result is not None
        assert "messages" in result
        assert len(result["messages"]) == 2

    def test_single_message_format(self, extractor):
        """Test single message with role/content treated as conversation."""
        data = {"role": "user", "content": "Single message"}
        result = extractor(data)
        assert result is not None
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Single message"

    def test_unsupported_format_returns_none(self, extractor):
        """Test unsupported format returns None."""
        data = {"some_random_key": "some_value", "another_key": 123}
        result = extractor(data)
        assert result is None

    def test_empty_messages_array(self, extractor):
        """Test empty messages array is still valid."""
        data = {"messages": []}
        result = extractor(data)
        assert result == data
        assert result["messages"] == []

    def test_messages_with_multimodal_content(self, extractor):
        """Test messages with list content (multimodal)."""
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "content": "Describe this image"},
                        {"type": "image", "content": "base64data..."},
                    ],
                }
            ]
        }
        result = extractor(data)
        assert result == data


class TestConversationViewerApp:
    """Tests for the ConversationViewerApp class."""

    @pytest.fixture
    def sample_jsonl_file(self):
        """Create a sample JSONL file for testing."""
        conversations = [
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi!"},
                ]
            },
            {
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Help me"},
                    {"role": "assistant", "content": "Sure!"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Search test content here"},
                    {"role": "assistant", "content": "Found it!"},
                ]
            },
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            for conv in conversations:
                f.write(json.dumps(conv) + "\n")
            temp_path = f.name

        yield temp_path

        # Cleanup
        Path(temp_path).unlink(missing_ok=True)

    def test_load_conversations(self, sample_jsonl_file):
        """Test that conversations are loaded correctly."""
        try:
            from oumi.cli.view_app import ConversationViewerApp

            app = ConversationViewerApp(file_path=sample_jsonl_file)
            app.load_conversations()

            assert len(app.conversations) == 3
            assert len(app.conversations[0].messages) == 2
            assert len(app.conversations[1].messages) == 3
        except ImportError:
            pytest.skip("textual not installed")

    def test_load_conversations_with_start_index(self, sample_jsonl_file):
        """Test starting from a specific index."""
        try:
            from oumi.cli.view_app import ConversationViewerApp

            app = ConversationViewerApp(file_path=sample_jsonl_file, start_index=1)
            app.load_conversations()

            assert app.current_index == 1
        except ImportError:
            pytest.skip("textual not installed")

    def test_load_conversations_handles_malformed_lines(self):
        """Test that malformed lines are skipped."""
        try:
            from oumi.cli.view_app import ConversationViewerApp

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False
            ) as f:
                f.write('{"messages": [{"role": "user", "content": "Valid"}]}\n')
                f.write("not valid json\n")
                f.write('{"messages": [{"role": "user", "content": "Also valid"}]}\n')
                temp_path = f.name

            app = ConversationViewerApp(file_path=temp_path)
            app.load_conversations()

            # Should have loaded 2 valid conversations, skipped the malformed one
            assert len(app.conversations) == 2

            Path(temp_path).unlink(missing_ok=True)
        except ImportError:
            pytest.skip("textual not installed")

    def test_from_stdin_flag(self, sample_jsonl_file):
        """Test that from_stdin flag is properly set."""
        try:
            from oumi.cli.view_app import ConversationViewerApp

            # Test with from_stdin=False (default)
            app1 = ConversationViewerApp(file_path=sample_jsonl_file)
            assert app1.from_stdin is False

            # Test with from_stdin=True
            app2 = ConversationViewerApp(
                file_path=sample_jsonl_file, from_stdin=True
            )
            assert app2.from_stdin is True
        except ImportError:
            pytest.skip("textual not installed")


class TestStatsComputation:
    """Tests for statistics computation."""

    @pytest.fixture
    def app_with_conversations(self):
        """Create an app with loaded conversations."""
        try:
            from oumi.cli.view_app import ConversationViewerApp

            conversations = [
                {
                    "messages": [
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi there!"},
                    ],
                    "metadata": {"source": "test1"},
                },
                {
                    "messages": [
                        {"role": "system", "content": "Be helpful"},
                        {"role": "user", "content": "Help"},
                        {"role": "assistant", "content": "Sure"},
                    ],
                    "metadata": {"source": "test2", "category": "help"},
                },
                {
                    "messages": [
                        {"role": "user", "content": "Bye"},
                        {"role": "assistant", "content": "Goodbye!"},
                    ]
                },
            ]

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False
            ) as f:
                for conv in conversations:
                    f.write(json.dumps(conv) + "\n")
                temp_path = f.name

            app = ConversationViewerApp(file_path=temp_path)
            app.load_conversations()

            yield app

            Path(temp_path).unlink(missing_ok=True)
        except ImportError:
            pytest.skip("textual not installed")

    def test_stats_screen_computation(self, app_with_conversations):
        """Test that StatsScreen computes statistics correctly."""
        try:
            from oumi.cli.view_app import StatsScreen

            stats_screen = StatsScreen(
                app_with_conversations.conversations,
                app_with_conversations.file_path,
            )
            stats = stats_screen._compute_stats()

            assert stats["total_conversations"] == 3
            assert stats["total_messages"] == 7  # 2 + 3 + 2
            assert stats["min_messages"] == 2
            assert stats["max_messages"] == 3
            assert stats["conversations_with_metadata"] == 2

            # Check role counts
            assert stats["role_counts"]["user"] == 3
            assert stats["role_counts"]["assistant"] == 3
            assert stats["role_counts"]["system"] == 1

            # Check metadata keys
            assert "source" in stats["metadata_keys"]
            assert "category" in stats["metadata_keys"]
        except ImportError:
            pytest.skip("textual not installed")


class TestSearchHighlight:
    """Tests for search highlighting functionality."""

    def test_highlight_search_basic(self):
        """Test basic search highlighting."""
        try:
            from oumi.cli.view_app import MessageWidget

            widget = MessageWidget(
                role="user",
                content="Hello world",
                search_term="world",
            )
            result = widget._highlight_search("Hello world")
            assert "[reverse]world[/reverse]" in result
            assert "Hello" in result
        except ImportError:
            pytest.skip("textual not installed")

    def test_highlight_search_case_insensitive(self):
        """Test case-insensitive search highlighting."""
        try:
            from oumi.cli.view_app import MessageWidget

            widget = MessageWidget(
                role="user",
                content="Hello WORLD",
                search_term="world",
            )
            result = widget._highlight_search("Hello WORLD")
            assert "[reverse]WORLD[/reverse]" in result
        except ImportError:
            pytest.skip("textual not installed")

    def test_highlight_search_multiple_matches(self):
        """Test highlighting multiple matches."""
        try:
            from oumi.cli.view_app import MessageWidget

            widget = MessageWidget(
                role="user",
                content="test test test",
                search_term="test",
            )
            result = widget._highlight_search("test test test")
            assert result.count("[reverse]test[/reverse]") == 3
        except ImportError:
            pytest.skip("textual not installed")

    def test_highlight_search_no_term(self):
        """Test no highlighting when search term is empty."""
        try:
            from oumi.cli.view_app import MessageWidget

            widget = MessageWidget(
                role="user",
                content="Hello world",
                search_term="",
            )
            result = widget._highlight_search("Hello world")
            assert result == "Hello world"
            assert "[reverse]" not in result
        except ImportError:
            pytest.skip("textual not installed")

    def test_highlight_search_special_chars(self):
        """Test highlighting with regex special characters."""
        try:
            from oumi.cli.view_app import MessageWidget

            widget = MessageWidget(
                role="user",
                content="Test (foo) [bar]",
                search_term="(foo)",
            )
            result = widget._highlight_search("Test (foo) [bar]")
            assert "[reverse](foo)[/reverse]" in result
        except ImportError:
            pytest.skip("textual not installed")


# =============================================================================
# Training Viewer Tests
# =============================================================================


class TestTrainingFolderDetection:
    """Tests for training folder detection."""

    def test_detects_training_folder_with_trainer_state(self, tmp_path):
        """Test that a folder with trainer_state.json is detected as training folder."""
        from oumi.cli.view import _is_training_folder

        # Create trainer_state.json
        trainer_state = {"global_step": 10, "max_steps": 100}
        (tmp_path / "trainer_state.json").write_text(json.dumps(trainer_state))

        assert _is_training_folder(tmp_path) is True

    def test_rejects_folder_without_trainer_state(self, tmp_path):
        """Test that a folder without trainer_state.json is not detected."""
        from oumi.cli.view import _is_training_folder

        # Create some other file
        (tmp_path / "some_file.txt").write_text("hello")

        assert _is_training_folder(tmp_path) is False

    def test_rejects_regular_file(self, tmp_path):
        """Test that a regular file is not detected as training folder."""
        from oumi.cli.view import _is_training_folder

        file_path = tmp_path / "not_a_folder.txt"
        file_path.write_text("hello")

        assert _is_training_folder(file_path) is False


class TestTrainingDataLoader:
    """Tests for training data loading."""

    @pytest.fixture
    def sample_training_folder(self, tmp_path):
        """Create a sample training folder with minimal data."""
        # Create trainer_state.json
        trainer_state = {
            "global_step": 10,
            "max_steps": 10,
            "epoch": 0.5,
            "total_flos": 1e12,
            "num_input_tokens_seen": 50000,
            "train_batch_size": 4,
            "log_history": [
                {"step": 5, "loss": 2.0, "learning_rate": 1e-4},
                {"step": 10, "loss": 1.5, "learning_rate": 5e-5},
            ],
        }
        (tmp_path / "trainer_state.json").write_text(json.dumps(trainer_state))

        return tmp_path

    def test_loads_basic_training_data(self, sample_training_folder):
        """Test loading basic training data from trainer_state.json."""
        from oumi.cli.view_training_app import load_training_data

        data = load_training_data(sample_training_folder)

        assert data.global_step == 10
        assert data.max_steps == 10
        assert data.epoch == 0.5
        assert data.num_input_tokens_seen == 50000
        assert len(data.log_history) == 2

    def test_handles_missing_optional_files(self, sample_training_folder):
        """Test that missing optional files are handled gracefully."""
        from oumi.cli.view_training_app import load_training_data

        data = load_training_data(sample_training_folder)

        # Optional files should be None
        assert data.telemetry is None
        assert data.metrics_summary is None
        assert data.training_config is None
        assert data.model_config is None

    def test_raises_on_missing_trainer_state(self, tmp_path):
        """Test that missing trainer_state.json raises an error."""
        from oumi.cli.view_training_app import load_training_data

        with pytest.raises(FileNotFoundError):
            load_training_data(tmp_path)

    def test_discovers_checkpoints(self, sample_training_folder):
        """Test that checkpoint directories are discovered."""
        from oumi.cli.view_training_app import load_training_data

        # Create a checkpoint directory
        checkpoint_dir = sample_training_folder / "checkpoint-5"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "model.safetensors").write_text("model data")
        (checkpoint_dir / "optimizer.pt").write_text("optimizer data")

        data = load_training_data(sample_training_folder)

        assert len(data.checkpoints) == 1
        assert data.checkpoints[0].step == 5
        assert data.checkpoints[0].has_model is True
        assert data.checkpoints[0].has_optimizer is True

    def test_determines_training_complete(self, sample_training_folder):
        """Test that is_complete is determined correctly."""
        from oumi.cli.view_training_app import load_training_data

        data = load_training_data(sample_training_folder)

        # global_step == max_steps means complete
        assert data.is_complete is True


class TestLogParsing:
    """Tests for log file parsing."""

    def test_parses_standard_log_line(self):
        """Test parsing a standard log line."""
        from oumi.cli.view_training_app import _parse_log_line

        line = (
            "[2025-11-21 09:59:33,027][oumi][rank0][pid:30141][MainThread][INFO]]"
            "[train.py:165] Starting training..."
        )
        entry = _parse_log_line(line)

        assert entry is not None
        assert entry.level == "INFO"
        assert entry.source == "train.py:165"
        assert "Starting training" in entry.message

    def test_parses_warning_log_line(self):
        """Test parsing a warning log line."""
        from oumi.cli.view_training_app import _parse_log_line

        line = (
            "[2025-11-21 09:59:36,311][oumi][rank0][pid:30141][MainThread][WARNING]]"
            "[callbacks.py:66] MFU logging is only supported on GPU."
        )
        entry = _parse_log_line(line)

        assert entry is not None
        assert entry.level == "WARNING"
        assert "MFU" in entry.message

    def test_returns_none_for_invalid_line(self):
        """Test that invalid lines return None."""
        from oumi.cli.view_training_app import _parse_log_line

        entry = _parse_log_line("This is not a valid log line")
        assert entry is None


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_format_size(self):
        """Test size formatting."""
        from oumi.cli.view_training_app import format_size

        assert "B" in format_size(100)
        assert "KB" in format_size(1024)
        assert "MB" in format_size(1024 * 1024)
        assert "GB" in format_size(1024 * 1024 * 1024)

    def test_format_number(self):
        """Test number formatting with suffixes."""
        from oumi.cli.view_training_app import format_number

        assert format_number(500) == "500"
        assert "K" in format_number(1500)
        assert "M" in format_number(1_500_000)
        assert "B" in format_number(1_500_000_000)
        assert "T" in format_number(1_500_000_000_000)
