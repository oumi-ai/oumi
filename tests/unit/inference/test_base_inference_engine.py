import tempfile
from pathlib import Path
from typing import Optional
from unittest import mock
from unittest.mock import patch

import jsonlines
import pytest

from oumi.core.configs import GenerationParams, InferenceConfig, ModelParams
from oumi.core.inference import BaseInferenceEngine
from oumi.core.types.conversation import Conversation, Message, Role


class MockInferenceEngine(BaseInferenceEngine):
    """Mock inference engine for testing scratch file functionality."""

    def get_supported_params(self) -> set[str]:
        return {"max_new_tokens", "temperature"}

    def _infer_online(
        self,
        input: list[Conversation],
        inference_config: Optional[InferenceConfig] = None,
    ) -> list[Conversation]:
        # Mock implementation that appends an assistant response
        results = []
        for i, conv in enumerate(input):
            new_conv = conv.model_copy(deep=True)
            new_conv.messages.append(
                Message(
                    role=Role.ASSISTANT,
                    content=f"Mock response {i}",
                )
            )
            results.append(new_conv)
            self._save_conversation_to_scratch(
                new_conv,
                inference_config.output_path if inference_config else None,
            )
        return results

    def _infer_from_file(
        self,
        input_filepath: str,
        inference_config: Optional[InferenceConfig] = None,
    ) -> list[Conversation]:
        return self.infer(inference_config=inference_config)


@pytest.fixture
def mock_engine():
    model_params = ModelParams(model_name="test-model")
    return MockInferenceEngine(model_params=model_params)


def create_test_conversation(idx: int) -> Conversation:
    """Creates a test conversation with a unique ID."""
    return Conversation(
        messages=[
            Message(
                role=Role.USER,
                content=f"Test message {idx}",
            )
        ],
        conversation_id=f"test-{idx}",
    )


def test_scratch_file_creation_and_cleanup(mock_engine):
    """Test that scratch files are created and cleaned up properly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = str(Path(temp_dir) / "output.jsonl")
        inference_config = InferenceConfig(
            output_path=output_path,
            generation=GenerationParams(max_new_tokens=10),
        )

        conversations = [create_test_conversation(1)]
        scratch_path = Path(temp_dir) / "scratch" / "output.jsonl"

        # Run inference with a patched _cleanup_scratch_file to prevent cleanup
        with patch.object(mock_engine, "_cleanup_scratch_file") as mock_cleanup:
            results = mock_engine.infer(
                input=conversations,
                inference_config=inference_config,
            )

            # Verify scratch file was created and exists
            assert scratch_path.exists(), "Scratch file should exist after inference"

            # Verify results
            assert len(results) == 1
            assert results[0].conversation_id == "test-1"
            assert len(results[0].messages) == 2  # Original + assistant response

        # Verify that cleanup was called
        mock_cleanup.assert_called_once_with(output_path)

        # Manually call cleanup since we prevented the automatic cleanup from the patch
        mock_engine._cleanup_scratch_file(output_path)


def test_infer_no_resume_from_scratch_on_success(mock_engine):
    """Test that inference processes all conversations each time since scratch is
    cleaned up after successful inference."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = str(Path(temp_dir) / "output.jsonl")
        inference_config = InferenceConfig(
            output_path=output_path,
            generation=GenerationParams(max_new_tokens=10),
        )

        # Create two conversations
        conversations = [
            create_test_conversation(1),
            create_test_conversation(2),
        ]

        scratch_path = Path(temp_dir) / "scratch" / "output.jsonl"

        with patch.object(
            mock_engine,
            "_infer_online",
            wraps=mock_engine._infer_online,
        ) as mock_infer:
            # Process only first conversation
            with patch.object(
                mock_engine,
                "_save_conversation_to_scratch",
                wraps=mock_engine._save_conversation_to_scratch,
            ) as mock_save:
                first_results = mock_engine.infer(
                    input=[conversations[0].model_copy(deep=True)],
                    inference_config=inference_config,
                )
                # Verify scratch file was created and saved to
                assert mock_save.called

            # Verify scratch file was cleaned up after first inference
            assert not scratch_path.exists()

            # Process all conversations
            with patch.object(
                mock_engine,
                "_save_conversation_to_scratch",
                wraps=mock_engine._save_conversation_to_scratch,
            ) as mock_save:
                all_results = mock_engine.infer(
                    input=conversations,  # Pass both conversations
                    inference_config=inference_config,
                )
                # Verify scratch file was created and saved to
                assert mock_save.called

            # Verify results
            assert len(first_results) == 1
            assert len(all_results) == 2
            assert all_results[0].conversation_id == "test-1"
            assert all_results[1].conversation_id == "test-2"

            # Each conversation should have the original message + assistant response
            assert len(all_results[0].messages) == 2
            assert len(all_results[1].messages) == 2

            # Verify infer_online was called with all inputs each time (no resuming)
            mock_infer.assert_has_calls(
                [
                    # First call with just the first conversation
                    mock.call(
                        [conversations[0].model_copy(deep=True)], inference_config
                    ),
                    # Second call with both conversations
                    mock.call(conversations, inference_config),
                ]
            )

            # Verify scratch file was cleaned up after final inference
            assert not scratch_path.exists()


def test_infer_resume_from_scratch_on_failure(mock_engine):
    """Test that inference resumes from scratch if it fails."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = str(Path(temp_dir) / "output.jsonl")
        inference_config = InferenceConfig(
            output_path=output_path,
            generation=GenerationParams(max_new_tokens=10),
        )

        # Create two conversations
        conversations = [
            create_test_conversation(1),
            create_test_conversation(2),
        ]

        scratch_path = Path(temp_dir) / "scratch" / "output.jsonl"

        # Run inference which fails on the second conversation
        def mock_infer_online(input_convs, config):
            # Process conversations one at a time
            results = []
            for i, conv in enumerate(input_convs):
                if i == 1:  # Fail on second conversation
                    raise RuntimeError("Failed processing second conversation")
                # Process first conversation normally
                new_conv = conv.model_copy(deep=True)
                new_conv.messages.append(
                    Message(
                        role=Role.ASSISTANT,
                        content=f"Mock response {conv.conversation_id}",
                    )
                )
                results.append(new_conv)
                # Save first conversation to scratch
                mock_engine._save_conversation_to_scratch(
                    new_conv,
                    config.output_path if config else None,
                )
            return results

        with patch.object(
            mock_engine,
            "_infer_online",
            side_effect=mock_infer_online,
        ) as mock_infer:
            # Should fail on second conversation
            with pytest.raises(RuntimeError):
                mock_engine.infer(
                    input=conversations,
                    inference_config=inference_config,
                )

            # Verify scratch file exists and contains first conversation
            assert scratch_path.exists()

            # Verify infer_online was called with both conversations
            mock_infer.assert_called_once()
            assert len(mock_infer.call_args[0][0]) == 2

            # Verify that the scratch file contains the first conversation
            with open(scratch_path) as f:
                lines = f.readlines()
                assert len(lines) == 1
                first_conv = Conversation.from_json(lines[0])
                assert (
                    first_conv.messages[0].content
                    == conversations[0].messages[0].content
                )
                assert first_conv.messages[-1].role == Role.ASSISTANT
                assert first_conv.messages[-1].content == "Mock response test-1"

        # Run inference again, this time with no errors
        # It should resume from scratch, only processing the second conversation
        with patch.object(
            mock_engine,
            "_infer_online",
            side_effect=mock_infer_online,
        ) as mock_infer:
            results = mock_engine.infer(
                input=conversations,
                inference_config=inference_config,
            )

            # Verify that results contain both conversations
            assert len(results) == 2
            assert results[0].conversation_id == "test-1"
            assert results[1].conversation_id == "test-2"
            assert len(results[0].messages) == 2
            assert len(results[1].messages) == 2

            # Verify that the scratch file was cleaned up
            assert not scratch_path.exists()

            # Verify that infer_online was called with only the second conversation
            mock_infer.assert_called_once()
            assert len(mock_infer.call_args[0][0]) == 1
            assert mock_infer.call_args[0][0][0].conversation_id == "test-2"

        # Verify that output file exists and contains expected conversations
        output_file_path = Path(output_path)
        assert output_file_path.exists(), (
            "Output file should exist after successful inference"
        )

        saved_conversations = []
        with jsonlines.open(output_path) as reader:
            for obj in reader:
                saved_conversations.append(Conversation.from_dict(obj))

        assert len(saved_conversations) == 2, (
            "Output file should contain all processed conversations"
        )

        # Verify that the saved conversations match the results
        for i, (result_conv, saved_conv) in enumerate(
            zip(results, saved_conversations)
        ):
            assert result_conv.conversation_id == saved_conv.conversation_id
            assert len(saved_conv.messages) == 2, (
                f"Conversation {i} should have original + assistant message"
            )

            # Verify original user message
            assert saved_conv.messages[0].role == Role.USER
            assert saved_conv.messages[0].content == f"Test message {i + 1}"

            # Verify assistant response was added
            assert saved_conv.messages[1].role == Role.ASSISTANT
            assert saved_conv.messages[1].content == f"Mock response test-{i + 1}"


def test_scratch_file_handling_with_errors(mock_engine):
    """Test that scratch files are handled properly even when errors occur."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = str(Path(temp_dir) / "output.jsonl")
        inference_config = InferenceConfig(
            output_path=output_path,
            generation=GenerationParams(max_new_tokens=10),
        )

        conversations = [create_test_conversation(1)]

        # Simulate an error during inference
        with patch.object(
            mock_engine, "_infer_online", side_effect=RuntimeError("Test error")
        ):
            with pytest.raises(RuntimeError):
                mock_engine.infer(
                    input=conversations,
                    inference_config=inference_config,
                )

        # Verify scratch file was cleaned up despite the error
        scratch_path = Path(temp_dir) / "scratch" / "output.jsonl"
        assert not scratch_path.exists()


def test_empty_scratch_file(mock_engine):
    """Ensure that inference works even if the scratch file is empty."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = str(Path(temp_dir) / "output.jsonl")
        inference_config = InferenceConfig(
            output_path=output_path,
            generation=GenerationParams(max_new_tokens=10),
        )

        # Create scratch directory and empty file
        scratch_path = Path(temp_dir) / "scratch"
        scratch_path.mkdir(parents=True)
        (scratch_path / "output.jsonl").touch()

        conversations = [create_test_conversation(1)]

        # Run inference
        results = mock_engine.infer(
            input=conversations,
            inference_config=inference_config,
        )

        # Verify results
        assert len(results) == 1
        assert results[0].conversation_id == "test-1"
        assert len(results[0].messages) == 2  # Original + assistant response


def test_full_scratch_file(mock_engine):
    """Validate that inference doesn't run if scratch file has all conversations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = str(Path(temp_dir) / "output.jsonl")
        inference_config = InferenceConfig(
            output_path=output_path,
            generation=GenerationParams(max_new_tokens=10),
        )

        conversations = [create_test_conversation(1)]

        # Create scratch directory and file with all conversations
        scratch_path = Path(temp_dir) / "scratch"
        scratch_path.mkdir(parents=True)
        (scratch_path / "output.jsonl").touch()
        with jsonlines.open(scratch_path / "output.jsonl", "w") as writer:
            writer.write(conversations[0].to_dict())

        # Run inference
        results = mock_engine.infer(
            input=conversations,
            inference_config=inference_config,
        )

        # Verify results
        assert len(results) == 1
        assert results[0].conversation_id == "test-1"
        assert len(results[0].messages) == 1  # Original, no assistant response in file


def test_final_conversations_saved_to_output_file(mock_engine):
    """Test that final conversations are saved to output file after inference."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = str(Path(temp_dir) / "output.jsonl")
        inference_config = InferenceConfig(
            output_path=output_path,
            generation=GenerationParams(max_new_tokens=10),
        )

        # Create test conversations
        conversations = [
            create_test_conversation(1),
            create_test_conversation(2),
        ]

        results = mock_engine.infer(
            input=conversations,
            inference_config=inference_config,
        )

        output_file_path = Path(output_path)
        assert output_file_path.exists(), (
            "Output file should exist after successful inference"
        )

        saved_conversations = []
        with jsonlines.open(output_path) as reader:
            for obj in reader:
                saved_conversations.append(Conversation.from_dict(obj))
        assert len(saved_conversations) == 2, (
            "Output file should contain all processed conversations"
        )

        # Verify that the saved conversations match the results
        for i, (result_conv, saved_conv) in enumerate(
            zip(results, saved_conversations)
        ):
            assert result_conv.conversation_id == saved_conv.conversation_id
            assert len(saved_conv.messages) == 2, (
                f"Conversation {i} should have original + assistant message"
            )

            # Verify original user message
            assert saved_conv.messages[0].role == Role.USER
            assert saved_conv.messages[0].content == f"Test message {i + 1}"

            # Verify assistant response was added
            assert saved_conv.messages[1].role == Role.ASSISTANT
            assert saved_conv.messages[1].content == f"Mock response {i}"

        # Verify that scratch file was cleaned up
        scratch_path = Path(temp_dir) / "scratch" / "output.jsonl"
        assert not scratch_path.exists(), (
            "Scratch file should be cleaned up after successful inference"
        )
