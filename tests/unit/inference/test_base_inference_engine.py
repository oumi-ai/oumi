import tempfile
from pathlib import Path
from unittest import mock
from unittest.mock import patch

import jsonlines
import pytest

from oumi.core.configs import GenerationParams, InferenceConfig, ModelParams
from oumi.core.inference import (
    BaseInferenceEngine,
    FailureDetail,
    InferenceErrorType,
    InferenceResult,
)
from oumi.core.types.conversation import Conversation, Message, Role


class MockInferenceEngine(BaseInferenceEngine):
    """Mock inference engine for testing scratch file functionality."""

    def get_supported_params(self) -> set[str]:
        return {"max_new_tokens", "temperature"}

    def _infer_online(
        self,
        input: list[Conversation],
        inference_config: InferenceConfig | None = None,
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
        inference_config: InferenceConfig | None = None,
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

        # Run inference with a patched _cleanup_scratch_file to prevent cleanup
        with patch.object(mock_engine, "_cleanup_scratch_file") as mock_cleanup:
            results = mock_engine.infer(
                input=conversations,
                inference_config=inference_config,
            )

            # Get the actual scratch path used by the engine
            scratch_path = Path(mock_engine._get_scratch_filepath(output_path))

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

            # Get scratch path and verify it was cleaned up after first inference
            scratch_path = Path(mock_engine._get_scratch_filepath(output_path))
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

            # Get scratch path and verify it was cleaned up after final inference
            scratch_path = Path(mock_engine._get_scratch_filepath(output_path))
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

            # Get scratch path and verify it exists and contains first conversation
            scratch_path = Path(mock_engine._get_scratch_filepath(output_path))
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

            # Get the actual scratch path and verify that it was cleaned up
            scratch_path = Path(mock_engine._get_scratch_filepath(output_path))
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

        # Get the actual scratch path and verify it was cleaned up despite the error
        scratch_path = Path(mock_engine._get_scratch_filepath(output_path))
        assert not scratch_path.exists()


def test_empty_scratch_file(mock_engine):
    """Ensure that inference works even if the scratch file is empty."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = str(Path(temp_dir) / "output.jsonl")
        inference_config = InferenceConfig(
            output_path=output_path,
            generation=GenerationParams(max_new_tokens=10),
        )

        # Create scratch directory and empty file using the engine's actual path
        actual_scratch_path = Path(mock_engine._get_scratch_filepath(output_path))
        actual_scratch_path.parent.mkdir(parents=True, exist_ok=True)
        actual_scratch_path.touch()

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

        # Create scratch file with all conversations using engine's actual path
        mock_engine._dataset_hash = (
            "8a03cc24121fd45c48ee2950b404ff17f66caa7a04284cd9b8b7aab9cf63996e"
        )
        actual_scratch_path = Path(mock_engine._get_scratch_filepath(output_path))
        actual_scratch_path.parent.mkdir(parents=True, exist_ok=True)
        actual_scratch_path.touch()
        with jsonlines.open(actual_scratch_path, "w") as writer:
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

        # Get the actual scratch path and verify that it was cleaned up
        scratch_path = Path(mock_engine._get_scratch_filepath(output_path))
        assert not scratch_path.exists(), (
            "Scratch file should be cleaned up after successful inference"
        )


def test_inference_result_error_messages_derived_from_failures():
    conversation = Conversation(
        messages=[Message(role=Role.USER, content="hi")],
    )
    result = InferenceResult(
        successful=[(0, conversation)],
        failures={
            1: FailureDetail(
                error_message="timeout", error_type=InferenceErrorType.RUNTIME
            ),
            2: FailureDetail(
                error_message="bad request",
                status_code=400,
                is_retryable=False,
                error_type=InferenceErrorType.API_STATUS,
            ),
        },
    )

    assert result.has_failures
    assert result.failed_indices == [1, 2]
    assert result.error_messages == {1: "timeout", 2: "bad request"}
    assert result.failures[2].status_code == 400
    assert not result.failures[2].is_retryable
    assert result.failures[1].is_retryable


def test_inference_result_no_failures():
    result = InferenceResult(successful=[], failures={})

    assert not result.has_failures
    assert result.failed_indices == []
    assert result.error_messages == {}


def test_failure_detail_defaults():
    detail = FailureDetail(error_message="boom")

    assert detail.status_code is None
    assert detail.is_retryable
    assert detail.error_type == InferenceErrorType.UNKNOWN


class FailAfterFirstMockEngine(MockInferenceEngine):
    """Mock engine that checkpoints the first conversation then fails."""

    def _infer_online(
        self,
        input: list[Conversation],
        inference_config: InferenceConfig | None = None,
    ) -> list[Conversation]:
        for idx, conv in enumerate(input):
            if idx >= 1:
                raise RuntimeError("Simulated failure")
            new_conv = conv.model_copy(deep=True)
            new_conv.messages.append(
                Message(role=Role.ASSISTANT, content=f"Mock response {idx}")
            )
            self._save_conversation_to_scratch(
                new_conv,
                inference_config.output_path if inference_config else None,
            )
        raise RuntimeError("Simulated failure")


@pytest.fixture
def failing_engine():
    return FailAfterFirstMockEngine(model_params=ModelParams(model_name="test-model"))


def test_infer_partial_all_success(mock_engine):
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = str(Path(temp_dir) / "output.jsonl")
        inference_config = InferenceConfig(
            output_path=output_path,
            generation=GenerationParams(max_new_tokens=10),
        )
        conversations = [create_test_conversation(idx) for idx in range(3)]

        result = mock_engine.infer_partial(
            input=conversations, inference_config=inference_config
        )

        assert not result.has_failures
        assert result.failed_indices == []
        assert result.error_messages == {}
        assert [idx for idx, _ in result.successful] == [0, 1, 2]
        for idx, conv in result.successful:
            assert conv.conversation_id == f"test-{idx}"
            assert conv.messages[-1].role == Role.ASSISTANT

        # Scratch cleaned up on success; output contains all successes.
        scratch_path = Path(mock_engine._get_scratch_filepath(output_path))
        assert not scratch_path.exists()
        with jsonlines.open(output_path) as reader:
            saved = [Conversation.from_dict(line) for line in reader]
        assert len(saved) == 3


def test_infer_partial_parity_with_infer(mock_engine):
    conversations = [create_test_conversation(idx) for idx in range(3)]

    infer_results = mock_engine.infer(
        input=[c.model_copy(deep=True) for c in conversations]
    )
    partial_result = mock_engine.infer_partial(
        input=[c.model_copy(deep=True) for c in conversations]
    )

    assert [c.conversation_id for c in infer_results] == [
        conv.conversation_id for _, conv in partial_result.successful
    ]
    assert [str(c.messages[-1].content) for c in infer_results] == [
        str(conv.messages[-1].content) for _, conv in partial_result.successful
    ]


def test_infer_partial_default_wrapper_credits_checkpointed_rows(failing_engine):
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = str(Path(temp_dir) / "output.jsonl")
        inference_config = InferenceConfig(
            output_path=output_path,
            generation=GenerationParams(max_new_tokens=10),
        )
        conversations = [create_test_conversation(idx) for idx in range(3)]

        result = failing_engine.infer_partial(
            input=conversations, inference_config=inference_config
        )

        assert result.has_failures
        assert [idx for idx, _ in result.successful] == [0]
        assert result.failed_indices == [1, 2]
        for idx in result.failed_indices:
            assert result.failures[idx].error_type == InferenceErrorType.ENGINE_FAILURE
            assert result.failures[idx].is_retryable
            assert "Simulated failure" in result.failures[idx].error_message

        # Scratch retained on failure for resume.
        scratch_path = Path(failing_engine._get_scratch_filepath(output_path))
        assert scratch_path.exists()

        # Output contains only the successful conversation.
        with jsonlines.open(output_path) as reader:
            saved = [Conversation.from_dict(line) for line in reader]
        assert len(saved) == 1
        assert saved[0].conversation_id == "test-0"


def test_infer_partial_resumes_from_scratch_after_failure(failing_engine, mock_engine):
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = str(Path(temp_dir) / "output.jsonl")
        inference_config = InferenceConfig(
            output_path=output_path,
            generation=GenerationParams(max_new_tokens=10),
        )
        conversations = [create_test_conversation(idx) for idx in range(3)]

        first = failing_engine.infer_partial(
            input=[c.model_copy(deep=True) for c in conversations],
            inference_config=inference_config,
        )
        assert first.failed_indices == [1, 2]

        # Re-run with a healthy engine: row 0 must come from scratch, only
        # the remaining rows are re-processed.
        with patch.object(
            mock_engine, "_infer_online", wraps=mock_engine._infer_online
        ) as mock_infer:
            second = mock_engine.infer_partial(
                input=[c.model_copy(deep=True) for c in conversations],
                inference_config=inference_config,
            )

        assert not second.has_failures
        assert [idx for idx, _ in second.successful] == [0, 1, 2]
        processed_ids = [c.conversation_id for c in mock_infer.call_args[0][0]]
        assert processed_ids == ["test-1", "test-2"]

        # Scratch cleaned after the fully successful run.
        scratch_path = Path(mock_engine._get_scratch_filepath(output_path))
        assert not scratch_path.exists()


def test_infer_partial_empty_input(mock_engine):
    result = mock_engine.infer_partial(input=[])

    assert result.successful == []
    assert result.failed_indices == []
    assert not result.has_failures


def test_infer_partial_duplicate_conversation_ids_raise(mock_engine):
    conversations = [create_test_conversation(1), create_test_conversation(1)]

    with pytest.raises(ValueError, match="is not unique"):
        mock_engine.infer_partial(input=conversations)


def test_infer_partial_writes_progress_file(mock_engine):
    import json as json_lib

    with tempfile.TemporaryDirectory() as temp_dir:
        progress_path = str(Path(temp_dir) / "progress.json")
        conversations = [create_test_conversation(idx) for idx in range(3)]

        mock_engine.infer_partial(input=conversations, progress_path=progress_path)

        with open(progress_path) as f:
            snapshot = json_lib.load(f)
        assert snapshot["total"] == 3
        assert snapshot["completed"] == 3
        assert snapshot["failed"] == 0


def test_infer_partial_progress_file_via_config_with_failures(failing_engine):
    import json as json_lib

    with tempfile.TemporaryDirectory() as temp_dir:
        progress_path = str(Path(temp_dir) / "progress.json")
        output_path = str(Path(temp_dir) / "output.jsonl")
        inference_config = InferenceConfig(
            output_path=output_path,
            progress_path=progress_path,
            generation=GenerationParams(max_new_tokens=10),
        )
        conversations = [create_test_conversation(idx) for idx in range(3)]

        result = failing_engine.infer_partial(
            input=conversations, inference_config=inference_config
        )

        assert result.failed_indices == [1, 2]
        with open(progress_path) as f:
            snapshot = json_lib.load(f)
        assert snapshot["total"] == 3
        assert snapshot["completed"] == 1
        assert snapshot["failed"] == 2
        assert snapshot["completed"] + snapshot["failed"] == snapshot["total"]


def test_infer_partial_progress_write_failure_does_not_raise(mock_engine):
    with tempfile.TemporaryDirectory() as temp_dir:
        progress_path = str(Path(temp_dir) / "progress.json")
        conversations = [create_test_conversation(idx) for idx in range(2)]

        with patch("os.replace", side_effect=OSError("disk full")):
            result = mock_engine.infer_partial(
                input=conversations, progress_path=progress_path
            )

        assert not result.has_failures
        assert len(result.successful) == 2


def test_infer_partial_outcome_count_mismatch_raises(mock_engine):
    conversations = [create_test_conversation(idx) for idx in range(2)]

    with patch.object(mock_engine, "_infer_online_partial", return_value=[]):
        with pytest.raises(RuntimeError, match="returned 0 outcomes"):
            mock_engine.infer_partial(input=conversations)


def test_existing_infer_tests_unaffected_by_partial_run(mock_engine):
    """infer() after infer_partial() on the same engine behaves normally."""
    conversations = [create_test_conversation(idx) for idx in range(2)]

    partial = mock_engine.infer_partial(
        input=[c.model_copy(deep=True) for c in conversations]
    )
    results = mock_engine.infer(input=[c.model_copy(deep=True) for c in conversations])

    assert len(partial.successful) == 2
    assert len(results) == 2
