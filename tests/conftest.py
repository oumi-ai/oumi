from pathlib import Path

import pytest

from oumi.core.types.conversation import Conversation, Message, Role
from oumi.utils.logging import get_logger


@pytest.fixture
def root_testdata_dir() -> Path:
    return Path(__file__).parent / "testdata"


@pytest.fixture(scope="session", autouse=True)
def setup_logging():
    """Fixture to set up logging for all tests.

    We want to propagate to the root logger so that
    pytest caplog can capture logs, and we can test
    logging for the default oumi logger.
    """
    logger = get_logger("oumi")
    logger.propagate = True
    return logger


@pytest.fixture(autouse=True)
def retain_logging_level():
    """Fixture to preserve the logging level between tests."""
    logger = get_logger("oumi")
    # Store the current log level
    log_level = logger.level
    yield
    # Rehydrate the log level
    logger.setLevel(log_level)


@pytest.fixture(autouse=True)
def cleanup_gpu_memory(request):
    """Automatically clean up GPU memory after GPU tests."""
    import os

    # Set memory defragmentation env var at fixture start
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    def _cleanup_gpu():
        """Perform comprehensive GPU cleanup."""
        try:
            import gc

            import torch

            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            # Silently ignore cleanup errors to avoid test failures
            pass

    # Determine if this test needs GPU cleanup
    gpu_markers = {"single_gpu", "multi_gpu"}
    test_markers = {mark.name for mark in request.node.iter_markers()}

    # Check for GPU-related decorators
    has_gpu_decorator = any(
        "requires_cuda" in str(mark) or "requires_gpu" in str(mark)
        for mark in request.node.iter_markers()
    )

    # Check if test function uses GPU device mapping or VLLM
    test_source = ""
    has_device_map = False
    has_vllm = False
    try:
        import inspect

        test_source = inspect.getsource(request.node.function)
        has_device_map = "get_default_device_map_for_inference" in test_source
        has_vllm = "vllm" in test_source.lower() or "VLLM" in test_source
    except Exception:
        pass

    # Check test path for VLLM or vision model indicators
    test_path = str(request.node.nodeid).lower()
    has_vllm_in_path = "vllm" in test_path
    has_vision_model = "with_images" in test_path or "vision" in test_path

    needs_cleanup = (
        gpu_markers.intersection(test_markers)
        or has_gpu_decorator
        or has_device_map
        or has_vllm
        or has_vllm_in_path
        or has_vision_model
    )

    # Use finalizer to ensure cleanup runs even on test failure
    if needs_cleanup:
        request.addfinalizer(_cleanup_gpu)

    yield  # Let the test run


@pytest.fixture
def single_turn_conversation():
    return Conversation(
        messages=[
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="Hi there!"),
        ]
    )


@pytest.fixture(autouse=True)
def cleanup_test_files(request):
    """Automatically clean up test-generated files after each test."""
    import tempfile
    from pathlib import Path

    # Get the current working directory at test start
    test_cwd = Path.cwd()

    def _cleanup_test_files():
        """Clean up test files in the current working directory."""
        test_file_patterns = [
            "test_output*.json",
            "test_*.json",
            "test_*.txt",
            "test_*.pdf",
            "test_*.csv",
            "test_*.md",
            "test_*.cast",
            "test_*.bin",
            "*_test_*.json",
            "*_test_*.txt",
            "*_test_*.pdf",
            "*_test_*.csv",
            "*_test_*.md",
            "*_test_*.cast",
            "*_test_*.bin",
            "stress_test_output*.json",
            "analysis_report*.md",
            "project_analysis*.md",
            "*_attachment*.txt",
            "*_cleanup_test_*.txt",
            "deeply_nested*.json",
            "sales_data*.json",
            "config*.json",
            "requirements*.txt",
            "readme*.md",
            "*_report*.md",
            # Command router test files
            "file1.json",
            "file2.json",
            "output.json",
            "file.txt",
            "test.json",
            "refinement_*.md",
            "demo.cast",
            # Malformed command test artifacts (these shouldn't be created!)
            "'mixed\"",
            '"unclosed',
        ]

        for pattern in test_file_patterns:
            for file_path in test_cwd.glob(pattern):
                try:
                    if file_path.is_file():
                        file_path.unlink()
                except Exception:
                    pass  # Ignore cleanup errors

        # Also clean up temp files from the system temp directory
        temp_dir = Path(tempfile.gettempdir())
        for pattern in ["tmp*test*", "*_test_*", "stress_test_*"]:
            for file_path in temp_dir.glob(pattern):
                try:
                    if file_path.is_file():
                        # Clean up files that are recent (from the current test session)
                        import time

                        current_time = time.time()
                        file_age = current_time - file_path.stat().st_mtime
                        if file_age < 3600:  # Files created in the last hour
                            file_path.unlink()
                except Exception:
                    pass  # Ignore cleanup errors

    # Use finalizer to ensure cleanup runs even on test failure
    request.addfinalizer(_cleanup_test_files)

    yield  # Let the test run
