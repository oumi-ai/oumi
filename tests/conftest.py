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
    yield  # Let the test run first

    # Only cleanup for GPU-related tests to avoid overhead
    gpu_markers = {"single_gpu", "multi_gpu"}
    test_markers = {mark.name for mark in request.node.iter_markers()}

    # Also check for GPU-related decorators in the test
    has_gpu_decorator = any(
        "requires_cuda" in str(mark) or "requires_gpu" in str(mark)
        for mark in request.node.iter_markers()
    )

    # Check if test function uses GPU device mapping
    test_source = ""
    try:
        import inspect

        test_source = inspect.getsource(request.node.function)
        has_device_map = "get_default_device_map_for_inference" in test_source
    except Exception:
        has_device_map = False

    if gpu_markers.intersection(test_markers) or has_gpu_decorator or has_device_map:
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


@pytest.fixture
def single_turn_conversation():
    return Conversation(
        messages=[
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="Hi there!"),
        ]
    )
