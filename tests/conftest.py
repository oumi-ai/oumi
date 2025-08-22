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

            if not torch.cuda.is_available():
                return

            # VLLM-specific cleanup
            try:
                import ray  # type: ignore[import-untyped]

                if ray.is_initialized():
                    # Shutdown any VLLM workers
                    ray.shutdown()
            except ImportError:
                pass

            # Clear VLLM engine cache if available
            try:
                from vllm.engine.llm_engine import LLMEngine  # type: ignore[import-untyped]

                if hasattr(LLMEngine, "_clear_cache"):
                    LLMEngine._clear_cache()  # type: ignore[attr-defined]
            except (ImportError, AttributeError):
                pass

            # Aggressive PyTorch cleanup
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            # Force garbage collection
            gc.collect()

            # Additional cache clearing after GC
            torch.cuda.empty_cache()

            # Try to clear any remaining allocated memory
            if hasattr(torch.cuda, "reset_accumulated_memory_stats"):
                torch.cuda.reset_accumulated_memory_stats()

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
