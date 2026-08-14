from pathlib import Path

import pytest
from datasets import config as datasets_config

from oumi.core.types.conversation import Conversation, Message, Role
from oumi.utils.logging import get_logger


def pytest_addoption(parser):
    parser.addoption(
        "--max-gpu-memory-gb",
        type=float,
        default=None,
        help="Skip tests whose requires_gpus(min_gb=...) exceeds this value.",
    )
    parser.addoption(
        "--above-gpu-memory-gb",
        type=float,
        default=None,
        help=(
            "Skip tests whose requires_gpus(min_gb=...) is at or below this "
            "value, i.e. tests a smaller machine already covers. Pair with "
            "--max-gpu-memory-gb so each machine runs one slice of the tests."
        ),
    )


def pytest_collection_modifyitems(config, items):
    """Selects tests by the VRAM they need, per --*-gpu-memory-gb."""
    max_gb = config.getoption("--max-gpu-memory-gb")
    above_gb = config.getoption("--above-gpu-memory-gb")
    if max_gb is None and above_gb is None:
        return

    eps = 1e-2  # Matches the tolerance requires_gpus uses.
    selected, deselected = [], []
    for item in items:
        marker = item.get_closest_marker("gpu_memory_gb")
        min_gb = marker.args[0] if marker and marker.args else 0.0
        too_big = max_gb is not None and min_gb > max_gb * (1 + eps)
        covered_by_smaller = above_gb is not None and min_gb <= above_gb * (1 + eps)
        if too_big or covered_by_smaller:
            deselected.append(item)
        else:
            selected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


@pytest.fixture(autouse=True)
def disable_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DO_NOT_TRACK", "1")


@pytest.fixture
def temp_hf_datasets_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Use a temporary cache directory for HuggingFace datasets.

    This prevents stale cache issues where cached datasets have outdated
    column names or formats that don't match the current code.

    Note: datasets.disable_caching() doesn't affect from_generator(),
    so we use a temp directory instead.
    """
    monkeypatch.setattr(datasets_config, "HF_DATASETS_CACHE", str(tmp_path))


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
