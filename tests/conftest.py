import asyncio
import inspect
from pathlib import Path
from unittest.mock import Mock

import aioresponses.core as aioresponses_core
import pytest
from aiohttp import ClientResponse, StreamReader
from aiohttp.client_proto import ResponseHandler
from datasets import config as datasets_config

from oumi.core.types.conversation import Conversation, Message, Role
from oumi.utils.logging import get_logger


def _patch_aioresponses_for_aiohttp_3_14() -> None:
    """Let aioresponses (<=0.7.9) build mocked responses on aiohttp 3.14+.

    aiohttp 3.14 added a required ``stream_writer`` argument to
    ``ClientResponse`` and made flow control call into the protocol's parser,
    which aioresponses never sets. Remove once aioresponses releases
    pnuckowski/aioresponses#288 and #292.
    """
    if "stream_writer" not in inspect.signature(ClientResponse).parameters:
        return

    class _ClientResponse(ClientResponse):
        def __init__(self, *args, **kwargs):
            # aiohttp only reads ``stream_writer.output_size``.
            kwargs.setdefault("stream_writer", Mock(output_size=0))
            super().__init__(*args, **kwargs)

    def _stream_reader_factory(loop: asyncio.AbstractEventLoop) -> StreamReader:
        protocol = ResponseHandler(loop=loop)
        protocol._parser = Mock()  # type: ignore[attr-defined]
        protocol._parser.feed_data.return_value = ([], False, b"")  # type: ignore[attr-defined]
        return StreamReader(protocol, limit=2**16, loop=loop)

    aioresponses_core.ClientResponse = _ClientResponse  # type: ignore[misc]
    aioresponses_core.stream_reader_factory = _stream_reader_factory


_patch_aioresponses_for_aiohttp_3_14()


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
