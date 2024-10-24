from pathlib import Path

import pytest

from oumi.utils.logging import get_logger


@pytest.fixture
def root_testdata_dir():
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
