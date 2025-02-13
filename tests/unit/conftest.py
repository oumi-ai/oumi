from unittest.mock import MagicMock

import pytest

from oumi.builders import build_chat_template
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer


@pytest.fixture
def mock_tokenizer():
    """Fixture that provides a mock tokenizer for testing purposes.

This fixture creates a mock instance of `BaseTokenizer` with predefined
attributes, suitable for use in unit tests that require a tokenizer.

Returns:
    MagicMock: A mock object with attributes `chat_template`, `pad_token_id`,
    and `model_max_length` set to simulate a tokenizer."""
    mock = MagicMock(spec=BaseTokenizer)
    mock.chat_template = build_chat_template(template_name="default")
    mock.pad_token_id = 32001
    mock.model_max_length = 1024
    return mock
