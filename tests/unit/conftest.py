from unittest.mock import MagicMock

import pytest

from oumi.builders import build_chat_template
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer


def create_mock_tokenizer(
    *,
    eos_token_id: int | None = None,
    pad_token_id: int | None = 32001,
    model_max_length: int = 1024,
    chat_template: str | None = None,
    tokenize_func=None,
) -> MagicMock:
    """Create a mock tokenizer that doesn't inherit from PreTrainedTokenizer."""
    mock = MagicMock(spec=BaseTokenizer)
    mock.eos_token_id = eos_token_id
    mock.pad_token_id = pad_token_id
    mock.model_max_length = model_max_length

    if chat_template is None:
        mock.chat_template = build_chat_template(template_name="default")
    else:
        mock.chat_template = chat_template

    if tokenize_func is not None:
        mock.side_effect = tokenize_func

    return mock


@pytest.fixture
def mock_tokenizer():
    return create_mock_tokenizer()
