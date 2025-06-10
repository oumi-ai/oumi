import functools
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from oumi.builders import build_tokenizer
from oumi.core.collators.text_completions_collator_with_padding import (
    TextCompletionsCollatorWithPadding,
)
from oumi.core.configs import ModelParams
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer
from oumi.utils import logging


@pytest.fixture
def mock_tokenizer():
    mock = MagicMock(spec=BaseTokenizer)
    mock.pad_token_id = 32001
    return mock


@functools.cache  # same as @cache added in Python 3.9
def create_test_tokenizer() -> tuple[BaseTokenizer, int]:
    tokenizer = build_tokenizer(
        ModelParams(
            model_name="MlpEncoder",
            torch_dtype_str="float16",
            trust_remote_code=False,
            tokenizer_name="gpt2",
            tokenizer_pad_token="<|endoftext|>",
        )
    )
    assert tokenizer.pad_token_id
    assert isinstance(tokenizer.pad_token_id, int)
    return tokenizer, int(tokenizer.pad_token_id)


def test_success_basic():
    tokenizer, pad_token_id = create_test_tokenizer()

    instruction_prefix = "ignore this and after me"
    response_prefix = "ignore this but not after me"

    instruction_prefix_tokens = tokenizer.encode(
        instruction_prefix, add_special_tokens=False
    )
    response_prefix_tokens = tokenizer.encode(response_prefix, add_special_tokens=False)

    collator = TextCompletionsCollatorWithPadding(
        tokenizer=tokenizer,
        instruction_prefix=instruction_prefix,
        response_prefix=response_prefix,
    )
    assert callable(collator)

    batch = [
        # Instructions with no response, all tokens are ignored
        {"input_ids": instruction_prefix_tokens + [101] + response_prefix_tokens},
        # Response with no instructions, only in-between tokens are used
        {
            "input_ids": (
                response_prefix_tokens
                + [201, 202, 203, 204]
                + instruction_prefix_tokens
            )
        },
        # No instructions or response, all tokens are ignored
        {"input_ids": [301, 302]},
        # Normal multi-turn conversation, only tokens after response are used
        {
            "input_ids": (
                instruction_prefix_tokens
                + [301, 302]
                + response_prefix_tokens
                + [303, 304]
                + instruction_prefix_tokens
                + [305, 306]
                + response_prefix_tokens
                + [307, 308]
            )
        },
    ]

    pad_length = max([len(b["input_ids"]) for b in batch])
    pad_tokens_per_batch = [
        [-100 for _ in range(pad_length - len(b["input_ids"]))] for b in batch
    ]

    collated_batch = collator(batch)
    instruction_labels = [-100 for _ in instruction_prefix_tokens]
    response_labels = [-100 for _ in response_prefix_tokens]

    expected_input_ids = [
        [
            batch[i]["input_ids"] + [pad_token_id for _ in pad_tokens_per_batch[i]]
            for i in range(len(batch))
        ]
    ]

    expected_attention_masks = [
        [
            [1 for _ in batch[i]["input_ids"]] + [0 for _ in pad_tokens_per_batch[i]]
            for i in range(len(batch))
        ]
    ]

    expected_labels = [
        instruction_labels + [-100] + response_labels + pad_tokens_per_batch[0],
        (
            response_labels
            + [201, 202, 203, 204]
            + instruction_labels
            + pad_tokens_per_batch[1]
        ),
        [-100, -100] + pad_tokens_per_batch[2],
        (
            instruction_labels
            + [-100, -100]
            + response_labels
            + [303, 304]
            + instruction_labels
            + [-100, -100]
            + response_labels
            + [307, 308]
            + pad_tokens_per_batch[3]
        ),
    ]

    assert "input_ids" in collated_batch
    assert len(collated_batch["input_ids"]) == len(batch)
    assert isinstance(collated_batch["input_ids"], torch.Tensor)
    assert np.all(
        collated_batch["input_ids"].numpy()
        == np.array(expected_input_ids, dtype=np.int32)
    )

    assert "attention_mask" in collated_batch
    assert len(collated_batch["attention_mask"]) == len(batch)
    assert isinstance(collated_batch["attention_mask"], torch.Tensor)
    assert np.all(
        collated_batch["attention_mask"].numpy()
        == np.array(expected_attention_masks, dtype=np.int32)
    )

    assert "labels" in collated_batch
    assert len(collated_batch["labels"]) == len(batch)
    assert isinstance(collated_batch["labels"], torch.Tensor)
    assert np.all(
        collated_batch["labels"].numpy() == np.array(expected_labels, dtype=np.int32)
    )


def test_debug_logging(caplog):
    """Test that example debugging logs are correctly generated when debug=True."""
    # Set the logging level to DEBUG for both caplog and the oumi logger
    caplog.set_level("DEBUG")

    # Get and configure the oumi logger to ensure debug messages are captured
    oumi_logger = logging.get_logger("oumi")
    oumi_logger.setLevel("DEBUG")
    oumi_logger.propagate = True  # Ensure propagation to root logger

    tokenizer, pad_token_id = create_test_tokenizer()

    instruction_prefix = "ignore this and after me"
    response_prefix = "ignore this but not after me"

    instruction_prefix_tokens = tokenizer.encode(
        instruction_prefix, add_special_tokens=False
    )
    response_prefix_tokens = tokenizer.encode(response_prefix, add_special_tokens=False)

    collator = TextCompletionsCollatorWithPadding(
        tokenizer=tokenizer,
        instruction_prefix=instruction_prefix,
        response_prefix=response_prefix,
        debug=True,
    )
    assert callable(collator)

    batch = [
        # Instructions with no response, all tokens are ignored
        {"input_ids": instruction_prefix_tokens + [101] + response_prefix_tokens},
        # Response with no instructions, only in-between tokens are used
        {
            "input_ids": (
                response_prefix_tokens
                + [201, 202, 203, 204]
                + instruction_prefix_tokens
            )
        },
        # No instructions or response, all tokens are ignored
        {"input_ids": [301, 302]},
        # Normal multi-turn conversation, only tokens after response are used
        {
            "input_ids": (
                instruction_prefix_tokens
                + [301, 302]
                + response_prefix_tokens
                + [303, 304]
                + instruction_prefix_tokens
                + [305, 306]
                + response_prefix_tokens
                + [307, 308]
            )
        },
    ]

    _ = collator(batch)

    # Check that debug logs were generated and verify their content
    log_text = caplog.text

    # Get the first example's token IDs for verification
    first_example_input_ids = batch[0]["input_ids"]

    # Verify raw example (decoded without special tokens)
    expected_raw_text = tokenizer.decode(
        first_example_input_ids, skip_special_tokens=True
    )
    assert f"Raw example: {expected_raw_text}" in log_text

    # Verify formatted example (decoded with special tokens)
    expected_formatted_text = tokenizer.decode(
        first_example_input_ids, skip_special_tokens=False
    )
    assert f"Formatted example: {expected_formatted_text}" in log_text

    # Verify tokenized example (list of tuples with token_id and decoded token)
    expected_tokenized = [
        (token_id, tokenizer.decode([token_id])) for token_id in first_example_input_ids
    ]
    assert f"Tokenized example: {expected_tokenized}" in log_text

    # Verify model input contains the expected structure
    assert "'input_ids':" in log_text
    assert "'attention_mask':" in log_text
    assert "'labels':" in log_text
