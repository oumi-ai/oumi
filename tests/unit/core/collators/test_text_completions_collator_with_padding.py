import functools
import warnings
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

import oumi.core.constants as constants
from oumi.builders import build_tokenizer
from oumi.core.collators.text_completions_collator_with_padding import (
    TextCompletionsCollatorWithPadding,
)
from oumi.core.configs import ModelParams
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer
from oumi.utils import logging

IGNORE = constants.LABEL_IGNORE_INDEX

# Template strings for span-masking tests — chosen to be unambiguous in GPT-2's vocab.
_RESP_STR = " ASSISTANT_RESPONSE_START"
_EOT_STR = " TURN_ENDS_HERE"
_TC_STR = " TOOL_CALL_BEGINS"

# Arbitrary token IDs used as "content" that must not appear in any template.
_SENTINELS = [601, 602, 603, 604, 605, 606, 607, 608]


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
            tokenizer_name="openai-community/gpt2",
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
        instruction_template=instruction_prefix,
        response_template=response_prefix,
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
        instruction_template=instruction_prefix,
        response_template=response_prefix,
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


# ===========================================================================
# Span-based masking tests (tool-aware collation)
# ===========================================================================


@functools.cache
def get_template_token_ids() -> tuple[list[int], list[int], list[int]]:
    """Return (resp_ids, eot_ids, tc_ids) encoded once and cached."""
    tokenizer, _ = create_test_tokenizer()
    resp = tokenizer.encode(_RESP_STR, add_special_tokens=False)
    eot = tokenizer.encode(_EOT_STR, add_special_tokens=False)
    tc = tokenizer.encode(_TC_STR, add_special_tokens=False)
    forbidden = set(resp) | set(eot) | set(tc)
    for sentinel in _SENTINELS:
        assert sentinel not in forbidden, (
            f"Sentinel {sentinel} collides with a template token ID. Adjust _SENTINELS."
        )
    return resp, eot, tc


def make_span_collator(
    mask_tool_calls: bool = False,
) -> TextCompletionsCollatorWithPadding:
    tokenizer, _ = create_test_tokenizer()
    masking_method = "assistant_turn_no_tools" if mask_tool_calls else "assistant_turn"
    return TextCompletionsCollatorWithPadding(
        tokenizer=tokenizer,
        response_template=_RESP_STR,
        masking_method=masking_method,
        end_of_turn_template=_EOT_STR,
        tool_call_start_template=_TC_STR if mask_tool_calls else None,
    )


def get_span_labels(collator, seq: list[int]) -> list[int]:
    return collator([{"input_ids": seq}])["labels"][0].tolist()


def flat(*parts: list[int]) -> list[int]:
    result = []
    for p in parts:
        result.extend(p)
    return result


# ---------------------------------------------------------------------------
# Single assistant turn
# ---------------------------------------------------------------------------


def test_span_single_turn_content_is_unmasked():
    resp, eot, _ = get_template_token_ids()
    prefix = [_SENTINELS[0], _SENTINELS[1]]
    content = [_SENTINELS[2], _SENTINELS[3]]
    seq = flat(prefix, resp, content, eot)

    labels = get_span_labels(make_span_collator(), seq)

    n_prefix = len(prefix) + len(resp)
    assert all(v == IGNORE for v in labels[:n_prefix])
    assert labels[n_prefix : n_prefix + len(content)] == content
    # EOT tokens are unmasked (model learns to produce the stop token)
    assert labels[n_prefix + len(content) : n_prefix + len(content) + len(eot)] == eot


def test_span_single_turn_response_template_tokens_are_masked():
    resp, eot, _ = get_template_token_ids()
    seq = flat(resp, [_SENTINELS[0]], eot)

    labels = get_span_labels(make_span_collator(), seq)

    for i in range(len(resp)):
        assert labels[i] == IGNORE, f"resp template token {i} should be masked"


def test_span_single_turn_eot_tokens_are_unmasked():
    resp, eot, _ = get_template_token_ids()
    content = [_SENTINELS[0]]
    seq = flat(resp, content, eot)

    labels = get_span_labels(make_span_collator(), seq)

    eot_start = len(resp) + len(content)
    assert labels[eot_start : eot_start + len(eot)] == eot


# ---------------------------------------------------------------------------
# Multiple assistant turns
# ---------------------------------------------------------------------------


def test_span_two_turns_both_unmasked():
    resp, eot, _ = get_template_token_ids()
    turn1 = [_SENTINELS[0], _SENTINELS[1]]
    middle = [_SENTINELS[2]]
    turn2 = [_SENTINELS[3], _SENTINELS[4]]
    seq = flat(resp, turn1, eot, middle, resp, turn2, eot)

    labels = get_span_labels(make_span_collator(), seq)

    t1_start = len(resp)
    t1_end = t1_start + len(turn1)
    assert labels[t1_start:t1_end] == turn1

    t2_start = t1_end + len(eot) + len(middle) + len(resp)
    t2_end = t2_start + len(turn2)
    assert labels[t2_start:t2_end] == turn2


def test_span_content_between_turns_is_masked():
    resp, eot, _ = get_template_token_ids()
    turn1 = [_SENTINELS[0]]
    between = [_SENTINELS[1], _SENTINELS[2]]
    turn2 = [_SENTINELS[3]]
    seq = flat(resp, turn1, eot, between, resp, turn2, eot)

    labels = get_span_labels(make_span_collator(), seq)

    between_start = len(resp) + len(turn1) + len(eot)
    for i in range(len(between)):
        assert labels[between_start + i] == IGNORE


# ---------------------------------------------------------------------------
# Tool result masking
# ---------------------------------------------------------------------------


def test_span_tool_result_is_masked():
    resp, eot, _ = get_template_token_ids()
    tool_call_content = [_SENTINELS[0], _SENTINELS[1]]
    tool_result = [_SENTINELS[2], _SENTINELS[3]]
    final_answer = [_SENTINELS[4], _SENTINELS[5]]
    seq = flat(resp, tool_call_content, eot, tool_result, resp, final_answer, eot)

    labels = get_span_labels(make_span_collator(), seq)

    tool_result_start = len(resp) + len(tool_call_content) + len(eot)
    for i in range(len(tool_result)):
        assert labels[tool_result_start + i] == IGNORE


def test_span_final_answer_after_tool_result_is_unmasked():
    resp, eot, _ = get_template_token_ids()
    tool_call_content = [_SENTINELS[0]]
    tool_result = [_SENTINELS[1]]
    final_answer = [_SENTINELS[2], _SENTINELS[3]]
    seq = flat(resp, tool_call_content, eot, tool_result, resp, final_answer, eot)

    labels = get_span_labels(make_span_collator(), seq)

    final_start = (
        len(resp) + len(tool_call_content) + len(eot) + len(tool_result) + len(resp)
    )
    assert labels[final_start : final_start + len(final_answer)] == final_answer


# ---------------------------------------------------------------------------
# mask_tool_calls option
# ---------------------------------------------------------------------------


def test_span_tool_call_turn_unmasked_by_default():
    resp, eot, tc = get_template_token_ids()
    tc_content = flat(tc, [_SENTINELS[0]])
    seq = flat(resp, tc_content, eot)

    labels = get_span_labels(make_span_collator(mask_tool_calls=False), seq)

    content_start = len(resp)
    assert labels[content_start : content_start + len(tc_content)] == tc_content


def test_span_tool_call_turn_masked_when_option_set():
    resp, eot, tc = get_template_token_ids()
    tc_content = flat(tc, [_SENTINELS[0]])
    seq = flat(resp, tc_content, eot)

    labels = get_span_labels(make_span_collator(mask_tool_calls=True), seq)

    content_start = len(resp)
    assert all(
        v == IGNORE for v in labels[content_start : content_start + len(tc_content)]
    )


def test_span_non_tool_call_turn_still_unmasked_when_mask_tool_calls_set():
    resp, eot, tc = get_template_token_ids()
    tc_content = flat(tc, [_SENTINELS[0]])
    final_answer = [_SENTINELS[1], _SENTINELS[2]]
    seq = flat(resp, tc_content, eot, resp, final_answer, eot)

    labels = get_span_labels(make_span_collator(mask_tool_calls=True), seq)

    final_start = len(resp) + len(tc_content) + len(eot) + len(resp)
    assert labels[final_start : final_start + len(final_answer)] == final_answer


def test_span_mask_tool_calls_requires_template():
    tokenizer, _ = create_test_tokenizer()
    with pytest.raises(ValueError, match="tool_call_start_template"):
        TextCompletionsCollatorWithPadding(
            tokenizer=tokenizer,
            response_template=_RESP_STR,
            masking_method="assistant_turn_no_tools",
            end_of_turn_template=_EOT_STR,
            tool_call_start_template=None,
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_span_no_response_template_all_masked():
    seq = [_SENTINELS[0], _SENTINELS[1], _SENTINELS[2]]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        labels = get_span_labels(make_span_collator(), seq)

    assert all(v == IGNORE for v in labels)
    assert any("response template" in str(w.message).lower() for w in caught)


def test_span_no_eot_unmasked_to_end_of_sequence():
    resp, _, _ = get_template_token_ids()
    content = [_SENTINELS[0], _SENTINELS[1]]
    seq = flat(resp, content)

    labels = get_span_labels(make_span_collator(), seq)

    assert labels[len(resp) :] == content


def test_span_empty_content_span():
    resp, eot, _ = get_template_token_ids()
    seq = flat(resp, eot)

    labels = get_span_labels(make_span_collator(), seq)

    assert all(v == IGNORE for v in labels)


def test_span_padding_matching_eot_does_not_false_match():
    """When pad_token_id matches the EOT token, padding must not be treated as
    a real end-of-turn boundary."""
    tokenizer, pad_token_id = create_test_tokenizer()
    resp, _, _ = get_template_token_ids()

    # Use pad_token_id itself as the EOT template — worst case scenario.
    eot_ids = [pad_token_id]
    content = [_SENTINELS[0], _SENTINELS[1]]
    # Sequence: [RESP] content (no real EOT), then padding
    seq = flat(resp, content) + [pad_token_id] * 5

    collator = TextCompletionsCollatorWithPadding(
        tokenizer=tokenizer,
        response_template=_RESP_STR,
        end_of_turn_template=str(tokenizer.decode(eot_ids)),
    )
    batch = collator([{"input_ids": seq}])
    labels = batch["labels"][0].tolist()

    # Content should be unmasked — the padding should not act as an EOT.
    content_start = len(resp)
    assert labels[content_start : content_start + len(content)] == content
    # Padding should be masked.
    assert all(v == IGNORE for v in labels[content_start + len(content) :])


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def test_span_batch_two_examples_processed_independently():
    resp, eot, _ = get_template_token_ids()
    _, pad_token_id = create_test_tokenizer()
    content_a = [_SENTINELS[0], _SENTINELS[1]]
    content_b = [_SENTINELS[2]]
    seq_a = flat(resp, content_a, eot)
    seq_b = flat(resp, content_b, eot)

    max_len = max(len(seq_a), len(seq_b))
    pad_a = [pad_token_id] * (max_len - len(seq_a))
    pad_b = [pad_token_id] * (max_len - len(seq_b))

    collator = make_span_collator()
    batch = collator([{"input_ids": seq_a + pad_a}, {"input_ids": seq_b + pad_b}])
    labels_a = batch["labels"][0].tolist()
    labels_b = batch["labels"][1].tolist()

    assert labels_a[len(resp) : len(resp) + len(content_a)] == content_a
    assert labels_b[len(resp) : len(resp) + len(content_b)] == content_b


def test_span_batch_bad_example_does_not_affect_others():
    resp, eot, _ = get_template_token_ids()
    _, pad_token_id = create_test_tokenizer()
    good_seq = flat(resp, [_SENTINELS[0]], eot)
    bad_seq = [_SENTINELS[1], _SENTINELS[2]]

    max_len = max(len(good_seq), len(bad_seq))
    pad_good = [pad_token_id] * (max_len - len(good_seq))
    pad_bad = [pad_token_id] * (max_len - len(bad_seq))

    collator = make_span_collator()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        batch = collator(
            [{"input_ids": good_seq + pad_good}, {"input_ids": bad_seq + pad_bad}]
        )

    assert batch["labels"][0].tolist()[len(resp)] == _SENTINELS[0]
    assert all(v == IGNORE for v in batch["labels"][1].tolist())


def test_span_output_labels_is_torch_tensor():
    resp, eot, _ = get_template_token_ids()
    seq = flat(resp, [_SENTINELS[0]], eot)
    batch = make_span_collator()([{"input_ids": seq}])
    assert isinstance(batch["labels"], torch.Tensor)


def test_span_labels_shape_matches_input_ids():
    resp, eot, _ = get_template_token_ids()
    seq = flat(resp, [_SENTINELS[0], _SENTINELS[1]], eot)
    batch = make_span_collator()([{"input_ids": seq}])
    assert batch["labels"].shape == batch["input_ids"].shape


def test_span_labels_numpy_values_match_expected():
    resp, eot, _ = get_template_token_ids()
    content = [_SENTINELS[0], _SENTINELS[1]]
    seq = flat(resp, content, eot)

    batch = make_span_collator()([{"input_ids": seq}])
    expected = [IGNORE] * len(resp) + content + eot
    assert np.all(batch["labels"].numpy() == np.array([expected], dtype=np.int32))
