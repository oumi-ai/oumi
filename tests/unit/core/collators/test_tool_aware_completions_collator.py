import functools
import warnings

import numpy as np
import pytest
import torch

from oumi.builders import build_tokenizer
from oumi.core.collators.tool_aware_completions_collator import (
    ToolAwareCompletionsCollator,
)
from oumi.core.configs import ModelParams
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer

IGNORE = -100

# Template strings — chosen to be unambiguous in GPT-2's vocabulary.
_RESP_STR = " ASSISTANT_RESPONSE_START"
_EOT_STR = " TURN_ENDS_HERE"
_TC_STR = " TOOL_CALL_BEGINS"

# Arbitrary token IDs used as "content" that must not appear in any template.
_SENTINELS = [601, 602, 603, 604, 605, 606, 607, 608]


@functools.cache
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


def make_collator(mask_tool_calls: bool = False) -> ToolAwareCompletionsCollator:
    tokenizer, _ = create_test_tokenizer()
    resp_ids, eot_ids, tc_ids = get_template_token_ids()
    return ToolAwareCompletionsCollator(
        response_template=resp_ids,
        end_of_turn_template=eot_ids,
        mask_tool_calls=mask_tool_calls,
        tool_call_start_template=tc_ids if mask_tool_calls else None,
        tokenizer=tokenizer,
    )


def get_labels(collator, seq: list[int]) -> list[int]:
    return collator.torch_call([{"input_ids": seq}])["labels"][0].tolist()


def flat(*parts: list[int]) -> list[int]:
    result = []
    for p in parts:
        result.extend(p)
    return result


# ---------------------------------------------------------------------------
# Single assistant turn
# ---------------------------------------------------------------------------


def test_single_turn_content_is_unmasked():
    resp, eot, _ = get_template_token_ids()
    prefix = [_SENTINELS[0], _SENTINELS[1]]
    content = [_SENTINELS[2], _SENTINELS[3]]
    seq = flat(prefix, resp, content, eot)

    labels = get_labels(make_collator(), seq)

    n_prefix = len(prefix) + len(resp)
    assert all(v == IGNORE for v in labels[:n_prefix])
    assert labels[n_prefix : n_prefix + len(content)] == content
    assert all(v == IGNORE for v in labels[n_prefix + len(content) :])


def test_single_turn_response_template_tokens_are_masked():
    resp, eot, _ = get_template_token_ids()
    seq = flat(resp, [_SENTINELS[0]], eot)

    labels = get_labels(make_collator(), seq)

    for i in range(len(resp)):
        assert labels[i] == IGNORE, f"resp template token {i} should be masked"


def test_single_turn_eot_tokens_are_masked():
    resp, eot, _ = get_template_token_ids()
    content = [_SENTINELS[0]]
    seq = flat(resp, content, eot)

    labels = get_labels(make_collator(), seq)

    eot_start = len(resp) + len(content)
    for i in range(len(eot)):
        assert labels[eot_start + i] == IGNORE, f"eot token {i} should be masked"


# ---------------------------------------------------------------------------
# Multiple assistant turns
# ---------------------------------------------------------------------------


def test_two_turns_both_unmasked():
    resp, eot, _ = get_template_token_ids()
    turn1 = [_SENTINELS[0], _SENTINELS[1]]
    middle = [_SENTINELS[2]]
    turn2 = [_SENTINELS[3], _SENTINELS[4]]
    seq = flat(resp, turn1, eot, middle, resp, turn2, eot)

    labels = get_labels(make_collator(), seq)

    t1_start = len(resp)
    t1_end = t1_start + len(turn1)
    assert labels[t1_start:t1_end] == turn1

    t2_start = t1_end + len(eot) + len(middle) + len(resp)
    t2_end = t2_start + len(turn2)
    assert labels[t2_start:t2_end] == turn2


def test_content_between_turns_is_masked():
    resp, eot, _ = get_template_token_ids()
    turn1 = [_SENTINELS[0]]
    between = [_SENTINELS[1], _SENTINELS[2]]
    turn2 = [_SENTINELS[3]]
    seq = flat(resp, turn1, eot, between, resp, turn2, eot)

    labels = get_labels(make_collator(), seq)

    between_start = len(resp) + len(turn1) + len(eot)
    for i in range(len(between)):
        assert labels[between_start + i] == IGNORE


# ---------------------------------------------------------------------------
# Tool result masking
# ---------------------------------------------------------------------------


def test_tool_result_is_masked():
    resp, eot, _ = get_template_token_ids()
    tool_call_content = [_SENTINELS[0], _SENTINELS[1]]
    tool_result = [_SENTINELS[2], _SENTINELS[3]]
    final_answer = [_SENTINELS[4], _SENTINELS[5]]
    seq = flat(resp, tool_call_content, eot, tool_result, resp, final_answer, eot)

    labels = get_labels(make_collator(), seq)

    tool_result_start = len(resp) + len(tool_call_content) + len(eot)
    for i in range(len(tool_result)):
        assert labels[tool_result_start + i] == IGNORE


def test_final_answer_after_tool_result_is_unmasked():
    resp, eot, _ = get_template_token_ids()
    tool_call_content = [_SENTINELS[0]]
    tool_result = [_SENTINELS[1]]
    final_answer = [_SENTINELS[2], _SENTINELS[3]]
    seq = flat(resp, tool_call_content, eot, tool_result, resp, final_answer, eot)

    labels = get_labels(make_collator(), seq)

    final_start = (
        len(resp) + len(tool_call_content) + len(eot) + len(tool_result) + len(resp)
    )
    assert labels[final_start : final_start + len(final_answer)] == final_answer


# ---------------------------------------------------------------------------
# mask_tool_calls option
# ---------------------------------------------------------------------------


def test_tool_call_turn_unmasked_by_default():
    resp, eot, tc = get_template_token_ids()
    tc_content = flat(tc, [_SENTINELS[0]])
    seq = flat(resp, tc_content, eot)

    labels = get_labels(make_collator(mask_tool_calls=False), seq)

    content_start = len(resp)
    assert labels[content_start : content_start + len(tc_content)] == tc_content


def test_tool_call_turn_masked_when_option_set():
    resp, eot, tc = get_template_token_ids()
    tc_content = flat(tc, [_SENTINELS[0]])
    seq = flat(resp, tc_content, eot)

    labels = get_labels(make_collator(mask_tool_calls=True), seq)

    content_start = len(resp)
    assert all(
        v == IGNORE for v in labels[content_start : content_start + len(tc_content)]
    )


def test_non_tool_call_turn_still_unmasked_when_mask_tool_calls_set():
    resp, eot, tc = get_template_token_ids()
    tc_content = flat(tc, [_SENTINELS[0]])
    final_answer = [_SENTINELS[1], _SENTINELS[2]]
    seq = flat(resp, tc_content, eot, resp, final_answer, eot)

    labels = get_labels(make_collator(mask_tool_calls=True), seq)

    final_start = len(resp) + len(tc_content) + len(eot) + len(resp)
    assert labels[final_start : final_start + len(final_answer)] == final_answer


def test_mask_tool_calls_requires_template():
    tokenizer, _ = create_test_tokenizer()
    resp_ids, eot_ids, _ = get_template_token_ids()
    with pytest.raises(ValueError, match="tool_call_start_template"):
        ToolAwareCompletionsCollator(
            response_template=resp_ids,
            end_of_turn_template=eot_ids,
            mask_tool_calls=True,
            tool_call_start_template=None,
            tokenizer=tokenizer,
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_no_response_template_all_masked():
    seq = [_SENTINELS[0], _SENTINELS[1], _SENTINELS[2]]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        labels = get_labels(make_collator(), seq)

    assert all(v == IGNORE for v in labels)
    assert any("response template" in str(w.message).lower() for w in caught)


def test_no_eot_unmasked_to_end_of_sequence():
    resp, _, _ = get_template_token_ids()
    content = [_SENTINELS[0], _SENTINELS[1]]
    seq = flat(resp, content)

    labels = get_labels(make_collator(), seq)

    assert labels[len(resp) :] == content


def test_empty_content_span():
    resp, eot, _ = get_template_token_ids()
    seq = flat(resp, eot)

    labels = get_labels(make_collator(), seq)

    assert all(v == IGNORE for v in labels)


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def test_batch_two_examples_processed_independently():
    resp, eot, _ = get_template_token_ids()
    _, pad_token_id = create_test_tokenizer()
    content_a = [_SENTINELS[0], _SENTINELS[1]]
    content_b = [_SENTINELS[2]]
    seq_a = flat(resp, content_a, eot)
    seq_b = flat(resp, content_b, eot)

    max_len = max(len(seq_a), len(seq_b))
    pad_a = [pad_token_id] * (max_len - len(seq_a))
    pad_b = [pad_token_id] * (max_len - len(seq_b))

    batch = make_collator().torch_call(
        [{"input_ids": seq_a + pad_a}, {"input_ids": seq_b + pad_b}]
    )
    labels_a = batch["labels"][0].tolist()
    labels_b = batch["labels"][1].tolist()

    assert labels_a[len(resp) : len(resp) + len(content_a)] == content_a
    assert labels_b[len(resp) : len(resp) + len(content_b)] == content_b


def test_batch_bad_example_does_not_affect_others():
    resp, eot, _ = get_template_token_ids()
    _, pad_token_id = create_test_tokenizer()
    good_seq = flat(resp, [_SENTINELS[0]], eot)
    bad_seq = [_SENTINELS[1], _SENTINELS[2]]

    max_len = max(len(good_seq), len(bad_seq))
    pad_good = [pad_token_id] * (max_len - len(good_seq))
    pad_bad = [pad_token_id] * (max_len - len(bad_seq))

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        batch = make_collator().torch_call(
            [{"input_ids": good_seq + pad_good}, {"input_ids": bad_seq + pad_bad}]
        )

    assert batch["labels"][0].tolist()[len(resp)] == _SENTINELS[0]
    assert all(v == IGNORE for v in batch["labels"][1].tolist())


def test_output_labels_is_torch_tensor():
    resp, eot, _ = get_template_token_ids()
    seq = flat(resp, [_SENTINELS[0]], eot)
    batch = make_collator().torch_call([{"input_ids": seq}])
    assert isinstance(batch["labels"], torch.Tensor)


def test_labels_shape_matches_input_ids():
    resp, eot, _ = get_template_token_ids()
    seq = flat(resp, [_SENTINELS[0], _SENTINELS[1]], eot)
    batch = make_collator().torch_call([{"input_ids": seq}])
    assert batch["labels"].shape == batch["input_ids"].shape


def test_labels_numpy_values_match_expected():
    resp, eot, _ = get_template_token_ids()
    content = [_SENTINELS[0], _SENTINELS[1]]
    seq = flat(resp, content, eot)

    batch = make_collator().torch_call([{"input_ids": seq}])
    expected = [IGNORE] * len(resp) + content + [IGNORE] * len(eot)
    assert np.all(batch["labels"].numpy() == np.array([expected], dtype=np.int32))
