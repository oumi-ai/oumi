import functools

import numpy as np
import pytest
import torch

import oumi.core.constants as constants
from oumi.builders import build_tokenizer
from oumi.core.collators.text_collator_with_padding import TextCollatorWithPadding
from oumi.core.configs import ModelParams
from oumi.core.tokenizers.base_tokenizer import BaseTokenizer


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
    collator = TextCollatorWithPadding(tokenizer, max_length=None)
    assert callable(collator)

    collated_batch = collator(
        [
            {"input_ids": [101, 102, 103, 104]},
            {"input_ids": [201, 202]},
        ]
    )

    assert "input_ids" in collated_batch
    assert isinstance(collated_batch["input_ids"], torch.Tensor)
    assert np.all(
        collated_batch["input_ids"].numpy()
        == np.array(
            [[101, 102, 103, 104], [201, 202, pad_token_id, pad_token_id]],
            dtype=np.int32,
        )
    )
    assert "attention_mask" in collated_batch
    assert isinstance(collated_batch["attention_mask"], torch.Tensor)
    assert np.all(
        collated_batch["attention_mask"].numpy()
        == np.array([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=np.int32)
    )
    assert "labels" not in collated_batch


def test_success_with_labels_and_max_length():
    tokenizer, pad_token_id = create_test_tokenizer()

    with pytest.raises(
        ValueError,
        match=(
            "Maximum sequence length exceeded. You should probably activate truncation"
        ),
    ):
        TextCollatorWithPadding(tokenizer, max_length=2)(
            [
                {"input_ids": [101], "labels": [101]},
                {"input_ids": [201, 202, 203, 204], "labels": [201, 202, 203, 204]},
                {"input_ids": [301, 302], "labels": [301, 302]},
            ]
        )

    with pytest.raises(
        ValueError,
        match=(
            "Maximum sequence length exceeded. You should probably activate truncation"
        ),
    ):
        TextCollatorWithPadding(tokenizer, max_length=2, truncation=False)(
            [
                {"input_ids": [101], "labels": [101]},
                {"input_ids": [201, 202, 203, 204], "labels": [201, 202, 203, 204]},
                {"input_ids": [301, 302], "labels": [301, 302]},
            ]
        )

    collator = TextCollatorWithPadding(tokenizer, max_length=2, truncation=True)
    assert callable(collator)

    collated_batch = collator(
        [
            {"input_ids": [101], "labels": [101]},
            {"input_ids": [201, 202, 203, 204], "labels": [201, 202, 203, 204]},
            {"input_ids": [301, 302], "labels": [301, 302]},
        ]
    )

    assert "input_ids" in collated_batch
    assert len(collated_batch["input_ids"]) == 3
    assert isinstance(collated_batch["input_ids"], torch.Tensor)
    assert np.all(
        collated_batch["input_ids"].numpy()
        == np.array([[101, pad_token_id], [201, 202], [301, 302]], dtype=np.int32)
    )

    assert "attention_mask" in collated_batch
    assert len(collated_batch["attention_mask"]) == 3
    assert isinstance(collated_batch["attention_mask"], torch.Tensor)
    assert np.all(
        collated_batch["attention_mask"].numpy()
        == np.array([[1, 0], [1, 1], [1, 1]], dtype=np.int32)
    )

    assert "labels" in collated_batch
    assert len(collated_batch["labels"]) == 3
    assert isinstance(collated_batch["labels"], torch.Tensor)
    assert np.all(
        collated_batch["labels"].numpy()
        == np.array([[101, pad_token_id], [201, 202], [301, 302]], dtype=np.int32)
    )


def test_success_label_ingnore_index():
    tokenizer, pad_token_id = create_test_tokenizer()

    collator = TextCollatorWithPadding(
        tokenizer, max_length=4, label_ignore_index=constants.LABEL_IGNORE_INDEX
    )
    assert callable(collator)

    collated_batch = collator(
        [
            {"input_ids": [101], "labels": [101]},
            {"input_ids": [201, 202, 203, 204], "labels": [201, 202, 203, 204]},
            {"input_ids": [301, 302], "labels": [301, 302]},
        ]
    )

    assert "input_ids" in collated_batch
    assert len(collated_batch["input_ids"]) == 3
    assert isinstance(collated_batch["input_ids"], torch.Tensor)
    assert np.all(
        collated_batch["input_ids"].numpy()
        == np.array(
            [
                [101, pad_token_id, pad_token_id, pad_token_id],
                [201, 202, 203, 204],
                [301, 302, pad_token_id, pad_token_id],
            ],
            dtype=np.int32,
        )
    )

    assert "attention_mask" in collated_batch
    assert len(collated_batch["attention_mask"]) == 3
    assert isinstance(collated_batch["attention_mask"], torch.Tensor)
    assert np.all(
        collated_batch["attention_mask"].numpy()
        == np.array([[1, 0, 0, 0], [1, 1, 1, 1], [1, 1, 0, 0]], dtype=np.int32)
    )

    assert "labels" in collated_batch
    assert len(collated_batch["labels"]) == 3
    assert isinstance(collated_batch["labels"], torch.Tensor)
    assert np.all(
        collated_batch["labels"].numpy()
        == np.array(
            [
                [101, -100, -100, -100],
                [201, 202, 203, 204],
                [301, 302, -100, -100],
            ],
            dtype=np.int32,
        )
    )
