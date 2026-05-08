import datasets

from oumi.core.datasets.pretraining_async_text_dataset import (
    PretrainingAsyncTextDataset,
)
from tests.unit.conftest import create_mock_tokenizer

_DATASET_LENGTH = 3
_BATCH_SIZE = 1
_NUM_TOKENS_PER_SAMPLE = 6
_SEQ_LEN = 10
_MOCK_TOKENS = list(range(1, _NUM_TOKENS_PER_SAMPLE + 1))


def _tokenize_to_mock_tokens(x, **kwargs):
    """Tokenize function that returns mock tokens for each input."""
    input_ids = []
    for _ in x:
        input_ids.append(_MOCK_TOKENS)
    return {"input_ids": input_ids, "labels": input_ids}


def test_iter():
    test_dataset = datasets.Dataset.from_list(
        [{"text": "T" * _NUM_TOKENS_PER_SAMPLE}] * _DATASET_LENGTH
    )
    tokenizer = create_mock_tokenizer(tokenize_func=_tokenize_to_mock_tokens)
    dataset = PretrainingAsyncTextDataset(
        tokenizer=tokenizer,
        dataset=test_dataset,
        formatting_func=lambda x: x,
        seq_length=_SEQ_LEN,
        sequence_buffer_size=_BATCH_SIZE * 2,
        pretokenized=False,
    )

    items = [x for x in dataset]

    assert len(items) == 2
    assert items[0]["input_ids"].tolist() == [1, 2, 3, 4, 5, 6, 0, 1, 2, 3]
    assert items[0]["labels"].tolist() == [1, 2, 3, 4, 5, 6, 0, 1, 2, 3]
    assert items[1]["input_ids"].tolist() == [4, 5, 6, 0, 1, 2, 3, 4, 5, 6]
    assert items[1]["labels"].tolist() == [4, 5, 6, 0, 1, 2, 3, 4, 5, 6]


def test_uses_tokenizer_eos_token_id():
    """Test that tokenizer's eos_token_id is used when available."""
    test_dataset = datasets.Dataset.from_list([{"text": "test"}])
    tokenizer = create_mock_tokenizer(eos_token_id=99)

    dataset = PretrainingAsyncTextDataset(
        tokenizer=tokenizer,
        dataset=test_dataset,
        formatting_func=lambda x: x,
        seq_length=_SEQ_LEN,
        pretokenized=False,
    )

    assert dataset.concat_token_id == 99


def test_uses_fallback_eos_token_id_when_tokenizer_has_none():
    """Test that fallback eos_token_id parameter is used when tokenizer has None."""
    test_dataset = datasets.Dataset.from_list([{"text": "test"}])
    tokenizer = create_mock_tokenizer(eos_token_id=None)

    dataset = PretrainingAsyncTextDataset(
        tokenizer=tokenizer,
        dataset=test_dataset,
        formatting_func=lambda x: x,
        seq_length=_SEQ_LEN,
        eos_token_id=42,
        pretokenized=False,
    )

    assert dataset.concat_token_id == 42


def test_uses_fallback_eos_token_id_when_no_tokenizer():
    """Test that fallback eos_token_id is used when tokenizer is None (pretokenized)."""
    test_dataset = datasets.Dataset.from_list([{"input_ids": [1, 2, 3]}])

    dataset = PretrainingAsyncTextDataset(
        tokenizer=None,
        dataset=test_dataset,
        seq_length=_SEQ_LEN,
        eos_token_id=77,
        pretokenized=True,
    )

    assert dataset.concat_token_id == 77
