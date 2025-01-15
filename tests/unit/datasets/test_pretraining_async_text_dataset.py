import datasets
import transformers

from oumi.core.datasets.pretraining_async_text_dataset import (
    PretrainingAsyncTextDataset,
)

_DATASET_LENGTH = 3
_BATCH_SIZE = 1
_NUM_TOKENS_PER_SAMPLE = 6
_SEQ_LEN = 10
_MOCK_TOKENS = list(range(1, _NUM_TOKENS_PER_SAMPLE + 1))


class MockTokenizer(transformers.PreTrainedTokenizer):
    def __init__(self):
        self.eos_token_id = None

    def __call__(self, x, **kwargs):
        input_ids = []
        for _ in x:
            input_ids.append(_MOCK_TOKENS)

        return {"input_ids": input_ids, "labels": input_ids}


def test_iter():
    test_dataset = datasets.Dataset.from_list(
        [{"text": "T" * _NUM_TOKENS_PER_SAMPLE}] * _DATASET_LENGTH
    )
    tokenizer = MockTokenizer()
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
