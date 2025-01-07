import tempfile
from pathlib import Path

import datasets
import pytest

from oumi.builders.data import build_dataset, build_dataset_mixture
from oumi.builders.models import build_tokenizer
from oumi.core.configs import (
    DataParams,
    DatasetParams,
    DatasetSplit,
    DatasetSplitParams,
    TrainingConfig,
)
from oumi.core.configs.params.model_params import ModelParams


@pytest.fixture
def gpt2_tokenizer():
    tokenizer = build_tokenizer(
        ModelParams(
            model_name="openai-community/gpt2",
            torch_dtype_str="float16",
            trust_remote_code=False,
            chat_template="default",
            tokenizer_pad_token="<|endoftext|>",
        )
    )
    assert tokenizer.pad_token_id is not None
    assert isinstance(tokenizer.pad_token_id, int)
    return tokenizer


@pytest.fixture
def sample_conversations_jsonl(single_turn_conversation):
    """Creates a temporary JSONL file with sample conversations."""
    conversations = [
        single_turn_conversation,
        single_turn_conversation,
    ]

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        import jsonlines

        with jsonlines.Writer(f) as writer:
            for conv in conversations:
                writer.write(conv.to_dict())

    yield Path(f.name)
    Path(f.name).unlink()  # Cleanup temp file


@pytest.mark.parametrize(
    "stream",
    [
        False,
        True,
    ],
)
def test_build_dataset_conversations(
    sample_conversations_jsonl, gpt2_tokenizer, stream: bool
):
    """Test building dataset from conversations format JSONL."""
    dataset = build_dataset(
        dataset_name="text_sft_jsonl",
        tokenizer=gpt2_tokenizer,
        dataset_path=str(sample_conversations_jsonl),
        stream=stream,
    )
    if stream:
        assert isinstance(dataset, datasets.IterableDataset)
    else:
        assert isinstance(dataset, datasets.Dataset)

    # Convert to list to access items
    items = list(dataset)
    assert len(items) == 2

    # Check first conversation
    assert isinstance(items[0], dict)
    assert isinstance(items[1], dict)


def test_build_dataset_invalid_path():
    """Test building dataset with invalid file path."""
    with pytest.raises(FileNotFoundError):
        build_dataset(
            dataset_name="text_sft_jsonl",
            tokenizer=None,
            dataset_path="nonexistent.jsonl",
        )


@pytest.mark.parametrize(
    "stream",
    [
        False,
        True,
    ],
)
def test_build_dataset_mixture(
    sample_conversations_jsonl, gpt2_tokenizer, stream: bool
):
    """Test building a mixture of datasets with specified proportions."""
    # Create config with dataset mixture
    config = TrainingConfig(
        data=DataParams(
            train=DatasetSplitParams(
                datasets=[
                    DatasetParams(
                        dataset_name="text_sft_jsonl",
                        dataset_path=str(sample_conversations_jsonl),
                        mixture_proportion=0.7,
                    ),
                    DatasetParams(
                        dataset_name="text_sft",
                        dataset_path=str(sample_conversations_jsonl),
                        mixture_proportion=0.3,
                    ),
                ],
                mixture_strategy="all_exhausted",
                seed=42,
                stream=stream,
            )
        )
    )

    dataset = build_dataset_mixture(
        config=config,
        tokenizer=gpt2_tokenizer,
        dataset_split=DatasetSplit.TRAIN,
    )
    if stream:
        assert isinstance(dataset, datasets.IterableDataset)
    else:
        assert isinstance(dataset, datasets.Dataset)

    # Convert to list to access items
    items = list(dataset)

    # Check that we have items from both datasets
    assert len(items) == 4
