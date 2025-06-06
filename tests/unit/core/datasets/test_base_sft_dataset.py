import pandas as pd
import pytest
import torch

from oumi.builders import build_tokenizer
from oumi.core.collators.text_completions_collator_with_padding import (
    TextCompletionsCollatorWithPadding,
)
from oumi.core.configs import ModelParams
from oumi.core.constants import LABEL_IGNORE_INDEX
from oumi.core.datasets.base_sft_dataset import BaseSftDataset
from oumi.core.types.conversation import Conversation, Message, Role

_INSTRUCTION_PREFIX = "USER:"
_RESPONSE_PREFIX = "ASSISTANT:"


def _get_hf_collator_result(conversation, tokenizer):
    batch = [
        tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
        )
    ]

    collator = TextCompletionsCollatorWithPadding(
        tokenizer=tokenizer,
        instruction_prefix=_INSTRUCTION_PREFIX,
        response_prefix=_RESPONSE_PREFIX,
    )

    return collator(batch)


class TestBaseSftDataset(BaseSftDataset):
    default_dataset = "test"

    def transform_conversation(self, example):
        return Conversation(
            messages=[
                Message(role=Role.USER, content="Hello"),
                Message(role=Role.ASSISTANT, content="Hi there!"),
            ]
        )

    def _load_data(self):
        return pd.DataFrame({"messages": [{"role": Role.USER, "content": "Hello"}]})


@pytest.fixture
def sft_dataset(gpt2_tokenizer):
    return TestBaseSftDataset(
        tokenizer=gpt2_tokenizer,
        assistant_only=True,
        response_template=_RESPONSE_PREFIX,
        instruction_template=_INSTRUCTION_PREFIX,
    )


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


def test_tokenize_conversation(
    single_turn_conversation, sft_dataset: TestBaseSftDataset
):
    result = sft_dataset.tokenize(single_turn_conversation)

    assert "input_ids" in result
    assert "attention_mask" in result
    assert "labels" in result
    assert (
        len(result["input_ids"])
        == len(result["attention_mask"])
        == len(result["labels"])
    )


def test_tokenize_assistant_template(sft_dataset, gpt2_tokenizer):
    assert not sft_dataset._is_template_compatible_with_completions_only_training

    enc = gpt2_tokenizer.encode(_RESPONSE_PREFIX, add_special_tokens=False)
    dec = gpt2_tokenizer.decode(enc)

    assert sft_dataset.response_token_ids == gpt2_tokenizer.encode(
        _RESPONSE_PREFIX, add_special_tokens=False
    )
    assert dec.strip() == sft_dataset._response_template

    turn = "Hello\nASSISTANT: Hi there!"
    enc = gpt2_tokenizer.encode(turn, add_special_tokens=False)
    dec = gpt2_tokenizer.decode(enc)

    assert dec == turn


def test_tokenize_long_input(sft_dataset: TestBaseSftDataset, gpt2_tokenizer):
    gpt2_tokenizer.model_max_length = 20
    conversation = Conversation(
        messages=[
            Message(
                role=Role.USER,
                content="This is a very long message that exceeds "
                "the model's maximum length.",
            ),
            Message(
                role=Role.ASSISTANT,
                content="This response is also very long and should be truncated.",
            ),
        ]
    )

    result = sft_dataset.tokenize(conversation)

    assert len(result["input_ids"]) == 20
    assert len(result["attention_mask"]) == 20
    assert len(result["labels"]) == 20


def test_tokenize_empty_conversation(sft_dataset: TestBaseSftDataset):
    conversation = Conversation(messages=[])

    result = sft_dataset.tokenize(conversation)

    assert "input_ids" in result
    assert "attention_mask" in result
    assert "labels" in result
    assert (
        len(result["input_ids"])
        == len(result["attention_mask"])
        == len(result["labels"])
    )


def test_tokenize_user_only_turn(sft_dataset, gpt2_tokenizer):
    assert not sft_dataset._is_template_compatible_with_completions_only_training
    conversation = Conversation(
        messages=[
            Message(role=Role.USER, content="Hello, oumi!"),
        ]
    )

    result = sft_dataset.tokenize(conversation)

    assert "input_ids" in result
    assert "attention_mask" in result
    assert "labels" in result
    assert (
        len(result["input_ids"])
        == len(result["attention_mask"])
        == len(result["labels"])
    )
    assert all(
        label == LABEL_IGNORE_INDEX for label in result["labels"]
    )  # All labels should be masked


def test_tokenize_assistant_only_turn_with_prefix(sft_dataset, gpt2_tokenizer):
    assert not sft_dataset._is_template_compatible_with_completions_only_training

    conversation = Conversation(
        messages=[
            Message(role=Role.SYSTEM, content="You are a helpful assistant."),
            Message(role=Role.ASSISTANT, content="Hi there!"),
        ]
    )

    result = sft_dataset.tokenize(conversation)

    assert "input_ids" in result
    assert "attention_mask" in result
    assert "labels" in result
    assert (
        len(result["input_ids"])
        == len(result["attention_mask"])
        == len(result["labels"])
    )

    # Masking with prefix is expected to be the same as the HF collator
    hf_batch = _get_hf_collator_result(conversation, gpt2_tokenizer)
    assert all(result["input_ids"] == hf_batch["input_ids"][0])
    assert all(result["attention_mask"] == hf_batch["attention_mask"][0])
    assert all(result["labels"] == hf_batch["labels"][0])

    # Note: THIS IS WRONG, but is expected because of the implementation
    # Correct behavior: Assistant tokens should NOT be masked,
    # Current behavior: Everything is masked, because we don't have a user turn
    assert all(label == LABEL_IGNORE_INDEX for label in result["labels"])


def test_tokenize_assistant_only_turn_with_template():
    tokenizer = build_tokenizer(
        ModelParams(
            model_name="openai-community/gpt2",
            torch_dtype_str="float16",
            trust_remote_code=False,
            chat_template="default_gen",
            tokenizer_pad_token="<|endoftext|>",
        )
    )
    sft_dataset = TestBaseSftDataset(
        tokenizer=tokenizer,
        assistant_only=True,
    )

    assert sft_dataset._is_template_compatible_with_completions_only_training

    conversation = Conversation(
        messages=[
            Message(role=Role.SYSTEM, content="You are a helpful assistant."),
            Message(role=Role.ASSISTANT, content="Hi there!"),
        ]
    )

    assistant_tokens = tokenizer.encode("Hi there!", add_special_tokens=False)
    assert len(assistant_tokens) == 3

    result = sft_dataset.tokenize(conversation)

    assert "input_ids" in result
    assert "attention_mask" in result
    assert "labels" in result
    assert (
        len(result["input_ids"])
        == len(result["attention_mask"])
        == len(result["labels"])
    )
    assert all(
        label == LABEL_IGNORE_INDEX
        for label in result["labels"][: -len(assistant_tokens)]
    )  # System turn + assistant prefix should be masked
    assert all(
        label != LABEL_IGNORE_INDEX
        for label in result["labels"][-len(assistant_tokens) :]
    )  # "Hi there!" should NOT be masked


def test_tokenize_return_tensors(gpt2_tokenizer):
    dataset = TestBaseSftDataset(
        tokenizer=gpt2_tokenizer,
        return_tensors=True,
        assistant_only=True,
        instruction_template=_INSTRUCTION_PREFIX,
        response_template=_RESPONSE_PREFIX,
    )
    conversation = Conversation(
        messages=[
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="Hi there!"),
        ]
    )

    result = dataset.tokenize(conversation)

    assert isinstance(result["input_ids"], torch.Tensor)
    assert isinstance(result["attention_mask"], torch.Tensor)
    assert isinstance(result["labels"], torch.Tensor)


def test_tokenize_invalid_input(sft_dataset):
    with pytest.raises(ValueError):
        sft_dataset.tokenize("invalid input")


def test_tokenize_no_return_tensors(gpt2_tokenizer):
    dataset = TestBaseSftDataset(tokenizer=gpt2_tokenizer, return_tensors=False)
    conversation = Conversation(
        messages=[
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="Hi there!"),
        ]
    )

    result = dataset.tokenize(conversation)

    for k, v in result.items():
        assert isinstance(v, list)


def test_hf_collator_with_padding(sft_dataset, gpt2_tokenizer):
    conversation = Conversation(
        messages=[
            Message(role=Role.ASSISTANT, content="Hello, oumi!"),
        ]
    )

    result = _get_hf_collator_result(conversation, gpt2_tokenizer)

    assert "input_ids" in result
    assert "attention_mask" in result
    assert "labels" in result


def test_with_generation_prompt():
    tokenizer = build_tokenizer(
        ModelParams(
            model_name="openai-community/gpt2",
            torch_dtype_str="float16",
            trust_remote_code=False,
            chat_template="default_gen",
            tokenizer_pad_token="<|endoftext|>",
        )
    )

    dataset = TestBaseSftDataset(tokenizer=tokenizer, assistant_only=True)

    conversation = Conversation(
        messages=[Message(role=Role.ASSISTANT, content="Hello, oumi!")]
    )

    result = dataset.tokenize(conversation)

    assert dataset._is_template_compatible_with_completions_only_training

    assert "input_ids" in result
    assert "attention_mask" in result
    assert "labels" in result
