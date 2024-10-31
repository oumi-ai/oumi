import pandas as pd
import pytest
import torch

from oumi.builders import build_tokenizer
from oumi.core.collators.text_completions_collator_with_padding import (
    TextCompletionsCollatorWithPadding,
)
from oumi.core.configs import ModelParams
from oumi.core.datasets.base_sft_dataset import BaseSftDataset
from oumi.core.types.conversation import Conversation, Message, Role


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
        response_template="\nASSISTANT: ",
        instruction_template="\nUSER: ",
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


def test_tokenize_multi_turn_conversation(sft_dataset, gpt2_tokenizer):
    conversation = Conversation(
        messages=[
            Message(role=Role.SYSTEM, content="You are a helpful assistant."),
            Message(role=Role.USER, content="What's the weather like today?"),
            Message(
                role=Role.ASSISTANT,
                content="I'm sorry, I don't have real-time information. "
                "How can I assist you with something else?",
            ),
            Message(role=Role.USER, content="Tell me a joke."),
            Message(
                role=Role.ASSISTANT,
                content="Why don't scientists trust atoms? "
                "Because they make up everything!",
            ),
        ]
    )

    result = sft_dataset.tokenize(conversation)

    prompt = sft_dataset.tokenize(conversation, tokenize=False)

    assert "input_ids" in result
    assert "attention_mask" in result
    assert "labels" in result
    assert (
        len(result["input_ids"])
        == len(result["attention_mask"])
        == len(result["labels"])
    )

    # Check that system and user messages are masked in labels
    system_and_user_tokens = gpt2_tokenizer.encode(
        "You are a helpful assistant.What's the weather like today?Tell me a joke.",
        add_special_tokens=False,
    )
    assert all(
        label == -100 for label in result["labels"][: len(system_and_user_tokens)]
    )

    # Check that assistant responses are not masked in labels
    assistant_tokens = gpt2_tokenizer.encode(
        "I'm sorry, I don't have real-time information. How can I assist you with "
        "something else?Why don't scientists trust atoms? "
        "Because they make up everything!",
        add_special_tokens=False,
    )
    assert any(label != -100 for label in result["labels"][-len(assistant_tokens) :])


def test_tokenize_assistant_template(sft_dataset, gpt2_tokenizer):
    enc = gpt2_tokenizer.encode(sft_dataset.response_template, add_special_tokens=False)
    dec = gpt2_tokenizer.decode(enc)

    assert sft_dataset.response_token_ids == gpt2_tokenizer.encode(
        sft_dataset.response_template.strip(), add_special_tokens=False
    )
    assert dec.strip() == sft_dataset.response_template.strip()

    turn = "Hello\nASSISTANT: Hi there!"
    enc = gpt2_tokenizer.encode(turn, add_special_tokens=False)
    dec = gpt2_tokenizer.decode(enc)

    assert sft_dataset._find_response_start(torch.tensor(enc)) == 2
    assert dec == turn


def test_tokenize_long_input(gpt2_tokenizer):
    gpt2_tokenizer.model_max_length = 20
    dataset = TestBaseSftDataset(tokenizer=gpt2_tokenizer)
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

    result = dataset.tokenize(conversation)

    assert len(result["input_ids"]) == 20
    assert len(result["attention_mask"]) == 20
    assert len(result["labels"]) == 20


def test_tokenize_empty_conversation(gpt2_tokenizer):
    dataset = TestBaseSftDataset(tokenizer=gpt2_tokenizer)
    conversation = Conversation(messages=[])

    result = dataset.tokenize(conversation)

    assert "input_ids" in result
    assert "attention_mask" in result
    assert "labels" in result
    assert (
        len(result["input_ids"])
        == len(result["attention_mask"])
        == len(result["labels"])
    )


def test_tokenize_user_only_turn(sft_dataset, gpt2_tokenizer):
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
        label == -100 for label in result["labels"]
    )  # All labels should be masked


def test_tokenize_assistant_only_turn(sft_dataset, gpt2_tokenizer):
    conversation = Conversation(
        messages=[
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
    assert all(label != -100 for label in result["labels"])


def test_tokenize_return_tensors(gpt2_tokenizer):
    dataset = TestBaseSftDataset(tokenizer=gpt2_tokenizer, return_tensors=True)
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

    batch = [
        gpt2_tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
        )
    ]

    instruction_prefix = sft_dataset.instruction_template
    response_prefix = sft_dataset.response_template

    collator = TextCompletionsCollatorWithPadding(
        tokenizer=gpt2_tokenizer,
        instruction_prefix=instruction_prefix,
        response_prefix=response_prefix,
    )

    result = collator(batch)

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

    dataset = TestBaseSftDataset(tokenizer=tokenizer)

    conversation = Conversation(
        messages=[Message(role=Role.ASSISTANT, content="Hello, oumi!")]
    )

    result = dataset._tokenize_for_completions_only_training_with_template(conversation)

    assert "input_ids" in result
    assert "attention_mask" in result
    assert "labels" in result
