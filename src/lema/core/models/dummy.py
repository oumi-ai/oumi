"""This is a dummy model that intends to demonstrate how users can define their own
custom model and configuration and subsequently fine-tune it or run inference.
This model is uniquely defined in our registry by `NAME`.
"""  # noqa: D205

from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

# Name that this model is registered with.
NAME = "learning-machines/dummy"


class DummyConfig(GPT2Config):
    """A dummy model config to be used for testing and as sample code."""

    pass


class DummyModel(GPT2LMHeadModel):
    """A dummy model to be used for testing and as sample code."""

    pass


def get_tokenizer():
    """Get the most appropriate tokenizer for `DummyModel`."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
