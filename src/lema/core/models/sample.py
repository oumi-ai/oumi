"""Sample custom model.

This is a sample model that intends to demonstrate how users can define their own
custom model and configuration and subsequently fine-tune it or run inference.
This model is uniquely defined in our registry by `NAME`.
"""

from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

# Name that this model is registered with.
NAME = "learning-machines/sample"


class SampleConfig(GPT2Config):
    """A sample model config to be used for testing and as sample code."""

    pass


class SampleModel(GPT2LMHeadModel):
    """A sample model to be used for testing and as sample code."""

    pass


def get_tokenizer():
    """Get the most appropriate tokenizer for `SampleModel`."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
