"""Sample custom model.

This is a sample model that intends to demonstrate how users can define their own
custom model and configuration and subsequently fine-tune it or run inference.
This model is uniquely defined in our registry by `NAME`.
"""

from transformers import GPT2Config, GPT2LMHeadModel

from lema.core.registry import RegistryType, register

# Name that this model is registered with.
NAME = "learning-machines/sample"


@register
class SampleConfig(GPT2Config):
    """A sample model config to be used for testing and as sample code."""

    registry_name = NAME
    registry_type = RegistryType.CONFIG_CLASS


@register
class SampleModel(GPT2LMHeadModel):
    """A sample model to be used for testing and as sample code."""

    registry_name = NAME
    registry_type = RegistryType.MODEL_CLASS
