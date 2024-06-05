"""Sample custom model.

This is a sample model that intends to demonstrate how users can define their own
custom model and configuration and subsequently fine-tune it or run inference.
This model is uniquely defined in our registry by `NAME`.
"""

import transformers

from lema.core import registry

# Name that this model is registered with.
NAME = "learning-machines/sample"


@registry.register(NAME, registry.RegistryType.MODEL_CONFIG_CLASS)
class SampleConfig(transformers.GPT2Config):
    """A sample model config to be used for testing and as sample code."""

    pass


@registry.register(NAME, registry.RegistryType.MODEL_CLASS)
class SampleModel(transformers.GPT2LMHeadModel):
    """A sample model to be used for testing and as sample code."""

    pass
