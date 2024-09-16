from dataclasses import dataclass
from typing import Optional

from oumi.core.configs.base_config import BaseConfig


@dataclass
class GenerationConfig(BaseConfig):
    # TODO: OPE-328 - Add more parameters to control text generation.
    max_new_tokens: int = 256
    """The maximum number of new tokens to generate.

    This limits the length of the generated text to prevent excessively long outputs.
    Default is 256 tokens.
    """

    batch_size: int = 2
    """The number of sequences to generate in parallel.

    Larger batch sizes can improve throughput but require more memory.
    Default is 2.
    """

    exclude_prompt_from_response: bool = True
    """Whether to trim the model's response and remove the prepended prompt."""

    input_filepath: Optional[str] = None
    """Path to the input file containing prompts for text generation."""

    output_filepath: Optional[str] = None
    """Path where the generated text will be saved."""
