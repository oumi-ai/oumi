from dataclasses import dataclass
from typing import List, Optional, Union

from oumi.core.configs.params.base_params import BaseParams
from oumi.core.configs.params.remote_params import RemoteParams


@dataclass
class GenerationParams(BaseParams):
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

    seed: Optional[int] = None
    """Seed to use for random number determinism.
    If specified, APIs may use this parameter to make a best-effort at determinism.
    """

    temperature: float = 1.0
    """Controls randomness in the output.

    Higher values (e.g., 1.0) make output more random, while lower values (e.g., 0.2)
    make it more focused and deterministic.
    """

    top_p: float = 1.0
    """An alternative to temperature, called nucleus sampling.

    It sets the cumulative probability threshold for token selection.
    For example, 0.9 means only considering the tokens comprising
    the top 90% probability mass.
    """

    frequency_penalty: float = 0.0
    """Positive values penalize new tokens based on their existing frequency in the text
    so far, decreasing the model's likelihood to repeat the same line verbatim.
    """

    presence_penalty: float = 0.0
    """Positive values penalize new tokens based on whether they appear in the text
    so far, increasing the model's likelihood to talk about new topics.
    """

    stop: Optional[Union[str, List[str]]] = None
    """list of sequences where the API will stop generating further tokens."""

    remote_params: Optional[RemoteParams] = None
    """Parameters for running inference against a remote API."""

    def __post_init__(self):
        """Validates generation-specific parameters."""
        if self.temperature < 0:
            raise ValueError("Temperature must be non-negative.")

        if not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1.")
