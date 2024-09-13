from dataclasses import dataclass
from typing import Optional

from lema.core.configs.base_config import BaseConfig


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

    api_url: Optional[str] = None
    """URL of the API endpoint to use for inference."""

    api_key: Optional[str] = None
    """API key to use for authentication."""

    max_retries: int = 3
    """Maximum number of retries to attempt when calling an API."""

    connection_timeout: float = 20.0
    """Timeout in seconds for a request to an API."""

    num_workers: int = 1
    """Number of workers to use for parallel inference."""

    politeness_policy: float = 0.0
    """Politeness policy to use when calling an API.

    If greater than zero, this is the amount of time in seconds a worker will sleep
    before making a subsequent request.
    """

    seed: Optional[int] = None
    """Seed to use for random number determinism.

    If specified, APIs may use this parameter to make a best-effort at determinism.
    """
