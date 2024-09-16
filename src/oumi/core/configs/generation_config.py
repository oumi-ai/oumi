from dataclasses import dataclass
from typing import Optional

import numpy as np

from oumi.core.configs.base_config import BaseConfig
from oumi.core.configs.params.remote_params import RemoteParams


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

    seed: Optional[int] = None
    """Seed to use for random number determinism.
    If specified, APIs may use this parameter to make a best-effort at determinism.
    """

    remote_params: Optional[RemoteParams] = None
    """Parameters for running inference against a remote API."""

    def __post_init__(self):
        """Verifies/populates params."""
        if self.remote_params is not None:
            if not self.remote_params.api_url:
                raise ValueError("The API URL must be provided in remote_params.")
            if self.remote_params.num_workers < 1:
                raise ValueError(
                    "Number of num_workers must be greater than or equal to 1."
                )
            if self.remote_params.politeness_policy < 0:
                raise ValueError(
                    "Politeness policy must be greater than or equal to 0."
                )
            if self.remote_params.connection_timeout < 0:
                raise ValueError(
                    "Connection timeout must be greater than or equal to 0."
                )
            if not np.isfinite(self.remote_params.politeness_policy):
                raise ValueError("Politeness policy must be finite.")
            if self.remote_params.max_retries < 0:
                raise ValueError("Max retries must be greater than or equal to 0.")
