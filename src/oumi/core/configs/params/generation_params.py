from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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

    stop_strings: Optional[List[str]] = None
    """List of sequences where the API will stop generating further tokens."""

    stop_token_ids: Optional[List[int]] = None
    """List of token ids for which the API will stop generating further tokens. This
    is only supported in `VLLMInferenceEngine` and `NativeTextInferenceEngine`."""

    remote_params: Optional[RemoteParams] = None
    """Parameters for running inference against a remote API."""

    logit_bias: Dict[Any, float] = field(default_factory=dict)
    """Modify the likelihood of specified tokens appearing in the completion.

    Keys are tokens (specified by their token ID in the tokenizer),
    and values are the bias (-100 to 100). Mathematically, the bias is added
    to the logits generated by the model prior to sampling. The exact effect will
    vary per model, but values between -1 and 1 should decrease or increase
    likelihood of selection; values like -100 or 100 should result in a ban or
    exclusive selection of the relevant token.
    """

    min_p: float = 0.0
    """Sets a minimum probability threshold for token selection.

    Tokens with probabilities below this threshold are filtered out before top-p
    or top-k sampling. This can help prevent the selection of highly improbable tokens.
    Default is 0.0 (no minimum threshold).
    """

    def __post_init__(self):
        """Validates generation-specific parameters."""
        if self.temperature < 0:
            raise ValueError("Temperature must be non-negative.")

        if not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1.")

        for token_id, bias in self.logit_bias.items():
            if not isinstance(token_id, (str, int)):
                raise ValueError(
                    f"Logit bias token ID {token_id} must be an integer or a string."
                )

            if not -100 <= bias <= 100:
                raise ValueError(
                    f"Logit bias for token {token_id} must be between -100 and 100."
                )

        if not 0 <= self.min_p <= 1:
            raise ValueError("min_p must be between 0 and 1.")
