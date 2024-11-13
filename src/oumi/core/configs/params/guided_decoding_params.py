import dataclasses
from typing import Any, Optional

from oumi.core.configs.params.base_params import BaseParams


@dataclasses.dataclass
class GuidedDecodingParams(BaseParams):
    """Parameters for guided decoding.

    The parameters are mutually exclusive. Only one of the parameters can be
    specified at a time.
    """

    # Should be Union[dict, BaseModel, str], but omegaconf does not like Union
    json: Optional[Any] = None
    """JSON schema, Pydantic model, or string to guide the output format.

    Can be a dict containing a JSON schema, a Pydantic model class, or a string
    containing JSON schema. Used to enforce structured output from the model.
    """

    regex: Optional[str] = None
    """Regular expression pattern to guide the output format.

    Pattern that the model output must match. Can be used to enforce specific
    text formats or patterns.
    """

    choice: Optional[list[str]] = None
    """List of allowed choices for the output.

    Restricts model output to one of the provided choices. Useful for forcing
    the model to select from a predefined set of options.
    """

    def __post_init__(self) -> None:
        """Validate parameters."""
        provided = sum(x is not None for x in [self.json, self.regex, self.choice])
        if provided > 1:
            raise ValueError(
                "Only one of 'json', 'regex', or 'choice' can be specified"
            )
