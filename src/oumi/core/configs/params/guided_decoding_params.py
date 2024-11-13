from typing import Any, Optional

from pydantic import BaseModel


class GuidedDecodingParams(BaseModel):
    """Parameters for guided decoding."""

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
