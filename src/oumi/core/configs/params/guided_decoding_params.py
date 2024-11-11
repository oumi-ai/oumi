from typing import Optional, Union

from pydantic import BaseModel


class GuidedDecodingParams(BaseModel):
    """Parameters for guided decoding.

    Args:
        json: JSON schema or Pydantic model to guide the output format.
        regex: Regular expression pattern to guide the output format.
        choice: List of allowed choices for the output.
        grammar: Grammar specification to guide the output format.
        json_object: Whether to force output to be a valid JSON object.
        backend: Backend to use for guided decoding.
        whitespace_pattern: Pattern for handling whitespace in the output.
    """

    json: Optional[Union[dict, BaseModel, str]] = None
    regex: Optional[str] = None
    choice: Optional[list[str]] = None
    grammar: Optional[str] = None
    json_object: Optional[bool] = None
    backend: Optional[str] = None
    whitespace_pattern: Optional[str] = None
