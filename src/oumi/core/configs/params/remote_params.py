from dataclasses import dataclass
from typing import Optional

from oumi.core.configs.params.base_params import BaseParams


@dataclass
class RemoteParams(BaseParams):
    """Parameters for running inference against a remote API."""

    api_url: Optional[str] = None
    """URL of the API endpoint to use for inference."""

    api_key: Optional[str] = None
    """API key to use for authentication."""

    api_key_env_varname: Optional[str] = None
    """Name of the environment variable containing the API key for authentication."""

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
