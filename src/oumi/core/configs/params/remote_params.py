from dataclasses import dataclass
from typing import Optional

import numpy as np

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

    batch_completion_window: Optional[str] = "24h"
    """Time window for batch completion. Currently only '24h' is supported.

    Only used for batch inference.
    """

    def __post_init__(self):
        """Validate the remote parameters."""
        if not self.api_url:
            raise ValueError("The API URL must be provided in remote_params.")
        if self.num_workers < 1:
            raise ValueError(
                "Number of num_workers must be greater than or equal to 1."
            )
        if self.politeness_policy < 0:
            raise ValueError("Politeness policy must be greater than or equal to 0.")
        if self.connection_timeout < 0:
            raise ValueError("Connection timeout must be greater than or equal to 0.")
        if not np.isfinite(self.politeness_policy):
            raise ValueError("Politeness policy must be finite.")
        if self.max_retries < 0:
            raise ValueError("Max retries must be greater than or equal to 0.")
