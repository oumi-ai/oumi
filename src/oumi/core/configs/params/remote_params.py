# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

    retry_backoff_base: float = 1.0
    """Base delay in seconds for exponential backoff between retries."""

    retry_backoff_max: float = 30.0
    """Maximum delay in seconds between retries."""

    connection_timeout: float = 300.0
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

    use_adaptive_concurrency: bool = True
    """Whether to use adaptive concurrency control.

    If True, the number of concurrent requests will be adjusted based on the error rate
    of the requests. As error rate increases above a threshold, the number of concurrent
    requests will decrease, and as error rate decreases below a threshold, the number of
    concurrent requests will increase.

    When this is enabled, users should set `num_workers` to the requests per minute
    (RPM/QPM) of the model/API, and the `politeness_policy` to 60s (as most APIs query
    limits are dictated by the number of requests per minute).

    The lowest concurrency can be is 1, and the highest concurrency is `num_workers`.
    Updates to concurrency will happen no sooner than `politeness_policy` seconds after
    the last update, and at least 10 requests must have been made since the last update.

    In the event that even 1 concurrency causes the error rate to exceed the threshold,
    it is recommended to increase the `politeness_policy` to allow more time between
    requests.
    """

    def __post_init__(self):
        """Validate the remote parameters."""
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
        if self.retry_backoff_base <= 0:
            raise ValueError("Retry backoff base must be greater than 0.")
        if self.retry_backoff_max < self.retry_backoff_base:
            raise ValueError(
                "Retry backoff max must be greater than or equal to retry backoff base."
            )


@dataclass
class AdaptiveConcurrencyParams(BaseParams):
    """Configuration for adaptive concurrency control."""

    min_concurrency: int = 5
    """Minimum number of concurrent requests to allow.

    Backoff throttling will never reduce concurrency below this value.
    """

    max_concurrency: int = 100
    """Maximum number of concurrent requests allowed.

    The concurrency will never be allowed to go above this value.
    """

    initial_concurrency_factor: float = 0.5
    """Initial concurrency factor.

    The initial concurrency will be set to (max_concurrency - min_concurrency) *
    initial_concurrency_factor + min_concurrency.

    Example:
    - min_concurrency = 5
    - max_concurrency = 100
    - initial_concurrency_factor = 0.5
    - initial_concurrency = (100 - 5) * 0.5 + 5 = 52.5 = 52
    """

    concurrency_step: int = 5
    """How much to increase concurrency during warmup.

    During warmup, concurrency will be increased by this amount. (i.e. if concurrency is
    50, and the concurrency step is 5, the concurrency will be increased to 55). This
    change will happen no sooner than min_update_time seconds after the last update.
    """

    min_update_time: float = 60.0
    """Minimum seconds between attempted updates.

    The concurrency will not be adjusted sooner than this time.
    """

    error_threshold: float = 0.01
    """Error rate threshold (0.01 = 1%) to trigger backoff.

    If the error rate is greater than this threshold, backoff will be triggered.

    Consider keeping this value low. Once a particular error rate is reached, there are
    already other requests in-flight which will likely fail, so the sooner concurrency
    is reduced, the better chance the system has of recovering.
    """

    backoff_factor: float = 0.8
    """Factor to multiply concurrency by during backoff.

    During backoff, the concurrency will be reduced by this factor. (i.e. if concurrency
    is 100, and the backoff factor is 0.8, the concurrency will be reduced to 80).
    """

    recovery_threshold: float = 0.00
    """Error rate threshold (0.00 = 0%) to allow recovery.

    If the error rate is less than this threshold, recovery will be triggered.
    """

    min_window_size: int = 10
    """Minimum number of recent requests to consider for error rate calculation.

    If the number of requests since the last update is less than this threshold, the
    concurrency will not be adjusted.
    """

    def __post_init__(self):
        """Validate the adaptive concurrency parameters."""
        if self.min_concurrency < 1:
            raise ValueError("Min concurrency must be greater than or equal to 1.")
        if self.max_concurrency < self.min_concurrency:
            raise ValueError(
                "Max concurrency must be greater than or equal to min concurrency."
            )
        if self.initial_concurrency_factor < 0 or self.initial_concurrency_factor > 1:
            raise ValueError("Initial concurrency factor must be between 0 and 1.")
        if self.concurrency_step < 1:
            raise ValueError("Concurrency step must be greater than or equal to 1.")
        if self.min_update_time <= 0:
            raise ValueError("Min update time must be greater than 0.")
        if self.error_threshold < 0 or self.error_threshold > 1:
            raise ValueError("Error threshold must be between 0 and 1.")
        if self.backoff_factor <= 0:
            raise ValueError("Backoff factor must be greater than 0.")
        if self.recovery_threshold < 0 or self.recovery_threshold > 1:
            raise ValueError("Recovery threshold must be between 0 and 1.")
        if self.recovery_threshold >= self.error_threshold:
            raise ValueError("Recovery threshold must be less than error threshold.")
        if self.min_window_size < 1:
            raise ValueError("Min window size must be greater than or equal to 1.")
