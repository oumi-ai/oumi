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

"""Anthropic-specific configuration parameters."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from oumi.core.configs.params.base_params import BaseParams


class CacheDuration(str, Enum):
    """Duration for prompt caching."""

    FIVE_MINUTES = "ephemeral"
    """5-minute cache (write cost: 1.25x, read cost: 0.1x)."""

    ONE_HOUR = "persistent"
    """1-hour cache (write cost: 2x, read cost: 0.1x)."""

    def __str__(self) -> str:
        """Return the string representation of the CacheDuration enum."""
        return self.value


@dataclass
class AnthropicParams(BaseParams):
    """Anthropic-specific parameters."""

    beta_features: list[str] = field(default_factory=list)
    """List of beta features to enable.

    Common beta features:
    - "token-efficient-tools-2025-02-19": Token-efficient tool use (up to 70% reduction)
    - "fine-grained-tool-streaming-2025-05-14": Fine-grained tool parameter streaming
    - "context-management-2025-06-27": Automatic tool call clearing
    """

    enable_prompt_caching: bool = False
    """Whether to enable prompt caching.

    When enabled, you can mark portions of the prompt to be cached, reducing
    costs by up to 90% and latency by up to 85% for repeated prompts.
    """

    cache_duration: CacheDuration = CacheDuration.FIVE_MINUTES
    """Duration for prompt caching.

    - FIVE_MINUTES (ephemeral): Lower write cost (1.25x), 5-minute TTL
    - ONE_HOUR (persistent): Higher write cost (2x), 1-hour TTL
    """

    cache_breakpoints: Optional[list[int]] = None
    """Message indices where cache breakpoints should be inserted.

    If None and enable_prompt_caching is True, a cache breakpoint will be
    automatically placed before the last user message. You can specify custom
    indices to control caching behavior.
    """

    def get_beta_header_value(self) -> Optional[str]:
        """Get the anthropic-beta header value.

        Returns:
            Comma-separated list of beta features, or None if no features are enabled.
        """
        if not self.beta_features:
            return None
        return ",".join(self.beta_features)
