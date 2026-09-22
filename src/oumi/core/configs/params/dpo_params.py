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
from typing import Any

from oumi.core.configs.params.base_params import BaseParams
from oumi.exceptions import OumiConfigError


@dataclass
class DpoParams(BaseParams):
    precompute_ref_log_probs: bool = False
    """Whether to compute reference-model log probabilities before training.

    Precomputation avoids keeping a reference model in memory during optimization.
    """

    precompute_ref_batch_size: int | None = None
    """Per-device batch size used to compute reference log probabilities.

    If unset, TRL uses the corresponding training or evaluation batch size.
    """

    ref_log_probs_cache_dir: str | None = None
    """Optional directory for reusable reference log-probability cache files.

    The caller is responsible for restoring and persisting this directory if cache
    entries should be shared across machines or jobs.
    """

    def __post_init__(self):
        """Validates DPO parameters."""
        if (
            self.precompute_ref_batch_size is not None
            and self.precompute_ref_batch_size <= 0
        ):
            raise OumiConfigError("precompute_ref_batch_size must be positive.")
        if self.ref_log_probs_cache_dir and not self.precompute_ref_log_probs:
            raise OumiConfigError(
                "ref_log_probs_cache_dir requires precompute_ref_log_probs=True."
            )

    def to_hf_trainer_kwargs(self) -> dict[str, Any]:
        """Converts DpoParams to TRL's DPOConfig kwargs."""
        result: dict[str, Any] = {}
        if self.precompute_ref_log_probs:
            result["precompute_ref_log_probs"] = True
        if self.precompute_ref_batch_size is not None:
            result["precompute_ref_batch_size"] = self.precompute_ref_batch_size
        return result
