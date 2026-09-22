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

import pytest

from oumi.core.configs.params.dpo_params import DpoParams
from oumi.exceptions import OumiConfigError


def test_to_hf_trainer_kwargs_enables_precomputation():
    params = DpoParams(
        precompute_ref_log_probs=True,
        precompute_ref_batch_size=4,
    )

    assert params.to_hf_trainer_kwargs() == {
        "precompute_ref_log_probs": True,
        "precompute_ref_batch_size": 4,
    }


def test_default_params_do_not_override_trainer_kwargs():
    assert DpoParams().to_hf_trainer_kwargs() == {}


def test_precompute_ref_batch_size_must_be_positive():
    with pytest.raises(OumiConfigError, match="must be positive"):
        DpoParams(precompute_ref_batch_size=0)


def test_cache_dir_requires_precomputation():
    with pytest.raises(OumiConfigError, match="requires precompute_ref_log_probs"):
        DpoParams(ref_log_probs_cache_dir="cache")
