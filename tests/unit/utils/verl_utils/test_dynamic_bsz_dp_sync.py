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

import importlib
import sys
from types import ModuleType
from typing import Any
from unittest.mock import patch

import torch.distributed as dist

from oumi.utils import packaging

_MODULE_NAME = "oumi.utils.verl_utils.dynamic_bsz_dp_sync"
_DP_ACTOR_MODULE_NAME = "verl.workers.actor.dp_actor"


def _import_fresh_module(verl_v0_8_or_later: bool = True):
    sys.modules.pop(_MODULE_NAME, None)
    with patch.object(
        packaging, "is_verl_v0_8_or_later", return_value=verl_v0_8_or_later
    ):
        return importlib.import_module(_MODULE_NAME)


def _fake_dp_actor_modules(prepare_dynamic_batch) -> dict[str, ModuleType]:
    dp_actor = ModuleType(_DP_ACTOR_MODULE_NAME)
    setattr(dp_actor, "prepare_dynamic_batch", prepare_dynamic_batch)
    return {
        "verl": ModuleType("verl"),
        "verl.workers": ModuleType("verl.workers"),
        "verl.workers.actor": ModuleType("verl.workers.actor"),
        _DP_ACTOR_MODULE_NAME: dp_actor,
    }


def _recording_prepare_dynamic_batch():
    calls: list[dict[str, Any]] = []

    def prepare_dynamic_batch(data, max_token_len, dp_group=None, **kwargs):
        calls.append(
            {"data": data, "max_token_len": max_token_len, "dp_group": dp_group}
        )
        return "micro_batches", "batch_idx_list"

    return prepare_dynamic_batch, calls


def test_passes_the_default_group_when_distributed_is_initialized():
    module = _import_fresh_module()
    prepare_dynamic_batch, calls = _recording_prepare_dynamic_batch()
    wrapped = module._with_default_dp_group(prepare_dynamic_batch)

    with (
        patch.object(dist, "is_available", return_value=True),
        patch.object(dist, "is_initialized", return_value=True),
    ):
        result = wrapped("batch", max_token_len=12288)

    assert result == ("micro_batches", "batch_idx_list")
    assert calls == [
        {"data": "batch", "max_token_len": 12288, "dp_group": dist.group.WORLD}
    ]


def test_keeps_an_explicit_group():
    module = _import_fresh_module()
    prepare_dynamic_batch, calls = _recording_prepare_dynamic_batch()
    wrapped = module._with_default_dp_group(prepare_dynamic_batch)
    explicit_group = object()

    with (
        patch.object(dist, "is_available", return_value=True),
        patch.object(dist, "is_initialized", return_value=True),
    ):
        wrapped("batch", max_token_len=12288, dp_group=explicit_group)
        wrapped("batch", 12288, explicit_group)

    assert [call["dp_group"] for call in calls] == [explicit_group, explicit_group]


def test_leaves_the_group_unset_without_distributed():
    module = _import_fresh_module()
    prepare_dynamic_batch, calls = _recording_prepare_dynamic_batch()
    wrapped = module._with_default_dp_group(prepare_dynamic_batch)

    with patch.object(dist, "is_initialized", return_value=False):
        wrapped("batch", max_token_len=12288)

    assert calls[0]["dp_group"] is None


def test_install_wraps_dp_actor_once():
    prepare_dynamic_batch, calls = _recording_prepare_dynamic_batch()
    fake_modules = _fake_dp_actor_modules(prepare_dynamic_batch)

    with patch.dict(sys.modules, fake_modules):
        module = _import_fresh_module(verl_v0_8_or_later=False)
        dp_actor = sys.modules[_DP_ACTOR_MODULE_NAME]
        wrapped = getattr(dp_actor, "prepare_dynamic_batch")
        module._install_patch()

        assert wrapped is not prepare_dynamic_batch
        assert getattr(dp_actor, "prepare_dynamic_batch") is wrapped
        with (
            patch.object(dist, "is_available", return_value=True),
            patch.object(dist, "is_initialized", return_value=True),
        ):
            wrapped("batch", max_token_len=12288)

    assert calls[0]["dp_group"] is dist.group.WORLD


def test_skips_install_on_verl_v0_8_or_later():
    prepare_dynamic_batch, _ = _recording_prepare_dynamic_batch()
    fake_modules = _fake_dp_actor_modules(prepare_dynamic_batch)

    with patch.dict(sys.modules, fake_modules):
        _import_fresh_module(verl_v0_8_or_later=True)
        dp_actor = sys.modules[_DP_ACTOR_MODULE_NAME]

        assert getattr(dp_actor, "prepare_dynamic_batch") is prepare_dynamic_batch
