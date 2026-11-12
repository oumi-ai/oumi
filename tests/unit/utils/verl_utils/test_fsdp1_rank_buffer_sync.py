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
from datetime import timedelta
from types import ModuleType
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from oumi.utils import packaging

_MODULE_NAME = "oumi.utils.verl_utils.fsdp1_rank_buffer_sync"


def _import_fresh_module():
    sys.modules.pop(_MODULE_NAME, None)
    return importlib.import_module(_MODULE_NAME)


def _run_distributed_buffer_sync(rank: int, world_size: int, init_method: str):
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=20),
    )
    try:
        with patch.object(packaging, "is_verl_v0_8_or_later", return_value=True):
            module = _import_fresh_module()

        actor_module = torch.nn.Module()
        actor_module.register_buffer(
            "shared",
            torch.tensor([1.0, 2.0]) if rank == 0 else torch.tensor([-1.0, -2.0]),
        )
        alias = torch.tensor([3.0, 4.0]) if rank == 0 else torch.tensor([-3.0, -4.0])
        actor_module.register_buffer("alias_a", alias)
        actor_module.register_buffer("alias_b", alias)
        actor_module.register_buffer("already_equal", torch.tensor([5.0]))
        if rank == 0:
            actor_module.register_buffer("rank0_only", torch.tensor([6.0]))
        else:
            actor_module.register_buffer("rank1_only", torch.tensor([7.0]))

        n_total, n_changed = module._sync_buffers_from_rank0(actor_module)

        assert n_total == 5
        assert n_changed == (0 if rank == 0 else 2)
        torch.testing.assert_close(actor_module.shared, torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(actor_module.alias_a, torch.tensor([3.0, 4.0]))
        torch.testing.assert_close(actor_module.alias_b, torch.tensor([3.0, 4.0]))
        assert actor_module.get_buffer("alias_a") is actor_module.get_buffer("alias_b")
        if rank == 1:
            torch.testing.assert_close(actor_module.rank1_only, torch.tensor([7.0]))
    finally:
        dist.destroy_process_group()


def test_skips_legacy_actor_import_on_verl_v0_8_or_later():
    with patch.object(packaging, "is_verl_v0_8_or_later", return_value=True):
        module = _import_fresh_module()

    assert module.is_verl_v0_8_or_later()
    sys.modules.pop(_MODULE_NAME, None)


def test_installs_patch_on_legacy_verl():
    class FakeDataParallelPPOActor:
        def __init__(self, config, actor_module, actor_optimizer=None, **kwargs):
            self.initialized = True

    original_init = FakeDataParallelPPOActor.__init__
    dp_actor_module = ModuleType("verl.workers.actor.dp_actor")
    dp_actor_module.DataParallelPPOActor = (  # pyright: ignore[reportAttributeAccessIssue]
        FakeDataParallelPPOActor
    )
    with (
        patch.object(packaging, "is_verl_v0_8_or_later", return_value=False),
        patch.dict(sys.modules, {"verl.workers.actor.dp_actor": dp_actor_module}),
    ):
        module = _import_fresh_module()

    assert FakeDataParallelPPOActor.__init__ is not original_init
    assert getattr(FakeDataParallelPPOActor, "_rank_buffer_sync_patched") is True

    with patch.object(module.dist, "is_available", return_value=False):
        actor = FakeDataParallelPPOActor(None, object())
    assert actor.initialized is True
    sys.modules.pop(_MODULE_NAME, None)


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_syncs_buffers_with_unequal_lists_across_processes(tmp_path):
    init_method = f"file://{tmp_path / 'distributed_init'}"

    mp.spawn(
        _run_distributed_buffer_sync,
        args=(2, init_method),
        nprocs=2,
        join=True,
    )
