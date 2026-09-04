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
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from oumi.utils import packaging

_MODULE_NAME = "oumi.utils.verl_utils.fsdp1_rank_buffer_sync"


def _import_fresh_module():
    sys.modules.pop(_MODULE_NAME, None)
    return importlib.import_module(_MODULE_NAME)


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
    actor_module = ModuleType("verl.workers.actor")
    actor_module.dp_actor = SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue]
        DataParallelPPOActor=FakeDataParallelPPOActor
    )

    with (
        patch.object(packaging, "is_verl_v0_8_or_later", return_value=False),
        patch.dict(sys.modules, {"verl.workers.actor": actor_module}),
    ):
        module = _import_fresh_module()

    assert FakeDataParallelPPOActor.__init__ is not original_init
    assert getattr(FakeDataParallelPPOActor, "_rank_buffer_sync_patched") is True

    with patch.object(module.dist, "is_available", return_value=False):
        actor = FakeDataParallelPPOActor(None, object())
    assert actor.initialized is True
    sys.modules.pop(_MODULE_NAME, None)
