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

"""Keep VERL FSDP1 dynamic-batch micro-batch counts equal across ranks.

With ``use_dynamic_bsz``, VERL 0.7's ``DataParallelPPOActor`` splits each batch into
micro-batches by token budget, but calls ``prepare_dynamic_batch`` without a
``dp_group``, so its ``same_micro_num_in_dp`` sync never runs. Each rank then picks
``ceil(local_tokens / budget)`` on its own, and when two ranks land on different
counts they run different numbers of FSDP forward and backward passes and deadlock
on the next collective. The patch passes the default process group, so every rank
uses the largest count. With sequence parallelism this is the world-wide maximum,
which is still the same on every rank.
"""

from collections.abc import Callable
from importlib import import_module
from typing import Any, cast

import torch.distributed as dist

from oumi.utils.packaging import is_verl_v0_8_or_later

# ``prepare_dynamic_batch(data, max_token_len, dp_group=None, ...)``
_DP_GROUP_POSITION = 2


def _with_default_dp_group(
    prepare_dynamic_batch: Callable[..., Any],
) -> Callable[..., Any]:
    def prepare_dynamic_batch_with_dp_group(*args: Any, **kwargs: Any) -> Any:
        dp_group_given = (
            len(args) > _DP_GROUP_POSITION or kwargs.get("dp_group") is not None
        )
        if not dp_group_given and dist.is_available() and dist.is_initialized():
            kwargs["dp_group"] = dist.group.WORLD
        return prepare_dynamic_batch(*args, **kwargs)

    return prepare_dynamic_batch_with_dp_group


def _install_patch() -> None:
    # This legacy actor module was removed in verl 0.8's worker-to-engine migration.
    dp_actor = cast(Any, import_module("verl.workers.actor.dp_actor"))
    if getattr(dp_actor, "_dynamic_bsz_dp_sync_patched", False):
        return
    dp_actor.prepare_dynamic_batch = _with_default_dp_group(
        dp_actor.prepare_dynamic_batch
    )
    dp_actor._dynamic_bsz_dp_sync_patched = True
    print("[dynamic_bsz_dp_sync] installed", flush=True)


if is_verl_v0_8_or_later():
    print(
        "[dynamic_bsz_dp_sync] skipped for verl >= 0.8; its FSDP engine passes the "
        "data-parallel group",
        flush=True,
    )
else:
    _install_patch()
