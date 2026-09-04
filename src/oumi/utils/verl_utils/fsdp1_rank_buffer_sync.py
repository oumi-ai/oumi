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

"""Synchronize VERL FSDP1 actor and reference buffers across ranks.

Some models can retain different non-sharded buffers, such as RoPE tables,
across ranks after VERL constructs an FSDP1 worker. This causes rank-dependent
log-probs. The patch wraps ``DataParallelPPOActor.__init__`` and broadcasts
rank 0's buffers by name, shape, and dtype, including aliased buffers.
"""

from typing import Any, cast

import torch
import torch.distributed as dist

from oumi.utils.packaging import is_verl_v0_8_or_later

_BufferSpec = tuple[str, tuple[int, ...], str]


def _sync_buffers_from_rank0(module: torch.nn.Module) -> tuple[int, int]:
    rank = dist.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.synchronize()
    local = dict(module.named_buffers(remove_duplicate=False))
    if rank == 0:
        spec: list[_BufferSpec] | None = [
            (n, tuple(b.shape), str(b.dtype)) for n, b in local.items() if b.numel() > 0
        ]
    else:
        spec = None
    holder: list[Any] = [spec]
    dist.broadcast_object_list(holder, src=0)
    received_spec = holder[0]
    if received_spec is None:
        raise RuntimeError("Rank 0 did not provide a module-buffer specification.")
    spec = cast(list[_BufferSpec], received_spec)
    n_total, n_changed = 0, 0
    with torch.no_grad():
        for name, shape, dtype_str in spec:
            dtype = getattr(torch, dtype_str.split(".")[-1])
            buf = local.get(name)
            usable = bool(
                buf is not None
                and tuple(buf.shape) == tuple(shape)
                and buf.dtype == dtype
            )
            if usable and buf is not None and buf.is_cuda:
                target = buf.data
                before = target.clone() if rank != 0 else None
            else:
                target = torch.empty(
                    shape, dtype=dtype, device=device
                )  # scratch: keep collectives aligned
                before = None
            dist.broadcast(target, src=0)
            if usable and buf is not None and not buf.is_cuda:
                buf.data.copy_(target.to(buf.device))
            n_total += 1
            if before is not None and not torch.equal(before, target):
                n_changed += 1
    torch.cuda.synchronize()
    return n_total, n_changed


def _install_patch() -> None:
    # This legacy actor module was removed in verl 0.8's worker-to-engine migration.
    from verl.workers.actor import dp_actor

    orig_init = dp_actor.DataParallelPPOActor.__init__

    def patched_init(
        self: Any,
        config: Any,
        actor_module: torch.nn.Module,
        actor_optimizer: Any = None,
        **kwargs: Any,
    ) -> None:
        cast(Any, orig_init)(self, config, actor_module, actor_optimizer, **kwargs)
        if (
            not (dist.is_available() and dist.is_initialized())
            or dist.get_world_size() == 1
        ):
            return
        n_total, n_changed = _sync_buffers_from_rank0(actor_module)
        print(
            f"[rank-buffer-sync] rank {dist.get_rank()}: "
            f"{n_total} buffers broadcast from rank 0, "
            f"{n_changed} had different content locally",
            flush=True,
        )

    if not getattr(dp_actor.DataParallelPPOActor, "_rank_buffer_sync_patched", False):
        setattr(dp_actor.DataParallelPPOActor, "__init__", patched_init)
        setattr(dp_actor.DataParallelPPOActor, "_rank_buffer_sync_patched", True)
        print("[fsdp1_rank_buffer_sync] installed", flush=True)


if is_verl_v0_8_or_later():
    print(
        "[fsdp1_rank_buffer_sync] skipped for verl >= 0.8; use the FSDP2 "
        "named-buffer synchronization path",
        flush=True,
    )
else:
    _install_patch()
