"""Re-broadcast every module buffer from rank 0 after verl builds an FSDP actor/ref.

Why: in the 2026-09-02 diagnostic run (`verl_logprob_dump_hook.py`) the actor's log-probs were
correct on rank 0 and wrong on ranks 1-3 for every sequence (seq-avg -logp 2.46 vs 0.46), while
vLLM agreed with rank 0. FSDP all-gathers *parameters* identically on all ranks, so a weight
problem would hit rank 0 too; only per-rank, non-sharded state (buffers) can differ. Comparing all
params+buffers across ranks after FSDP construction found exactly three mismatches, gemma-4's RoPE
tables `rotary_emb.{sliding_attention,full_attention}_inv_freq` (+ the `original_inv_freq`
alias). transformers keeps those as aliased buffers; FSDP's coalesced `sync_module_states`
broadcast de-duplicates aliases per tensor object, and the ranks do not end up with the same
buffer list, so the tables land in the wrong slots (non-deterministically: some launches are fine).

How: wraps DataParallelPPOActor.__init__ (verl constructs it right after FSDP wrapping, for both
the actor and the ref). Rank 0 publishes its buffer list BY NAME (with shapes/dtypes) via
broadcast_object_list; every rank then takes part in exactly the same sequence of broadcasts,
writing into its own buffer of that name (aliases included, `remove_duplicate=False`) or into a
scratch tensor if it has no such buffer, so the collectives can never misalign or deadlock.
Loaded via external_lib (bundled in gemma4_verl_fixes.py). Prints per rank how many buffers were
overwritten with different content, so the effect is visible in the run log.
"""

import torch
import torch.distributed as dist
from verl.workers.actor import dp_actor

_orig_init = dp_actor.DataParallelPPOActor.__init__


def _sync_buffers_from_rank0(module) -> tuple[int, int]:
    rank = dist.get_rank()
    device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.synchronize()
    local = dict(module.named_buffers(remove_duplicate=False))
    if rank == 0:
        spec = [
            (n, tuple(b.shape), str(b.dtype)) for n, b in local.items() if b.numel() > 0
        ]
    else:
        spec = None
    holder = [spec]
    dist.broadcast_object_list(holder, src=0)
    spec = holder[0]
    n_total, n_changed = 0, 0
    with torch.no_grad():
        for name, shape, dtype_str in spec:
            dtype = getattr(torch, dtype_str.split(".")[-1])
            buf = local.get(name)
            usable = (
                buf is not None
                and tuple(buf.shape) == tuple(shape)
                and buf.dtype == dtype
            )
            if usable and buf.is_cuda:
                target = buf.data
                before = target.clone() if rank != 0 else None
            else:
                target = torch.empty(
                    shape, dtype=dtype, device=device
                )  # scratch: keep collectives aligned
                before = None
            dist.broadcast(target, src=0)
            if usable and not buf.is_cuda:
                buf.data.copy_(target.to(buf.device))
            n_total += 1
            if before is not None and not torch.equal(before, target):
                n_changed += 1
    torch.cuda.synchronize()
    return n_total, n_changed


def __init__(self, config, actor_module, actor_optimizer=None, **kwargs):
    _orig_init(self, config, actor_module, actor_optimizer, **kwargs)
    if (
        not (dist.is_available() and dist.is_initialized())
        or dist.get_world_size() == 1
    ):
        return
    n_total, n_changed = _sync_buffers_from_rank0(actor_module)
    print(
        f"[rank-buffer-sync] rank {dist.get_rank()}: {n_total} buffers broadcast from rank 0, "
        f"{n_changed} had different content locally",
        flush=True,
    )


if not getattr(dp_actor.DataParallelPPOActor, "_rank_buffer_sync_patched", False):
    dp_actor.DataParallelPPOActor.__init__ = __init__
    dp_actor.DataParallelPPOActor._rank_buffer_sync_patched = True
    print("[verl_rank_buffer_sync] installed", flush=True)
