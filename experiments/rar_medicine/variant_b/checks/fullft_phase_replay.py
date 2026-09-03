"""Replay the full-FT actor+ref worker's step-1 phases on 4 GPUs and account for GPU memory.

Mirrors verl 0.7.1 fsdp_workers for train_verl.yaml (lora_rank 0): actor = fp32 HF model in
FSDP1 FULL_SHARD with bf16 MixedPrecision (no offload), ref = same model in FSDP1 with
CPUOffload(offload_params=True), both in ONE process per GPU as verl colocates them. Then:
  1. rollout_mode's weight collection: actor.state_dict() (SHARDED_STATE_DICT / DTensor) and a
     full_tensor() all-gather of every tensor, as the per-tensor streaming generator does
  2. compute_log_prob (old log-probs): micro-batch MB_LP x (P+R) tokens, entropy on
  3. compute_ref_log_prob: micro-batch MB_REF, entropy off
and prints live memory + a per-call-site breakdown of live blocks at each boundary.

Run (4 idle GPUs, ~8 min):
  CUDA_VISIBLE_DEVICES=0,1,2,3 MB_LP=8 MB_REF=16 torchrun --nproc_per_node 4 fullft_phase_replay.py
"""

import os
from collections import defaultdict

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardedStateDictConfig
from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM
from verl.utils import torch_functional as verl_F
from verl.utils.fsdp_utils import get_fsdp_wrap_policy

GiB = 2**30
MB_LP = int(os.environ.get("MB_LP", "8"))
MB_REF = int(os.environ.get("MB_REF", "16"))
P, R = int(os.environ.get("P", "512")), 1024
RECORD = os.environ.get("RECORD", "1") == "1"

dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
local = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local)
dev = torch.device("cuda", local)
mesh = init_device_mesh("cuda", mesh_shape=(world,), mesh_dim_names=("fsdp",))
if RECORD:
    torch.cuda.memory._record_memory_history(max_entries=400000)


def p(msg):
    if rank == 0:
        print(msg, flush=True)


def mem(tag):
    torch.cuda.synchronize()
    a = torch.cuda.memory_allocated() / GiB
    r = torch.cuda.memory_reserved() / GiB
    m = torch.cuda.max_memory_allocated() / GiB
    p(f"[mem] {tag:60s} allocated {a:6.2f}  reserved {r:6.2f}  peak {m:6.2f} GiB")


def attribute(frames):
    keys = [
        "modeling_gemma4",
        "_flat_param",
        "fsdp/_runtime_utils",
        "fsdp/_init_utils",
        "fsdp/_state_dict_utils",
        "fsdp/_unshard_param_utils",
        "fsdp_utils",
        "torch_functional",
        "fullft_phase_replay",
        "_optimizer_utils",
        "distributed/tensor",
        "dtensor",
    ]
    for fr in frames:
        fn = fr.get("filename", "")
        if any(k in fn for k in keys):
            return f"{os.path.basename(fn)}:{fr.get('name')}:{fr.get('line')}"
    for fr in frames:
        fn = fr.get("filename", "")
        if "site-packages/torch/" not in fn and fn:
            return f"{os.path.basename(fn)}:{fr.get('name')}:{fr.get('line')}"
    return "unattributed"


def live_breakdown(tag, top=12):
    if not RECORD or rank != 0:
        return
    snap = torch.cuda.memory._snapshot()
    agg = defaultdict(int)
    total = 0
    for seg in snap["segments"]:
        for blk in seg["blocks"]:
            if blk["state"] != "active_allocated":
                continue
            agg[attribute(blk.get("frames", []))] += blk["size"]
            total += blk["size"]
    print(
        f"[live] {tag}: {total / GiB:.2f} GiB in active blocks, by allocating call site:",
        flush=True,
    )
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1])[:top]:
        if v / GiB >= 0.05:
            print(f"      {v / GiB:6.2f} GiB  {k}", flush=True)


def build(role):
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-E2B-it", dtype=torch.float32, attn_implementation="sdpa"
    )
    if role == "actor":
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    policy = get_fsdp_wrap_policy(
        module=model, config={"min_num_params": 0}, is_lora=False
    )
    mp = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    )
    fsdp = FSDP(
        model,
        cpu_offload=None if role == "actor" else CPUOffload(offload_params=True),
        auto_wrap_policy=policy,
        device_id=dev,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp,
        sync_module_states=True,
        device_mesh=mesh,
        use_orig_params=False,
        forward_prefetch=False,
    )
    return fsdp


ref = build("ref")
mem("after ref FSDP init (CPUOffload)")
actor = build("actor")
opt = torch.optim.AdamW(actor.parameters(), lr=1e-6)
FSDP.set_state_dict_type(
    actor,
    state_dict_type=StateDictType.SHARDED_STATE_DICT,
    state_dict_config=ShardedStateDictConfig(),
)
n_units = len(actor._all_handles) if hasattr(actor, "_all_handles") else -1
mem("after actor FSDP init + AdamW (lazy)")
torch.cuda.empty_cache()
mem("after empty_cache (start of step 1)")

# ---- 1. rollout_mode weight collection -------------------------------------------------
torch.cuda.reset_peak_memory_stats()
params = actor.state_dict()
kinds = defaultdict(int)
for v in params.values():
    kinds[type(v).__name__] += 1
p(f"[state_dict] {len(params)} tensors, types {dict(kinds)}")
mem("after actor.state_dict()")
biggest = 0
for name, v in params.items():
    t = v.to(dev, non_blocking=True).full_tensor() if isinstance(v, DTensor) else v
    biggest = max(biggest, t.numel() * t.element_size())
    del t
mem(f"after full_tensor() stream of all params (largest {biggest / GiB:.2f} GiB)")
del params
torch.cuda.empty_cache()
mem("after del state_dict + empty_cache")
live_breakdown("after weight-sync emulation")

# ---- 2. old log-probs ----------------------------------------------------------------
torch.manual_seed(0)
actor.eval()


def fwd(model, mb, entropy):
    input_ids = torch.randint(0, 262144, (mb, P + R), device=dev)
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(P + R, device=dev).unsqueeze(0).expand(mb, -1)
    responses = input_ids[:, -R:]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        logits = out.logits[:, -R - 1 : -1, :]
        lp = verl_F.logprobs_from_logits(logits, responses)
        ent = verl_F.entropy_from_logits(logits) if entropy else None
    return lp, ent


torch.cuda.reset_peak_memory_stats()
outs = []
for i in range(2):
    outs.append(fwd(actor, MB_LP, entropy=True))
    mem(f"old_log_prob: after micro-batch {i + 1} (mb {MB_LP}, entropy on)")
torch.cuda.synchronize()
mem("old_log_prob: end")
live_breakdown("after old_log_prob")

# ---- 3. ref log-probs ---------------------------------------------------------------
ref.eval()
torch.cuda.reset_peak_memory_stats()
try:
    lp, _ = fwd(ref, MB_REF, entropy=False)
    mem(f"ref: after micro-batch 1 (mb {MB_REF}, entropy off)")
    lp, _ = fwd(ref, MB_REF, entropy=False)
    mem("ref: after micro-batch 2")
except torch.OutOfMemoryError as e:
    p("[oom] ref forward: " + str(e).split("If reserved")[0].strip())
live_breakdown("after ref forward")
dist.barrier()
dist.destroy_process_group()
