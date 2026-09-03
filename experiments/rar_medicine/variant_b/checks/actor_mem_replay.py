"""Replay verl 0.7.1's actor path for train_verl.yaml on 4 GPUs and account for GPU memory.

Mirrors fsdp_workers._build_model_optimizer (fp32 model, PEFT LoRA, FSDP1 FULL_SHARD,
bf16 MixedPrecision, verl wrap policy with is_lora=True, use_orig_params=False),
init_model's offload, rollout_mode's load -> collect_lora_params(base_sync_done=False)
-> offload, then compute_log_prob's load -> _forward_micro_batch (non-rmpad branch).
Prints allocated memory after each stage and a per-call-site breakdown of live blocks
right before entropy_from_logits, i.e. the state the OOM message reports.

Run (4 idle GPUs, ~6 min): CUDA_VISIBLE_DEVICES=4,5,6,7 MB=32 torchrun --nproc_per_node 4 actor_mem_replay.py
Optional: SNAP=/path/snapshot.pickle writes a torch memory snapshot for pytorch.org/memory_viz.
"""

import os
from collections import defaultdict

import torch
import torch.distributed as dist
from peft import LoraConfig, TaskType, get_peft_model
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from transformers import AutoModelForCausalLM
from verl.utils import torch_functional as verl_F
from verl.utils.fsdp_utils import (
    collect_lora_params,
    get_fsdp_wrap_policy,
    load_fsdp_model_to_gpu,
    offload_fsdp_model_to_cpu,
)

GiB = 2**30
MB = int(os.environ.get("MB", "32"))
P, R = 1024, 1024
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
    p(f"[mem] {tag:56s} allocated {a:6.2f}  reserved {r:6.2f}  peak {m:6.2f} GiB")


def handle_report(fsdp, tag):
    """Per-FSDP-unit view: which flat params are resident/unsharded on this rank."""
    if rank != 0:
        return
    rows = []
    tot_shard = tot_unsharded = tot_mp = 0
    for h in fsdp._all_handles:
        fp = h.flat_param
        shard_b = (
            fp._local_shard.numel() * fp._local_shard.element_size()
            if fp._local_shard.is_cuda
            else 0
        )
        unsharded = fp.data.data_ptr() != fp._local_shard.data_ptr() and fp.data.is_cuda
        uns_b = fp.data.numel() * fp.data.element_size() if unsharded else 0
        padded = getattr(fp, "_full_param_padded", None)
        pad_b = (
            padded.untyped_storage().nbytes()
            if padded is not None and padded.is_cuda
            else 0
        )
        mp = getattr(fp, "_mp_shard", None)
        mp_b = mp.untyped_storage().nbytes() if mp is not None and mp.is_cuda else 0
        tot_shard += shard_b
        tot_unsharded += max(uns_b, pad_b)
        tot_mp += mp_b
        name = type(h._fully_sharded_module).__name__
        if unsharded or pad_b:
            rows.append(
                f"      unit {name:32s} numel {fp.numel() / 1e9:6.3f}B  unsharded/padded buf {max(uns_b, pad_b) / GiB:5.2f} GiB  dtype {fp.data.dtype}"
            )
    print(
        f"[fsdp] {tag}: {len(fsdp._all_handles)} units; local fp32 shards {tot_shard / GiB:.2f} GiB; "
        f"unsharded buffers resident {tot_unsharded / GiB:.2f} GiB; mp shards {tot_mp / GiB:.2f} GiB",
        flush=True,
    )
    for r_ in rows:
        print(r_, flush=True)


def attribute(frames):
    """Pick the most informative frame for a live block."""
    keys = [
        "modeling_gemma4",
        "peft/tuners",
        "_flat_param",
        "fsdp/_runtime_utils",
        "fsdp/_init_utils",
        "fsdp/_unshard_param_utils",
        "fsdp_utils",
        "torch_functional",
        "replay_actor_mem",
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


def live_breakdown(tag, top=14):
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
        print(f"      {v / GiB:6.2f} GiB  {k}", flush=True)


# ---- build like fsdp_workers._build_model_optimizer -------------------------------
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-E2B-it", dtype=torch.float32, attn_implementation="sdpa"
)
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
model.enable_input_require_grads()
lora = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    exclude_modules="(.*vision_tower.*)|(.*audio_tower.*)",
    bias="none",
)
model = get_peft_model(model, lora)
n_total = sum(q.numel() for q in model.parameters())
n_train = sum(q.numel() for q in model.parameters() if q.requires_grad)
p(
    f"[model] {type(model.base_model.model).__name__}: {n_total / 1e9:.3f} B params, {n_train / 1e6:.1f} M trainable (LoRA r=64)"
)
policy = get_fsdp_wrap_policy(module=model, config={"min_num_params": 0}, is_lora=True)
mp = MixedPrecision(
    param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32
)
fsdp = FSDP(
    model,
    cpu_offload=None,
    auto_wrap_policy=policy,
    device_id=dev,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    mixed_precision=mp,
    sync_module_states=True,
    device_mesh=mesh,
    use_orig_params=False,
    forward_prefetch=False,
)
mem("after FSDP init (fp32 shards resident)")
offload_fsdp_model_to_cpu(fsdp)
mem("init_model: after offload_fsdp_model_to_cpu")

# ---- rollout_mode, step 1 (base_sync_done=False) ------------------------------------
load_fsdp_model_to_gpu(fsdp)
mem("rollout_mode: after load_fsdp_model_to_gpu")
torch.cuda.reset_peak_memory_stats()
params = collect_lora_params(module=fsdp, layered_summon=False, base_sync_done=False)
mem("rollout_mode: after collect_lora_params(base_sync_done=False)")
handle_report(fsdp, "after collect_lora_params")
del params
torch.cuda.empty_cache()
mem("rollout_mode: after del params + empty_cache")
offload_fsdp_model_to_cpu(fsdp)
mem("rollout_mode: after offload_fsdp_model_to_cpu")

# ---- compute_log_prob --------------------------------------------------------------
load_fsdp_model_to_gpu(fsdp)
mem("compute_log_prob: after load_fsdp_model_to_gpu")
torch.manual_seed(0)
input_ids = torch.randint(0, 262144, (MB, P + R), device=dev)
attention_mask = torch.ones_like(input_ids)
position_ids = torch.arange(P + R, device=dev).unsqueeze(0).expand(MB, -1)
responses = input_ids[:, -R:]
fsdp.eval()
torch.cuda.reset_peak_memory_stats()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    out = fsdp(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    logits = out.logits
    p(
        f"[logits] {logits.dtype} {tuple(logits.shape)} = {logits.numel() * logits.element_size() / GiB:.2f} GiB"
    )
    mem(f"after model forward, micro-batch {MB}")
    handle_report(fsdp, "after forward")
    logits.div_(1.0)
    logits = logits[:, -R - 1 : -1, :]
    log_probs = verl_F.logprobs_from_logits(logits, responses)
    mem("after logprobs_from_logits (state the OOM message reports)")
    live_breakdown("before entropy_from_logits")
    try:
        ent = verl_F.entropy_from_logits(logits)
        mem("after entropy_from_logits")
    except torch.OutOfMemoryError as e:
        p("[oom] reproduced: " + str(e).split("If reserved")[0].strip())
    del out, logits, log_probs
torch.cuda.synchronize()
if RECORD and rank == 0:
    path = os.environ.get("SNAP", "/dev/null")
    if path != "/dev/null":
        torch.cuda.memory._dump_snapshot(path)
        p(f"[snapshot] written to {path}")
dist.barrier()
dist.destroy_process_group()
