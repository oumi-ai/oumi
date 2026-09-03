# GPU memory budget: gemma-4-E2B-it LoRA GRPO on 4× H100 80GB

> **Superseded 2026-09-02 for the current recipe.** `train_verl.yaml` is now a
> full fine-tune with remove-padding, actor offload and the gemma-4 KV-share
> patch; the measured budget, the OOM post-mortem and the tuning log are in
> `MEMORY_TUNING.md`. The LoRA-path numbers below remain valid for a LoRA
> configuration (its step-1 gather peak was measured at 43.9 GiB, see
> `MEMORY_TUNING.md` §2), but note that vLLM LoRA is a no-op on gemma-4
> (issue 15) and that the "full-finetune baseline" section here predates the
> measurements in `MEMORY_TUNING.md` §4.

Working for `train_verl.yaml` as launched by `run.sh` (GPUs 0-3, verl 0.7.1
FSDP1 hybrid engine, vLLM 0.19.1 TP=2, LoRA r=16, no actor offload). Every
number below is derived from the model config or the verl source; the two
observed numbers it is checked against are called out as such.

## Constants

| Quantity | Value | Where it comes from |
|---|---|---|
| H100 usable memory | 79.65 GiB | `nvidia-smi`: 81 559 MiB |
| Model parameters | 5.53 B | `AutoModelForCausalLM.from_config` on the meta device, `sum(p.numel())` |
| …of which `embed_tokens_per_layer` | 2.35 B → **8.75 GiB fp32** | 262 144 × (35 layers × 256) — one tensor; the reason `update_weights_bucket_megabytes` is 10 240 |
| …of which embeddings + tied head + norms (root FSDP unit) | ~2.75 B → 5.5 GiB bf16 | everything not inside a decoder-layer FSDP unit (`wrap_policy.min_num_params: 0` wraps per layer) |
| Full model, fp32 / bf16 | 20.6 GiB / 10.3 GiB | 5.53 B × 4 / × 2 |
| fp32 shard per GPU at 4 GPUs | **5.15 GiB** | 20.6 / 4 |
| LoRA trainable params (r=16, 7 projections × 35 layers, towers excluded) | 25.3 M (0.46 %) | `get_peft_model` with the exact `LoraConfig` verl builds |
| LoRA fp32 weights / grads / AdamW (2 moments) | 97 MiB / 97 MiB / 193 MiB | 25.3 M × 4 / × 4 / × 8 |
| vLLM bf16 weights per GPU at TP=2 | 5.15 GiB | 10.3 / 2 |
| vLLM budget per GPU at `gpu_memory_utilization: 0.4` | 31.9 GiB | 0.4 × 79.65 |
| Sequence length | 2 048 | `max_prompt_length` 1 024 + `max_completion_length` 1 024 |
| Sequences per step / per GPU | 512 / 128 | `train_batch_size` 64 × `n` 8, over 4 DP ranks |
| Sequences per vLLM replica | 256 | 512 over 2 TP=2 replicas |

The actor keeps fp32 master weights (verl default `model_dtype` for the actor).
FSDP mixed precision runs forward/backward in bf16 by all-gathering a bf16
copy of each FSDP unit on demand and freeing it afterwards
(`reshard_after_forward: True`).

## The full-finetune baseline this replaces (why fix #11 was needed)

`GEMMA4_VERL_GRPO_FIXES.md` #11 observed ~61.5 GiB held by the FSDP actor plus
~17 GiB vLLM at the first weight sync on 4 GPUs. Reconstructing it:

| Term (full finetune, per GPU) | GiB |
|---|---|
| fp32 shard | 5.15 |
| AdamW moments, fp32, sharded (2 × 20.6 / 4) | 10.3 |
| fp32 grads, sharded | 5.15 |
| **Resident** | **20.6** |
| in flight: largest tensor moved to GPU (`embed_tokens_per_layer`, fp32) | 8.75 |
| in flight: sender bucket (`torch.empty(10 240 MiB, device=cuda)`, `bucketed_weight_transfer.py:165`) | 10.0 |
| in flight: receiver clone of the bucket, same GPU (`:251`) | 10.0 |
| **Modelled peak** | **49.4** |
| Observed | 61.5 |

The ~12 GiB gap is consistent with PyTorch's caching allocator holding blocks
freed after the training phase (nvidia-smi and the OOM message report
*reserved* memory). Treat "+12 GiB allocator slack" as an empirical fudge
factor and carry it through below. 49.4 + 12 + 17 (vLLM) + ~1 (two CUDA
contexts) ≈ 79.6 — the card was exactly full, which is what was seen.

The fix at the time was `param_offload` + `optimizer_offload`, which removes
the 20.6 GiB resident term during sync. We do not use it here: the terms that
made it necessary are the two LoRA deletes.

## LoRA, phase by phase

### Resident at all times

| Term | GiB |
|---|---|
| fp32 shard of the frozen base | 5.15 |
| LoRA weights + grads + AdamW (0.1 + 0.1 + 0.2) | 0.4 |
| **Resident** | **5.55** |

Versus 20.6 GiB for full finetune: −15 GiB per GPU. This is the whole story.

### Phase A — weight sync, step 1 only (`base_sync_done = False`)

`rollout.load_format` is `dummy_dtensor`, so vLLM starts with random weights
and the first sync ships the full base (`fsdp_workers.rollout_mode` →
`collect_lora_params(base_sync_done=False)`, `utils/fsdp_utils.py:660`).
Two sub-peaks that do **not** overlap in time:

**A1 — gather.** `FSDP.summon_full_params` materialises the full fp32 model
on every rank while the shard stays allocated, then copies it to CPU.

| Term | GiB |
|---|---|
| resident | 5.55 |
| full fp32 model gathered | 20.6 |
| vLLM asleep at level 1 (weights on CPU, KV freed; context + cudagraph pool) | ~2-3 |
| **A1 peak** | **~29** (+12 slack → ~41) |

Host RAM: each of 4 ranks holds a 20.6 GiB CPU copy for the duration →
82 GiB. Box has 2 TB; irrelevant, noted because it is new with LoRA (the
full-FT path streamed straight from the sharded state dict).

**A2 — stream.** vLLM resumes its weights, then tensors move CPU → GPU one at
a time into the sender bucket; the receiver clones each bucket on the same
GPU and casts into vLLM's bf16 buffers.

| Term | GiB |
|---|---|
| resident | 5.55 |
| largest tensor on GPU (`embed_tokens_per_layer` fp32) | 8.75 |
| sender bucket | 10.0 |
| receiver clone | 10.0 |
| vLLM weights (bf16, TP=2) | 5.15 |
| CUDA contexts (actor + vLLM worker) | ~1 |
| **A2 peak** | **~40.5** (+12 slack → ~53) |

Headroom at the single worst moment of the run: **~27 GiB**. Full finetune at
the same moment modelled at 49.4 + 5.15 + 1 = 55.5 before slack.

### Phase B — weight sync, step 2 onward (`base_sync_done = True`)

Only adapter tensors cross the bus: `get_peft_model_state_dict` under the
same `summon_full_params`, 97 MiB fp32, delivered to vLLM as a
`TensorLoRARequest` (`add_lora`, no bucket streaming).

| Term | GiB |
|---|---|
| resident | 5.55 |
| full fp32 model gathered (summon; `layered_summon: False`) | 20.6 |
| vLLM asleep | ~2-3 |
| **B peak** | **~29** (+12 slack → ~41) |

The 20.6 GiB summon is the cost of not using `layered_summon`, which would
need `load_format: safetensors`. Not worth it at this headroom.

### Phase C — rollout (vLLM awake, actor idle)

| Term | GiB |
|---|---|
| resident actor shard (no offload) | 5.55 |
| vLLM budget (weights 5.15 + KV cache + activations, capped by `0.4`) | ≤ 31.9 |
| vLLM LoRA slot (`max_loras: 1`, rank 16: ~25 M × 2 B) | 0.05 |
| CUDA contexts | ~1 |
| **C peak** | **≤ 38.5** |

KV demand is tiny for this architecture: 1 KV head × head_dim 256 → 1 KiB per
token per caching layer in bf16; 20 of 35 layers reuse an earlier layer's KV
(`num_kv_shared_layers: 20`), leaving 15 caching layers, 12 of them
512-token sliding windows. Even the naive upper bound (15 layers × full
2 048 context) is 30 MiB/sequence → 256 sequences per replica = 7.5 GiB per
replica = **3.8 GiB per GPU** at TP=2, against ~25 GiB available inside the
0.4 budget. `gpu_memory_utilization` could drop to 0.25 without preemption;
it is left at 0.4 because Phase C is nowhere near the binding constraint.

### Phase D — training (vLLM asleep at level 1)

Three passes per step: old-log-prob (`rollout.log_prob_micro_batch_size_per_gpu: 4`,
with entropy), ref-log-prob (adapter disabled, `ref.log_prob_micro_batch_size_per_gpu: 2`,
no entropy), and the PPO update (`ppo_micro_batch_size_per_gpu: 4`, backward).
The update pass is the peak. Per GPU, micro-batch of 4 × 2 048 tokens:

| Term | GiB |
|---|---|
| resident | 5.55 |
| root FSDP unit gathered in bf16 for forward/backward (embeddings + tied head) | 5.5 |
| one decoder layer gathered in bf16 (80 M params) | 0.15 |
| checkpointed layer inputs, 35 × 4 × 2 048 × 1 536 × 2 B | 0.82 |
| per-layer input embeddings, 4 × 2 048 × 35 × 256 × 2 B | 0.14 |
| recompute of one layer (MLP intermediate 6 144, sdpa) | ~0.3 |
| logits path, conservative: bf16 full logits [4, 2 048, 262 144] 4.0 + fp32 log-softmax over the 1 024 response positions 4.0 + fp32 softmax for entropy 4.0 + bf16 logits grad 4.0 | ≤ 16 |
| vLLM asleep | ~2-3 |
| **D peak** | **≤ ~31** (+12 slack → ~43) |

This phase is per-GPU-count invariant except the shard (2.6 → 5.15 GiB):
the same micro-batch sizes ran for 575 steps on 8 GPUs (full finetune, so with
10.3 + 5.15 GiB more resident than here).

## Summary

| Phase | Modelled peak | +12 GiB slack | Headroom to 79.6 |
|---|---|---|---|
| A1 gather (step 1) | 29 | 41 | 39 |
| A2 stream (step 1) | 40.5 | 53 | **27** ← worst moment |
| B sync (step ≥ 2) | 29 | 41 | 39 |
| C rollout | ≤ 38.5 | — (vLLM pre-reserves) | 41 |
| D training | ≤ 31 | 43 | 37 |

## Observed (run #3, 2026-08-26, nvidia-smi every 20 s, GPUs 0-3)

nvidia-smi reports *reserved* memory (PyTorch caching allocator + vLLM's
pre-allocated pool), so these sit above the modelled *allocated* peaks by the
slack discussed above.

| Time | Per-GPU (MiB) | Phase | Modelled |
|---|---|---|---|
| 00:47:29 | 1 053 | CUDA contexts, Ray workers up | — |
| 00:48:10 → 00:49:30 | 13 649 (flat) | actor loaded + FSDP-sharded, LoRA applied, vLLM not started. 13.6 GiB = 5.15 GiB fp32 shard + ~8 GiB allocator cache left from the 36.6 GiB fp32 load transient (seen on run #1) | 5.55 resident |
| 00:50:10 | 24 264 – 27 431 | vLLM engines loading (dummy weights, TP=2) | — |
| 00:50:51 | 40 748 – 44 908 | vLLM fully up: 13.6 + 31.9 budget (weights + pre-allocated KV) | C ≤ 38.5 + slack |
| 00:51:31 | 46 836 – **50 220** | + cudagraph capture / profiling before `Starting verl training` (00:51:48). **Run peak.** | — |
| 00:52:11 | 34 259 – 35 137 | vLLM asleep (level 1) + `summon_full_params` gather: 13.6 + 20.6 = 34.2 ✓ | A1 29 (+5 slack) |
| 00:52:52 | 34 383 / 34 379 / **42 975** / 35 261 | A2 stream in progress on GPU 2 when `load_weights` raised (issue 14); stream never completed | A2 40.5 (+slack) |
| 00:53:32 → 05:10 | 42 969 – 42 995 (flat) | trainer hung on the failed RPC, nothing freed | — |

Take-aways: the A1 gather lands within 5 GiB of the model; the largest number
of the run (50.2 GiB) is vLLM's full 0.4 reservation plus the resident actor,
i.e. the rollout-phase footprint, with ~29 GiB to spare. Phases B, C (with
generation) and D were not reached in this run — extend the table when they
are.

## If it OOMs anyway

In order of preference:

1. **Step 1, in `receive_weights → clone` or `.to(device)`:** the streaming
   peak. `rollout.load_format: safetensors` makes vLLM load the base from the
   HF cache at start-up, so Phase A never runs (only Phase B, 97 MiB). This
   also enables `layered_summon: True`, which replaces the 20.6 GiB summon in
   Phase B with per-layer gathers. Cost: a different first-sync path from the
   one every gemma-4 run here has used; verify rollouts are not gibberish
   (`critic/score` not pinned, `actor/pg_loss ≠ 0`).
2. **Step 1 only, marginal:** `update_weights_bucket_megabytes: 9216` — the
   bucket must hold the 8.75 GiB tensor (verl asserts, no chunking), so this
   is the floor; saves 2 GiB on each side of the transfer.
3. **Training phase, in the logits path:** `ppo_micro_batch_size_per_gpu: 2`
   and `rollout.log_prob_micro_batch_size_per_gpu: 2` halve the 16 GiB term;
   `ppo_mini_batch_size` unchanged so the optimisation is identical, just
   twice the accumulation steps.
4. **Last:** `actor.fsdp_config.param_offload: True` — removes the 5.15 GiB
   shard from every phase at the cost of an H2D copy per phase switch. Kept
   off deliberately for speed.

## Reproducing the constants

```bash
# parameter counts (meta device, no weights loaded)
HF_HUB_OFFLINE=1 python - <<'EOF'
import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model
cfg = AutoConfig.from_pretrained("google/gemma-4-E2B-it")
with init_empty_weights():
    m = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
total = sum(p.numel() for p in m.parameters())
pm = get_peft_model(m, LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    exclude_modules="(.*vision_tower.*)|(.*audio_tower.*)", bias="none"))
print(total/1e9, "B total;", sum(p.numel() for p in pm.parameters() if p.requires_grad)/1e6, "M trainable")
t = cfg.text_config
print("kv heads", t.num_key_value_heads, "head_dim", t.head_dim, "kv_shared_layers", t.num_kv_shared_layers,
      "layer_types", {x: t.layer_types.count(x) for x in set(t.layer_types)}, "window", t.sliding_window)
EOF
```

Source anchors (verl 0.7.1 site-packages): `workers/fsdp_workers.py`
`rollout_mode` (~L750) for the sync sequence; `utils/fsdp_utils.py`
`collect_lora_params` (L633) for the two sync paths;
`workers/rollout/vllm_rollout/bucketed_weight_transfer.py` L165/L251 for the
sender/receiver buffers; `workers/actor/dp_actor.py` for the `.eval()`
log-prob passes and `.train()` update.
