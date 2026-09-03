# Where the 74 GiB went, what else was wrong, and the retuned recipe (2026-09-02)

The 08:08 UTC launch of `train_verl.yaml` (gemma-4-E2B-it, LoRA r=64, 4× H100)
died at step 1 with

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 32.00 GiB. GPU 0 has a
total capacity of 79.18 GiB of which 842.75 MiB is free. Process 834991 has 520.00 MiB
memory in use. Process 852079 has 75.72 GiB memory in use. Process 859101 has 2.06 GiB
memory in use. Of the allocated memory 74.09 GiB is allocated by PyTorch, and 89.93 MiB
is reserved by PyTorch but unallocated.
```

inside `verl/utils/torch_functional.py:entropy_from_logits` → `softmax`, called from
the actor's old-log-prob pass (`dp_actor.py:373`, the `use_remove_padding: False`
branch). This file records what held the memory (measured, not modelled), which of
it was waste, two unrelated correctness bugs found while fixing it, the recipe that
replaced it, and the tuning log for the overnight run.

Companion files: `GPU_MEMORY.md` (the earlier arithmetic budget for the LoRA
recipe), `RUN_REPORT.md` (the 08-26 LoRA run), `gemma4_kv_share_patch.py` (§0),
`checks/actor_mem_replay.py` + `actor_mem_replay_mb32.log` (§2),
`checks/fullft_phase_replay.py` (§4.2), `checks/kv_share_patch_check.py`,
`checks/rmpad_fa2_numerics_check.py` (§7).

## 0. Read this first: every gemma-4 verl forward so far was garbage

Found while validating flash-attn numerics (§7), unrelated to memory, and it
matters more than the OOM.

gemma-4-E2B has `num_kv_shared_layers: 20` of 35: those layers must reuse the K/V
of the last non-shared layer of their type. transformers 5.5.1 implements the
reuse **only through the cache object** (`modeling_gemma4.py:1198`,
`if self.is_kv_shared_layer and past_key_values is not None`); without a cache the
20 layers compute K/V from their own `k_proj`/`v_proj`, which were never trained
for that role. verl calls every actor/ref forward with `use_cache=False`
(`dp_actor._forward_micro_batch`), and `GradientCheckpointingLayer.__call__`
(`transformers/modeling_layers.py:60-79`) strips `past_key_values` under gradient
checkpointing regardless. Measured on a real chat prompt + response
(`checks/kv_share_patch_check.py`):

| forward | mean log-prob of the response | greedy next tokens |
|---|---|---|
| `use_cache=True` (what vLLM / `generate()` effectively do) | −3.23 | fluent |
| `use_cache=False` (what verl does) | **−13.27** | `'it ** ** ** ** ** **h ** **'` |

So the old-log-prob, ref and PPO-update passes of every gemma-4 verl run in this
repo computed nonsense while vLLM (its own gemma-4 implementation shares K/V
natively) produced fine rollouts. It was visible in hindsight:
`actor/entropy` 11.7 in the letter-counting full-FT run (uniform over 262 144
tokens is 12.5), `training/rollout_probs_diff_mean` 0.76 and
`rollout_actor_probs_pearson_corr` 0.09 in the 08-26 LoRA run (actor vs vLLM
log-probs on the same tokens, should be ~1e-3 / ~1.0). Those were attributed to
the LoRA no-op (issue 15), which is real but secondary.

**Fix (no site-packages edit):** `gemma4_kv_share_patch.py`, loaded by verl's
`actor_rollout_ref.model.external_lib` hook in every FSDP worker. It substitutes a
per-forward holder for the missing cache: `Gemma4TextModel.forward` resets it,
`Gemma4TextAttention.forward` receives it whenever it would have received `None`;
the holder only carries `shared_layers` and a no-op `update()`, so there is no
cache bookkeeping and no interaction with mask construction. Validated
(`checks/kv_share_patch_check.py`): cache-less eval / train / train + gradient
checkpointing forwards are **bit-identical** to the cached forward, and gradient
norms with and without checkpointing match to 0.0. Cost: the K/V of the two
producing layers stay alive between forwards (tens of MiB per micro-batch).

Consequence for old results: treat every gemma-4 GRPO run before 2026-09-02 as
untrained (the rollout side was fine, the gradient side was noise).

## 1. Decoding the error message

| Quantity | Value | Meaning |
|---|---|---|
| total capacity | 79.18 GiB | one H100 80 GB (81 559 MiB) |
| process 852079 | 75.72 GiB | **the FSDP actor worker itself** (host-namespace PID; we run in a container so the PIDs do not match Ray's) |
| ├ allocated by PyTorch | 74.09 GiB | live tensors, itemised in §2 |
| ├ reserved but unallocated | 0.09 GiB | the allocator had already released its cache and retried |
| └ remainder | ~1.5 GiB | CUDA context, NCCL buffers, cuBLAS workspace (outside the PyTorch allocator) |
| process 859101 | 2.06 GiB | the vLLM engine core, asleep at level 1 (weights on CPU, KV cache freed; CUDA context + cudagraph pool remain) |
| process 834991 | 0.52 GiB | the Ray driver / vLLM HTTP server CUDA context |
| free | 0.82 GiB | 79.18 − 75.72 − 2.06 − 0.52 |
| requested | 32.00 GiB | 32 × 1024 × 262 144 × 4 bytes |

The request is one fp32 tensor of shape (`log_prob_micro_batch_size_per_gpu` = 32,
`max_completion_length` = 1024, gemma-4 `vocab_size` = 262 144). All four workers
failed on the same allocation with the same numbers; each reports its own GPU as
"GPU 0" because Ray sets `CUDA_VISIBLE_DEVICES` per worker. vLLM is not the
problem here: it held 2 GiB.

## 2. What the 74 GiB was (measured)

`checks/actor_mem_replay.py` rebuilds the actor exactly as
`fsdp_workers._build_model_optimizer` does (fp32 weights, PEFT LoRA r=64, FSDP1
FULL_SHARD with verl's LoRA wrap policy → 554 units, bf16 `MixedPrecision`,
`use_orig_params=False`), runs `init_model`'s offload, `rollout_mode`'s
load → `collect_lora_params(base_sync_done=False)` → offload, then
`compute_log_prob`'s load → `_forward_micro_batch` at micro-batch 32 on
1024 + 1024 tokens, with `torch.cuda.memory._record_memory_history` on. It
reproduces the OOM to the second decimal ("74.10 GiB is allocated by PyTorch").

Live tensors at the moment of the failing allocation, by allocating call site
(`actor_mem_replay_mb32.log`):

| GiB | Tensor | Allocated by | Why it exists |
|---|---|---|---|
| **32.00** | bf16 logits for the **whole padded sequence**, shape (32, 2048, 262144) | `modeling_gemma4.py:2451` (`logits * final_logit_softcapping`) | HF returns logits for every position; verl slices the response half afterwards, so the prompt half (16 GiB) is computed and never read |
| **32.00** | fp32 copy of the response-slice logits, (32, 1024, 262144) | `torch.autocast` inside `F.softmax` | `_forward_micro_batch` runs under `autocast(bf16)`; `softmax` is on autocast's fp32 list, so the input is upcast first. This copy succeeded; the *output* of the softmax (another 32 GiB) is what failed |
| 5.19 | FSDP root unit, unsharded bf16 flat param (embed_tokens, `embed_tokens_per_layer`, vision/audio towers, norms, tied lm_head: 2.79 B params) | `_flat_param.py:_alloc_padded_unsharded_flat_param` | FSDP1 never reshards the root unit after a forward ("needed for backward"), even under `no_grad` |
| 4.88 | fp32 local shards of all 554 units (¼ of 5.2 B params) | `_flat_param.py:flat_param_to` (the reload from CPU offload) | the model |
| 0.03 | per-layer-input projection, batch tensors, the (32, 1024) log-probs | `modeling_gemma4.py:project_per_layer_inputs` etc. | |
| **74.10** | | | |

What the step would have needed, per GPU, had the softmax fit: the softmax output
(32 GiB), then `pd * logits` promotes `logits` to fp32 (another 32 GiB) and
materialises the product (32 GiB). Measured at micro-batch 2 and scaled: **4 GiB
per sequence** at 2048 padded tokens, so micro-batch 32 needed ~138 GiB on an
80 GB card. The forward alone already peaked at 74.3 GiB (gemma-4's logit
softcapping holds two full-size bf16 logits tensors at once) and only fit because
the softmax had not happened yet.

Other measured peaks from the same replay (LoRA path): FSDP init 25.6 GiB;
`collect_lora_params(base_sync_done=False)` (the step-1 base sync gather, and
every step's gather under the sleep-level mismatch in `RUN_REPORT.md` §5.5)
43.9 GiB in the actor process.

### Why the referenced recipe fits and this one did not

The blog recipe (Qwen2.5-3B-Instruct, GSM8K, 4× A100 80 GB,
`log_prob_micro_batch_size_per_gpu=40`) differs in three ways that compound:

| | blog | this run | factor |
|---|---|---|---|
| `use_remove_padding` | True (flash-attn installed) | False (sdpa) | blog computes logits for **real tokens only** and uses the flash-attn cross-entropy + compiled/chunked entropy; this path pads every sequence to `max_prompt_length + max_response_length` and runs the plain, unfused entropy |
| vocab | 151 936 | 262 144 | 1.73× per position |
| padded length | 512 + 1024 | 1024 + 1024 | 1.33× padded; GSM8K sequences are a few hundred real tokens, and with remove-padding that is all the blog pays for |

Per sequence in the micro-batch, this run paid ~4 GiB versus well under 1 GiB in
the blog. Vocabulary explains 1.7× of it; padding and the non-remove-padding code
path explain the rest.

## 3. Inefficiencies in the 74 GiB

Ranked by bytes.

1. **64 of 74 GiB are full-vocabulary tensors whose useful content is ~256 KiB.**
   The pass needs one log-prob per response token (32 × 1024 floats) and one
   entropy per token. Instead the 262k-wide distribution is materialised four
   times (bf16 logits, fp32 cast, fp32 softmax, fp32 `pd*logits`). In the
   remove-padding path verl uses flash-attn's fused cross-entropy and a
   chunked/compiled entropy; the padded path at `dp_actor.py:373` calls the plain
   `entropy_from_logits`, so `entropy_from_logits_with_chunking`,
   `entropy_checkpointing` and `use_torch_compile` all have **no effect** there.
2. **Half of the logits are for prompt positions that are discarded** (16 GiB of
   the 32): HF computes `lm_head` over all 2048 positions, verl slices
   `[-1025:-1]`. Beyond that, the padded length is 2048 while real length averages
   ~500 (prompt mean 147, response mean 345, `RUN_REPORT.md`): ~24 % token
   utilisation in every actor/ref forward.
3. **The entropy is computed for a logging metric.** `entropy_coeff` is 0, so
   entropy never enters the loss; verl 0.7.1 hard-codes
   `calculate_entropy = not is_lora` in `fsdp_workers.compute_log_prob` for the
   old-log-prob pass (the `is_lora` flag there marks the *reference* pass, not
   "model has LoRA"). That metric costs 3 GiB per sequence here.
4. **Autocast upcasts** `softmax`, `logsumexp`, `log_softmax` to fp32: double the
   bytes plus a cast copy. Numerically right, but only cheap when chunked.
5. **FSDP root unit stays unsharded**: 5.19 GiB of bf16 embeddings/towers resident
   after every no-grad forward, for FSDP1's assumption that the root is about to
   run backward. With a separate ref worker (full FT) there are two of them:
   10.4 GiB (`checks/fullft_phase_replay.py`).
6. **Gemma-4 final-logit softcapping** makes two extra full-size bf16 copies
   transiently (`logits / cap`, `tanh`, `* cap`), so the forward's own peak is
   2× the logits.
7. **LoRA sync (now moot)**: `collect_lora_params(base_sync_done=False)` summons
   the full fp32 model and round-trips it through CPU, peaking at 43.9 GiB in the
   actor process while vLLM's weights are awake.

Items 1, 2 and 4 are removed by the remove-padding path (§7), item 3 is shrunk by
it (chunked entropy), item 5 is FSDP1 behaviour, item 6 is HF code.

## 4. The retuned recipe

### 4.1 Full fine-tune instead of LoRA

`GEMMA4_VERL_GRPO_FIXES.md` issue 15 (open) and
`checks/vllm_lora_adapter_check.py`: on this install (vllm 0.19.1 + the
`SupportsLoRA` backport) a gemma-4 LoRA adapter loads into vLLM but never changes
its output, bit-identical to the base even at 100× alpha. In a hybrid GRPO run
that means every rollout is a **base-model sample** regardless of training, so a
LoRA r=64 run tonight would have produced a non-result even with §0 fixed. Full
fine-tuning streams the actual weights to vLLM each step, so rollouts are
on-policy; `rollout.calculate_log_probs: True` keeps the
`training/rollout_probs_diff_*` guard that must now sit at bf16 noise.

Consequences: LR 1e-6 (the letter-counting full-FT value), a separate ref worker
(FSDP CPUOffload), ~60 GB per verl checkpoint (fp32 params + AdamW, sharded) so
`trainer.max_actor_ckpt_to_keep: 2`.

### 4.2 Measured full-FT residency, per GPU (`checks/fullft_phase_replay.py`)

Actor (fp32, FSDP1, no offload) + ref (FSDP1 CPUOffload) in one process as verl
colocates them, 4 GPUs, 512 + 1024 padded tokens:

| Point | live | peak in phase | what is live |
|---|---|---|---|
| after both FSDP inits | 4.77 | 25.5 | actor fp32 shards |
| after `state_dict()` + `full_tensor()` stream (weight sync) | 5.15 | 28.7 | shards + DTensor placement metadata; largest streamed tensor 8.75 GiB |
| after old-log-prob, padded mb 8, entropy on | 10.37 | 40.4 | + actor root unit unsharded 5.19 |
| after ref, padded mb 16, no entropy | 15.57 | 39.6 | + ref root unit unsharded 5.19 |

The AdamW states (9.5 GiB) appear after the first optimizer step, fp32 grads
(4.77) during the update.

### 4.3 What attempt 1 (padded path) showed

Attempt 1 ran the padded recipe (log-prob mb 8, ref mb 16, PPO mb 8, no actor
offload, vLLM 0.55) without §0. It got through generation, weight sync and the
old-log-prob pass and OOMed in the **ref** forward at the logit-softcapping copy:
"Tried to allocate 12.00 GiB ... 58.07 GiB is allocated by PyTorch, 6.44 GiB
reserved but unallocated". 12 GiB is one bf16 (16, 1536, 262144) logits tensor;
58 GiB live at that point is ~35 GiB more than the replay above predicts for the
same phase, and the padded budget in the first version of this file had no room
for that. The per-phase `log_gpu_memory_usage` lines verl emits at DEBUG level
stopped after the vLLM build (the vLLM import reconfigures logging), so
`gemma4_kv_share_patch.py` gained an opt-in memory trace
(`GEMMA4_PATCH_MEMLOG=1`) that prints live/peak memory at every text-model
forward; attempt 2 uses it (§5).

nvidia-smi during attempt 1 (5 s samples, GPUs 0-3): p95 59.8 GiB, max
70.4–76.4 GiB, rollout-phase utilisation only ~10 % (vLLM generating 128
sequences per GPU is latency-bound, not memory-bound).

### 4.4 Changes, one line each

| Setting | was | now | why |
|---|---|---|---|
| `model.external_lib` | — | `gemma4_kv_share_patch` | §0 |
| `model.lora_rank` | 64 | 0 (full FT) | §4.1 |
| `learning_rate` | 1e-5 | 1e-6 | full-FT value |
| `actor.optim.{lr_warmup_steps, lr_scheduler_type, min_lr_ratio}` | (top-level oumi keys, silently ignored by the VERL_GRPO trainer) | 20 / cosine / 0.1 | the config dump showed `lr_scheduler_type: constant, lr_warmup_steps: -1` |
| `data.train.sample_count` | 20000 | removed | oumi oversamples above the split size (17 926 + 2 074 duplicates) |
| `data.max_prompt_length` | 1024 | 512 | p99 is 383, max 1000; 37 of 20 000 train rows and 5 of 1 000 val rows filtered |
| `model.use_remove_padding` | False | True (sdpa kernel) | §7; logits/entropy cost scales with real tokens (~30 % of padded) |
| `actor.use_dynamic_bsz` / `ppo_max_token_len_per_gpu` | False / (mb 32) | True / 16384 | token-budgeted micro-batches |
| `actor.entropy_from_logits_with_chunking` | False | True | honoured by the rmpad branch |
| `rollout.log_prob_use_dynamic_bsz` / `log_prob_max_token_len_per_gpu` | False / (mb 32) | True / 24576 | |
| `ref.log_prob_use_dynamic_bsz` / `log_prob_max_token_len_per_gpu` | False / (mb 32) | True / 24576 | |
| `actor.fsdp_config.{param,optimizer}_offload` | param True | **both True** | attempt 2: without them the actor process held 60.7 GiB during the step-1 weight sync and vLLM's receiver OOMed (issue #11 again); from step 2 AdamW adds 9.5 GiB there |
| `rollout.checkpoint_engine.update_weights_bucket_megabytes` | 10240 | 9216 | must hold the 8960 MiB embed tensor; saves 1 GiB on sender and receiver |
| `vllm_gpu_memory_utilization` | 0.4 | 0.55 | rollout-phase headroom |
| `save_steps` / `max_actor_ckpt_to_keep` | 100 / ∞ | 50 / 2 | a usable checkpoint every ~1.5 h, ≤ 120 GB on disk |
| `num_train_epochs` / `max_steps` | 3 / — | 1 / set from measured step time | finish overnight; raise `max_steps` and re-run to continue (auto-resume) |
| `rollout.engine_kwargs.vllm.lora_target_modules` | 4 fused modules | removed | LoRA-only |

## 5. Tuning log

GPU samples: `nvidia-smi ... -l 5` on GPUs 0-3 (scratch CSVs, summarised here);
actor-process peaks: verl's `perf/max_memory_allocated_gb` per step and the
`[gemma4-memlog]` lines.

| attempt | recipe | outcome |
|---|---|---|
| 1 (08:49) | padded sdpa, log-prob mb 8 / ref mb 16 / PPO mb 8, no §0 patch | OOM in ref forward at step 1, 58 GiB live (§4.3) |
| 2 (09:16) | §0 patch, remove-padding sdpa, dynamic bsz, token caps 12288 (diagnostic), no actor offload | OOM in the **vLLM receiver** during the step-1 weight sync (`bucketed_weight_transfer.py:251 tensor.clone()`, 72 MiB): actor process 60.73 GiB, vLLM worker 17.80 GiB (weights 9.5 + received bucket), driver 0.5 → card full. verl streams fp32 (the bf16 cast in the sender is commented out), disables expandable segments for the IPC buffer, and FSDP1 leaves the root unit unsharded after `state_dict()`, so the actor sits at ~28 GiB allocated / ~60 GiB reserved while sending |
| 3 (09:26) | as 2 + actor `param_offload` + `optimizer_offload` | **works.** Steps 2-4: 72 / 66 / 72 s (gen 20-24, old_log_prob 5, ref 5, update_actor 30-32, update_weights 6.5), peak allocated 50.1 GiB every step, reserved 74.6; entropy 0.71-0.75, ppl 1.47-1.54, kl 0.0098, score mean 0.48-0.59. Step 1: 103 s/step (gen 35.5, old_log_prob 12.8, ref 15.3, update_actor 33.0, update_weights 6.2); `perf/max_memory_allocated_gb` 48.0, reserved 73.5; entropy 0.73, ppl 1.53, kl 0.0098, score mean 0.48 (min 0, max 1), response mean 320 tokens; `rollout_probs_diff_mean` 0.179, pearson 0.64 (see §5.1) |

| final, 1st launch (09:39) | yaml: log-prob/ref caps 24576, update cap 12288, 280 steps, `val_before_train` on, KV-share patch only | ran 44 steps at 72-82 s/step, peak 52 GiB, baseline val 0.5308; **stopped at 10:46** because 3 of 4 ranks computed wrong log-probs (§8); output moved to `output/medqa_gemma4-e2b-it_fullft_run1_broken_ranks` |
| final, 2nd launch (10:47) | same yaml + first version of `verl_rank_buffer_sync` (per-rank `named_buffers()` loop) | **deadlocked at worker init** (NCCL collective timeout): the ranks do not even have identical buffer *lists* in the live worker, so per-rank loops issue different numbers of broadcasts; killed 11:02 |
| final, 3rd launch (11:03) | same yaml + `verl_rank_buffer_sync` v2 | healthy metrics from step 1 (§6), 76-98 s/step, peak 50 GiB; **crashed at step 20 (11:47)**: NCCL watchdog timeout, ranks 0-1 in an FSDP all-gather (an extra forward) while ranks 2-3 were in an all-reduce. `dp_actor.compute_log_prob` calls `prepare_dynamic_batch(data, max_token_len)` without `dp_group`, so `use_dynamic_bsz` cannot equalise micro-batch counts across DP ranks; once responses lengthened (mean 368 at step 19) the counts diverged. No checkpoint yet (first at step 50) |
| final, 4th launch (18:10) | dynamic batching OFF: fixed `log_prob_micro_batch_size_per_gpu` 16 (actor+ref), `ppo_micro_batch_size_per_gpu` 8, remove-padding kept. Worst case per micro-batch = 24576 / 12288 tokens = the proven caps; counts identical on all ranks (8 / 4) | healthy from step 1 (§6); steps 2+ ~55 s (fixed micro-batches pack ~7.5k real tokens, less allocator churn than the 24k dynamic packs), peak 39-42 GiB; _(running; log `logs/medqa_gemma4-e2b-it_fullft.log`)_ |

Why the final caps: raising the update cap 12288 → 16384 would move the step peak
from 50 to ~59 GiB for a ~4 % step-time gain (update_actor is ~30 s of ~70; the
rest is generation, which the caps do not touch), so the proven value stays.
The log-prob and ref passes have no entropy-free headroom problem: at 24576 tokens
their dynamic memory is ~31 GiB (bf16 logits 12 + softcapping copy 12 + chunked
entropy 4 + activations) on a 10-17 GiB base, below the update's peak.

### 5.1 Attempt 3, step 1, per-forward memory trace (rank 0, `GEMMA4_PATCH_MEMLOG=1`)

| phase (token cap 12288) | forwards | live entering each forward | peak in phase |
|---|---|---|---|
| weight sync (actor offloaded) | — | — | 28.7 GiB allocated (`state_dict` root unit 5.2 + `full_tensor` 8.75 + bucket 9 + shard 4.8) |
| old log-prob (entropy on) | 5 | 10.5 GiB (fp32 shard 4.77 + actor root unit 5.19) | ≤ 28.7 (did not exceed the sync peak) |
| ref (CPUOffload worker) | 5 | 5.8 → 11.5 → 17.3 → 5.8 → 11.5 GiB (root-unit-sized buffers accumulate, then reshard) | ≤ 28.7 |
| PPO update (grad on) | 6 | 10.5 → 19.1 (AdamW loaded, +9.5) → 23.4 (grads, +4.3) | **48.0 GiB** allocated, reserved climbs to 69-71 |

So the dynamic part of the update forward+backward is ~25-29 GiB at 12288 tokens
(bf16 logits 6 GiB, softcapping copy, flash-attn CE in-place grad, logits grad,
checkpointed activations); the log-prob passes are ≤ 18 GiB at the same cap. The
gap between allocated (48) and reserved (71) is caching-allocator fragmentation
from variable-size micro-batches; it is soft (released on an allocation retry),
the hard number is allocated + vLLM asleep (2.6) + contexts (~2).

nvidia-smi over attempt 3's steady-state steps (5 s samples, GPUs 0-3): memory
p95 74-77 GiB, max 78.5 GiB (the caching allocator holds freed blocks until an
allocation fails), mean 40 GiB. SM utilisation mean 38-42 % with a median of 0:
about half of the wall clock has the GPUs idle (vLLM's decode of a 5 B model at
128 concurrent sequences is latency-bound, plus phase switches, the CPU-offload
copies and the wait for the last judge verdicts). Memory is no longer the
bottleneck for this recipe; per-step time is.


## 6. Results

Final run `medqa_gemma4-e2b-it_fullft` (relaunched 18:10 UTC with both fixes and fixed micro-batches; log
`logs/medqa_gemma4-e2b-it_fullft.log`; checkpoints under
`tmp/rar_medicine/variant_b/output/medqa_gemma4-e2b-it_fullft/verl_output/`).

| | value |
|---|---|
| baseline (`val_before_train`, 995 val prompts, greedy, judge = gpt-4.1-mini meta-rubric) | **0.5308** |
| baseline, 3rd launch (995 val prompts, greedy) | **0.5325** (judge noise between launches: 0.5308 / 0.5325 on the same base model) |
| step 1 (1st launch, ranks 1-3 broken) | 93 s; peak 47.8 GiB; score mean 0.466; `rollout_probs_diff_mean` 0.066, pearson 0.84; `training_log_ppl` 0.79 vs `rollout_log_ppl` 0.46 |
| step 1 (3rd launch, both fixes) | peak 48.4 GiB; score mean 0.460; entropy 0.47; **`rollout_probs_diff_mean` 0.0075, pearson 0.9987, `training_log_ppl` 0.468 vs `rollout_log_ppl` 0.467, k3 KL 0.0012** |
| step 2 (3rd launch) | peak 50.2 GiB; score mean 0.582; 0.0072 / 0.9987 / 0.453 vs 0.452 |
| baseline, 4th launch | 0.5281 (three baselines of the same base model: 0.5308 / 0.5325 / 0.5281 → judge noise ≈ ±0.003) |
| step 1 (4th launch, fixed micro-batches) | 112 s (first-step compile); peak 38.7 GiB; score mean 0.471; 0.0076 / 0.9987 / 0.455 vs 0.454 |
| step 2 (4th launch) | **55 s**; peak 41.9 GiB; score mean 0.563; 0.0072 / 0.9987 / 0.437 vs 0.436 |

At worker init the sync reported 3 buffers with different content on two of the
four ranks (the RoPE tables) and 0 on the others, i.e. the fault was present in
this launch too and was repaired before the first forward.

_(training in progress; final table filled in when the run ends)_

## 7. flash-attn / flex-attention / remove-padding

* No prebuilt flash-attn 2 wheel exists for torch 2.10 (`v2.8.3.post1` ships
  cu12 wheels for torch 2.4–2.8 and a cu13 wheel for 2.9). `nvcc` 12.8 is at
  `/usr/local/cuda`, so `pip install flash-attn --no-build-isolation --no-deps`
  built 2.8.3.post1 from source in 10 min (MAX_JOBS=48). Nothing else in the env
  changed. verl's `logprobs_from_logits` now takes flash-attn's Triton
  cross-entropy on every path; checked bit-identical to `log_softmax` on real
  gemma-4 logits (`scratch logprob_kernel_check.py`).
* **FlashAttention-2 kernels cannot run gemma-4**: "FlashAttention forward only
  supports head dimension at most 256"; gemma-4-E2B's global-attention layers use
  `global_head_dim: 512`. So `attn_implementation: flash_attention_2` is out.
* **Remove-padding works with sdpa.** verl's rmpad branch only needs
  `flash_attn.bert_padding` (pure Python) and a model that handles packed
  sequences from `position_ids`; transformers' `masking_utils` builds the
  block-diagonal causal / sliding-window masks from position-id resets for sdpa
  (and flex). verl's `apply_monkey_patch` touches attention only for the Qwen-VL /
  GLM4V / Kimi families, so gemma-4 runs HF's own path.
  `checks/rmpad_fa2_numerics_check.py` (with §0 applied, 8 real prompts, 4 884
  real tokens of 12 288 padded):

  | comparison on response-token log-probs | mean abs diff | p99 | max |
  |---|---|---|---|
  | noise floor: eager padded vs sdpa padded | 0.0148 | 0.295 | 0.965 |
  | **packed sdpa vs padded sdpa** | 0.0139 | 0.249 | 0.685 |
  | packed vs padded, entropy | 0.0038 | 0.067 | 0.205 |

  Packed differs from padded by less than two attention kernels differ from each
  other, so it is bf16 noise. (Without §0 both paths are garbage and differ by
  0.27 mean, which is how the KV-sharing bug was found.)
* flex-attention needs no install but brings nothing here: the memory is in the
  padded logits, not in attention, and sdpa already accepts the packed masks.

## 7b. Does a newer vLLM make gemma-4 LoRA work? Tested: no (0.21.0)

vLLM PR #39291 (merged 2026-04-17, first in 0.20.0) adds native LoRA wiring to
`Gemma4ForConditionalGeneration` (`SupportsLoRA`, `packed_modules_mapping`
qkv/gate_up, language-model targets only). The installed 0.19.1 (2026-04-18)
predates it, and the env's hand-backported mixin only made the engine accept the
adapter. Every release from 0.20.0 to 0.21.0 (last inside oumi's `vllm<0.22`) pins
**torch 2.11.0** (installed: 2.10.0), so this is a stack move, not a pip bump.

Tested in a cloned env (`conda create --clone oumi -n oumi-vllm021`, then
`pip install vllm==0.21.0`, which pulled torch 2.11.0+cu130 and silently
downgraded transformers to 4.57.6 (no gemma-4) — restored to 5.5.1, which vLLM
0.21 allows). Needed `VLLM_USE_DEEP_GEMM=0` (engine start probes an FP8 backend
gemma-4 does not use) and cuDNN SDPA off for the HF reference (torch 2.11's cuDNN
kernel rejects the 512-wide heads). `checks/vllm_lora_adapter_check.py` with the
08-26 r=16 adapter:

| | result |
|---|---|
| HF+PEFT vs HF base | adapter changes the greedy output within 96 tokens on 7/8 prompts (median common prefix 38.5 tokens) |
| vLLM+LoRA vs vLLM base | **identical on 8/8 prompts** |
| vLLM+LoRA tracks HF+LoRA better than vLLM-base | 0/7 |

So the upstream support does not make the adapter act on gemma-4 either, and the
LoRA recipe stays blocked on vLLM. **Mechanism** (probe via `collective_rpc` on the
0.21 worker, `scratch lora_attach_probe.py`): vLLM's `gemma4.py` builds the 35
decoder layers once as `layers` and then re-registers the same layer objects
under two more module paths, `self_decoder.decoder_layers[0:15]` and
`cross_decoder.decoder_layers[0:20]` (the KV-shared block), which is what the
forward actually runs. The LoRA manager therefore sees **422** LoRA-capable module
names for ~212 real linears and wraps them by name; the adapter's 140 weights
bind (by the HF→vLLM name mapping) to `language_model.model.layers.N.{qkv_proj,
o_proj,gate_up_proj,down_proj}`, while the wrappers that execute are the ones
registered under `…_decoder.decoder_layers.N.*`. After a LoRA-tagged generation
every executed module's `lora_a_stacked` is all zeros. Same architecture in
0.19.1 → same no-op. This is a vLLM bug to report upstream (module aliasing in
`Gemma4Model` vs the LoRA manager's name-based wrapping); a config-only
workaround does not exist.
The full fine-tune is the working path. The conda env `oumi-vllm021` (a clone of
`oumi` with vLLM 0.21.0 / torch 2.11) is kept for further probing; the live env
was not touched.

## 8. Open item (resolved): the actor disagreed with vLLM by more than bf16 noise

verl logs, per step, the agreement between the actor's old-log-prob pass and the
log-probs vLLM reported for the same sampled tokens. After the §0 fix:

| run | log-prob cap | `rollout_probs_diff_mean` (mean abs prob diff) | pearson | actor seq-avg −logp (`training_log_ppl`) | vLLM seq-avg −logp (`rollout_log_ppl`) |
|---|---|---|---|---|---|
| before §0 (08-26 LoRA) | padded mb 4 | 0.76 | 0.09 | — | — |
| attempt 3 | 12288 | 0.179–0.186 (stable over 4 steps) | 0.62–0.64 | 1.53 | 0.44–0.46 |
| final run | 24576 | 0.064–0.068 (stable over 12+ steps) | 0.84 | 0.75–0.79 | 0.42–0.46 |
| healthy target | — | ~0.007 | ~0.999 | 0.43 | 0.43 |

vLLM is right (its 0.43–0.46 matches HF offline). The in-run actor scores its own
rollouts ~0.35 nats/token (final) to ~1.1 nats/token (attempt 3) too low, and the
error depends on the actor's micro-batch token cap, which points at the actor side.
Everything I could isolate reproduces **correctly** (all with the §0 patch,
sdpa, bf16 autocast, real vLLM samples of the val prompts):

| isolated component | result |
|---|---|
| HF plain forward vs vLLM (raw and `processed_logprobs` modes), 5 084 tokens | mean abs prob diff 0.0075, pearson 0.997 |
| dummy-init vLLM + `load_weights` from the checkpoint (verl's step-1 sync path): all 460 vLLM params loaded, vs normal engine | 0.006, pearson 0.998 (`checks/vllm_dummy_sync_fidelity_check.py`) |
| verl-style packed forward, 48 sequences in 12288- or 24576-token packs, vs per-sequence HF | 0.007, pearson 0.999, 0 bad sequences |
| verl's real `DataParallelPPOActor.compute_log_prob` on a 4-way FSDP1 actor built like the worker (mixed precision, dynamic bsz, rmpad, compiled chunked entropy), 48 and 128 sequences (16 prompts × 8), with the offload → state_dict/full_tensor → offload → reload sequence emulated | 0.006–0.007, pearson 0.999, seq-avg −logp 0.41–0.43 = vLLM |

All of those checks looked only at **rank 0**. A one-step diagnostic run with
`verl_logprob_dump_hook.py` (dumps each rank's real batch, the actor's returned
log-probs and vLLM's) settled it:

| rank | in-run actor vs vLLM, mean abs prob diff | actor seq-avg −logp | vLLM seq-avg −logp | bad sequences |
|---|---|---|---|---|
| 0 | 0.0073 | 0.437 | 0.437 | 0 / 128 |
| 1 | 0.349 | 2.455 | 0.461 | 128 / 128 |
| 2 | 0.348 | 2.466 | 0.460 | 128 / 128 |
| 3 | 0.352 | 2.493 | 0.466 | 128 / 128 |

Rank 0 is exact; ranks 1-3 are wrong on every sequence; the run's own logged
`training_log_ppl` 2.06 is the 4-rank average. FSDP all-gathers *parameters*
identically on every rank, so weights cannot make ranks differ; only per-rank,
non-sharded state can. Comparing all 2 032 parameters and buffers across ranks
after FSDP construction (`scratch verl_actor_logprob_check.py`, `STATE_CMP=1`)
found exactly three mismatches, all in `model.language_model.rotary_emb`:
`sliding_attention_inv_freq` (128 values) and `full_attention_inv_freq` (256)
hold wrong values on ranks ≠ 0 and `sliding_attention_original_inv_freq` is
zeros. **Ranks 1-3 ran attention with wrong RoPE tables**, which is exactly
"moderately wrong on every sequence", and the wrongness depends on what FSDP's
coalesced `sync_module_states` broadcast happened to write, hence the different
severities per run (1.53 / 0.79 / 2.06) and the apparent dependence on the
micro-batch cap (a coincidence of runs). The exact fault inside FSDP1's coalesced
`sync_module_states` broadcast was not pinned down; what is established is that
(a) transformers keeps the RoPE tables as aliased non-persistent buffers
(`inv_freq` / `original_inv_freq`), (b) after construction the live worker's ranks
do not hold identical buffer lists (a per-rank `named_buffers()` broadcast loop
deadlocks with an NCCL count mismatch, see §5), and (c) the standalone 4-GPU
rebuild reproduced the three RoPE mismatches in one launch and none in the next,
so the corruption is launch-dependent.

**Fix:** `verl_rank_buffer_sync.py` re-broadcasts every buffer from rank 0 right
after verl constructs `DataParallelPPOActor` for the actor and the ref: rank 0
publishes its buffer names/shapes/dtypes, and every rank then runs the identical
sequence of broadcasts, writing into its own buffer of that name (aliases
included) or into a scratch tensor when it has no such buffer. Bundled with the KV-share patch as
`gemma4_verl_fixes.py` (`model.external_lib`). Validation in the 4-GPU harness
(FSDP actor built like the worker, verl's real `compute_log_prob`, 48 real
samples): the sync reported 3 buffers with different content on ranks 1 and 2 and
0 on rank 3 in that launch (launch-dependent, as observed), the post-sync
cross-rank comparison shows 0 mismatches in 2 032 tensors, and **every rank**
scores the samples at mean abs prob diff 0.007 / pearson 0.999 vs vLLM, seq-avg
−logp 0.429 vs 0.428. The live run prints the same `[rank-buffer-sync]` line per
rank at worker init.

Consequence: the 09:39 final run trained for ~45 steps with 3 of 4 ranks
producing wrong log-probs and gradients; it was stopped and relaunched with the
fix (§6). Every earlier gemma-4 verl run in this repo had this second bug on top
of §0.
