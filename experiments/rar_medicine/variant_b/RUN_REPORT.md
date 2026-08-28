# RaR-Medicine Variant B — LoRA GRPO on gemma-4-E2B-it: run report (2026-08-26)

Covers the 05:24 launch that hung for 2 h 21 min, its root cause and fix, the
22:34 relaunch (training normally as this is written), how LoRA is wired into
this verl run, the per-step time budget and why `update_weights` is ~4× slower
than the full-finetune letter-counting run, and the checks and measurements made
along the way. Companion docs: `train_verl.yaml` (inline rationale),
`GPU_MEMORY.md` (memory budget), `../../../GEMMA4_VERL_GRPO_FIXES.md` (issues 1–14
and the site-packages patches), `checks/` (standalone vLLM/LoRA checks).

## TL;DR

| | |
|---|---|
| Why the 05:24 run "failed" | It never crashed — it deadlocked at step 1 inside the reward. All 8 verl `RewardLoopWorker`s built the `SimpleJudge`, one judge call per worker returned, every later call parked forever. GPUs sat at 0 % util for 2 h 21 min until the raylet was killed. |
| Root cause | oumi `RemoteInferenceEngine` with the default `remote_params.use_adaptive_concurrency: True` is not safe to share across threads: each `judge()` call runs in its own event loop (`safe_asyncio_run`), but all loops share one `AdaptiveConcurrencyController` (capacity `num_workers`=1, `asyncio.Lock` + waiter futures bound to one loop). Reproduced outside Ray: 16 threads → 1 returns, 15 hang. |
| Fix | `judge_config.yaml` → `remote_params.use_adaptive_concurrency: False` (32 threads → 32/32 return in ~5 s). One line. |
| Relaunch | 22:34 UTC. Step 1 at 22:42. 137 s/step, 64 steps ≈ 2 h 25 min. Judge scores 0.46–0.60 mean with full 0.1–1.0 spread, zero judge retries/failures, entropy drifting 3.03 → 2.47 by step 20. |
| Slowness | `update_weights` 37 s/step vs 10 s in letter counting. **Root cause: verl re-streams the entire 20 GiB base model to vLLM every step in LoRA hybrid runs** — the actor-side `ServerAdapter` assumes vLLM sleeps at level 2 (`VLLM_SLEEP_LEVEL`), the vLLM server actually sleeps at level 1 for LoRA, so `fsdp_workers.rollout_mode` takes its "re-send the base" branch each step for nothing (§5.5). **Validated:** same repro with the actor-side level forced to 1 → `update_weights` **4.7 s** (was 37.8–42 s), step 102 s vs 120–123 s. Fix = 1-line verl patch (or `layered_summon: True` + `load_format: safetensors`, config only, slower). `update_actor` is 2× letter counting mainly because prompts are padded to 1024 while p99 is 383 tokens. |

## 1. What the run is

GRPO (verl 0.7.1, FSDP1 hybrid engine, vLLM 0.19.1 async server, TP=2 → two
replicas on 4× H100) on `anisha2102/RaR-Medicine` (4 096-sample subset, 64
prompts × 8 rollouts per step, 64 steps). Reward = oumi `SimpleJudge`
(gpt-4.1-mini, meta-rubric, integer 0–10 → /10). Policy = LoRA r=16 over frozen
`google/gemma-4-E2B-it`.

## 2. How LoRA is wired in (and what it took)

oumi's top-level `peft:` block is not plumbed into the VERL_GRPO trainer, so the
adapter is declared with verl's own keys under `actor_rollout_ref.model`:

```yaml
lora_rank: 16
lora_alpha: 32
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
exclude_modules: "(.*vision_tower.*)|(.*audio_tower.*)"   # ONE regex string, not a list
```

Consequences verl derives from `lora_rank > 0` (all confirmed in the live run):

1. **Actor** is wrapped in a PEFT model; FSDP's LoRA wrap policy makes every
   trainable leaf its own FSDP unit → **554 FSDP units** (vs ~36 for full FT).
2. **Reference policy** = the same actor with the adapter disabled; no ref worker
   is created (the `ref.fsdp_config.param_offload: True` line is inert). Visible as
   `timing_s/ref` 7.7 s here vs 9.6–14.9 s in letter counting (which reloaded an
   offloaded ref model every step).
3. **vLLM** is launched with `--enable-lora --max-loras 1 --max-lora-rank 16`.
   After step 1 only adapter tensors cross the bus (`TensorLoRARequest` →
   `add_lora`).

Things that had to be true for this to work on gemma-4 (details in
`GEMMA4_VERL_GRPO_FIXES.md`):

| Obstacle | Fix | Where |
|---|---|---|
| vLLM 0.19.1 `Gemma4ForConditionalGeneration` did not declare `SupportsLoRA` → every worker died at model load, trainer hung waiting | backport the `SupportsLoRA` mixin from vLLM main (Patch 5; erased by a vllm reinstall) | `vllm/model_executor/models/gemma4_mm.py` |
| With `--enable-lora` vLLM wraps *every* LoRA-capable linear (`<m>.base_layer.weight`), but verl's step-1 base sync renames only q/k/v/o/gate/up/down → `KeyError: layers.0.per_layer_input_gate.weight`, trainer hangs (observed 4 h) | restrict what vLLM wraps to the four fused modules the adapter touches | `rollout.engine_kwargs.vllm.lora_target_modules: [qkv_proj, o_proj, gate_up_proj, down_proj]` |
| PEFT cannot adapt the towers' `Gemma4ClippableLinear`, which share the projection names | `exclude_modules` regex (a list would be suffix-matched and exclude nothing) | yaml |
| Zero-init adapter + full-FT LR 1e-6 barely moves | `learning_rate: 1.0e-5` | yaml |
| Weight-sync bucket must hold the 8.75 GiB fp32 `embed_tokens_per_layer` for the step-1 base sync | `checkpoint_engine.update_weights_bucket_megabytes: 10240` | yaml |
| No validation that adapter tensors land in vLLM (verl's `from_lora_tensors` hijack skips module-name checks) | `rollout.calculate_log_probs: True` → `training/rollout_probs_diff_*` per step | yaml (but see §4.2) |

Standalone checks written for it, all passing on 2026-08-26: `checks/seed_adapter.py`
(zero-init adapter with verl's exact `LoraConfig`), `checks/vllm_lora_check.py`
(patched vLLM class loads it; outputs identical with/without adapter),
`checks/vllm_base_sync_check.py` (replays the step-1 base sync with verl's renaming).

### 2.1 Trainable-parameter accounting (re-measured)

`get_peft_model` with verl's exact `LoraConfig` on the meta device: **25 337 856
trainable = 25.3 M**, i.e. 0.46 % of the 5.53 B-param `Gemma4ForConditionalGeneration`
the actor instantiates (0.49 % of the 5.12 B text-only model). A LoRA on a
`in→out` linear adds `r·(in+out)`; gemma-4-E2B's layers are not uniform, which is
why "7 projections × 35 layers" with headline shapes gives only 18.9 M:

| module | layers | (in, out) | params each |
|---|---|---|---|
| q_proj / o_proj | 28 sliding (head_dim 256) | 1536 ↔ 2048 | 57 344 |
| q_proj / o_proj | 7 full-attention (`global_head_dim` 512) | 1536 ↔ 4096 | 90 112 |
| k_proj / v_proj | 28 / 7 | 1536 → 256 / 512 | 28 672 / 32 768 |
| gate/up/down_proj | 15 | 1536 ↔ 6144 | 122 880 |
| gate/up/down_proj | 20 | 1536 ↔ 12 288 | 221 184 |

→ q 2 236 416 + k 1 032 192 + v 1 032 192 + o 2 236 416 + 3 × 6 266 880 = 25 337 856.
Memory: ×4 B = 97 MiB fp32 weights, 97 MiB grads, 193 MiB AdamW.

### 2.2 Memory rows people asked about

* "fp32 / bf16 = 5.53 B × 4 / × 2" — bytes per parameter. The FSDP actor keeps
  fp32 master weights (20.6 GiB, 5.15 GiB/GPU sharded); forward/backward and
  vLLM run in bf16 (10.3 GiB; 5.15 GiB per GPU at TP=2).
* `gpu_memory_utilization: 0.4` → vLLM may use 0.4 × 79.65 = 31.9 GiB of the
  card for weights + profiled activations + KV cache; it must be *free* at
  start. Prediction for the rollout phase (vLLM awake + actor resident + allocator
  slack) ≈ 46 GiB; observed 57 % of 80 GiB = 45.8 GiB. Generous for this model
  (1 KV head, 20/35 layers KV-shared, sliding windows): 0.25–0.3 would also work.

## 3. Failure: the 05:24 run hung for 2 h 21 min

### 3.1 Evidence chain

| time (UTC) | event | source |
|---|---|---|
| 05:29:15 | `Starting verl training...` | run log |
| 05:30:29 | vLLM receives first generate request (so the step-1 LoRA base sync — the issue-14 hang — succeeded) | run log |
| 05:30:33 (±0.4 s) | all 8 `RewardLoopWorker`s log `Building SimpleJudge` — **and never log again** (no retries, no "Judge failed") | `/tmp/ray/session_2026-08-26_05-24-41_*/logs/worker-*-27764xx.err` |
| 05:31 → 07:51 | all 4 GPUs 0 % util, memory flat at 57 % (vLLM still awake ⇒ still inside the rollout/reward phase) | wandb system metrics, run `4g7c5q58` |
| 07:51:34 | raylet terminated (external kill; cgroup `oom_kill 0`, no exception anywhere) | run log, `/sys/fs/cgroup/memory.events` |

Ruled out: OOM (host and cgroup), NCCL/weight-sync deadlock (would show 100 %
util), vLLM engine death (no error in any vLLM worker log), API key/model
(a single sequential judge call works, reward 1.0 in 4 s).

### 3.2 Mechanism

`rar_medicine_grpo.py` shares one `SimpleJudge` per reward worker across a
16-way thread semaphore (verl runs sync reward functions in the actor's default
thread pool). Every `engine.infer()` runs in a fresh thread + event loop
(`oumi/core/async_utils.py: safe_asyncio_run`). `RemoteInferenceEngine._query_api`
routes each request through the **engine-level** `AdaptiveConcurrencyController`
(default `use_adaptive_concurrency: True`), whose `AdaptiveSemaphore` holds an
`asyncio.Lock` and per-waiter futures. Capacity = `num_workers` = 1:

1. thread A acquires the slot and completes;
2. threads B…P queue waiter futures created on *their* loops;
3. A's `release()` resolves those futures with `call_soon` from a foreign thread —
   which never wakes the other loops' `select()`. No request is ever sent, so the
   aiohttp `connection_timeout` never starts. Silent, permanent.

Repro (`scratchpad/judge_repro.py`, real gpt-4.1-mini calls): 16 threads on one
judge → 1 returns, 15 hang; `faulthandler` shows 15 idle event loops in
`selectors.select`. Matches the log: one call per worker, then silence.

### 3.3 Fix and verification

`judge_config.yaml`:

```yaml
remote_params:
  use_adaptive_concurrency: False
```

`_try_record_success/_error` are guarded by the same flag, `PoliteAdaptiveSemaphore`
and the aiohttp session are per-call, and `_rate_limiter` is None (no RPM set) —
nothing else is shared across loops. Same repro at 32 threads: 32/32 return in
5–6 s each. Better long-term: one `judge.judge(list)` per reward chunk with
`num_workers: N` (single loop; the adaptive controller then works as designed),
or a per-thread engine. Worth an oumi issue: the engine could bind the controller
per loop or refuse cross-loop use instead of hanging.

Also cleaned up before relaunch: 6 orphaned `ray::_run_verl_train` dataloader
workers (PPID 1) left by the dead run — they hold `/dev/nvidia*` fds but no memory,
so `run.sh`'s pre-flight does not catch them.

## 4. Relaunch (22:34 UTC) — status

`oumi train` pid 3440617, log `logs/rar_medicine_grpo_verl_variant_b.log` (the
hung run's log is `…20260826-223438.log`). Training start 22:39:10, step 1 at
≈22:42.

### 4.1 Per-step metrics (steps 1–31, half the run)

| step | score mean | resp len | entropy | rollout_probs_diff mean | pearson | update_weights s | step s |
|---|---|---|---|---|---|---|---|
| 1 | 0.496 | 348 | 3.03 | 0.760 | 0.091 | 36.9 | 141 |
| 2 | 0.486 | 310 | 3.02 | 0.757 | 0.093 | 39.2 | 134 |
| 3 | 0.537 | 352 | 2.99 | 0.755 | 0.101 | 37.1 | 143 |
| 4 | 0.541 | 310 | 2.98 | 0.763 | 0.087 | 36.4 | 140 |
| 5 | 0.462 | 340 | 2.95 | 0.752 | 0.097 | 36.6 | 145 |
| 6 | 0.561 | 330 | 2.93 | 0.756 | 0.090 | 36.4 | 144 |
| 7 | 0.502 | 341 | 2.90 | 0.755 | 0.090 | 36.5 | 142 |
| 8 | 0.510 | 366 | 2.85 | 0.767 | 0.093 | 36.3 | 136 |
| 9 | 0.510 | 332 | 2.91 | 0.767 | 0.085 | 34.6 | 133 |
| 10 | 0.597 | 327 | 2.82 | 0.762 | 0.091 | 34.2 | 137 |
| 11 | 0.475 | 323 | 2.82 | 0.763 | 0.082 | 36.9 | 135 |
| 12 | 0.510 | 315 | 2.78 | 0.753 | 0.082 | 32.2 | 129 |
| 13 | 0.461 | 374 | 2.81 | 0.768 | 0.084 | 32.1 | 132 |
| 14 | 0.551 | 354 | 2.76 | 0.762 | 0.093 | 32.9 | 164 |
| 15 | 0.418 | 353 | 2.70 | 0.762 | 0.082 | 35.5 | 146 |
| 16 | 0.486 | 349 | 2.72 | 0.764 | 0.083 | 33.2 | 130 |
| 17 | 0.477 | 396 | 2.69 | 0.766 | 0.081 | 35.0 | 134 |
| 18 | 0.540 | 337 | 2.61 | 0.776 | 0.076 | 34.3 | 130 |
| 19 | 0.528 | 324 | 2.54 | 0.779 | 0.065 | 33.6 | 132 |
| 20 | 0.518 | 333 | 2.47 | 0.769 | 0.072 | 33.8 | 132 |
| 21 | 0.538 | 323 | 2.47 | 0.763 | 0.076 | 36.0 | 159 |
| 22 | 0.524 | 352 | 2.41 | 0.775 | 0.082 | 34.0 | 145 |
| 23 | 0.490 | 329 | 2.35 | 0.769 | 0.061 | 36.0 | 154 |
| 24 | 0.533 | 335 | 2.32 | 0.775 | 0.069 | 33.4 | 130 |
| 25 | 0.575 | 324 | 2.31 | 0.777 | 0.064 | 36.4 | 134 |
| 26 | 0.528 | 350 | 2.20 | 0.780 | 0.076 | 35.5 | 132 |
| 27 | 0.448 | 350 | 2.25 | 0.776 | 0.061 | 41.7 | 154 |
| 28 | 0.496 | 350 | 2.17 | 0.779 | 0.058 | 36.3 | 148 |
| 29 | 0.535 | 356 | 2.11 | 0.771 | 0.058 | 33.5 | 149 |
| 30 | 0.488 | 323 | 2.13 | 0.775 | 0.057 | 35.1 | 144 |
| 31 | 0.534 | 350 | 2.05 | 0.782 | 0.051 | 34.0 | 145 |

Healthy signs: judge scores span 0.1–1.0 within every batch (so GRPO has
non-zero advantages; garbage rollouts would pin at ~0), 0.8 % of responses hit
the 1 024-token cap, `timing_s/reward ≈ 0` (rewards fully overlapped with
generation), zero `Retrying request` / `Judge failed` lines through step 31,
entropy falling steadily (3.03 → 2.05). Not yet visible: an upward score trend —
batch means sit in 0.42–0.60 with no drift over 31 steps (each batch is 64
prompts, so ±0.05 is noise; compare the step-50 validation instead). The Pearson
guard drifts 0.09 → 0.05 while `rollout_probs_diff_mean` stays 0.75–0.78 (§4.2).

**Step-50 validation** (256 val prompts, greedy, 1 judge call each, 17 s):
`val-core/anisha2102/RaR-Medicine/acc/mean@1 = 0.535`. Training-batch means over
steps 32–51 average ≈0.52 (sampled at T=1), so greedy 0.535 is in line, not above
it. There is no step-0 baseline to compare against (`val_before_train: False`), so
this number alone cannot show learning; entropy has fallen 3.03 → 1.23 by step 51,
which says the policy *is* moving. To get the baseline, either judge the base model
on the same 256 prompts (greedy) after the run, or set `val_before_train: True`
next time (~256 extra judge calls, ≈$0.15).

### 4.2 Open item: `training/rollout_probs_diff_mean ≈ 0.76` from step 1

This metric (|exp(actor log-prob) − exp(vLLM log-prob)| over response tokens)
was added as the guard for the LoRA sync path: it should sit at bf16 noise
(~1e-3) and *grow* if the adapter stopped reaching vLLM. It is 0.75–0.78 with
Pearson ≈ 0.09 **already at step 1**, when the adapter is zero-initialised and
actor ≡ base model — while vLLM's outputs are demonstrably coherent (judge
scores) and its base weights were verified by `vllm_base_sync_check.py`. So as
an absolute it is not measuring what we assumed in this stack (candidates:
actor padded-sdpa forward vs vLLM for gemma-4's sliding/KV-shared layers, or a
positional misalignment in verl's comparison). Its *trend* is flat, which is
what the guard was for, but it should be calibrated: run the base model through
HF (bf16, sdpa, padded) and vLLM on the same sampled sequence and compare
per-token log-probs offline. Not done yet.

## 4.3 End of run (01:12 UTC, 2026-08-27)

64/64 steps, 2 h 33 min wall, zero judge retries/failures throughout. Final-step
validation `val-core/anisha2102/RaR-Medicine/acc/mean@1 = 0.552` (0.535 at step 50);
entropy 3.03 → 0.95. Checkpoint `verl_output/global_step_64/` (22 GB: 4 FSDP
shards + optimizer + `actor/huggingface/` config + **`actor/lora_adapter/`** — the
PEFT adapter, 101 MB, r=16/α=32, correct target/exclude modules). Ray/vLLM
teardown printed the usual `EngineCore died unexpectedly` / `resource_tracker
KeyError` noise; the process exited and all GPUs are free.

**Caveat — the top-level export is not a usable merged model.** oumi's final
"Saving final model" wrote `output/model.safetensors` (11.1 GB) with **2,502
PEFT-prefixed keys** (`base_model.model.model.…`) including the 490 un-merged
`lora_A`/`lora_B` tensors, next to a plain `Gemma4ForConditionalGeneration`
`config.json`. `from_pretrained` on that directory will not match keys.
`src/oumi/utils/verl_model_merger.py` has no LoRA/PEFT handling (it was written
for full-FT checkpoints). To get a merged model: load the base, attach
`actor/lora_adapter/` with `PeftModel.from_pretrained`, `merge_and_unload()`,
`save_pretrained` — or serve the adapter directly in vLLM (`--enable-lora`).
Worth a repo fix: detect PEFT keys in the merger and either strip the prefix +
merge, or just copy `lora_adapter/` to the output.

## 5. Slowness: why `update_weights` takes 37 s (letter counting: 10 s)

### 5.1 Per-step budget vs the letter-counting run

| phase (s) | rar LoRA, 4 GPU (steps 1–3) | letter counting full-FT, 8 GPU (steps 1–5) |
|---|---|---|
| `gen` (generation + waiting for the last rewards) | 25 / 20 / 30 | 20 / 5.7 / 5.6 / 5.3 / 5.5 |
| `old_log_prob` | 16 / 15 / 15 | 7.2 / 7.0 / 6.9 / 7.2 / 7.0 |
| `ref` | 7.8 / 7.7 / 7.7 | 14.8 / 9.5 / 9.6 / 9.9 / 9.6 |
| `update_actor` | 55 / 52 / 52 | 27.5 / 26.9 / 26.7 / 27.1 / 26.7 |
| **`update_weights`** | **36.8 / 39.2 / 37.0** | **9.4 / 10.0 / 10.4 / 10.0 / 9.6** |
| step | 141 / 134 / 143 | 79 / 60 / 60 / 60 / 59 |

Same sequences per GPU (128), so per-GPU work differs only by sequence length
and by the LoRA path:

* **`update_actor` / `old_log_prob` 2×**: padded forwards (`use_remove_padding:
  False`, no flash-attn) run on `max_prompt_length + response_length` = 1024 +
  1024 = 2048 tokens per sequence vs 256 + 1024 = 1280 in letter counting.
  Real token use is ~500/2048 (24 %). Tokenised prompt lengths over the 4 096
  training rows: mean 147, p50 134, p90 196, p99 383, max 1000 (>384: 40 rows,
  >512: 7, >768: 2; val max 593). `max_prompt_length: 512` with
  `data.filter_overlong_prompts: True` (drops 7 + 1 rows) cuts the padded
  length by 25 %; installing flash-attn and `use_remove_padding: True` is the
  real fix (~3–4× less actor compute).
* **`gen` = ~8 s of generation + ~20 s waiting for the judge tail** (GPUs 0 %,
  all processes idle; visible in the traces). Inherent to per-sample async
  rewards with ~5 s API latency; only concurrency or a faster judge shortens it.
* **`ref` faster**: no offloaded ref model to reload (adapter-disabled actor).
* **`update_weights` 4× slower while shipping 200× less data** — investigated below.

### 5.2 What `update_weights` does in the LoRA hybrid path (verl 0.7.1)

`ray_trainer` → `CheckpointEngineManager.update_weights` (naive backend) →
every FSDP rank runs `fsdp_workers.rollout_mode()`:

1. `aggressive_empty_cache(force_sync=True)` (gc.collect + empty_cache, ≤3×)
2. `collect_lora_params(layered_summon=False, base_sync_done=True)`:
   `FSDP.summon_full_params(whole model)` → `get_peft_model_state_dict` → 490
   adapter tensors `.cpu()` (step 1 instead: full base state dict, 20.6 GiB
   round-trip through CPU, keys renamed to `.base_layer`)
3. `convert_weight_keys`, `set_expandable_segments(False)`
4. vLLM `wake_up(["weights"])` (sleep level 1 → restore 5.15 GiB from CPU)
5. `rollout.update_weights`: `collective_rpc("update_weights_from_ipc")` → vLLM
   workers `remove_lora`, then `BucketedWeightSender` (10 GiB CUDA-IPC bucket)
   → `BucketedWeightReceiver` clones → `add_lora(TensorLoRARequest)`
6. `aggressive_empty_cache`, vLLM `wake_up(["kv_cache"])`, `clear_kv_cache`,
   `set_global_steps`, `set_expandable_segments(True)`

### 5.3 What was measured (each in isolation, idle GPUs 4–7, same model/wrap/config)

| component | measured | notes |
|---|---|---|
| `collect_lora_params(layered_summon=False)` | **1.2–1.4 s** at world size 4 (0.6–1.1 s at ws 1) | peak +21 GiB (the whole-model summon); 554 FSDP units |
| `collect_lora_params(layered_summon=True)` | **11.4 s** at ws 4 (9.6 s at ws 1) | peak only 5.1 GiB — a memory knob, ~9× slower |
| full-FT style `state_dict()` + `full_tensor()` per tensor | 1.7 s at ws 4 | for comparison |
| `gc.collect` / `aggressive_empty_cache` (PEFT+FSDP object graph, 30 GiB cached) | 0.2–0.4 s / ≤0.16 s | expandable segments on or off |
| `set_expandable_segments`, `torch.empty(10 GiB)` bucket | 0.00–0.05 s | |
| CUDA-IPC bucket transfer of the real 490-tensor adapter (two processes, same GPU) | **0.55–0.6 s** sender, 1.1 s receiver | identical for 10 GiB and 512 MiB buckets; `cudaIpcOpenMemHandle` 0.00 s |
| vLLM `sleep(1)` / `wake_up(weights)` / `remove_lora` / `add_lora` / `wake_up(kv_cache)` | 0.25 / 0.3 / 0.00 / 0.06–0.10 / 0.01 s | vLLM 0.19.1, gemma-4, cudagraphs on |

Sum of the parts ≈ 3–4 s. The step timer says 37 s.

### 5.4 Where the time actually goes (live traces of the running job)

1 Hz `nvidia-smi` + `/proc` CPU accounting across full steps (`scratchpad/timeline*.tsv`,
`threads.tsv`):

* The 35 s between the end of `update_actor` and the first generation kernel
  splits into **~20 s** where all four FSDP ranks' **main Python thread is at
  ~100 % CPU in lockstep**, GPU util 3–8 %, memory flat, and **every vLLM process
  (2 EngineCores, 4 TP workers) at 0.0 CPU** — followed by **~8 s** of staggered
  IPC transfers/wake-ups (memory bouncing in 10 GiB bucket steps at different
  moments on different GPUs, TP workers active) and ~4 s of the rest.
* The ~20 s phase is therefore actor-side, *before* the first vLLM call, and
  not GPU- or bandwidth-bound. The +20 GiB whole-model summon spike is *not*
  present during it, so it is not the gather itself.
* **Step 1 (full 20 GiB base sync) took the same 36.8 s as steps 2–3 (97 MiB
  adapter)** → the dominant cost is payload-independent.
* Not the allocator (§5.3), not gc, not IPC, not vLLM, not `convert_weight_keys`
  / `set_expandable_segments` (trivial code). The standalone replicas of the
  actor-side calls are fast, so the cost depends on live-process state that the
  replicas lack.
* Stack sampling of the live workers is blocked here (`py-spy` → ptrace scope 1,
  no sudo; Ray dashboard profiling disabled; no `perf`). A faithful repro with
  `VERL_LOGGING_LEVEL=INFO` (verl prints timestamped `log_gpu_memory_usage`
  markers between every phase of `rollout_mode` and `update_weights done, time
  cost`) is running on GPUs 4–7: `scratchpad/letter_counting_lora_repro.yaml`,
  log `scratchpad/lora_repro.log`. Its breakdown goes in §5.5.

### 5.5 Root cause: the base model is re-synced every step

Repro (`scratchpad/letter_counting_lora_repro.yaml`: letter-counting task, no judge,
this run's exact LoRA block, 4 GPUs, `VERL_LOGGING_LEVEL=INFO`) reproduces the
slow `update_weights` (35–41 s between the `aggressive_empty_cache` marker that
opens `rollout_mode` and the one right after the transfer), so it is the LoRA
sync path, not the dataset or judge. The vLLM TP workers' INFO lines give it away
— every step, per worker:

```
23:36:48  Loading standard weights (non-FP8, async)   <- BucketedWeightReceiver bucket 1: full base model
23:36:50  Loading standard weights (non-FP8, async)   <- bucket 2
23:36:51  Loading standard weights (non-FP8, async)   <- bucket 3 ...
23:36:51  vLLM load weights, loaded_params: 490       <- and only then the adapter
```

`Loading standard weights` is the **non-LoRA** branch of `_update_weights`
(`model.load_weights(...)`): the whole base model is being streamed again. The
code path (`verl/workers/fsdp_workers.py: rollout_mode`, lines 784–836):

```python
if peft_config is not None and getattr(self.rollout, "sleep_level", None) == 2 and free_cache_engine:
    base_model_params = collect_lora_params(module, layered_summon=..., base_sync_done=False)  # full model -> CPU -> GPU
    ...
    await self.rollout.update_weights(per_tensor_base_params, base_sync_done=False)          # streams ~20 GiB
await self.rollout.update_weights(per_tensor_param, peft_config=peft_config, base_sync_done=True)  # then 97 MiB adapter
```

`self.rollout` is the actor-side `ServerAdapter`, whose `sleep_level` is
`VLLM_SLEEP_LEVEL` (= 2 for vllm ≥ 0.8.5) unless `rollout.layered_summon` is set.
But the vLLM server (`vllm_async_server.py: sleep()`) uses **level 1 for LoRA**
("lora only update adapter weights") — and this env's gemma-4 patch forces level 1
for everything — so the base weights survive every sleep and the re-sync is pure
waste. Two halves of verl disagree about the sleep level; the actor side pays for
the pessimistic assumption on every step:

| per-step cost of the wasted branch | evidence |
|---|---|
| `collect_lora_params(base_sync_done=False)`: `summon_full_params` of 5.5 B fp32, `model.to("cpu")`, `state_dict()` with a `.cpu()` per tensor, `model.to(gpu)` — ~20 GiB down and back up, per rank | the ~20 s main-thread CPU-bound phase with vLLM idle (§5.4); host RAM transiently +4 × 20 GiB |
| streaming ~20 GiB fp32 through 10 GiB CUDA-IPC buckets, receiver clone, `load_weights` of 670 tensors | the 10 GiB memory bounces staggered per rank; 3+ `Loading standard weights` per worker per step; the ~8 s vLLM-active phase |
| same work at step 1 and at step 2+ | `update_weights` 36.8 s vs 39.2 / 37.0 s |

Letter counting (full FT) never enters this branch (`peft_config is None`): it
streams the sharded state dict once, without the CPU round trip → 10 s.

**Fixes**

1. *verl patch (recommended, 1 line):* in `verl/workers/rollout/vllm_rollout/vllm_rollout.py`
   `ServerAdapter.__init__`, use `sleep_level = 1` when the run is LoRA (mirror the
   server's `lora_as_adapter` test), or set `VLLM_SLEEP_LEVEL = 1` in
   `verl/third_party/vllm/__init__.py` (this env forces level 1 server-side
   anyway for gemma-4). Expected per step: whole-model summon 1.3 s + adapter
   transfer 0.6 s + vLLM wake/add_lora ~1 s ≈ 4–5 s instead of 37 s (~−24 % step
   time). **Validated** on the same repro with the actor-side level forced to 1 via a
   `sitecustomize.py` on the repro's PYTHONPATH (`scratchpad/fixpath/`, nothing
   in site-packages touched):

   | | unfixed repro | fixed repro |
   |---|---|---|
   | `timing_s/update_weights` | 42.0 s (step 1), 37.8 s (step 2) | **4.7 s** (step 4) |
   | `timing_s/step` | 123 / 120 s | **102 s** |
   | `Loading standard weights` lines (base-model buckets) | 3+ per worker **every step** | 12 total = 3 buckets × 4 workers, **step 1 only** |
   | `rollout_mode` bracket, step 2 (cleanup → adapter at vLLM → cleanup) | 35–41 s | 23:51:13.5 → 23:51:16.6 → 23:51:17.4 = **3.8 s** |

   Applied to production (137 s/step, 37 s of it `update_weights`) this is ≈ −32 s
   per step, ≈ −35 min over the 64-step run.
2. *Config only:* `rollout.layered_summon: True` (sets the actor-side level to 1)
   **plus** `rollout.load_format: safetensors` (required: the layered path cannot
   do the step-1 base sync, vLLM loads the base from disk). Per step ≈ 11.4 s
   layered collection + ~2 s. Also lowers the sync peak from +21 GiB to +5 GiB,
   and lets `update_weights_bucket_megabytes` drop to a few hundred MB.

### 5.6 What `rollout.layered_summon` is

For LoRA runs verl needs the adapter tensors as full (unsharded) tensors every
step. Default (`False`): one `FSDP.summon_full_params` over the **whole** model —
all 5.5 B fp32 params materialise on every rank (+20.6 GiB peak), then the 490
LoRA tensors are picked out. `True`: walk the FSDP sub-units (per decoder layer /
module) and summon **one at a time**, collecting each unit's LoRA tensors, so the
peak is one unit instead of the whole model. It is a **memory** knob, not a speed
one: measured 11.4 s vs 1.3 s (554 separate summon contexts, each with a
`state_dict()` and `empty_cache`). It also requires the base weights to already
be in vLLM (`rollout.load_format: safetensors`), because the layered path can only
ever collect adapter tensors — hence the yaml comment tying the two together. With
~27 GiB of headroom at the worst moment (`GPU_MEMORY.md`), this run does not need it.

## 6. Recommendations (in order)

1. Keep `use_adaptive_concurrency: False` for any oumi judge used from threads or
   from a verl reward function (memory note saved: `oumi-remote-engine-thread-deadlock`).
2. Run the next production run with `VERL_LOGGING_LEVEL=INFO` — it costs nothing
   and gives the `update_weights` breakdown for free.
3. Actor compute: `max_prompt_length: 512` + `filter_overlong_prompts: True` now;
   flash-attn + `use_remove_padding: True` when the env allows.
4. Judge failures currently return reward 0.0 silently after retries — make them
   raise (or count them) so a rate-limited step cannot train on zeros unnoticed.
5. Calibrate `rollout_probs_diff` offline (HF vs vLLM on the same tokens) before
   relying on it as the adapter-sync guard.
6. Before the next run, apply the `update_weights` fix from §5.5 as verl patch #6:
   simplest is `VLLM_SLEEP_LEVEL = 2` → `1` in `verl/third_party/vllm/__init__.py`
   (consistent with the server-side gemma-4 patch 2b, which already forces level 1),
   or the narrower LoRA-only test in `ServerAdapter.__init__`. Document it in
   `GEMMA4_VERL_GRPO_FIXES.md` next to patches 1–5 (also lost on a verl reinstall).
   Not applied here — the live run is unaffected either way (modules already imported).

## 7. Artefacts

* Run logs: `logs/rar_medicine_grpo_verl_variant_b.log` (live),
  `logs/rar_medicine_grpo_verl_variant_b.20260826-223438.log` (the hang),
  `…-052412.log` (issue 14), `…-004609.log` (port collision), `…-003910.log` (issue 13).
* Ray worker logs of the hang: `/tmp/ray/session_2026-08-26_05-24-41_442093_2761014/logs/`.
* Scratch scripts (session scratchpad `…/b0e1e938-…/scratchpad/`): `judge_repro.py`
  (deadlock repro), `lora_sync_bench.py` / `lora_sync_bench_dist.py`
  (`collect_lora_params` timing, ws 1 / ws 4), `gc_bench.py`, `alloc_bench.py`,
  `ipc_bench2.py` (bucket transfer), `vllm_side_bench.py` (sleep/wake/add_lora),
  `timeline*.tsv`, `threads.tsv` (live traces), `letter_counting_lora_repro.yaml` + `lora_repro.log` (unfixed repro), `fixpath/sitecustomize.py` + `lora_repro_fix.log` (fix validation).
