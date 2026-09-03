# RaR-Medicine GRPO — Variant B judge (gpt-4.1-mini)

Trains gemma-4-E2B-it on `anisha2102/RaR-Medicine` with verl GRPO, using ONE
LLM judge that applies the fixed meta rubric (Variant B / implicit aggregation,
see `../META_RUBRIC.md`) instead of the dataset's per-sample rubrics.
Reward = judge's holistic 0-10 score / 10, conditioned on the sample's
`reference_answer`. Sized for 4 GPUs.

**2026-09-02:** the recipe is now a **full fine-tune** with remove-padding,
actor offload and `gemma4_kv_share_patch.py` (vLLM applies gemma-4 LoRA as a
no-op, and without the patch every verl forward on gemma-4 is garbage). Read
`MEMORY_TUNING.md` first: it has the OOM post-mortem, the KV-sharing bug, the
measured per-phase memory budget and the tuning log. The LoRA (r=16) history is
in `RUN_REPORT.md` and `GPU_MEMORY.md`.

## Files

- `rar_medicine_grpo.py` — everything registered in one importable module:
  - dataset `anisha2102/RaR-Medicine` (verl-format rows; question goes into
    `extra_info` for the judge, reference answer into `reward_model.ground_truth`)
  - reward `rar_medicine_verl` — verl-style reward that runs one shared oumi
    **`SimpleJudge`** per reward worker (verl 0.7's reward loop dispatches sync
    rewards to a thread pool, so judge calls run concurrently; per-worker
    concurrency capped by a semaphore)
- `judge_config.yaml` — the judge itself (SimpleJudge config): meta-rubric
  system instruction, `{question}`/`{reference_answer}`/`{response}` prompt
  template, `judgment_type: INT` (0-10, reward = judgment/10), JSON response
  enforced via guided decoding, gpt-4.1-mini (non-reasoning) over the OPENAI engine
- `train_verl.yaml` — training config (adapted from
  `configs/examples/letter_counting/grpo/train_verl.yaml`, incl. the gemma-4
  workarounds), LoRA declared with verl's own `actor_rollout_ref.model.lora_*`
  keys (oumi's `peft:` block is not wired into the VERL_GRPO trainer)
- `MEMORY_TUNING.md` — 2026-09-02 OOM post-mortem, gemma-4 KV-sharing bug, measured full-FT budget, tuning log and results
- `gemma4_kv_share_patch.py` — REQUIRED for gemma-4 training forwards (loaded via `actor_rollout_ref.model.external_lib`)
- `checks/kv_share_patch_check.py`, `checks/rmpad_fa2_numerics_check.py`, `checks/actor_mem_replay.py`, `checks/fullft_phase_replay.py` — the measurements behind `MEMORY_TUNING.md`
- `GPU_MEMORY.md` — per-phase GPU memory budget for the earlier LoRA config on 4x H100 (superseded, see banner)
  (why no actor offload is needed with LoRA, and what to change if it OOMs)
- `checks/` — standalone vLLM check that a PEFT adapter with this exact
  `LoraConfig` loads into the (patched) vLLM gemma-4 class and is a no-op when
  zero-initialised; run it after any vllm reinstall
- `oumi_extra_deps.txt` — pointed to by `OUMI_EXTRA_DEPS_FILE` so the oumi
  driver imports the module (running the `@register` decorators) before the
  config is resolved
- `run.sh` — pins `CUDA_VISIBLE_DEVICES=0,1,2,3` (must match
  `trainer.n_gpus_per_node`), copies `INTERNAL_OPENAI_API_KEY` from the
  repo-root `.env` into `OPENAI_API_KEY` (only that variable — the file also
  holds unrelated credentials that must not reach the Ray workers), sets
  `OUMI_EXTRA_DEPS_FILE` + `PYTHONPATH` (verl's reward-loop Ray actors import
  the reward as `pkg://rar_medicine_grpo`, so the dir must be on `PYTHONPATH`),
  and launches `oumi train` with all output in `logs/<run_name>.log` at the
  repo root (previous log rotated to `<run_name>.<timestamp>.log`)
- `test_judge.py` — smoke test: registry/dataset/parse checks offline, plus
  live judge sanity checks when `OPENAI_API_KEY` is set (reference answer must
  outscore a wrong answer)

## Usage

```bash
# 1. Smoke-test the judge first (2 samples, ~6 API calls):
cd /workspace/persist/shanghong/oumi/experiments/rar_medicine/variant_b
OPENAI_API_KEY=sk-... OUMI_EXTRA_DEPS_FILE=$PWD/oumi_extra_deps.txt python test_judge.py

# 2. Train on GPUs 0-3, key taken from the repo .env, log in ../../../logs/:
bash run.sh
tail -f ../../../logs/rar_medicine_grpo_verl_variant_b.log

# Other GPUs: CUDA_VISIBLE_DEVICES=4,5,6,7 bash run.sh
```

Requires the vllm site-packages patch that lets `Gemma4ForConditionalGeneration`
take LoRA (GEMMA4_VERL_GRPO_FIXES.md, issue 13 / Patch 5) — without it vLLM
dies at start with "does not support LoRA yet" and the trainer hangs.
The yaml also restricts vLLM's LoRA wrapping to the four fused modules verl
knows how to sync (`engine_kwargs.vllm.lora_target_modules`; issue 14) — without
it the first weight sync dies on gemma-4's `per_layer_input_gate` and the
trainer hangs with no error at the driver. `checks/vllm_base_sync_check.py`
reproduces that sync standalone.

## Evaluation inference (`eval/`)

Runs the trained policy and the untrained base model over a fixed 1000-sample
subset of the hub `test` split with `oumi infer`, so the two output files can be
judged side by side (same prompts, same greedy decoding, same 1024-token cap).

- `eval/prepare_eval_set.py` — downloads the test parquet (2,242 rows) and
  samples 1000 rows (`DataFrame.sample(random_state=42)`, sorted by position).
  Writes `output/rar_medicine_grpo_verl_variant_b/eval/test_1000.parquet` (all columns + `idx`, the row position
  in the hub split — the join key, since there is no id column and some
  questions repeat) and `output/rar_medicine_grpo_verl_variant_b/eval/test_1000.jsonl` (oumi `Conversation`s:
  the training system prompt from `rar_medicine_grpo.py` + the question, with
  `idx` / `question_source` / `reference_answer` in `metadata`). The raw hub parquet stays in
  `../data/` (gitignored); `output/` is gitignored too; the script is deterministic.
- `eval/infer_base.yaml` — `oumi infer`, VLLM engine, `google/gemma-4-E2B-it`,
  greedy, `max_new_tokens: 1024` (= training `max_completion_length`). Output:
  `output/rar_medicine_grpo_verl_variant_b/eval/outputs/base_gemma4_e2b_it.jsonl` — the input conversations with
  the assistant turn appended plus `metadata.finish_reason` / `usage`.
- `eval/infer_trained.yaml` — same prompts and decoding, **NATIVE engine
  (transformers + PEFT)** with `model.adapter_model` pointing at verl's
  `verl_output/global_step_64/actor/lora_adapter/`, i.e. bf16 base + fp32
  adapter applied separately — the actor's own parameterization. Output:
  `output/rar_medicine_grpo_verl_variant_b/eval/outputs/grpo_variant_b_step64.jsonl`. Why not vLLM:
  - **vLLM 0.19.1 LoRA is a silent no-op for gemma-4** (even with the
    `SupportsLoRA` backport): the adapter loads into the right fused modules but
    outputs are bit-identical to the base, even at 100x `lora_alpha`
    (`checks/vllm_lora_adapter_check.py`; `GEMMA4_VERL_GRPO_FIXES.md` issue 15).
    This is the same mechanism verl used for rollouts, so training-time rollouts
    were base-model samples.
  - **The trainer's HF export is not merged**: `FSDPModelMerger` has no LoRA
    handling, so `tmp/rar_medicine/variant_b/output/model.safetensors` is the
    PEFT-wrapped state dict verbatim (`base_model.model.*.lora_A.default.weight`,
    ...) and loads nowhere.
  - **Merging into bf16 is lossy**: the adapter delta is ~5e-4 of the weight
    norm, under bf16 resolution for most elements — only ~82% of the delta norm
    survives the cast and 85% of merged elements equal the base.
  - `generation.use_cache: True` is mandatory: oumi defaults it to False, and
    gemma-4's KV-shared layers read K/V from the cache, so cache-less HF
    generation produces multilingual gibberish.
- `eval/merge_lora.py` + `eval/infer_trained_merged_vllm.yaml` — optional: a
  proper fp32 merge of the adapter into the base saved as a bf16 HF checkpoint
  (+ tokenizer + `processor_config.json`, which vLLM needs) at
  `tmp/rar_medicine/variant_b/merged_model/`, and a vLLM config that serves it.
  Fast, but approximate for the reason above; not run by `run_infer.sh`.
- `eval/run_infer.sh` — builds the eval set if missing, then runs base (vLLM,
  ~2 min) and trained (NATIVE, ~15-25 min) concurrently, one GPU each
  (`BASE_GPU`/`TRAINED_GPU`, default 0/1; `ONLY=base|trained` for one). Sets
  `VLLM_WORKER_MULTIPROC_METHOD=spawn` (oumi initialises CUDA before vLLM forks
  its engine core) and pins `CUDA_VISIBLE_DEVICES` per run (oumi's VLLM engine
  otherwise tensor-parallelises across every visible GPU). Logs:
  `logs/rar_medicine_infer_{base,trained}.log`. Re-running resumes from the
  output file. Ends with `summarize_outputs.py --fix-finish-reason`.
- `eval/summarize_outputs.py` — per file: rows, response-token stats, how many
  hit the 1024-token cap, "final answer" presence; trained-vs-base identical
  count. Recomputes truncation from token counts because the NATIVE engine
  labels every sequence of a batch `length` when any member hits the cap
  (`native_text_inference_engine.py`: `len(seq)` over the still-padded
  `output_batch.data`); `--fix-finish-reason` rewrites the labels in place,
  keeping the engine's as `metadata.finish_reason_engine`.
- `checks/vllm_lora_adapter_check.py` — the HF+PEFT vs vLLM+LoRA comparison
  behind the no-op finding; exit 1 while vLLM LoRA stays broken for gemma-4.

```bash
bash experiments/rar_medicine/variant_b/eval/run_infer.sh
```

Result of the 2026-08-27 run (greedy, 1000 prompts): base 335 mean response
tokens, trained 354; both truncate 12/1000 at the cap and say "final answer" in
997/1000; only 9/1000 trained responses are byte-identical to the base — the
adapter changes wording almost everywhere, as expected from a small delta at
near-tie tokens under greedy decoding. Whether it changes *quality* is the
judge's job (`judge_config.yaml` over `reference_answer` in each row's metadata).

## Knobs

Judge model, generation limits, and API retries live in `judge_config.yaml`
(edit there to change model, max tokens, timeouts). Env vars:

| Var | Default | Meaning |
|---|---|---|
| `RAR_JUDGE_CONFIG` | `judge_config.yaml` (next to the module) | alternate SimpleJudge YAML |
| `RAR_JUDGE_MAX_CONCURRENCY` | `16` | in-flight judge calls per reward worker (x `reward.num_workers` = 8) |
| `RAR_JUDGE_MAX_RETRIES` | `2` | outer retries around a judge call (engine retries transient API errors itself); reward 0.0 after final failure |

## Cost / throughput

Each step judges `train_batch_size * rollout.n` = 64 * 8 = 512 responses
(~$0.30/step at gpt-4.1-mini pricing); each eval judges the 256-sample val
subset once. Train is subsampled to 4096 (~64 steps/epoch) — raise
`sample_count` in `train_verl.yaml` for full runs (17,926 examples).

## Notes

- The judge is gpt-4.1-mini, a non-reasoning model, at `temperature: 0.0` so
  the same response always gets the same score. (gpt-5 models would pin
  temperature to 1.0 — oumi's OpenAI engine enforces that for o1/o3/o4/gpt-5
  names — and spend hidden reasoning tokens on a single-integer verdict.)
- SimpleJudge enforces the JSON response schema (explanation + integer
  judgment) via guided decoding, so parsing is structurally guaranteed.
- Judge failures (after retries) and empty responses get reward 0.0 and a
  warning in the reward-worker logs.
- verl config knob `reward.num_workers` (default 8) controls how many Ray
  reward actors share the judging.
- With `lora_rank > 0` verl runs no separate reference worker: the reference
  log-probs are the actor's with the adapter disabled. vLLM receives the full
  base weights once (step 1) and only adapter tensors afterwards.
- `rollout.calculate_log_probs: True` makes verl log
  `training/rollout_probs_diff_*` (actor vs vLLM on the same tokens) each
  step. It should sit at bf16 noise; a growing value means vLLM is not
  applying the adapter (verl's tensor-based LoRA sync does not validate
  module names, so that failure is otherwise silent).
