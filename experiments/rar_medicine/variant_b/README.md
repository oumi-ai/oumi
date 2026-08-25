# RaR-Medicine GRPO — Variant B judge (gpt-5-mini)

Trains gemma-4-E2B-it on `anisha2102/RaR-Medicine` with verl GRPO, using ONE
LLM judge that applies the fixed meta rubric (Variant B / implicit aggregation,
see `../META_RUBRIC.md`) instead of the dataset's per-sample rubrics.
Reward = judge's holistic 0-10 score / 10, conditioned on the sample's
`reference_answer`.

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
  enforced via guided decoding, gpt-5-mini over the OPENAI engine
- `train_verl.yaml` — training config (adapted from
  `configs/examples/letter_counting/grpo/train_verl.yaml`, incl. the gemma-4
  workarounds)
- `oumi_extra_deps.txt` — pointed to by `OUMI_EXTRA_DEPS_FILE` so the oumi
  driver imports the module (running the `@register` decorators) before the
  config is resolved
- `run.sh` — sets `OUMI_EXTRA_DEPS_FILE` + `PYTHONPATH` (verl's reward-loop Ray
  actors import the reward as `pkg://rar_medicine_grpo`, so the dir must be on
  `PYTHONPATH`), checks `OPENAI_API_KEY`, launches `oumi train`
- `test_judge.py` — smoke test: registry/dataset/parse checks offline, plus
  live judge sanity checks when `OPENAI_API_KEY` is set (reference answer must
  outscore a wrong answer)

## Usage

```bash
export OPENAI_API_KEY=sk-...

# 1. Smoke-test the judge first (2 samples, ~6 API calls):
cd /workspace/persist/shanghong/oumi/tmp/rar_medicine/variant_b
OUMI_EXTRA_DEPS_FILE=$PWD/oumi_extra_deps.txt python test_judge.py

# 2. Train:
bash run.sh
```

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
(~$0.15-0.30/step at gpt-5-mini pricing); each eval judges the 256-sample val
subset once. Train is subsampled to 4096 (~64 steps/epoch) — raise
`sample_count` in `train_verl.yaml` for full runs (17,926 examples).

## Notes

- gpt-5 models only accept `temperature=1.0` and use `max_completion_tokens`;
  oumi's OpenAI engine handles both automatically for gpt-5 model names.
- SimpleJudge enforces the JSON response schema (explanation + integer
  judgment) via guided decoding, so parsing is structurally guaranteed.
- Judge failures (after retries) and empty responses get reward 0.0 and a
  warning in the reward-worker logs.
- verl config knob `reward.num_workers` (default 8) controls how many Ray
  reward actors share the judging.
