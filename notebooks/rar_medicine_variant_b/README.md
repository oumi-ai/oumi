# RL-trained Gemma (RaR-Medicine GRPO) vs baseline Gemma

Evaluates  GRPO checkpoints against the untrained
`google/gemma-4-E2B-it` baseline, scored with the **same LLM judge that was used
as the training reward**, and looks for reward hacking in the RL policy.
Reuses the rollout + judge pipeline in [`../rar_medicine_judges/`](../rar_medicine_judges/).

Checkpoints (all under
`s3://oumi-science-donotdelete/shanghong/runpod-node1/oumi/experiments/rar_medicine/variant_b/output/`):

| S3 folder | local folder | what it is | status |
|---|---|---|---|
| `medqa_gemma4-e2b-it_fullft/` | `models/medqa_gemma4-e2b-it_fullft/` | **full fine-tune** GRPO run, the one we want to study (280 steps, lr 1e-6 cosine, KL 0.001, 8 rollouts/prompt, full RaR-Medicine train split) | downloaded 2026-09-08; plain `model.*` keys, no merge needed; `processor_config.json` was missing and copied from base |
| `unmerged_model/` | `models/variant_b/` (+ `models/variant_b_merged/`) | earlier **LoRA** run (r=16, 64 steps, lr 1e-5); PEFT save that needed merging | evaluated (== base), local copies deleted; results kept in `results/test1k/` |

Python env: `PY=/root/miniconda3/envs/oumi/bin/python`. OpenAI key comes from the
repo `.env`. 8 H100s; each stage below says which GPU it uses.

## What to run, in order

| # | file | run when | what it does | output |
|---|---|---|---|---|
| 1 | `download_model.py` | once per checkpoint | pulls every object under the S3 prefix into `models/<run>/` (git-ignored) | `models/medqa_gemma4-e2b-it_fullft/` |
| 2 | *(check)* | after download | make sure `processor_config.json` is present and the safetensors keys are plain `model.*` (see "Checkpoint gotchas") | – |
| 3 | `merge_adapter.py` | **only** if step 2 shows a PEFT save (`adapter_config.json` or `base_model.model.*` keys) | merges the LoRA adapter into the base and writes a vLLM-loadable checkpoint | `models/<run>_merged/` |
| 4 | `eval_test1k.py generate` | once per model | greedy answers from one model on the 1000 test prompts (vLLM, 1 GPU, ~3 min) | `results/test1k/responses__<model>.jsonl` |
| 5 | `eval_test1k.py judge` | after step 4 for both models | scores every response with the training-reward judge (gpt-4.1-mini, OpenAI, ~5 min / model) | `results/test1k/judge__<model>__<variant>__<judge>.jsonl` |
| 6 | `eval_test1k.py summarize` | after step 5 | paired comparison + reward-hacking indicators | `results/test1k/SUMMARY.md`, `all_samples.jsonl` |
| 6b | `checkpoint_curve.py` | after judging intermediate checkpoints (`merge_verl_ckpt.py` -> `generate` -> `judge`) | training-judge vs gold-judge scores per training step | `results/test1k/CURVE.md` |
| 6c | `inspect_disagreements.py` | any time after step 5 | cases where the training judge scores 9-10 but strict judges score low, with all explanations | `results/test1k/DISAGREEMENTS__<model>.md` |
| 7 | `run_pipeline.sh` | optional, second view | GRPO-style rollouts (T=1, 4 samples) on 64 *val* prompts, scored by training judge **and** held-out per-question-rubric judges; reports the transfer ratio | `results/COMPARE.md` |

Steps 4-6 are cached: an existing output file is skipped unless `--overwrite`.
The **base model is already done** for steps 4-5 (`responses__base.jsonl`,
`judge__base__...jsonl`), so a new checkpoint only needs its own generate + judge.

### Commands for the full-FT checkpoint

```bash
cd notebooks/rar_medicine_variant_b
PY=/root/miniconda3/envs/oumi/bin/python

# 1. download (needs AWS creds that can read the bucket; see "Access note")
$PY download_model.py --profile <aws-profile>          # or env AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY
#   add --dry-run to list first; equivalent CLI:
#   aws s3 sync s3://oumi-science-donotdelete/shanghong/runpod-node1/oumi/experiments/rar_medicine/variant_b/output/medqa_gemma4-e2b-it_fullft/ models/medqa_gemma4-e2b-it_fullft/ --region us-west-2

# 2. check the download (see gotchas below)
ls models/medqa_gemma4-e2b-it_fullft/                  # must include processor_config.json
$PY -c "from safetensors import safe_open; f=safe_open('models/medqa_gemma4-e2b-it_fullft/model.safetensors','pt'); print(list(f.keys())[:3])"
#   -> keys should start with 'model.' ; if 'base_model.model.' run step 3 and use models/..._merged below

# 3. (only for a PEFT save)
$PY merge_adapter.py --src models/medqa_gemma4-e2b-it_fullft --dst models/medqa_gemma4-e2b-it_fullft_merged

# 4-6. evaluate against the cached base results
$PY eval_test1k.py generate  --model fullft --gpu 0    # greedy, 1 sample / prompt
$PY eval_test1k.py judge     --models base fullft      # base is cached, only fullft is judged
$PY eval_test1k.py summarize --models base fullft      # -> results/test1k/SUMMARY.md

# any other checkpoint: --model NAME --path /dir/or/hf-id on generate, then NAME in --models
```

Variants of step 4: `--temperature 1.0 --num-generations 4` mimics GRPO rollout
sampling (use this if greedy looks flat). Variants of steps 5-6: add a held-out
grader with `--judges variant_b_training_reward:gpt-4.1-mini fixed_rubric_implicit:gpt-4.1`
(the test file has no per-question rubrics, so only reference-anchored variants apply).

### Commands for the val-set rollout comparison (step 7)

```bash
MODEL_PATH=models/medqa_gemma4-e2b-it_fullft POLICY_NAME=fullft bash run_pipeline.sh
#   POLICY_GPU=0 JUDGE_GPU=1 by default; JUDGES="variant:model ..." (first = training reward)
```

## Checkpoint gotchas (learned on the LoRA run)

* **`processor_config.json` must be present.** Gemma 4 is multimodal; vLLM loads
  the processor even for text and dies without it. If the S3 folder lacks it,
  copy it from the base:
  `cp /workspace/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/3e22461f65e89153144f8adb70e3b8c2cc9845a7/processor_config.json models/<run>/`
* **A single `model.safetensors` is not proof of a merged model.** The
  `unmerged_model/` save had keys `base_model.model.model.*` plus `lora_A/lora_B`
  tensors (PEFT wrapper saved as-is). vLLM rejects that; `merge_adapter.py`
  applies the adapter from `verl_output/global_step_N/actor/lora_adapter/` to the
  base (verified: merged == base + BA*alpha/r to bf16 precision). A true full-FT
  save has plain `model.*` keys and loads directly.
* `verl_output/` (optimizer + FSDP shards per saved step; 111 GB for the full-FT
  run, steps 50-250) and `logs/`, `telemetry/` are training state, not needed for
  eval. `telemetry/training_config.yaml` is the record of hyper-parameters;
  `rank_0000.log` has no reward metrics (those went to wandb / tensorboard).
  The top-level `model.safetensors` is the final save. **It is bit-identical to
  `verl_output/global_step_250`** (all 2012 tensors, max abs diff 0), so the
  "final" model is the step-250 weights, not step 280. Only steps 200 and 250
  still have actor weights (`max_actor_ckpt_to_keep: 2`); `merge_verl_ckpt.py`
  converts them to HF (patches verl's merger, which chokes on Gemma 4's 0-dim
  audio buffers).
* Full-FT weight change vs base is small in norm (~4-6e-4 relative on
  language-model matrices, 0 on the vision/audio towers), so do not judge
  "did it move" from weights; judge it from the eval.

## Data

* `medqa_test_1k.jsonl` – 1000 RaR-Medicine **test** prompts in oumi conversation
  format: the exact training system prompt + question, reference answer in
  `metadata`, no per-question rubric. Used by `eval_test1k.py`.
* `anisha2102/RaR-Medicine` **val** split (64 seeded prompts, with rubrics) –
  used by `run_pipeline.sh` via `../rar_medicine_judges/generate_rollouts.py`.
  The RL runs trained on the `train` split and validated on `val` (256 rows for
  the LoRA run, 1000 for full-FT), so `val` is not fully held out; `test` is.

## Judges

| role | variant | model | why |
|---|---|---|---|
| training reward | `variant_b_training_reward` | gpt-4.1-mini | exact prompt / model / T=0 used as the GRPO reward: holistic 0-10 meta-rubric vs the reference answer, reward = score/10 |
| held-out | `fixed_rubric_implicit` | gpt-4.1 | same meta-rubric, stronger model (usable on the test file) |
| held-out | `rubric_explicit` | gemma-4-E4B-it (local) / gpt-4.1 | per-question dataset rubric, per-item verdicts; different prompt and (for gemma) model family (val set only) |

`variant_b_training_reward.yaml` lives in `../rar_medicine_judges/judge_prompts/`
(same system prompt as `fixed_rubric_implicit`, plain QUESTION:/REFERENCE
ANSWER:/RESPONSE: template as in the training SimpleJudge config).

## What the summaries report

`results/test1k/SUMMARY.md` (step 6): mean reward per model, paired bootstrap CI
on the delta, per-prompt win rate, 0-10 score histogram, wrong-answer rate
(score <= 3) and 9-10 rate, breakdown by question source, surface stats that
move under reward hacking (length, truncation, bold density, score/length
correlation), **answer-commitment bins** (single / repeated_identical /
conflicting / none, one fixed extraction rule, plus the base -> trained
transition matrix), **per-question flip tables** (fixed / regressed / net) for
every judge, and the largest per-prompt swings. Ambiguous responses are dumped
to `audit_ambiguous.jsonl` for manual review; audit notes go in `AUDIT.md`.

`results/COMPARE.md` (step 7): the same under training + held-out judges, plus
reference / mismatch control anchors and the **transfer ratio** = held-out gain
/ training-judge gain. A ratio well below 1 is the primary reward-hacking
signal. Reference point: gemma-4-E4B-it vs E2B-it (a model that is simply
better) gives 0.91.

## What is (not) in git

Code, judge prompts and READMEs are committed. Everything generated is
git-ignored (repo `.gitignore`): `models/`, `results/`, `logs/`,
`medqa_test_1k.jsonl`, and in `../rar_medicine_judges/` the `rollouts/`,
`judge_outputs/` and `logs/` folders. The repo's pre-commit hook rejects files
over 500 KB. To reproduce, rerun the steps above; to share results, copy
`results/test1k/` to the S3 bucket next to the checkpoints.

## Access note

The default role on this box (`staging-remote-compute-role`, account
470584372089) is denied `s3:ListBucket` and `s3:GetObject` on the bucket, so
downloads need a profile / env credentials with access, or a policy change by
the bucket owner.

## Results log

**2026-09-08, full-FT run (`medqa_gemma4-e2b-it_fullft`, step 280), greedy,
training judge gpt-4.1-mini, n=1000 test prompts:** clear gain. Mean reward
0.515 -> 0.640 (paired 95% CI [+0.104, +0.146], win rate 0.64); wrong-answer
rate (score <= 3) 57.2% -> 43.1%; perfect 10s 290 -> 450. Gains are uniform
across question sources. Surface drift is large: mean length 336 -> 664 tokens,
truncation 1.5% -> 8.6%, bold markers 2x. 47.7% of responses contain two
"The final answer is" lines, but that is almost entirely the *same* answer
restated back-to-back at the end (a learned tic, 44.8% of responses), not
hedging between alternatives. Within fullft, committed single-answer responses
score 0.71 vs 0.60 for the restated ones and 0.50 for truncated ones, so the
judge is not rewarding the tic or the length (corr(score, length) = -0.26).
Held-out check (`fixed_rubric_implicit:gpt-4.1`, same meta-rubric, stronger
model): 0.478 -> 0.575 (+0.097, CI [+0.076, +0.116], win rate 0.62). Transfer
ratio 0.78 vs 0.91 for the "simply better model" reference (E4B vs E2B), i.e.
most of the gain is real, ~20% of it is specific to the training judge. Genuine
hedging (>1 *different* final answers) rose 2.3% -> 14.0%; the judge scores
those lower, so it is a side effect, not an exploit. Truncation at 1024 tokens
(8.6%) is the most costly drift: truncated responses average 0.50 vs 0.65.

*Cross-family and frontier judges* (same meta-rubric, `fixed_rubric_implicit`,
all 1000 test prompts x 2 models; SUMMARY.md has the full table with paired CIs):

| judge family | judges | transfer ratio (held-out gain / training gain) | bias vs training judge |
|---|---|---|---|
| OpenAI, same tier as reward | gpt-4.1 | 0.77 | -0.05 |
| OpenAI frontier | gpt-5.5, gpt-5.6-luna/sol/terra | **0.55-0.60** | -0.11 to -0.14 (strictest) |
| Google | gemini-2.5-pro, gemini-3.5-flash | 0.79-0.80 | -0.10 / -0.06 |
| local, other families | gemma-4-E4B-it, Llama-3.1-8B | 0.77 / 0.79 | -0.05 / -0.02 |
| local | Qwen3-4B | 1.02 | +0.04 (most lenient) |
| Anthropic | claude-opus-5, claude-sonnet-5 | **0.41 / 0.48** (strictest of all) | -0.17 / -0.12 |

Every judge agrees the model gets *more answers right* (three fixed per one
regressed; binary kappa with the training judge 0.77-0.87 for all but Llama).
The disagreement is about *quality of a correct answer*: the frontier OpenAI
judges hand out far fewer 9-10s to the trained model (24% vs the training
judge's 49%) because its answers are long, restate the final answer, sometimes
give a combined/soft answer, and contain incidental inaccuracies (rubric E2 and
P2). gpt-4.1-mini forgives that; gpt-5.5/5.6 do not. Claude Opus 5 goes
furthest: it gives 9-10 to 18% of responses for *both* models, i.e. it credits
the trained model with more correct answers and with no style gain at all. The
ordering is monotone in judge strength: Qwen-4B sees 102% of the training-judge
gain, mid-tier judges ~78%, frontier OpenAI ~57%, Claude Sonnet 5 48%, Claude
Opus 5 41%. So roughly half of the training reward gain is judge-independent
and half is style the lenient training judge over-rewards: a mild
reward-hacking signal, not a crude length/keyword exploit.

Anthropic judges refused 3-5 pathogen-biology questions per model (safety
classifier; recorded as null, excluded from means). Claude 5 models reject
temperature, so their scores are not deterministic; gpt-5.x and Gemini reason
internally, so `--max-new-tokens 4096` is needed or the JSON gets truncated.

*Checkpoint curve* (`checkpoint_curve.py` -> `results/test1k/CURVE.md`; only
steps 200 and 250 survive `max_actor_ckpt_to_keep: 2`, and the final save is
bit-identical to step 250, so the curve is base / 200 / 250):

| step | training judge | gpt-5.6-terra | claude-sonnet-5 | transfer (5.6 / sonnet) | repeated final line | truncated | training 9-10s judged wrong by 5.6 |
|---|---|---|---|---|---|---|---|
| 0 | 0.515 | 0.399 | 0.434 | | 0.2% | 1.5% | 11 |
| 200 | 0.638 | 0.483 | 0.496 | 0.69 / 0.50 | 35% | 7.4% | 34 |
| 250 | 0.640 | 0.474 | 0.494 | 0.60 / 0.48 | 45% | 8.6% | 37 |

All of the correctness gain is in place by step 200. Between 200 and 250 the
training reward is flat (+0.002, CI [-0.013, +0.019]) while both gold judges
tick down (gpt-5.6 -0.010, CI [-0.024, +0.005]; 555/1000 responses change
score) and the surface drift keeps growing. That window is the clearest
reward-hacking evidence in the study: the policy kept moving after its reward
plateaued, and every move was neutral for its own judge and neutral-to-negative
for independent ones. If the run had kept steps 50-150, the onset could be
located; it did not.

*Judge reliability, following "Judge's Verdict" (arXiv 2510.09738):*
Krippendorff's alpha across 13 judges 0.822 (their human-human baseline was
0.80); quadratic-weighted kappa with the training judge 0.84-0.91 for every
judge except Llama-3.1-8B (0.70, binary kappa 0.49, consensus z = -2.8: drop it
as a judge). No human anchor yet; ~100 hand-graded responses would enable the
paper's actual human-likeness test.

*Per-question flips* (wrong = judge score <= 3): under the training judge
203 questions fixed, 62 regressed, net +141 (accuracy 42.8% -> 56.9%); under
the held-out judge 178 fixed, 74 regressed, net +104 (42.0% -> 52.4%).

*Answer commitment* (one fixed extraction rule, `results/test1k/AUDIT.md`):
base single 95% / repeated 2% / conflicting 2% / none 0.4%; fullft single 48% /
repeated 37% / conflicting 9% / none 6%. Manual audit of 37 fullft "conflicting"
cases found 1-2 genuine two-answer hedges; the rest are the same answer
paraphrased or a restatement of the question followed by the answer. Genuine
hedging is <= 1%. The "none" bin is almost entirely truncation at 1024 tokens.
So the formatting drift is real but does not change the accuracy picture.

Next: per-question-rubric judges on val (`run_pipeline.sh`) for a rubric-
independent held-out view, and sampled (T=1) generation to match training.

**2026-09-08, LoRA run (`unmerged_model` -> `variant_b_merged`), greedy, training
judge:** no measurable difference from base. Mean reward 0.516 vs 0.516 (paired
95% CI [-0.009, +0.010], win rate 0.50); wrong-answer rate 57.7% -> 57.0%;
length +2.5%; hedged multiple "final answer" lines 4.1% -> 6.0%. 78% of greedy
responses are >0.9 similar to the base's; first line identical on 782/1000.
LoRA delta ~0.03% of base weight norm (64 steps, lr 1e-5, KL 0.001). Nothing to
reward-hack in that checkpoint. Superseded by the full-FT run.

## Reward-hacking follow-ups (not yet implemented)

* More held-out graders in `compare.py` (`rubric_explicit_v2`, `reference_only`)
  and a human spot-check of the top-10 per-prompt gains.
* Swap-rubric test: judge each response against a *different* question's rubric;
  a policy that echoes generic rubric vocabulary scores high anyway.
