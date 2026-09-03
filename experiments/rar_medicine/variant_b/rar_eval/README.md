# RaR-Medicine meta-rubric evaluation

Scores the Variant B eval generations in [`../eval_outputs/`](../eval_outputs/) with the
**training judge itself** — [`../judge_config.yaml`](../judge_config.yaml), gpt-4.1-mini at
temperature 0, one holistic 0–10 integer per response conditioned on the sample's
`reference_answer` (the RaR-Implicit meta rubric in
[`../../META_RUBRIC.md`](../../META_RUBRIC.md)). The reported `mean_score` is
`judgment / 10`, i.e. exactly the GRPO reward, measured on 1,000 held-out test prompts.

This is the metric the policy was optimised for, so it answers "did training move the
reward on unseen prompts?" — not "is the model better at medicine?". For the latter see
[`../HEALTHBENCH_SUMMARY.md`](../HEALTHBENCH_SUMMARY.md).

## Run

```bash
bash experiments/rar_medicine/variant_b/rar_eval/run_eval.sh                  # both arms, then compare
bash experiments/rar_medicine/variant_b/rar_eval/run_eval.sh --num_samples 20 # paired pilot
python experiments/rar_medicine/variant_b/rar_eval/compare_runs.py            # compare only
```

`run_eval.sh` copies `INTERNAL_OPENAI_API_KEY` from the repo `.env` into `OPENAI_API_KEY`
(the variable the judge config reads) and runs `oumi evaluate` on `eval_base.yaml` and
`eval_trained.yaml`. No GPU is used: the configs point `responses_path` at the existing
generations and set `inference_engine: OPENAI` only because the task signature asks for
an engine and that one is free to construct. Cost is ~1,000 gpt-4.1-mini calls per arm,
under $1 each.

Pilots and interrupted runs are cheap to resume: judgments are appended to
`artifacts/<arm>/judgments.jsonl` keyed by `conversation_id`, and the harness skips
anything already there. A 20-sample pilot's calls are reused by the full run.

## Harness

`src/oumi/evaluation/registry/rar_medicine_task.py`, registered as the custom task
`rar_medicine`. Per run it:

1. loads `responses_path` (an `oumi infer` output: one `Conversation` per line with a
   final assistant turn and `metadata.reference_answer`), or generates responses from a
   prompt-only `dataset_path` with the configured engine when no file exists;
2. maps each row onto the judge's `{question}` (last user turn), `{reference_answer}`
   and `{response}` — the same mapping as `rar_medicine_grpo.judge_response`;
3. judges in resumable batches with `SimpleJudge.judge_partial`, overriding only the
   judge config's request concurrency (`judge_num_workers`); blank responses score 0
   without a call, as in training;
4. writes `sample_results.jsonl` and `summary.json`.

Only the judge's request concurrency is changed in memory; `judge_config.yaml` stays
byte-identical to what training used.

## Artifacts

Per arm, in `artifacts/{base,trained}/`:

| file | contents |
| --- | --- |
| `judgments.jsonl` | one row per conversation: `judgment`, `explanation`, raw judge JSON (the resumable cache) |
| `sample_results.jsonl` | `conversation_id`, `idx`, `question_source`, `judgment`, `score`, `response`, `explanation` |
| `summary.json` | `mean_score` ± bootstrap std, `mean_judgment`, `judgment_histogram`, `frac_correct_conclusion`, `by_question_source`, judge provenance |
| `judge_progress.json` | live counters written by the inference engine |

`compare_runs.py` joins the two `sample_results.jsonl` on `conversation_id` and writes
`artifacts/comparison.json`: paired delta with bootstrap 95% CI and paired t for the
score, the mean judgment and the "correct final conclusion" rate (judgment ≥ 4 — the
rubric caps wrong conclusions at 3), per-sample movement, the two judgment histograms,
and per-`question_source` deltas. It first checks how many trained responses are
byte-identical to base, the adapter-activity test that exposed vLLM's silent LoRA no-op
on gemma-4.

## Reading the numbers

- **The judge is nearly binary.** With a rubric that caps wrong conclusions at 3 and
  reserves 9–10 for correct, well-reasoned answers, gpt-4.1-mini mostly emits 2–3 or
  10; the 4–9 range is thin. `frac_correct_conclusion` and the histogram therefore
  carry most of the information; `mean_score` is roughly `0.1 × P(wrong) × 2.5 + P(right)`.
- **Decoder noise floor.** The base arm was decoded by vLLM and the trained arm by the
  NATIVE engine (vLLM's LoRA path is a no-op for gemma-4). On HealthBench, re-decoding
  the *same* model across those two engines moved 30% of per-sample scores by >0.10 and
  was indistinguishable from the base-vs-trained delta. Treat any small delta here the
  same way until a same-engine base run exists.
- **Judge == reward.** A gain here can be reward hacking as easily as improvement; the
  per-sample `explanation` fields and the HealthBench results are the cross-check.

## Results (2026-09-02, 1,000 paired test prompts, gpt-4.1-mini judge)

| arm | mean score | bootstrap sd | mean judgment | judgment ≥ 4 |
| --- | ---: | ---: | ---: | ---: |
| base `gemma-4-E2B-it` (vLLM, greedy) | 0.5097 | 0.0121 | 5.10 | 40.1% |
| Variant B step 64 (NATIVE + LoRA, greedy) | 0.5096 | 0.0120 | 5.10 | 39.7% |
| **paired delta** | **−0.0001** | 95% CI [−0.0096, +0.0098] | −0.001 | −0.4 pt, t = −0.53 |

The GRPO run did not move its own reward on held-out prompts. Per sample, 144 responses
scored higher, 141 lower, 715 unchanged (mean |Δ| 0.60 judgment points, sd 1.58) — the
same symmetric churn HealthBench showed. The adapter is active (9/1,000 responses
byte-identical to base, median character similarity 0.395), so this is a null result, not
a no-op.

The judgment distribution is bimodal in both arms: 2 (≈41%) and 10 (≈31%) hold most of
the mass, 4–8 together under 6%. The judge is effectively grading "is the final answer
right", which the training reward inherited. Per source: the 896 `medical-o1-reasoning-SFT`
prompts move −0.004 (CI [−0.014, +0.005]); the 91 `General/VNet` prompts move +0.042
(CI [−0.007, +0.091]) — suggestive but not significant, and the two arms were decoded by
different engines (see "Reading the numbers").

Full tables: `artifacts/comparison.json`; per-sample grades with judge explanations:
`artifacts/{base,trained}/sample_results.jsonl`.
