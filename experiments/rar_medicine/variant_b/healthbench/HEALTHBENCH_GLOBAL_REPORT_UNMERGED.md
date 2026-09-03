# HealthBench with a consolidated dataset-level rubric

**google/gemma-4-E2B-it** (base) vs **google/gemma-4-E2B-it** (RaR-Medicine GRPO Variant B, LoRA merged), graded on 1000 HealthBench conversations.

## What this measures

HealthBench ships a bespoke physician-written rubric per example: 57237 rubric items over 5000 examples, 48568 of them unique. That makes per-criterion statistics incomparable across the dataset and costs one judge call per rubric item. Those rubrics were consolidated into **15 criteria shared by every sample**, each graded 0-4, weighted by that axis's share of positive point mass macro-averaged over examples to match HealthBench's own averaging. That is the leading term of the benchmark score's axis decomposition, not an exact reproduction of it: an axis whose items in a given example are all negative-point gets zero weight while its points still count in the benchmark's numerator, which affects 34.7% of examples. Recomputing the headline under |points|-mass, pooled or equal weights moves the delta by at most 0.0008, so the convention is not load-bearing here. Grading is one call per sample.

- Judge: `gpt-4o-mini` at temperature 0.0, rubric `v2` (sha `545c42e5b843a796`).
- Score: `sum(weight * grade / 4) / sum(weight)` per sample, averaged over samples. Both models graded by the same judge, same rubric, same prompts.

## Headline

| | base | trained | delta |
| --- | ---: | ---: | ---: |
| Consolidated rubric score | 0.4994 | 0.5031 | **+0.0037** |
| 95% CI on delta (paired bootstrap) | | | [-0.0032, +0.0106] |
| t (paired) | | | +1.07 |

The 95% CI **includes zero**: on this metric the two models are not distinguishable.

## Bottom line

Neither the consolidated metric nor a ground-truth comparison separates these two checkpoints.

## Per-axis decomposition

The scalar hides the mechanism. Axis deltas of opposite sign mean the training moved different capabilities in different directions, and the headline is only their weighted residue.

| axis | weight | base | trained | delta | t |
| --- | ---: | ---: | ---: | ---: | ---: |
| accuracy | 0.323 | 0.4740 | 0.4811 | +0.0071 | +1.72 |
| completeness | 0.397 | 0.6026 | 0.6063 | +0.0037 | +0.66 |
| context awareness | 0.168 | 0.1415 | 0.1407 | -0.0008 | -0.20 |
| communication quality | 0.068 | 0.7586 | 0.7559 | -0.0028 | -0.47 |
| instruction following | 0.044 | 0.7200 | 0.7276 | +0.0076 | +1.12 |

No axis shows a significant move in both directions.

## Per-criterion

| id | axis | criterion | base | trained | delta | t | ceiling |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| AC1 | accuracy | Evidence-based Recommendations | 1.69 | 1.74 | +0.049 | +2.07 | 7% |
| AC2 | accuracy | Accurate Medical Information | 2.88 | 2.90 | +0.025 | +0.90 | 44% |
| AC3 | accuracy | Accurate Dosage and Treatment Details | 0.75 | 0.76 | +0.003 | +0.13 | 1% |
| AC4 | accuracy | Proactive Harm Mitigation | 2.26 | 2.29 | +0.036 | +1.20 | 30% |
| CP1 | completeness | Identify Red Flags | 1.92 | 1.95 | +0.021 | +0.63 | 30% |
| CP2 | completeness | Provide Actionable Steps | 2.74 | 2.74 | +0.004 | +0.15 | 36% |
| CP3 | completeness | Address Safety Concerns | 2.27 | 2.30 | +0.027 | +0.87 | 30% |
| CP4 | completeness | Omission of Critical Information | 2.71 | 2.71 | +0.007 | +0.22 | 44% |
| CX1 | context awareness | Seek and Use Context | 0.14 | 0.15 | +0.001 | +0.06 | 1% |
| CX2 | context awareness | Tailor to User's Context | 1.27 | 1.26 | -0.006 | -0.19 | 7% |
| CX3 | context awareness | Proactive Contextual Inquiry | 0.28 | 0.28 | -0.005 | -0.21 | 1% |
| CM1 | communication quality | Clarity and Structure | 2.83 | 2.82 | -0.017 | -0.64 | 41% |
| CM2 | communication quality | Tone and Register | 3.23 | 3.23 | -0.005 | -0.17 | 60% |
| IF1 | instruction following | Adherence to Task and Format | 2.64 | 2.64 | -0.001 | -0.03 | 41% |
| IF2 | instruction following | Relevance and Language Consistency | 3.12 | 3.18 | +0.062 | +2.02 | 57% |

Grades are means on the 0-4 scale. `ceiling` is the share of all gradings at the top grade, pooled over both models: a criterion near 100% there cannot separate models, which is why the rubric was gated on it.

## By HealthBench theme

| theme | n | base | trained | delta |
| --- | ---: | ---: | ---: | ---: |
| hedging | 229 | 0.4960 | 0.5109 | +0.0149 |
| global_health | 209 | 0.4989 | 0.4994 | +0.0006 |
| communication | 180 | 0.4880 | 0.4832 | -0.0048 |
| context_seeking | 111 | 0.5246 | 0.5193 | -0.0053 |
| health_data_tasks | 109 | 0.4129 | 0.4030 | -0.0099 |
| emergency_referrals | 96 | 0.6190 | 0.6398 | +0.0208 |
| complex_responses | 66 | 0.4701 | 0.4815 | +0.0114 |

## Does this track real HealthBench?

Not available for this run. The ground-truth per-sample-rubric judgments were produced against a **different set of responses** (the vLLM / merged-bf16 generations). Comparing them with grades of these responses would match on prompt_id while silently comparing two different models' answers. The validation lives in the merged-path report; `compare_merge_paths.py` compares the two serving paths directly.
## Controls

- **Negative control** (base model's own responses split into two random halves): delta -0.0161, null spread [-0.0240, +0.0246]. The metric does not manufacture a difference from sampling alone.
- **Label-permutation test** (10000 sign flips of the paired difference): p = 0.2889.

## Caveats

- **The trained checkpoint is the merged bf16 model.** This repo's own `eval_configs/infer_trained.yaml` records that only ~82% of the LoRA adapter delta norm survives the bf16 merge, and that vLLM 0.19.1's LoRA path is a silent no-op for Gemma-4. The faithful policy is NATIVE + runtime adapter; these numbers grade an approximation of it. Base and trained responses do differ on 4,822 of 5,000 prompts, so the merge is not a no-op.
- **The consolidated rubric is a proxy, not HealthBench.** It shares the benchmark's prompts, axes and weighting, but its criteria are synthesised abstractions, so a score here is not comparable to a published HealthBench number. The validation section bounds how far the two diverge.
- **Judge noise is not in the confidence intervals.** The bootstrap CI captures sampling variation over the 1000 prompts only. The judge's own test-retest variability is unmeasured here and is a separate error term.
- **Criterion selection used a 200-sample pilot** drawn from the same dataset. Two criteria were rewritten after they graded 95%+ of responses at the ceiling for both models; none were selected or dropped on the basis of which model they favoured.

Artifacts: `artifacts/native_base` and `artifacts/native_trained` (`criterion_grades.jsonl`, `sample_results.jsonl`, `summary.json`); rubric `global_rubric_v2.json`; every grade row is stamped with judge model, engine, temperature and prompt hash, and aggregation refuses to mix them.
