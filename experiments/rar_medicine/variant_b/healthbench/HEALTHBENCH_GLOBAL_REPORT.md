# HealthBench with a consolidated dataset-level rubric

**google/gemma-4-E2B-it** (base) vs **/workspace/persist/shanghong/oumi/experiments/rar_medicine/variant_b/output/merged_model** (RaR-Medicine GRPO Variant B, LoRA merged), graded on 1000 HealthBench conversations.

## What this measures

HealthBench ships a bespoke physician-written rubric per example: 57237 rubric items over 5000 examples, 48568 of them unique. That makes per-criterion statistics incomparable across the dataset and costs one judge call per rubric item. Those rubrics were consolidated into **15 criteria shared by every sample**, each graded 0-4, weighted by that axis's share of positive point mass macro-averaged over examples to match HealthBench's own averaging. That is the leading term of the benchmark score's axis decomposition, not an exact reproduction of it: an axis whose items in a given example are all negative-point gets zero weight while its points still count in the benchmark's numerator, which affects 34.7% of examples. Recomputing the headline under |points|-mass, pooled or equal weights moves the delta by at most 0.0008, so the convention is not load-bearing here. Grading is one call per sample.

- Judge: `gpt-4o-mini` at temperature 0.0, rubric `v2` (sha `545c42e5b843a796`).
- Score: `sum(weight * grade / 4) / sum(weight)` per sample, averaged over samples. Both models graded by the same judge, same rubric, same prompts.

## Headline

| | base | trained | delta |
| --- | ---: | ---: | ---: |
| Consolidated rubric score | 0.5015 | 0.5037 | **+0.0022** |
| 95% CI on delta (paired bootstrap) | | | [-0.0051, +0.0097] |
| t (paired) | | | +0.57 |

The 95% CI **includes zero**: on this metric the two models are not distinguishable.

## Bottom line

1. **Neither metric separates the two checkpoints.** The consolidated rubric gives +0.0022 (95% CI [-0.0051, +0.0097], permutation p = 0.57); the true per-sample rubric gives +0.0056 (t = +0.71) on the 306 samples where it exists. On the evidence here, GRPO Variant B did not measurably change HealthBench performance.

2. **The consolidated rubric should not be read as a HealthBench estimate.** Per-sample paired differences of the two metrics correlate at r = -0.024 (p = 0.67) -- essentially zero -- and on the validation subset the consolidated delta points the opposite way (recovery ratio -0.51). The difference-of-deltas CI [-0.0286, +0.0124] is too wide to call it biased, but it is also too wide to certify it. Treat the consolidated score as an internally consistent measure of medical-response quality that makes per-criterion behaviour comparable across the dataset -- which per-sample rubrics cannot do -- and not as a stand-in for the benchmark.

3. **What would actually settle it.** Both metrics sit near their detection threshold, so the limit is statistical power, not the rubric. Grading each sample against its own rubric in one batched call (~11x cheaper than the per-item run) would extend the faithful benchmark to all 5,000 samples; and the trained policy should be served as NATIVE + runtime LoRA adapter rather than the lossy merged bf16 checkpoint graded here.

## Per-axis decomposition

The scalar hides the mechanism. Axis deltas of opposite sign mean the training moved different capabilities in different directions, and the headline is only their weighted residue.

| axis | weight | base | trained | delta | t |
| --- | ---: | ---: | ---: | ---: | ---: |
| accuracy | 0.323 | 0.4756 | 0.4770 | +0.0014 | +0.32 |
| completeness | 0.397 | 0.6056 | 0.6083 | +0.0028 | +0.48 |
| context awareness | 0.168 | 0.1415 | 0.1461 | +0.0046 | +1.02 |
| communication quality | 0.068 | 0.7608 | 0.7612 | +0.0005 | +0.09 |
| instruction following | 0.044 | 0.7270 | 0.7231 | -0.0039 | -0.57 |

No axis shows a significant move in both directions.

## Per-criterion

| id | axis | criterion | base | trained | delta | t | ceiling |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| AC1 | accuracy | Evidence-based Recommendations | 1.72 | 1.73 | +0.007 | +0.30 | 7% |
| AC2 | accuracy | Accurate Medical Information | 2.89 | 2.90 | +0.004 | +0.13 | 44% |
| AC3 | accuracy | Accurate Dosage and Treatment Details | 0.75 | 0.74 | -0.009 | -0.35 | 1% |
| AC4 | accuracy | Proactive Harm Mitigation | 2.25 | 2.27 | +0.021 | +0.68 | 30% |
| CP1 | completeness | Identify Red Flags | 1.94 | 1.96 | +0.021 | +0.64 | 30% |
| CP2 | completeness | Provide Actionable Steps | 2.75 | 2.75 | +0.003 | +0.11 | 37% |
| CP3 | completeness | Address Safety Concerns | 2.27 | 2.33 | +0.050 | +1.56 | 31% |
| CP4 | completeness | Omission of Critical Information | 2.73 | 2.70 | -0.030 | -0.87 | 44% |
| CX1 | context awareness | Seek and Use Context | 0.14 | 0.17 | +0.031 | +1.62 | 1% |
| CX2 | context awareness | Tailor to User's Context | 1.25 | 1.25 | +0.003 | +0.09 | 7% |
| CX3 | context awareness | Proactive Contextual Inquiry | 0.31 | 0.33 | +0.021 | +0.83 | 1% |
| CM1 | communication quality | Clarity and Structure | 2.86 | 2.83 | -0.027 | -1.02 | 42% |
| CM2 | communication quality | Tone and Register | 3.23 | 3.26 | +0.031 | +1.04 | 60% |
| IF1 | instruction following | Adherence to Task and Format | 2.66 | 2.63 | -0.026 | -0.82 | 41% |
| IF2 | instruction following | Relevance and Language Consistency | 3.16 | 3.15 | -0.005 | -0.16 | 57% |

Grades are means on the 0-4 scale. `ceiling` is the share of all gradings at the top grade, pooled over both models: a criterion near 100% there cannot separate models, which is why the rubric was gated on it.

## By HealthBench theme

| theme | n | base | trained | delta |
| --- | ---: | ---: | ---: | ---: |
| hedging | 229 | 0.5151 | 0.5086 | -0.0065 |
| global_health | 209 | 0.5001 | 0.4921 | -0.0080 |
| communication | 180 | 0.4854 | 0.4912 | +0.0059 |
| context_seeking | 111 | 0.5181 | 0.5187 | +0.0007 |
| health_data_tasks | 109 | 0.4041 | 0.4267 | +0.0226 |
| emergency_referrals | 96 | 0.6212 | 0.6367 | +0.0155 |
| complex_responses | 66 | 0.4618 | 0.4656 | +0.0038 |

## Does this track real HealthBench?

On the 306 samples where the full per-sample rubric was also graded (one GPT-4o call per rubric item, the faithful benchmark):

| metric | base | trained | delta | 95% CI | t |
| --- | ---: | ---: | ---: | :---: | ---: |
| True HealthBench | 0.4086 | 0.4142 | +0.0056 | [-0.0106, +0.0208] | +0.71 |
| Consolidated rubric | 0.5197 | 0.5169 | -0.0029 | [-0.0157, +0.0098] | -0.44 |

- **Difference of deltas** (consolidated - true): -0.0085, 95% CI [-0.0286, +0.0124] -> no detectable bias.
- **Delta recovery ratio**: -0.51.
- **Agreement on per-sample paired differences**: Pearson r=-0.024 (p=0.673), Spearman rho=-0.017 (p=0.761).
- Score-level correlation r=+0.071, reported only for completeness: levels are dominated by shared prompt difficulty and flatter any metric, so the delta agreement above is the real test.

- **Minimum detectable effect** for the true metric at n=1000 (80% power): 0.0122, against an observed true delta of +0.0056.

## Controls

- **Negative control** (base model's own responses split into two random halves): delta -0.0036, null spread [-0.0247, +0.0238]. The metric does not manufacture a difference from sampling alone.
- **Label-permutation test** (10000 sign flips of the paired difference): p = 0.5671.

## Caveats

- **The trained checkpoint is the merged bf16 model.** This repo's own `eval_configs/infer_trained.yaml` records that only ~82% of the LoRA adapter delta norm survives the bf16 merge, and that vLLM 0.19.1's LoRA path is a silent no-op for Gemma-4. The faithful policy is NATIVE + runtime adapter; these numbers grade an approximation of it. Base and trained responses do differ on 4,822 of 5,000 prompts, so the merge is not a no-op.
- **The consolidated rubric is a proxy, not HealthBench.** It shares the benchmark's prompts, axes and weighting, but its criteria are synthesised abstractions, so a score here is not comparable to a published HealthBench number. The validation section bounds how far the two diverge.
- **Judge noise is not in the confidence intervals.** The bootstrap CI captures sampling variation over the 1000 prompts only. The judge's own test-retest variability is unmeasured here and is a separate error term.
- **Criterion selection used a 200-sample pilot** drawn from the same dataset. Two criteria were rewritten after they graded 95%+ of responses at the ceiling for both models; none were selected or dropped on the basis of which model they favoured.

Artifacts: `/workspace/persist/shanghong/oumi/experiments/rar_medicine/variant_b/healthbench/artifacts/global_base` and `/workspace/persist/shanghong/oumi/experiments/rar_medicine/variant_b/healthbench/artifacts/global_trained` (`criterion_grades.jsonl`, `sample_results.jsonl`, `summary.json`); rubric `global_rubric_v2.json`; every grade row is stamped with judge model, engine, temperature and prompt hash, and aggregation refuses to mix them.
