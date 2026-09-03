# HealthBench consolidated-rubric evaluation — summary

RaR-Medicine GRPO Variant B vs untrained `google/gemma-4-E2B-it`, 1,000 HealthBench
conversations, graded against one consolidated 15-criterion rubric by gpt-4o-mini.

Reports: [`healthbench/HEALTHBENCH_GLOBAL_REPORT_UNMERGED.md`](healthbench/HEALTHBENCH_GLOBAL_REPORT_UNMERGED.md)
(runtime adapter — the faithful policy) ·
[`healthbench/HEALTHBENCH_GLOBAL_REPORT.md`](healthbench/HEALTHBENCH_GLOBAL_REPORT.md)
(merged bf16, includes the ground-truth validation) ·
rubric [`HEALTHBENCH_RUBRIC.md`](HEALTHBENCH_RUBRIC.md)

## Headline

| serving path | base | trained | delta | 95% CI | t |
| --- | ---: | ---: | ---: | :---: | ---: |
| **NATIVE + runtime LoRA adapter** (faithful) | 0.4994 | 0.5031 | **+0.0037** | [−0.0032, +0.0106] | +1.07 |
| vLLM + merged bf16 (lossy) | 0.5015 | 0.5037 | +0.0022 | [−0.0051, +0.0097] | +0.57 |
| *difference of the two deltas* | | | +0.0016 | [−0.0086, +0.0115] | |

Neither path shows a significant effect. The unmerged path gives a delta ~1.7× the
merged path's, the direction expected if the bf16 merge attenuates a real but small
effect — but the difference between paths is itself not significant, so the merge
caveat does not change the conclusion at n = 1,000.

The adapter is definitely active: 41/1,000 responses identical to base (4.1%), median
character similarity 0.446. (This was worth checking — vLLM 0.19.1's LoRA path is a
silent no-op for Gemma-4, which is why both arms were generated with NATIVE.)

## The measurement's noise floor — the most important result

Running the **same base model** through two different decoders (vLLM vs NATIVE, both
greedy) produces:

| contrast | mean delta | sd | mean abs delta | samples moving >0.10 |
| --- | ---: | ---: | ---: | ---: |
| **same model**, different decoder → *noise* | +0.0021 | 0.1172 | 0.0814 | 29.9% |
| different model, same decoder → *"effect"* | +0.0022 | 0.1210 | 0.0814 | 29.8% |

These are indistinguishable. The rubric-plus-judge pipeline cannot tell "base vs
trained" apart from "base vs the same base re-decoded". Any future claim of an effect
on this metric has to clear this floor, and ±0.002-scale bootstrap CIs badly understate
the real uncertainty because they capture prompt sampling only.

## Corrections to earlier reporting

- **"The model changed a great deal, the changes just cancelled out" was not
  supported.** That rested on per-sample score movement (sd 0.1377, 29% of samples
  moving >0.10). The noise-floor measurement above shows pure re-decoding with *zero*
  model change produces the same movement. The movement is rephrasing noise, not
  offsetting quality changes.
- What does survive: the adapter rewrites nearly every response (within one engine,
  greedy decoding at fixed batch composition is deterministic, so the 0.446 similarity
  is attributable to the adapter). Rewriting is not the same as changing quality.
- **The "more complete, less accurate" trade does not reproduce** on the faithful path.
  The merged-path ground truth showed accuracy −0.026 (t = −2.12) against completeness
  +0.033 (t = +2.18); with the runtime adapter, accuracy is the *largest positive* axis
  (+0.0071, t = +1.72). Two things differ between those measurements (metric and
  serving path), so this is not a clean contradiction — but the earlier trade-off story
  should not be relied on.

## What still holds

- **Consolidation works as a rubric but is not a HealthBench estimate.** Against the
  true per-sample rubric on 306 samples, per-sample paired deltas correlate at
  r = −0.024 (p = 0.67). It makes per-criterion behaviour comparable across the whole
  dataset, which per-sample rubrics cannot; it does not stand in for the benchmark.
- **Generic criteria die unless calibrated.** HealthBench's own 33 sample-agnostic
  criteria score 398/436 for *both* models — bit-identical. Rubric v1 reproduced this
  (AC4 95.2%, CX3 97.5% at ceiling); the pilot gate caught it and the rewrite to grade
  degree-of-active-behaviour restored discrimination (AC4: 95% → 30% ceiling).
- **Underpowered.** MDE for the true metric at n = 1,000 is 0.0122 against an observed
  true delta of +0.0056.

## How the axis weights were derived

Splitting HealthBench's per-example score by axis gives
`score_i = sum_a w_ia * s_ia`, where `w_ia` is axis *a*'s share of example *i*'s
**positive** point mass and `s_ia` is that axis's sub-score (negative-point items live
inside `s_ia`). The single shared weight is the mean of `w_ia` over examples —
macro-averaged, because HealthBench macro-averages. Each criterion then carries
`axis_weight / criteria_in_that_axis`.

| axis | macro (used) | pooled | macro \|points\| | % examples with the axis |
| --- | ---: | ---: | ---: | ---: |
| accuracy | 0.3230 | 0.3162 | 0.3417 | 83.0% |
| completeness | 0.3969 | 0.4279 | 0.3864 | 82.9% |
| context awareness | 0.1681 | 0.1640 | 0.1554 | 66.7% |
| communication quality | 0.0684 | 0.0549 | 0.0738 | 44.5% |
| instruction following | 0.0436 | 0.0370 | 0.0427 | 20.4% |

**Known imprecision.** That decomposition is the leading term, not an identity. An axis
whose items in a given example are *all* negative-point has zero positive mass, so it
gets zero weight while its points still count in the benchmark's numerator. This affects
**34.7% of examples (1,735/5,000)**, and the omitted term reaches 0.357 on individual
samples. (Verified: restoring it makes the identity exact to 1.1e-16.)

**It does not move the result.** Recomputed from identical grades: |points|-mass weights
give +0.0038, pooled +0.0038, equal weights +0.0030, against the +0.0037 reported. With
every per-criterion delta near zero, no weighted average of them can be far from zero.
The convention *would* be load-bearing for a model that moved axes in opposing
directions — it is not for this one.

## Where things live

| what | path |
| --- | --- |
| Generations, merged path (vLLM, 5,000) | `artifacts/{base,trained}/model_responses.jsonl` |
| Generations, unmerged path (NATIVE, 1,000) | `healthbench/artifacts/native_{base,trained}/model_responses.jsonl` |
| LoRA adapter (unmerged, step 64) | `output/unmerged_model/verl_output/global_step_64/actor/lora_adapter` |
| Consolidated rubric | `healthbench/global_rubric_v2.json`, `HEALTHBENCH_RUBRIC.md` |
| Grades and summaries | `healthbench/artifacts/{global,native}_{base,trained}/` |
| Ground-truth per-item judgments | `healthbench/artifacts/{base_gemma4_e2b_it,trained_merged_model}/` |
| Harness | `src/oumi/evaluation/registry/healthbench_global_task.py`, `healthbench_common.py` |
| Scripts | `healthbench/{consolidate_rubrics,revise_rubric,validate_consolidation,make_global_report,compare_merge_paths,grade_subset}.py` |
| Eval configs | `healthbench/eval_{base,trained}_{global,native}.yaml` |
