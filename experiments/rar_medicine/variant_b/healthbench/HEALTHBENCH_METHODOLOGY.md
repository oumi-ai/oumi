# Consolidating HealthBench's per-sample rubrics into a dataset-level rubric

Methodology and design record. Companion to
[`HEALTHBENCH_SUMMARY.md`](HEALTHBENCH_SUMMARY.md) (results),
[`HEALTHBENCH_RUBRIC.md`](HEALTHBENCH_RUBRIC.md) (the criteria), and the generated
reports under [`healthbench/`](healthbench/).

---

## 1. Problem

HealthBench ships a bespoke, physician-written rubric for **every** conversation:
5,000 examples, 57,237 rubric items, 48,568 of them textually unique. An example's
score is

```
score_i = (Σ points of met rubric items) / (Σ positive points in example i)
```

and the dataset score is the mean of `score_i`, clipped to [0, 1].

Two consequences follow, and they are the reason for this work:

1. **No criterion is shared between examples**, so there is no such thing as a
   dataset-wide pass rate for any criterion. "The model got better at flagging red
   flags" is not an expressible claim, let alone a testable one.
2. **Grading costs one judge call per rubric item** — 57,237 per model. An earlier run
   reached ~6–8% in an hour and stalled.

The goal: one rubric shared by all samples, so grading is one call per sample and each
criterion has a distribution over the whole dataset.

**The transformation, stated plainly.** HealthBench asks *"did this response satisfy the
20 specific things a physician wanted for **this** conversation?"* The consolidated
rubric asks *"how well does this response do on the 15 things physicians ask for across
**the corpus**?"* Everything below is how those 15 were derived, calibrated, and
checked.

---

## 2. Corpus structure (measured, not assumed)

| quantity | value |
| --- | ---: |
| examples | 5,000 |
| rubric items | 57,237 (39,662 positive-point, 17,575 negative) |
| unique criterion strings | 48,568 |
| `level:example` items | 49,184 → 48,529 unique |
| `level:cluster` items | 8,053 → **33 unique**, across 37 cluster tags |
| items with no axis tag | 0 / 57,237 |
| points range | example items −10…+10; **cluster items always exactly +5** |

Every item carries exactly one `axis:` tag from {accuracy, completeness,
context_awareness, communication_quality, instruction_following}.

**The `level:cluster` layer matters.** HealthBench groups conversations into clusters
sharing a theme *and* a situational condition
(`cluster:emergency_referrals_emergent_emergency_behavior`,
`cluster:hedging_any-reducible-uncertainty_seeks_context`), and applies the same
criteria to every example in a cluster. Those 33 strings **are already the artifact this
project set out to build**: a small, sample-agnostic, shared rubric. That made them a
free pilot, and the pilot failed — see D4.

---

## 3. Design decisions

Each records the decision, what else was considered, and the evidence.

### D1 — Grade 0–4, not binary

**Decision.** Every criterion is graded on an integer 0–4 scale with written anchors for
0, 2 and 4. HealthBench itself is binary (met / not met).

**Why.** Of the 306 examples fully judged under the real rubrics for both models, 204
carry cluster items → 436 cluster gradings. **Base met 398. Trained met 398.** Not
close — identical, at 91.3%. HealthBench's own sample-agnostic criteria have *zero*
discriminating power between these two checkpoints, because a competent
instruction-tuned model satisfies nearly all of them. A binary consolidated rubric would
have reproduced that and reported a delta of zero regardless of the models compared.

Grading 0–4 creates headroom above "acceptable" so that two competent models can be
separated. The cost is a departure from the benchmark's scoring semantics, which is one
reason the result is never reported as a HealthBench number (§7).

### D2 — Axis weights from macro-averaged positive point mass

**Decision.** `w_a = mean over examples of (axis a's share of that example's positive
point mass)`, normalised. Each criterion then carries `w_a / (criteria in axis a)`.

| axis | macro (used) | pooled | macro \|points\| | % examples with the axis |
| --- | ---: | ---: | ---: | ---: |
| accuracy | 0.3230 | 0.3162 | 0.3417 | 83.0% |
| completeness | 0.3969 | 0.4279 | 0.3864 | 82.9% |
| context awareness | 0.1681 | 0.1640 | 0.1554 | 66.7% |
| communication quality | 0.0684 | 0.0549 | 0.0738 | 44.5% |
| instruction following | 0.0436 | 0.0370 | 0.0427 | 20.4% |

**Why macro, not pooled.** Splitting the HealthBench score by axis gives
`score_i = Σ_a w_ia · s_ia`, and the benchmark macro-averages over examples, so the
matching single weight is the mean of `w_ia` — not the corpus-wide ratio. They differ
most where axis coverage is uneven: communication quality appears in 44.5% of examples
and is 25% heavier under macro, because macro gives every example an equal vote instead
of letting rubric-dense examples dominate.

**Known imprecision.** That decomposition is its **leading term, not an identity**. An
axis whose items in a given example are *all* negative-point has zero positive mass, so
`w_ia = 0` while its points still count in the benchmark's numerator. This affects
**34.7% of examples (1,735/5,000)**, and the omitted term reaches 0.357 on individual
samples. Restoring it makes the identity exact to 1.1e-16, confirming that is the whole
gap.

**Sensitivity — it is not load-bearing here.** Recomputed from identical grades:

| convention | delta | t |
| --- | ---: | ---: |
| macro positive-mass (used) | +0.0037 | +1.07 |
| macro \|points\|-mass | +0.0038 | +1.09 |
| pooled positive-mass | +0.0038 | +1.06 |
| equal weights | +0.0030 | +0.86 |

At most 0.0008 of movement. When every per-criterion delta sits near zero, no weighted
average of them can be far from zero. **The convention would be load-bearing for a model
that moved axes in opposing directions** — this is a property of this result, not a
general reassurance.

### D3 — Negatives stay inside their axis; three criteria are harm-avoidance

**Decision.** Negative-point items are not given their own axis or a separate safety
score. They are folded into the axis they belong to, and three of the fifteen criteria
are written as harm-avoidance (grade 4 = fully avoided, 0 = clearly exhibited),
allocated to the top three axes by negative point mass (accuracy .381, completeness
.385, context .121).

**Why.** Negatives carry roughly half the observable signal: on the ground-truth
samples, the negative-item met rate moves −1.06pp between models while the positive-item
rate moves +0.82pp. Splitting them into a separate metric would discard half the effect;
merging them into a positive accuracy criterion would blend two very different base
rates.

**Consequence.** Because grades are bounded at 0, a sample cannot score negative, unlike
real HealthBench where 9–11% of per-example scores are below zero and one reaches −0.63.
This is a deliberate departure that keeps the score in [0, 1] and interpretable.

### D4 — Cluster criteria used for wording only, never for coverage

**Decision.** The 33 `level:cluster` strings seed the synthesis prompts as style
exemplars, but are not allowed to determine which criteria exist.

**Why.** Their axis distribution is close to the inverse of the corpus: 43% accuracy,
25% context, 21% communication, and only **4% completeness** — while completeness is the
single heaviest axis at 0.397. They also contain **zero** negative-point items, so they
offer no template for harm-avoidance. Seeding coverage from them would starve the
heaviest axis and omit half the signal.

### D5 — 15 criteria, allocated 4 / 4 / 3 / 2 / 2

**Decision.** Total 15, proportional to axis weight, clamped to [2, 4], remainder
assigned by largest unmet demand (`raw − allocated`).

**Why 15.** It has to fit in one prompt alongside the conversation and still be graded
attentively. The rubric block runs ~1.8k tokens; the anchors are what make it long.
Fewer than ~10 loses diagnostic resolution; well past ~20 the later items degrade as the
instruction block starts to compete with the response for attention.

**Why clamped.** The floor keeps a diagnostic foothold on light axes;
the cap stops the heaviest axis fragmenting into near-duplicates. Since influence is
carried by the weights, clamping changes granularity only, not the score's composition.

An earlier version distributed the remainder by largest *fractional part*, which handed
the spare slot to instruction-following (weight 0.044) over context-awareness (0.168) —
the fractional part of an axis already pushed up to the floor says nothing about need.
Fixed to unmet demand.

### D6 — One judge call per sample, keyed by criterion id

**Decision.** One call grades all 15 criteria. The judge writes per-criterion evidence
first into `explanation`, then emits `judgment` as `AC1=3,AC2=2,…`.

**Why one call.** 1,000 calls per model instead of 15,000, and the conversation is sent
once rather than fifteen times — the dominant token cost. The shared rubric block is
placed **ahead** of the conversation so the ~1.8k-token prefix is byte-identical across
every call and eligible for provider prompt caching.

**Why id-keyed, not positional.** A dropped, duplicated or reordered criterion becomes a
parse error instead of silently shifting every grade after it.

**Bias mitigations.** Per-criterion evidence *before* the verdict (a global preamble is
where halo bias is manufactured — the prompt bans one); fixed criterion order, identical
for both models, so position bias becomes a constant per-criterion offset that cancels in
the paired delta. Randomising order per sample was considered and rejected: it converts a
cancelling systematic bias into added variance, which is the wrong trade at this effect
size.

**Accepted risk.** Batching invites halo and anchoring effects that independent calls
avoid. Not separately quantified; §8.

### D7 — Freeze the rubric to a versioned artifact

**Decision.** Synthesis output is written to `global_rubric_v{1,2}.json` with provenance
(dataset SHA-256, item counts, point masses, synthesis model) and committed. Evaluation
loads the file; it never re-synthesises.

**Why.** Otherwise the measuring instrument changes between runs and no two numbers are
comparable.

### D8 — Judge provenance stamped on every grade, mixing refused

**Decision.** Every grade row carries judge model, engine, temperature and a hash of the
prompt template. Aggregation raises rather than combining rows that disagree.

**Why.** Two judge endpoints with documented failover exist in this project
(`judge_gpt4o.yaml`, `judge_gpt4o_openrouter.yaml`), and the resume cache makes silent
mixing easy. Grading one model through one endpoint and the other through the other would
manufacture a difference larger than anything being measured.

### D9 — Resume cache keyed by prompt_id, not position

**Decision.** The grade cache keys on `prompt_id`.

**Why.** `num_samples` below the dataset size draws a *random* subset, so row N of a
200-sample pilot and row N of a 1,000-sample run are different prompts. A positional
cache would serve one prompt's grades for another. The same keying lets one expensive
generation pass serve every later subset.

### D10 — Both arms on the same inference engine

**Decision.** For the headline comparison, base and trained are generated with the same
engine (NATIVE), so the LoRA adapter is the only difference.

**Why.** The merged bf16 checkpoint loses ~18% of the adapter delta norm to rounding
(85% of merged elements come back equal to base), so it serves a different policy than
the runtime adapter. Regenerating only the trained arm on a different engine would have
swapped one confound for another — and §6.3 shows the engine swap alone moves scores as
much as the model change does. vLLM + runtime adapter is not an option: vLLM 0.19.1's
LoRA path is a silent no-op for `Gemma4ForConditionalGeneration`, bit-identical to base
even at 100× alpha.

---

## 4. Pipeline

| stage | what happens | key parameters | code |
| --- | --- | --- | --- |
| 1 · Bucket | dedupe by (text, axis, sign) → 48,568 records with counts and themes; split into 5 axes × 2 signs | — | `consolidate_rubrics.py: analyze_corpus` |
| 2 · Weight | macro-averaged positive-mass share per axis (D2) | — | `analyze_corpus` |
| 3 · Allocate | 15 criteria across axes; 3 harm-avoidance slots by negative mass (D5, D3) | `TOTAL_CRITERIA=15`, clamp [2,4] | `allocate_criteria`, `allocate_harm_criteria` |
| 4 · Map | sample 10,000 criteria proportional to bucket size, theme-stratified round-robin; batch 200 → 55 GPT-4o calls, 6 candidates each → **330 candidates** | `--sample-total 10000 --batch-size 200 --per-batch-target 6` | `map_stage` |
| 5 · Reduce | one GPT-4o call per axis over its candidates, with axis definition, target count, harm quota, calibration instruction, distinctness requirement → final criteria with 0/2/4 anchors | — | `reduce_stage` |
| 6 · Freeze | ids, weights, provenance → `global_rubric_v1.json` | — | `assemble_rubric` |
| 7 · Gate | grade 200 samples × 2 models; per-criterion saturation and discrimination (§5) | thresholds in §5 | `validate_consolidation.py gate` |
| 8 · Revise | rewrite failing criteria using their observed grade histogram + sibling criteria → `global_rubric_v2.json`; re-gate | — | `revise_rubric.py` |
| 9 · Score | one judge call per sample; weighted mean | `judge_batch_size 250` | `healthbench_global_task.py` |

Cost of synthesis: 60 GPT-4o calls, 363k input / 14k output tokens (~$1).

**Scoring.**

```
per-sample  score = Σ_c (weight_c · grade_c / 4) / Σ_c weight_c     ∈ [0,1]
dataset     score = mean over samples          (Σ weight_c = 1.0000)
```

The denominator is now **constant across samples**, which is precisely what makes
per-criterion rates comparable dataset-wide — and also what removes HealthBench's
per-example weighting. That trade is the subject of §6.

---

## 5. Calibration (internal): the saturation gate

This is what the rubric was calibrated *against*. It uses **no information from the
per-sample rubric** — only the distribution of the rubric's own grades.

**Procedure.** Grade 200 samples for both models. For each criterion, pool all 400
gradings and compute the ceiling rate (share at grade 4), floor rate (share at 0),
standard deviation, and the base-vs-trained paired *t*. Pool across models deliberately:
a criterion only earns its weight if it varies across the responses being compared.

**Rule.** A criterion fails if it is pinned at one end of the scale
(ceiling > 0.90, floor > 0.90, or sd < 0.40) **and** fails to separate the models
(|t| < 1.0). Separately, any criterion pair correlating above |r| = 0.90 fails as
redundant, since each carries its own weight and would double-count one behaviour.

**Why the conjunction.** The rule originally failed on extremeness alone. That flagged
`CX1` (92% at floor) — which had the *highest* t in the entire rubric at +3.03. A rare
behaviour can be the signal precisely because it is rare: if the few responses that
exhibit it concentrate in one model, the criterion discriminates well despite an extreme
marginal distribution. The purpose of the gate is to remove criteria that cannot
contribute, not ones that are rare-but-informative. Note this rule was refined *after*
seeing data; the decision it changed was whether to *delete* an existing criterion, and
keeping it is the null action, so no criterion's content was selected on outcome.

**v1 result — FAIL.**

| criterion | ceiling (base / trained / pooled) | t | verdict |
| --- | ---: | ---: | --- |
| `AC4` Avoidance of harmful practices | 95.0% / 95.5% / **95.2%** | +0.00 | SATURATED |
| `CX3` Avoid unsupported assumptions | 98.0% / 97.0% / **97.5%** | −1.00 | SATURATED |

Both models saturated near-identically — the exact failure D1 predicted from the cluster
criteria.

**Revision.** Each failing criterion was rewritten by one GPT-4o call carrying its
observed grade histogram and the other 14 criteria (added after a first attempt drifted
onto ground `CX1` already covered). The instruction: the criterion names a *rare fault*,
so re-scope it onto the **degree of active management** of the same concern — grade 4
must require something only a minority of responses do, grade 2 the typical response that
merely avoids the fault, grade 0 the fault present.

**v2 result — PASS.** `AC4` moved from 95.2% ceiling / t = 0.00 to **29.8% ceiling, sd
1.37**. No criterion saturated, floored, variance-free, or redundant. `AC4`~`CP3`
correlate at r = 0.87, below the 0.90 line but close enough to note (§8).

---

## 6. Validation (external): agreement with the per-sample rubric

### 6.1 What is being asked, and what was *not* done

The per-sample rubric was **never a fitting target**. No weight, criterion, or anchor was
tuned to improve agreement with it. Correlation was computed once, after the rubric was
frozen, as a check. Fitting the consolidated rubric to the true score on the same 306
samples would produce an in-sample fit with no evidential value, and would require a
held-out split the ground truth is too small to afford. §9 records what that would take.

**Data.** 306 samples have both metrics: fully judged per-sample rubrics for *both*
models (the expensive run reached sample 306 for base and 396 for trained — the
comparison intersects on the 306; comparing 306 against 396 at this effect size would be
an artifact by itself), plus consolidated grades, topped up for exactly those prompt_ids
via `grade_subset.py`.

### 6.2 Estimand: paired deltas, not score levels

The claim under test is *"the trained model beats the base model"*, so the quantity to
validate is the **paired difference**, not the level. Level correlation is dominated by
prompt difficulty, which any two metrics share, and flatters everything. Both are
reported below, but they answer different questions and — as it turns out — have
opposite reliability properties.

### 6.3 Measurement-noise model

Interpreting any correlation requires knowing how much of the variance is noise. That is
measurable here, using a control whose true answer is exactly zero: **the same base model
decoded by two different engines** (vLLM and NATIVE, both greedy).

The control is well matched to the contrast of interest — it changes the response text by
about as much as the adapter does (median character similarity 0.433 cross-engine vs
0.446 base-vs-adapter), while changing the model not at all.

| contrast | mean Δ | sd | mean \|Δ\| | moving >0.10 |
| --- | ---: | ---: | ---: | ---: |
| **same model**, different decoder — *noise* | +0.0021 | 0.1172 | 0.0814 | 29.9% |
| different model, same decoder — *"effect"* | +0.0022 | 0.1210 | 0.0814 | 29.8% |

Indistinguishable. Implied noise for a **single** score: sd **0.0829**
(= 0.1172/√2).

### 6.4 Reliability and the attenuation ceiling

Observed correlations are attenuated by measurement error:
`r_observed = r_true · √(reliability₁ · reliability₂)`. With reliability
`= (observed variance − noise variance) / observed variance`:

| quantity | observed sd | noise sd | reliability | max observable \|r\| |
| --- | ---: | ---: | ---: | ---: |
| per-sample **level** | 0.1876 | 0.0829 | **0.805** | 0.897 (if truth perfect) / 0.805 (if equal) |
| per-sample **delta** | 0.1139 | 0.1172 | **≈ 0.000** | ≈ 0 |

This is the central methodological finding of the validation, and it splits the result in
two:

- **Levels are reliably measured** (0.805). A low level correlation is therefore real
  information.
- **Per-sample deltas are pure noise.** The expected noise sd (0.1172) *exceeds* the
  observed delta sd (0.1139). There is no detectable per-sample signal at all, in either
  direction, so a per-sample delta correlation **cannot** succeed regardless of how well
  the rubrics agree. No sample size fixes this; only reducing per-sample noise does.

### 6.5 Results

| test | statistic | reading |
| --- | --- | --- |
| dataset-level delta, true rubric | +0.0056, 95% CI [−0.0106, +0.0208], t = +0.71 | n.s. |
| dataset-level delta, consolidated | −0.0029, 95% CI [−0.0157, +0.0098], t = −0.44 | n.s. |
| **difference of deltas** | −0.0085, 95% CI [−0.0286, +0.0124] | contains 0 → no detectable bias, but far too wide to certify |
| delta recovery ratio | −0.51 | sign not recovered on this subset |
| per-sample **delta** correlation | Pearson r = −0.024 (p = 0.67), Spearman ρ = −0.017 | **uninformative** — ceiling ≈ 0 (§6.4) |
| per-sample **level** correlation | Pearson r = +0.071 (p = 0.079), Spearman ρ = +0.099 | **real disagreement** — 0.09× the same-reliability ceiling of 0.805 |
| theme-aggregated delta correlation | Pearson r = **+0.736** (p = 0.06), 7 themes | suggestive agreement on coarse structure |
| random-bin delta correlation | Pearson r = +0.007 (p = 0.99), 10 bins of ~30 | null, as the reliability analysis predicts |
| MDE, true metric at n = 1,000 | 0.0122 | vs observed true delta +0.0056 → underpowered ~4× in n |

Aggregation was tried two ways (theme bins and random bins). Only theme bins were
positive. With n = 7 and p = 0.06 that is one marginal result out of two attempts, and it
is reported as suggestive, not established.

### 6.6 Interpretation

Reading the four correlation rows together:

1. **Response by response, the two rubrics disagree about quality.** Level correlation is
   +0.071 against a ceiling of 0.805 — the levels *are* measured well enough to correlate
   strongly, and they don't. The consolidated rubric ranks individual responses
   differently from HealthBench.
2. **Response by response, neither rubric can measure *change* at all.** Delta
   reliability ≈ 0. This is a fact about the whole measurement setup, not about
   consolidation: re-decoding the same model moves per-sample scores as much as changing
   the model does.
3. **At the level of broad themes the two may agree** (r = +0.736), which is the
   granularity at which the consolidated rubric's noise averages down enough to leave
   signal.
4. **At the dataset level, agreement is untested rather than confirmed.** The
   difference-of-deltas interval spans ±0.02 around a target effect of 0.005.

Therefore the consolidated score is reported as its own metric — an internally consistent
measure of medical-response quality whose criteria are comparable across the entire
dataset, which per-sample rubrics cannot offer — and **never as a HealthBench estimate**.

An earlier version of this write-up said the near-zero delta correlation showed the proxy
"carries essentially no information about the real one." That over-claimed: the test had
no power to detect agreement even if it existed. §6.4 is the correction.

---

## 7. Controls

| control | result | purpose |
| --- | --- | --- |
| Same-model, different decoder | +0.0021 vs the +0.0022 "effect" | the strongest control: the pipeline cannot separate a model change from a re-decode |
| Negative control (base split into random halves) | −0.0161, null spread [−0.0240, +0.0246] | the metric does not manufacture differences from sampling |
| Label permutation, 10,000 sign flips | p = 0.29 | observed delta sits inside the null |
| Adapter liveness | 41/1,000 identical, median similarity 0.446 | the adapter is active, unlike vLLM's LoRA path for Gemma-4 |
| Weighting sensitivity | ≤ 0.0008 across four conventions | the weight convention is not driving the result |
| Serving-path comparison | difference of deltas +0.0016, CI [−0.0086, +0.0115] | the lossy merge does not measurably change the effect |

---

## 8. Threats to validity

- **Judge test–retest is unmeasured.** The cross-engine control bounds decode + judge
  noise *together*; the judge's own flip rate at fixed input was never isolated. It is the
  one error term with no number attached, and §6.4 shows this class of noise is the
  binding constraint.
- **Batching bias.** One call grading 15 criteria invites halo and anchoring effects that
  independent calls avoid. Mitigated by evidence-before-verdict and fixed ordering (D6);
  not quantified.
- **Bounded score.** Grades floor at 0, so no sample can score negative, unlike real
  HealthBench where 9–11% do (D3).
- **`AC4`~`CP3` correlate at r = 0.87**, under the 0.90 rejection line — some safety
  behaviour is counted twice in the weighting.
- **Synthesis is LLM-generated.** The criteria are GPT-4o abstractions of physician
  writing, not physician-authored. Frozen and versioned (D7), but not clinically reviewed.
- **Ground truth covers a prefix.** The 306 validation samples are the first rows the
  expensive run reached, not a random draw, so they are not guaranteed representative.
- **Two criteria clear |t| > 2 individually** (`AC1` +2.07, `IF2` +2.02) but neither
  survives a Bonferroni correction across 15 criteria (|t| ≥ 2.94).
- **Gate threshold refined after seeing data** (§5). Direction of the change and why it
  does not constitute outcome selection are recorded there.

---

## 9. Deliberately not done

- **Fitting the rubric to the per-sample score.** Weights, criteria, or anchors could be
  optimised to maximise agreement on the 306 ground-truth samples. Not done: it needs a
  held-out split the ground truth cannot afford, and an in-sample fit would carry no
  evidential value. Doing it properly means first extending ground truth (below), then
  splitting it.
- **Extending the faithful benchmark to all 5,000 samples.** Grading each sample against
  *its own* rubric in one batched call costs ~10,000 calls rather than the ~114,000 a
  per-item run needs — the same order of saving consolidation was built for, with none of
  the construct risk. This is the highest-value next step, and it would also give the
  validation real power.
- **Measuring judge test–retest.** Re-grading ~200 samples at a second seed gives the
  per-criterion flip rate and the noise floor's judge component, currently entangled with
  decode noise.
- **More than two checkpoints.** Rank agreement between two models is one bit at a 50%
  baseline. Generating for steps 0/16/32/48/64 and correlating ranks across five points
  is the cheapest real validity evidence available.

---

## 10. Reproduction

```bash
# 0. environment: the OpenAI key here is INTERNAL_OPENAI_API_KEY, not OPENAI_API_KEY
set -a; . ./.env; set +a; export OPENAI_API_KEY="$INTERNAL_OPENAI_API_KEY"
HB=experiments/rar_medicine/variant_b/healthbench

# 1. corpus statistics only (no API calls)
python $HB/consolidate_rubrics.py --stats-only

# 2. synthesise the rubric  (60 GPT-4o calls, ~$1)
python $HB/consolidate_rubrics.py --out $HB/global_rubric_v1.json

# 3. pilot both arms at 200 samples, then gate
for s in base trained; do
  oumi evaluate -c $HB/eval_${s}_native.yaml --tasks.0.num_samples 200
done
python $HB/validate_consolidation.py gate --rubric $HB/global_rubric_v1.json

# 4. revise whatever failed, then re-gate
python $HB/revise_rubric.py                     # -> global_rubric_v2.json

# 5. full run, both arms, same engine (D10)
for s in base trained; do oumi evaluate -c $HB/eval_${s}_native.yaml; done

# 6. top up ground-truth prompt_ids, then validate + report
python $HB/grade_subset.py --global-dir $HB/artifacts/global_base \
                           --truth-dir  $HB/artifacts/base_gemma4_e2b_it
python $HB/validate_consolidation.py validate
python $HB/compare_merge_paths.py
python $HB/make_global_report.py --skip-validation \
  --base $HB/artifacts/native_base --trained $HB/artifacts/native_trained \
  --out $HB/HEALTHBENCH_GLOBAL_REPORT_UNMERGED.md

# unit tests
python -m pytest tests/unit/core/evaluation/ -k healthbench -q
```

Reusing an existing generation cache: pass `--inference_engine OPENAI` so no model
weights are loaded (the engine is constructed but never called).
