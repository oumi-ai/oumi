# RaR-Medicine Meta Rubric (single-judge GRPO reward)

Replaces the per-sample rubrics in `anisha2102/RaR-Medicine` with one fixed rubric.
The per-sample specificity is recovered by conditioning the judge on the sample's
`reference_answer` — the same source GPT-4o used to generate the per-sample rubrics.

## Why these dimensions

Analysis of all 171,770 rubric items across the 17,926 train samples (~9.6 items each)
shows a rigid four-tier anatomy:

| Tier      | Share | Weights | Recurring content                                                        |
|-----------|-------|---------|--------------------------------------------------------------------------|
| Essential | 21.6% | always 5 | correct final answer/diagnosis, explicitly stated                       |
| Important | 36.0% | 3 or 4  | correct mechanism/rationale, key supporting facts, link to patient context |
| Optional  | 28.2% | 1 or 2  | conciseness, clarity, logical flow, patient-centered tone, extra insight |
| Pitfall   | 14.2% | -1 or -2 | does not assert wrong alternatives, no misdiagnosis, no harmful/misleading content |

The meta rubric below mirrors those tiers and weight bands, so the reward scale is
comparable to what the per-sample rubrics would have produced.

## The meta rubric

**Essential (w=5 each)**
- **E1 — Correct final answer.** The response's ultimate conclusion (diagnosis, selected
  option, value, or recommendation) agrees with the reference answer's conclusion.
- **E2 — Explicit commitment.** The response clearly states that single final answer;
  it does not hedge between alternatives or leave the conclusion implicit.

**Important**
- **I1 (w=4) — Sound rationale.** The justification is medically correct and consistent
  with the reference answer's reasoning (mechanism, significance, "why this and not that").
- **I2 (w=3) — Key supporting facts.** Includes the load-bearing supporting facts the
  reference gives, and ties the answer to the specifics of the question (patient's
  presentation, context, constraints).

**Optional**
- **O1 (w=2) — Clarity and concision.** Well-organized, logically ordered, no substantial
  padding or irrelevant digressions.
- **O2 (w=1) — Appropriate communication.** Tone and level fit the audience (professional,
  or patient-appropriate when the question implies a patient); adds genuinely useful
  context rather than jargon.

**Pitfall (deductions; "triggered" = bad)**
- **P1 (w=-2) — Incorrect or harmful content.** Endorses an incorrect alternative answer,
  contradicts the reference conclusion anywhere in the response, or makes an unsafe or
  clinically harmful recommendation.
- **P2 (w=-1) — Misleading additions.** Fabricated or wrong incidental claims, or
  extraneous content that muddies or undercuts the correct answer.

## Two ways to score it — both are ONE judge, ONE call per rollout

The 8 criteria are never 8 separate judges. The variants differ only in what the single
judge call outputs:

- **Variant A — explicit aggregation:** judge outputs 8 binary fields in one JSON;
  the reward function computes the weighted sum in code. Auditable and controllable,
  but the RaR paper found fixed weighted sums "offer more control but can be brittle".
- **Variant B — implicit aggregation (RaR paper's best variant):** judge sees the same
  weighted criteria and holistically integrates them into a single 0–10 score;
  `reward = score / 10`. RaR-Implicit was the paper's strongest variant (up to +31%
  on HealthBench over Direct-Likert) and aligned better for smaller judges.

Recommendation: start with **Variant B** to match the paper's best setup; keep Variant A
for debugging reward hacking (its per-criterion bits tell you *which* dimension the
policy is gaming).

## Variant A — explicit aggregation

Compute in the reward function, not in the judge:

```
raw    = 5*E1 + 5*E2 + 4*I1 + 3*I2 + 2*O1 + 1*O2 - 2*P1 - 1*P2
reward = clip(raw / 20, 0.0, 1.0)          # 20 = total positive weight
```

Optional gate (harsher, more GRPO signal on correctness): if `E1 == 0`, cap reward at
`0.25` so eloquent-but-wrong responses can't reach mid-range scores.

### Judge prompt template (Variant A)

Run at temperature 0. Fill `{question}`, `{reference_answer}`, `{response}`.

```text
You are a strict medical answer grader. Grade the RESPONSE against the REFERENCE ANSWER
for the given QUESTION. The reference answer is the ground truth; where the response
disagrees with it, the response is wrong.

Evaluate each criterion independently and answer 1 (met / triggered) or 0 (not):

E1: The response's final conclusion (diagnosis, chosen option, value, or recommendation)
    agrees with the reference answer's conclusion.
E2: The response explicitly and unambiguously commits to that single final answer
    (no hedging between alternatives, no implicit-only conclusion).
I1: The reasoning given for the conclusion is medically sound and consistent with the
    reference answer's rationale.
I2: The response includes the key supporting facts present in the reference answer and
    connects the answer to the specifics of the question (patient context, constraints).
O1: The response is clear, logically organized, and concise, without substantial
    irrelevant content or padding.
O2: The tone and level of explanation fit the intended audience and add useful context.
P1: The response endorses an incorrect alternative answer, contradicts the reference
    conclusion anywhere, or makes an unsafe or harmful recommendation. (1 = this problem
    is present.)
P2: The response contains fabricated or incorrect incidental claims, or misleading
    extraneous content. (1 = this problem is present.)

Rules:
- Judge content, not style preferences beyond O1/O2.
- E1 is about the final conclusion only; partial overlap of reasoning does not make a
  wrong conclusion correct.
- If the response gives no final answer, E1 = 0 and E2 = 0.
- Be strict on P1: any harmful or contradicting statement triggers it even if the final
  answer is correct.

QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

RESPONSE:
{response}

Output exactly one JSON object on a single line, nothing else:
{"why": "<one short sentence overall>", "E1": 0|1, "E2": 0|1, "I1": 0|1, "I2": 0|1, "O1": 0|1, "O2": 0|1, "P1": 0|1, "P2": 0|1}
```

## Variant B — implicit aggregation (single holistic score)

One judge call outputs one score; `reward = score / 10`. No weighted sum in code —
the weights are shown to the judge and it integrates them holistically. This mirrors
RaR-Implicit, the paper's best-performing variant.

### Judge prompt template (Variant B)

Run at temperature 0. Fill `{question}`, `{reference_answer}`, `{response}`.

```text
You are a strict medical answer grader. Grade the RESPONSE against the REFERENCE ANSWER
for the given QUESTION. The reference answer is the ground truth; where the response
disagrees with it, the response is wrong.

Weigh the response against these criteria. Higher-weight criteria matter more; the
Pitfall criteria are deductions:

Essential (weight 5 each):
- E1: The response's final conclusion (diagnosis, chosen option, value, or
  recommendation) agrees with the reference answer's conclusion.
- E2: The response explicitly and unambiguously commits to that single final answer
  (no hedging between alternatives, no implicit-only conclusion).
Important:
- I1 (weight 4): The reasoning given for the conclusion is medically sound and
  consistent with the reference answer's rationale.
- I2 (weight 3): The response includes the key supporting facts present in the
  reference answer and connects the answer to the specifics of the question.
Optional:
- O1 (weight 2): The response is clear, logically organized, and concise.
- O2 (weight 1): The tone and level of explanation fit the intended audience.
Pitfalls (deduct when present):
- P1 (weight -2): The response endorses an incorrect alternative answer, contradicts
  the reference conclusion anywhere, or makes an unsafe or harmful recommendation.
- P2 (weight -1): The response contains fabricated or incorrect incidental claims,
  or misleading extraneous content.

Rules:
- A response whose final conclusion is wrong (E1 fails) cannot score above 3.
- A response that is correct and explicit but gives no reasoning should land mid-range.
- Reserve 9-10 for responses that are correct, well-reasoned, complete, and clean.
- Judge content, not style preferences beyond O1/O2.

QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

RESPONSE:
{response}

Considering all criteria and their weights, rate the response on a scale of 0 to 10.
Output exactly one JSON object on a single line, nothing else:
{"why": "<one short sentence overall>", "score": <integer 0-10>}
```

## GRPO integration notes

- Both variants: one judge call per rollout. Parse the JSON; on parse failure, retry
  once, then fall back to reward 0 (and log).
- Variant A discriminates deterministically (weighted sum in code) and tells you which
  dimension a policy is gaming; Variant B matches the paper's best results and lets the
  judge handle partial credit within a criterion, at the cost of auditability.
- Rewards land in [0, 1] either way; GRPO's group-relative normalization handles scale.
  Variant B's integer 0-10 scale gives 11 levels — watch for tie-heavy groups early in
  training if most rollouts cluster at the same score.
- ~88% of samples come from `medical-o1-reasoning-SFT` (many MCQ-style); E1/E2 handle
  both MCQ ("identifies option B") and open-ended ("states one to three weeks") forms.
