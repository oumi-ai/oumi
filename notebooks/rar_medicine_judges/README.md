# LLM judges as GRPO rewards on RaR-Medicine

Experiments for choosing (a) a judge system prompt and (b) a judge model to use
as the reward signal when training Gemma with GRPO on
[anisha2102/RaR-Medicine](https://huggingface.co/datasets/anisha2102/RaR-Medicine)
(medical QA with a reference answer and a weighted rubric per question).

There is no rule-based reward to compare against here, so judge quality is
measured with label-free controls, GRPO-relevant signal shape, and agreement
with a strong gold judge. See the docstring in `analyze.py` for every metric.

## Layout

```
common.py            record formats, rubric rendering, explicit-rubric scoring
common_controls.py   nearest-neighbour "mismatch" control selection
generate_rollouts.py sample N prompts x 4 rollouts from a policy (vLLM) + controls
refresh_controls.py  recompute the mismatch controls of an existing rollout file
run_judges.py        score rollouts with (prompt variant x judge model); cached
analyze.py           metrics, inter-judge matrix, disagreement digest, report
judge_prompts/       one YAML per judge prompt variant (model-agnostic)
rescore_explicit.py  re-aggregate an explicit run offline (no judge calls)
rollouts/            <policy>.jsonl  (384 rows = 64 groups x (4 samples + 2 controls))
judge_outputs/       <policy>/<variant>__<model>[__rep1].jsonl, aligned to rollouts
results/             <policy>/summary.{md,csv}, judge_spearman_matrix.csv, scores.parquet
```

## Judge prompt variants

| variant | judge sees | output | notes |
|---|---|---|---|
| `rubric_explicit` | question, rubric | met/not-met per item -> weighted sum in [0,1] | RaR explicit aggregation |
| `rubric_implicit` | question, rubric | holistic 0-10 | RaR implicit aggregation |
| `rubric_and_reference` | + reference answer | holistic 0-10 | does the reference help? |
| `reference_only` | question, reference | holistic 0-10 | no-rubric baseline |
| `essential_gate` | question, rubric | 4-way label -> {1, .5, 0, 0} | correctness-only, flat reward |
| `fixed_rubric_implicit` | question, reference | holistic 0-10 | fixed 8-criterion rubric, same for every question |
| `fixed_rubric_explicit` | question, reference | met/not-met on 8 fixed criteria -> [0,1] | same criteria, aggregated in code |

The two `fixed_rubric_*` variants share one hand-written rubric (E1, E2 essential;
I1, I2 important; O1, O2 optional; P1, P2 pitfalls) applied to every question, with
the reference answer as ground truth. Unlike the `rubric_*` variants they need no
per-question rubric, so they transfer to datasets that do not ship one. The explicit
version declares `fixed_rubric` (ids and weights) and `cap_if_unmet` in its YAML;
the cap expresses "a wrong final conclusion cannot score above 3/10", which a linear
weighted sum cannot.

`--explanation off` produces `<variant>_noexpl` runs (no chain-of-thought field).

## Controls

Every group of 4 policy samples carries two extra rows:

* **reference** (`gen_idx=-1`): the dataset's reference answer.
* **mismatch** (`gen_idx=-2`): the reference answer of the most similar *other*
  question in the split: fluent and on-topic but wrong.

A judge that ranks the mismatch below every policy sample, and does not punish
the terse reference answer, is behaving sensibly without any human labels.

## Run

```bash
PY=/root/miniconda3/envs/oumi/bin/python   # env with oumi + vllm
$PY generate_rollouts.py --policy google/gemma-4-E2B-it --gpu 0
$PY run_judges.py --policy gemma-4-E2B-it --models gpt-4o-mini gpt-4.1 --engine OPENAI
CUDA_VISIBLE_DEVICES=2 $PY run_judges.py --policy gemma-4-E2B-it \
    --models Qwen/Qwen3-4B-Instruct-2507 --engine VLLM
$PY run_judges.py --policy gemma-4-E2B-it --models gpt-4.1 --engine OPENAI \
    --variants rubric_explicit rubric_implicit --rep 1      # self-consistency
$PY analyze.py --policy gemma-4-E2B-it --gold rubric_explicit__gpt-4.1
```

`rescore_explicit.py` re-aggregates a finished explicit run from its stored per-item
verdicts (drop the cap, drop a weight class) without re-calling any judge:

```bash
$PY rescore_explicit.py --policy gemma-4-E2B-it --variant fixed_rubric_explicit \
    --suffix nocap --no-cap
```

`run_judges.export_simple_judge_config(...)` writes a `SimpleJudge` YAML for any
single-score variant, ready for a GRPO judge reward like
`judge_count_letters_verl`.
