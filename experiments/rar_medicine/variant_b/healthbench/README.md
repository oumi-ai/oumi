# HealthBench evaluation

This Oumi custom evaluation compares the merged Variant B checkpoint with the
untrained `google/gemma-4-E2B-it` model on all 5,000 examples from
`openai/healthbench`. GPT-4o judges every sample-specific rubric independently.

Set `OPENAI_API_KEY`, then run:

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
  oumi evaluate -c experiments/rar_medicine/variant_b/healthbench/eval_base.yaml

CUDA_VISIBLE_DEVICES=1 VLLM_WORKER_MULTIPROC_METHOD=spawn \
  oumi evaluate -c experiments/rar_medicine/variant_b/healthbench/eval_trained.yaml

python experiments/rar_medicine/variant_b/healthbench/make_report.py
```

If the direct OpenAI account has no credits, use GPT-4o through OpenRouter by setting
`OPENROUTER_API_KEY` and overriding the judge config with
`judge_gpt4o_openrouter.yaml`. The saved rubric cache makes provider failover resumable.

Large artifacts go to `/tmp/oumi_healthbench` because the workspace filesystem is
currently full. Each model directory contains:

- `model_responses.jsonl`: generated HealthBench completions;
- `rubric_judgments.jsonl`: one resumable GPT-4o judgment per rubric;
- `sample_results.jsonl`: responses with scored rubrics and normalized sample scores;
- `summary.json`: dataset score, bootstrap standard deviation, and tag breakdowns.

The score matches the HealthBench reference implementation. Met positive and negative
criteria contribute signed points. A sample's achieved points are divided by the sum
of its positive rubric points, then sample scores are averaged and clipped to `[0, 1]`.

For a smoke test, override the sample count and artifact directory:

```bash
oumi evaluate -c experiments/rar_medicine/variant_b/healthbench/eval_base.yaml \
  --tasks.0.num_samples 10 \
  --tasks.0.eval_kwargs.artifact_dir /tmp/oumi_healthbench_smoke/base
```
