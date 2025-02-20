# Leaderboards

Leaderboards provide a structured, transparent, and competitive environment for evaluating Large Language Models (LLMs), helping to guide the development of more powerful, reliable, and useful models while fostering collaboration and innovation within the field. This page discusses how to evaluate models on popular leaderboards.

## HuggingFace Leaderboard V2

As of early 2025, the most popular standardized benchmarks, used across academia and industry, are the benchmarks introduced by HuggingFace's latest (V2) leaderboard. HuggingFace has posted [a blog](https://huggingface.co/spaces/open-llm-leaderboard/blog) elaborating on why these benchmarks have been selected, while EleutherAI has also provided a comprehensive [README](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/leaderboard/README.md) discussing the benchmark evaluation goals, coverage, and applicability.

- MMLU-Pro (Massive Multitask Language Understanding) [[paper](https://arxiv.org/abs/2406.01574)]
- GPQA (Google-Proof Q&A Benchmark) [[paper](https://arxiv.org/abs/2311.12022)]
- MuSR (Multistep Soft Reasoning) [[paper](https://arxiv.org/abs/2310.16049)]
- MATH (Mathematics Aptitude Test of Heuristics, Level 5). [[paper](https://arxiv.org/abs/2103.03874)]
- IFEval (Instruction Following Evaluation) [[paper](https://arxiv.org/abs/2311.07911)]
- BBH (Big Bench Hard) [[paper](https://arxiv.org/abs/2210.09261)]

You can evaluate a model on Hugging Face's latest leaderboard by creating a yaml file and invoking the CLI with the following command:

````{dropdown} configs/recipes/smollm/evaluation/135m/leaderboards/huggingface_leaderboard_v2_eval.yaml
```{literalinclude} /../configs/recipes/smollm/evaluation/135m/leaderboards/huggingface_leaderboard_v2_eval.yaml
:language: yaml
```
````

```bash
oumi launch up -c configs/recipes/smollm/evaluation/135m/leaderboards/huggingface_leaderboard_v2_eval.yaml
```

A few things to pay attention to:

- **GPQA Gating**. Access to GPQA is restricted through gating mechanisms, to minimize the risk of data contamination. Before running the leaderboard evaluation, you must first log in to HuggingFace and accept the [terms of use for QPQA](https://huggingface.co/datasets/Idavidrein/gpqa). In addition, you need to authenticate on the Hub using HuggingFace's [User Access Token](https://huggingface.co/docs/hub/security-tokens#user-access-tokens) when launching the evaluation job. You can do so either by setting the environmental HuggingFace token variable [HF_TOKEN](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hftoken) or by storing its value at [HF_TOKEN_PATH](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hftokenpath) (default location is `~/.cache/huggingface/token`).
- **Dependencies**. This leaderboard (specifically the `IFEval` and `MATH` benchmarks) requires specific packages to be deployed to function correctly. You can either install all Oumi evaluation packages with `pip install oumi[evaluation]`, or explore the required packages for each benchmark at {gh}`src/oumi/evaluation/platform_prerequisites.py` and only install the packages needed for your specific case.

## HuggingFace Leaderboard V1

Before HuggingFace's leaderboard V2 was introduced, the most popular benchmarks were captured in the [V1 leaderboard](https://huggingface.co/docs/leaderboards/en/open_llm_leaderboard/archive). Note that due to the fast advancement of AI models, many of these benchmarks have been saturated (i.e., they became too easy to measure meaningful improvements for recent models) while newer models also showed signs of contamination, indicating that data very similar to these benchmarks may exist in their training sets.

- ARC (AI2 Reasoning Challenge) [[paper](https://arxiv.org/abs/1803.05457)]
- MMLU (Massive Multitask Language Understanding) [[paper](https://arxiv.org/abs/2009.03300)]
- Winogrande (Adversarial Winograd Schema Challenge at Scale) [[paper](https://arxiv.org/abs/1907.10641)]
- HellaSwag (Harder Endings, Longer contexts, and Low-shot Activities for Situations With Adversarial Generations) [[paper](https://arxiv.org/abs/1905.07830)]
- GSM 8K (Grade School Math) [[paper](https://arxiv.org/abs/2110.14168)]
- TruthfulQA (Measuring How Models Mimic Human Falsehoods) [[paper](https://arxiv.org/abs/2109.07958)]

You can evaluate a model on Hugging Face's V1 leaderboard by creating a yaml file and invoking the CLI with the following command:

````{dropdown} configs/recipes/smollm/evaluation/135m/leaderboards/huggingface_leaderboard_v1_eval.yaml
```{literalinclude} /../configs/recipes/smollm/evaluation/135m/leaderboards/huggingface_leaderboard_v1_eval.yaml
:language: yaml
```
````

```bash
oumi launch up -c configs/recipes/smollm/evaluation/135m/leaderboards/huggingface_leaderboard_v1_eval.yaml
```

## Running Remotely

Running leaderboard evaluations can be resource-intensive, particularly when working with large models that require GPU acceleration. As such, you may need to execute on remote machines with the necessary hardware resources. Provisioning and running leaderboard evaluations on a remote GCP machine can be achieved with the following sample yaml code.

- HuggingFace Leaderboard V2:

````{dropdown} configs/recipes/smollm/evaluation/135m/leaderboards/huggingface_leaderboard_v2_gcp_job.yaml
```{literalinclude} /../configs/recipes/smollm/evaluation/135m/leaderboards/huggingface_leaderboard_v2_gcp_job.yaml
:language: yaml
```
````

```bash
oumi launch up -c configs/recipes/smollm/evaluation/135m/leaderboards/huggingface_leaderboard_v2_gcp_job.yaml
```

- HuggingFace Leaderboard V1:

````{dropdown} configs/recipes/smollm/evaluation/135m/leaderboards/huggingface_leaderboard_v1_gcp_job.yaml
```{literalinclude} /../configs/recipes/smollm/evaluation/135m/leaderboards/huggingface_leaderboard_v1_gcp_job.yaml
:language: yaml
```
````

```bash
oumi launch up -c configs/recipes/smollm/evaluation/135m/leaderboards/huggingface_leaderboard_v1_gcp_job.yaml
```

```{tip}
In addition to GCP, Oumi supports out-of-the-box various cloud providers (including AWS, Azure, Runpod, Lambda) or even your own custom cluster. To explore these, visit the {doc}`running code on clusters </user_guides/launch/launch>` page.
```

A few things to pay attention to:

- **Output folder**. When executing in a remote machine that is not accessible after the evaluation completes, you need to re-direct your output to persistent storage. For GCP, you can store your output into a mounted GCS Bucket. For example, assuming your bucket is `gs://my-gcs-bucket`, mount to it and set `output_dir` as shown below.

```yaml
storage_mounts:
  /my-gcs-bucket:
    source: gs://my-gcs-bucket
    store: gcs

output_dir: "/my-gcs-bucket/huggingface_leaderboard"
```

- **HuggingFace Access Token**. If you need to authenticate on the HuggingFace Hub to access private or gated models, datasets, or other resources that require authorization, you need to cache HuggingFace's [User Access Token](https://huggingface.co/docs/hub/security-tokens#user-access-tokens) in the remote machine. This token is acting as a HuggingFace login credential to interact with the platform beyond publicly available content. To do so, mount the locally cached token file (by default `~/.cache/huggingface/token`) to the remote machine, as shown below.

```yaml
file_mounts:
  ~/.cache/huggingface/token: ~/.cache/huggingface/token # HF credentials
```

- **W&B Credentials**. If you are using [Weights & Biases](https://wandb.ai/site/) for experiment tracking, make sure you mount the locally cached credentials file (by default `~/.netrc`) to the remote machine, as shown below.

```yaml
file_mounts:
  ~/.netrc: ~/.netrc
```

- **Dependencies**. If you need to deploy packages in the remote machine, such as Oumi's evaluation packages, make sure that these are installed in the setup script, which is executed before the job starts (typically during cluster creation).

```yaml
setup: |
  pip install oumi[evaluation]
```

```{tip}
To learn more on running jobs remotely, including attaching to various storage systems and mounting local files, visit the {doc}`running code on clusters </user_guides/launch/launch>` page.
```
