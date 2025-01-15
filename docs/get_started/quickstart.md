# Quickstart

## Pre-requisites

Let's start by installing Oumi. You can easily install the latest stable version of Oumi with the following commands:

```bash
pip install oumi

# Optional: If you have an Nvidia or AMD GPU, you can install the GPU dependencies
pip install oumi[gpu]
```

If you need help setting up your environment (python, pip, git, etc), you can find detailed instructions in the {doc}`/development/dev_setup` guide. The {doc}`installation guide </get_started/installation>` offers more details on how to install Oumi for your specific environment and use case.

## Introduction

Now that we have Oumi installed, let's get started with the basics! We're going to use the `oumi` command-line interface (CLI) to train, evaluate, and run inference with a model.

We'll use a small model (`SmolLM-135M`) so that the examples can run fast on both CPU and GPU. `SmolLM` is a family of state-of-the-art small models with 135M, 360M, and 1.7B parameters, trained on a new high-quality dataset. You can learn more about about them in [this blog post](https://huggingface.co/blog/smollm).

For a full list of recipes, including larger models like Llama 3.2, you can explore the {doc}`recipes page </resources/recipes>`.

## Oumi CLI

The general structure of Oumi CLI commands is:

```bash
oumi <command> [options]
```

For detailed help on any command, you can use the `--help` option:

```bash
oumi --help            # for general help
oumi <command> --help  # for command-specific help
```

The available commands are:

- `train`
- `evaluate`
- `infer`
- `launch`
- `judge`

Let's go through some examples of each command.

## Training

You can quickly start training a model using any of existing {doc}`recipes </resources/recipes>` or your own {doc}`custom configs </user_guides/train/configuration>`. The following command will start training using the recipe in `configs/recipes/smollm/sft/135m/quickstart_train.yaml`:

````{dropdown} configs/recipes/smollm/sft/135m/quickstart_train.yaml
```{literalinclude} ../../configs/recipes/smollm/sft/135m/quickstart_train.yaml
:language: yaml
```
````

```bash
oumi train -c configs/recipes/smollm/sft/135m/quickstart_train.yaml
```

You can easily override any parameters directly in the command line, for example:

```bash
oumi train -c configs/recipes/smollm/sft/135m/quickstart_train.yaml \
  --training.max_steps 5 \
  --training.learning_rate 1e-4 \
  --training.output_dir output/smollm-135m-sft
```

To run the same recipe on your own dataset, you can override the dataset name and path:

```bash
oumi train -c configs/recipes/smollm/sft/135m/quickstart_train.yaml \
  --data.train.datasets "[{dataset_name: text_sft, dataset_path: /path/to/local/dataset}]" \
  --training.output_dir output/smollm-135m-sft-custom
```

You can also run training on multiple GPUs (make sure to [install the GPU dependencies](/get_started/installation.md#optional-dependencies) if not already installed).

For example, if you have a machine with 4 GPUs, you can run:

```bash
oumi distributed torchrun -m \
  oumi train -c configs/recipes/smollm/sft/135m/quickstart_train.yaml \
  --training.output_dir output/smollm-135m-sft-dist
```


## Evaluation

To evaluate a trained model:

````{dropdown} configs/recipes/smollm/evaluation/135m/quickstart_eval.yaml
```{literalinclude} ../../configs/recipes/smollm/evaluation/135m/quickstart_eval.yaml
:language: yaml
```
````

Using a model downloaded from HuggingFace:

```bash
oumi evaluate -c configs/recipes/smollm/evaluation/135m/quickstart_eval.yaml \
  --model.model_name HuggingFaceTB/SmolLM2-135M-Instruct
```

Or with our newly trained model saved on disk:

```bash
oumi evaluate -c configs/recipes/smollm/evaluation/135m/quickstart_eval.yaml \
  --model.model_name output/smollm-135m-sft
```

To explore the benchmarks that our evaluations support, including HuggingFace leaderboards and AlpacaEval, visit our {doc}`evaluation guide </user_guides/evaluate/evaluate>`.

## Inference

To run inference with a trained model:

````{dropdown} configs/recipes/smollm/inference/135m_infer.yaml
```{literalinclude} ../../configs/recipes/smollm/inference/135m_infer.yaml
:language: yaml
```
````

Using a model downloaded from HuggingFace:

```bash
oumi infer -c configs/recipes/smollm/inference/135m_infer.yaml \
  --generation.max_new_tokens 40 \
  --generation.temperature 0.7 \
  --interactive
```

Or with our newly trained model saved on disk:

```bash
oumi infer -c configs/recipes/smollm/inference/135m_infer.yaml \
  --model.model_name output/smollm-135m-sft \
  --generation.max_new_tokens 40 \
  --generation.temperature 0.7 \
  --interactive
```

To learn more about running inference locally or remotely (including OpenAI, Google, Anthropic APIs) and leveraging inference engines to parallelize and speed up your jobs, visit our {doc}`inference guide </user_guides/infer/infer>`.

## Launching Jobs

So far we have been using the `train`, `evaluate`, and `infer` commands to run jobs locally.
To launch a distributed training job:

````{dropdown} configs/recipes/smollm/sft/135m/quickstart_gcp_job.yaml
```{literalinclude} ../../configs/recipes/smollm/sft/135m/quickstart_gcp_job.yaml
:language: yaml
```
````

```bash
oumi launch up -c configs/recipes/smollm/sft/135m/quickstart_gcp_job.yaml
```

To launch an evaluation job:

````{dropdown} configs/recipes/smollm/evaluation/135m/quickstart_gcp_job.yaml
```{literalinclude} ../../configs/recipes/smollm/evaluation/135m/quickstart_gcp_job.yaml
:language: yaml
```
````

```bash
oumi launch up -c configs/recipes/smollm/evaluation/135m/quickstart_gcp_job.yaml
```

To explore the Cloud providers that we support for running jobs on remote clusters, details on remote job management, and authoring configuration files, visit {doc}`running jobs remotely </user_guides/launch/remote>`.

## 🔗 Community

⭐ If you like our project, please give it a star on [GitHub](https://github.com/oumi-ai/oumi).

👋 If you are interested in contributing, please read the [Contributor’s Guide](/development/contributing).
