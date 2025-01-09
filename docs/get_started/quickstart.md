# Quickstart

Now that we have Oumi installed, let's get started with the basics! We're going to use the `oumi` CLI to train, evaluate, and run inference with a model.

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

To start training a model:

```bash
oumi train -c configs/recipes/smollm/sft/135m/quickstart_train.yaml
```

This uses the configuration in `configs/recipes/smollm/sft/135m/quickstart_train.yaml`:

````{dropdown} configs/recipes/smollm/sft/135m/quickstart_train.yaml
```{literalinclude} ../../configs/recipes/smollm/sft/135m/quickstart_train.yaml
:language: yaml
```
````

You can easily override any parameters:

```bash
oumi train -c configs/recipes/smollm/sft/135m/quickstart_train.yaml \
  --training.max_steps 5 \
  --training.learning_rate 1e-4 \
  --training.output_dir output/smollm-135m-sft
```

To run the same recipe on your own dataset, you can override the dataset name:

```bash
oumi train -c configs/recipes/smollm/sft/135m/quickstart_train.yaml \
  --data.train.datasets "[{dataset_name: text_sft, dataset_path: /path/to/local/dataset}]" \
  --training.output_dir output/smollm-135m-sft-custom
```

You can also run training on multiple GPUs. For example, if you have a machine with 4 GPUs, you can run:

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
  --tasks "[{evaluation_platform: lm_harness, task_name: m_mmlu_en}]"
```

Or with our newly trained model saved on disk:

```bash
oumi evaluate -c configs/recipes/smollm/evaluation/135m/quickstart_eval.yaml \
  --tasks "[{evaluation_platform: lm_harness, task_name: m_mmlu_en}]" \
  --model.model_name output/smollm-135m-sft
```

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

If you like our project, please give it a star on GitHub. If you are interested in contributing, please read Contributor’s Guide.
