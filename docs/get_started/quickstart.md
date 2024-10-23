# Quickstart

Now that we have Oumi installed, let's get started with the basics! We're going to use the `oumi` CLI to train, evaluate, and run inference with a model.

We'll use a small model (`SmolLM-135M`) so that the examples can be run on CPU. `SmolLM` is a family of state-of-the-art small models with 135M, 360M, and 1.7B parameters, trained on a new high-quality dataset. You can learn more about about them in [this blog post](https://huggingface.co/blog/smollm).

For a full list of recipes, including larger models like Llama 3.2, you can explore the {doc}`recipes page <../models/recipes>`.

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

```{termynal} termynal:oumi-train
---
typeDelay: 40
lineDelay: 700
---
- value: oumi train -c configs/recipes/smollm/smollm-135m_fft.yaml
  type: input
- Loading configuration...
- "Initializing model: SmolLM-135M"
- type: progress
- Starting training...
- "Epoch 1/3: 100%|██████████| 1000/1000 [00:45<00:00, 22.22it/s]"
- "Epoch 2/3: 100%|██████████| 1000/1000 [00:44<00:00, 22.73it/s]"
- "Epoch 3/3: 100%|██████████| 1000/1000 [00:44<00:00, 22.73it/s]"
- "Training complete!"
- "Saving model to output/smollm-135m-fft..."
```

This uses the configuration in `configs/recipes/smollm/smollm-135m_fft.yaml`.

You can easily override any parameters:

```bash
oumi train -c configs/recipes/smollm/fft/smollm-135m.yaml \
  --training.num_train_epochs=5 \
  --training.learning_rate=1e-4 \
  --output_dir=output/smollm-135m-fft
```

To run the same recipe on a different dataset, you can override the dataset name:

```bash
oumi train -c configs/recipes/smollm/fft/smollm-135m.yaml \
  --data.train.datasets.0.dataset_name text_jsonl_dataset \
  --data.train.datasets.0.dataset_path "/path/to/local/dataset" \
  --output_dir output/smollm-135m-fft-custom
```

We can also run training on multiple GPUs. For example, to run on 4 GPUs:

```bash
torchrun --nproc_per_node=4 oumi train \
  -c configs/recipes/smollm/fft/smollm-135m.yaml \
  --output_dir output/smollm-135m-fft-dist
```

## Evaluation

To evaluate a trained model:

```bash
oumi evaluate -c configs/recipes/smollm/evaluation/smollm-135m.yaml \
  --model.model_name=output/smollm-135m-fft  # the path to our trained model \
  --lm_harness_params.tasks=["mmlu", "hellaswag"]
```

## Inference

To run inference with a trained model:

```bash
oumi infer -c configs/recipes/smollm/inference/smollm-135m.yaml \
  --model.model_name=output/smollm-135m-fft \
  --generation.max_new_tokens=100 \
  --generation.temperature=0.7
```

## Launching Jobs

So far we have been using the `train`, `evaluate`, and `infer` commands to run jobs locally.
To launch a distributed training job:

```bash
oumi launch -c configs/recipes/smollm/launch/gcp_train.yaml
```

To launch an evaluation job:

```bash
oumi launch -c configs/recipes/smollm/launch/gcp_evaluate.yaml
```
