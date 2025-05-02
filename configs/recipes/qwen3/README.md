# Qwen3

## Summary

Configs for Alibaba's Qwen3 model family. See the [blog post](https://qwenlm.github.io/blog/qwen3/) for more information. Models in this family include:

- Mixture of Experts
  - [Qwen/Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B)
  - [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B)
- Dense
  - [Qwen/Qwen3-30b-a3b](https://huggingface.co/Qwen/Qwen3-30b-a3b)
  - [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B)
  - [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
  - [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)
  - [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)
  - [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)

## Quickstart

1. Follow our [quickstart](https://oumi.ai/docs/en/latest/get_started/quickstart.html) for installation.
2. (Optional) if you wish to kick off jobs on a remote cluster, follow our [job launcher setup guide](https://oumi.ai/docs/en/latest/user_guides/launch/launch.html#setup).
3. Run your desired oumi command (examples below)!
   - Note that installing the Oumi repository is **not required** to run the commands. We fetch the latest Oumi config remotely from GitHub thanks to the `oumi://` prefix.
4. (Optional) If you with to do deeper experimentation, follow our [instructions](https://oumi.ai/docs/en/latest/development/dev_setup.html) to clone the Oumi repository locally.
   - Make sure to delete the `oumi://` prefix when running Oumi commands, to disable fetching the latest configs from GitHub!

## Example Commands

### Training

To launch Qwen3 30B A3B LoRA training locally:

```shell
oumi train -c oumi://configs/recipes/qwen3/sft/30b_a3b_lora/train.yaml
```

To launch Qwen3 30B A3B LoRA training on a remote GCP 4x A100 cluster:

```shell
oumi launch up -c oumi://configs/recipes/qwen3/sft/30b_a3b_lora/gcp_job.yaml --cluster qwen3-30b-a3b-lora
```

### Evaluation

To evaluate Qwen3 30B A3B locally:

```shell
oumi evaluate -c oumi://configs/recipes/qwen3/evaluation/30b_a3b_eval.yaml
```

To instead use the vLLM engine for inference during evaluation:

```shell
oumi evaluate -c oumi://configs/recipes/qwen3/evaluation/30b_a3b_eval.yaml --inference_engine VLLM
```

To evaluate Qwen3 30B A3B on a remote GCP 4x A100 cluster:

```shell
oumi launch up -c oumi://configs/recipes/qwen3/evaluation/30b_a3b_gcp_job.yaml --cluster qwen3-30b-a3b-eval
```

### Inference

To run interactive inference on Qwen3 30B A3B locally:

```shell
oumi infer -i -c oumi://configs/recipes/qwen3/inference/30b_a3b_infer.yaml
```

To instead use the vLLM engine for inference:

```shell
oumi infer -i -c oumi://configs/recipes/qwen3/inference/30b_a3b_infer.yaml --engine VLLM
```
