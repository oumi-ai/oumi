# Qwen3.5

## Summary

Configs for Alibaba's Qwen3.5 model family. See the [Hugging Face collection](https://huggingface.co/collections/Qwen/qwen35) for more information. Qwen3.5 is a hybrid architecture: three of every four layers are Gated DeltaNet (linear attention) and the fourth is full attention. The checkpoints are vision-language models with a chat template that enables thinking by default. Models in this family include:

- Dense
  - [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) — **GRPO example available**
  - [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B)
  - [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) — **FFT + LoRA configs available**

## Quickstart

1. Follow our [quickstart](https://oumi.ai/docs/en/latest/get_started/quickstart.html) for installation.
2. (Optional) if you wish to kick off jobs on a remote cluster, follow our [job launcher setup guide](https://oumi.ai/docs/en/latest/user_guides/launch/launch.html#setup).
3. Run your desired oumi command (examples below)!
   - Note that installing the Oumi repository is **not required** to run the commands. We fetch the latest Oumi config remotely from GitHub thanks to the `oumi://` prefix.
4. (Optional) If you wish to do deeper experimentation, follow our [instructions](https://oumi.ai/docs/en/latest/development/dev_setup.html) to clone the Oumi repository locally.
   - Make sure to delete the `oumi://` prefix when running Oumi commands, to disable fetching the latest configs from GitHub!

## Example Commands

### Training

To launch Qwen3.5 0.8B FFT training locally:

```shell
oumi train -c oumi://configs/recipes/qwen3_5/sft/0.8b_full/train.yaml
```

### LoRA Training

To launch Qwen3.5 0.8B LoRA training locally:

```shell
oumi train -c oumi://configs/recipes/qwen3_5/sft/0.8b_lora/train.yaml
```

### GRPO (RL with an LLM judge)

To full-fine-tune Qwen3.5 4B on RaR-Medicine with verl GRPO and a gpt-4.1-mini judge reward (4 GPUs, needs `OPENAI_API_KEY`):

```shell
oumi train -c oumi://configs/examples/grpo_verl_medqa/train_qwen3_5_4b.yaml
```

The config header lists the extra packages the hybrid architecture needs (`flash-linear-attention`, `causal-conv1d`, `tilelang`, `qwen-vl-utils`) and explains why remove-padding is off and thinking is disabled.

### Inference

To run inference locally:

```shell
oumi infer -i -c oumi://configs/recipes/qwen3_5/inference/0.8b_infer.yaml
```

To run inference with vLLM:

```shell
oumi infer -i -c oumi://configs/recipes/qwen3_5/inference/0.8b_vllm_infer.yaml
```
