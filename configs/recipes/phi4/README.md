# Phi4

## Summary

Configs for Microsoft's Phi4 model family. See the [blog post](https://azure.microsoft.com/en-us/blog/one-year-of-phi-small-language-models-making-big-leaps-in-ai/) for more information. Models in this family include:

- [microsoft/Phi-4-mini-reasoning](https://huggingface.co/microsoft/Phi-4-mini-reasoning): A 3.8B parameter reasoning model.
- [microsoft/Phi-4-reasoning](https://huggingface.co/microsoft/Phi-4-reasoning): A 14B parameter reasoning model.
- [microsoft/Phi-4-reasoning-plus](https://huggingface.co/microsoft/Phi-4-reasoning-plus): A 14B parameter reasoning model trained with RL.

## Quickstart

1. Follow our [quickstart](https://oumi.ai/docs/en/latest/get_started/quickstart.html) for installation.
2. (Optional) if you wish to kick off jobs on a remote cluster, follow our [job launcher setup guide](https://oumi.ai/docs/en/latest/user_guides/launch/launch.html#setup).
3. Run your desired oumi command (examples below)!
   - Note that installing the Oumi repository is **not required** to run the commands. We fetch the latest Oumi config remotely from GitHub thanks to the `oumi://` prefix.
4. (Optional) If you with to do deeper experimentation, follow our [instructions](https://oumi.ai/docs/en/latest/development/dev_setup.html) to clone the Oumi repository locally.
   - Make sure to delete the `oumi://` prefix when running Oumi commands, to disable fetching the latest configs from GitHub!

## Example Commands

The following commands can also be used for `microsoft/Phi-4-reasoning` by updating `model.model_name`, since it's the same size as `microsoft/Phi-4-reasoning-plus`.

### Training

To launch Phi-4-reasoning-plus LoRA training locally:

```shell
oumi train -c oumi://configs/recipes/phi4/sft/reasoning_plus/lora_train.yaml
```

To instead launch it on a remote GCP 4x A100 cluster:

```shell
oumi launch up -c oumi://configs/recipes/phi4/sft/reasoning_plus/lora_gcp_job.yaml --cluster phi-4-reasoning-plus-lora
```

### Evaluation

To evaluate Phi-4-reasoning-plus locally:

```shell
oumi evaluate -c oumi://configs/recipes/phi4/evaluation/reasoning_plus_eval.yaml
```

To instead use the vLLM engine for inference during evaluation:

```shell
oumi evaluate -c oumi://configs/recipes/phi4/evaluation/reasoning_plus_eval.yaml --inference_engine VLLM
```

To evaluate Phi-4-reasoning-plus on a remote GCP 4x A100 cluster:

```shell
oumi launch up -c oumi://configs/recipes/phi4/evaluation/reasoning_plus_gcp_job.yaml --cluster phi-4-reasoning-plus-eval
```

### Inference

To run interactive inference on Phi-4-reasoning-plus locally:

```shell
oumi infer -i -c oumi://configs/recipes/phi4/inference/reasoning_plus_infer.yaml
```

To instead use the vLLM engine for inference:

```shell
oumi infer -i -c oumi://configs/recipes/phi4/inference/reasoning_plus_infer.yaml --engine VLLM
```
