# GLM-4

## Summary

Configs for THUDM/Zhipu AI's GLM-4 model family. Models in this family include:

- [THUDM/glm-4-9b-chat](https://huggingface.co/THUDM/glm-4-9b-chat)
- [zai-org/GLM-4.7](https://huggingface.co/zai-org/GLM-4.7) (via Fireworks API)

## Quickstart

1. Follow our [quickstart](https://oumi.ai/docs/en/latest/get_started/quickstart.html) for installation.
2. (Optional) If you wish to kick off jobs on a remote cluster, follow our [job launcher setup guide](https://oumi.ai/docs/en/latest/user_guides/launch/launch.html#setup).
3. Run your desired oumi command (examples below)!
   - Note that installing the Oumi repository is **not required** to run the commands. We fetch the latest Oumi config remotely from GitHub thanks to the `oumi://` prefix.

## Example Commands

### Training

To launch GLM-4 LoRA training locally:

```shell
oumi train -c oumi://configs/recipes/glm4/sft/4p7_lora_train.yaml
```

To launch GLM-4 LoRA training with multiple GPUs:

```shell
oumi distributed torchrun -m oumi train -c oumi://configs/recipes/glm4/sft/4p7_lora_train.yaml
```

### Inference

To run interactive inference on GLM-4.7 via Fireworks API:

```shell
export FIREWORKS_API_KEY=your_api_key
oumi infer -i -c oumi://configs/recipes/glm4/inference/4p7_fireworks_infer.yaml
```

To run interactive inference on GLM-4 Air via vLLM:

```shell
oumi infer -i -c oumi://configs/recipes/glm4/inference/air_vllm_infer.yaml
```
