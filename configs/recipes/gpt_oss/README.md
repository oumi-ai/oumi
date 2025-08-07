# gpt-oss

## Summary

Configs for OpenAI's gpt-oss model family. See the [blog post](https://huggingface.co/blog/welcome-openai-gpt-oss) for more information. The models in this family, both of which are MoE architectures, are:

- [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b): 21B total/3.6B active parameters
- [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b): 117B total/5.1B active parameters

## Quickstart

> **Note**: It is recommended to run these models on Hopper-family GPUs. We have tested on H100s with CUDA 12.8. The models can still run on other hardware, but you may need to deviate from our instructions below. See the blog above for more details.

1. Follow our [quickstart](https://oumi.ai/docs/en/latest/get_started/quickstart.html) for installation.
2. As mentioned in the blog above, gpt-oss models require some of the latest versions of packages to run. Run the following command:

   ```bash
   uv pip install -U accelerate transformers kernels torchvision git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels
   ```

3. (optional) If you want to run gpt-oss on vLLM, run the following:

   ```bash
   uv pip install -U accelerate transformers kernels torchvision git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels
   ```

4. Run your desired oumi command (examples below)!
   - Note that installing the Oumi repository is **not required** to run the commands. We fetch the latest Oumi config remotely from GitHub thanks to the `oumi://` prefix.

## Example Commands

### Inference

To run interactive inference on gpt-oss-120b locally:

```shell
oumi infer -i -c oumi://configs/recipes/gpt_oss/inference/120b_infer.yaml
```

To run interactive inference on gpt-oss-120b locally using the vLLM inference engine:

```shell
oumi infer -i -c oumi://configs/recipes/gpt_oss/inference/120b_infer.yaml --engine VLLM
```

To run inference on gpt-oss-120b on a Lambda cluster using the vLLM inference engine:

```shell
oumi launch up -c oumi://configs/recipes/gpt_oss/inference/120b_vllm_lambda_job.yaml --cluster gpt-oss-120b-vllm-infer
```

To run interactive remote inference on gpt-oss-120b on Together AI:

```shell
oumi infer -i -c oumi://configs/recipes/gpt_oss/inference/120b_together_infer.yaml
```
