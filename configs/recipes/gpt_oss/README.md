# gpt-oss

## Summary

Configs for OpenAI's gpt-oss model family and community variants. See the [blog post](https://huggingface.co/blog/welcome-openai-gpt-oss) for more information. The models in this family, which are MoE architectures, are:

## Official OpenAI Models
- [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b): 21B total/3.6B active parameters
- [openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b): 117B total/5.1B active parameters

## Community Variants
- [huihui-ai/Huihui-gpt-oss-20b-BF16-abliterated](https://huggingface.co/huihui-ai/Huihui-gpt-oss-20b-BF16-abliterated): Abliterated version of gpt-oss-20b with modified safety restrictions

## Quickstart

> **Note**: It is recommended to run these models on Hopper-family GPUs. We have tested on H100s with CUDA 12.8. The models can still run on other hardware, but you may need to deviate from our instructions below. See the blog above for more details.

1. Follow our [quickstart](https://oumi.ai/docs/en/latest/get_started/quickstart.html) for installation.
2. As mentioned in the blog above, gpt-oss models require some of the latest versions of packages to run. Run the following command:

   ```bash
   uv pip install -U accelerate transformers kernels torchvision git+https://github.com/triton-lang/triton.git@main#subdirectory=python/triton_kernels
   ```

3. Run your desired oumi command (examples below)!
   - Note that installing the Oumi repository is **not required** to run the commands. We fetch the latest Oumi config remotely from GitHub thanks to the `oumi://` prefix.

## Example Commands

### Inference

#### Original OpenAI Models

To run interactive inference on gpt-oss-120b locally:

```shell
oumi infer -i -c oumi://configs/recipes/gpt_oss/inference/120b_infer.yaml
```

To run interactive remote inference on gpt-oss-120b on Together AI:

```shell
oumi infer -i -c oumi://configs/recipes/gpt_oss/inference/120b_together_infer.yaml
```

To run interactive inference on gpt-oss-20b locally:

```shell
oumi infer -i -c oumi://configs/recipes/gpt_oss/inference/20b_infer.yaml
```

#### Community Variants

To run interactive inference on Huihui GPT-OSS-20B Abliterated locally:

```shell
oumi infer -i -c oumi://configs/recipes/gpt_oss/inference/20b_huihui_abliterated_infer.yaml
```

To run memory-efficient GGUF inference on macOS:

```shell
oumi infer -i -c oumi://configs/recipes/gpt_oss/inference/20b_huihui_abliterated_gguf_macos_infer.yaml
```

To run GPU-accelerated vLLM inference:

```shell
oumi infer -i -c oumi://configs/recipes/gpt_oss/inference/20b_huihui_abliterated_vllm_infer.yaml
```
