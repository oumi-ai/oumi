![# Oumi: Open Universal Machine Intelligence](docs/_static/logo/header_logo.png)

[![Documentation](https://img.shields.io/badge/Documentation-oumi-blue.svg)](https://oumi.ai/docs/latest/index.html)
![Discord](https://img.shields.io/discord/1286348126797430814?label=Discord)
[![PyPI version](https://badge.fury.io/py/oumi.svg)](https://badge.fury.io/py/oumi)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Tests](https://github.com/oumi-ai/oumi/actions/workflows/pretest.yaml/badge.svg?branch=main)](https://github.com/oumi-ai/oumi/actions/workflows/gpu_tests.yaml)
[![GPU Tests](https://github.com/oumi-ai/oumi/actions/workflows/gpu_tests.yaml/badge.svg?branch=main)](https://github.com/oumi-ai/oumi/actions/workflows/pretest.yaml)
![GitHub Repo stars](https://img.shields.io/github/stars/oumi-ai/oumi)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

### Everything you need to build state-of-the-art foundation models, end-to-end.

Oumi is an end-to-end platform that streamlines the entire lifecycle of foundation models - from data preparation and training to evaluation and deployment. Whether you're experimenting on a laptop or deploying models in production, Oumi provides the tools and workflows you need.

With Oumi, you can:

- 🚀 Train and fine-tune models from 10M to 405B parameters using state-of-the-art techniques (SFT, LoRA, QLoRA, DPO, and more)
- 🤖 Work with both text and multimodal models (Llama, Qwen, Phi, and others)
- 🔄 Synthesize and curate training data with LLM judges
- ⚡️ Deploy models efficiently with popular inference engines (vLLM, SGLang)
- 📊 Evaluate models comprehensively across standard benchmarks
- 🌎 Run anywhere - from laptops to clusters to clouds (AWS, Azure, GCP, Lambda, and more)
- 🔌 Integrate with both open models and commercial APIs (OpenAI, Anthropic, Vertex AI, Parasail, ...)

All with one consistent API, a focus on flexibility & reproducibility for experimentation, and production-grade reliability.

## Why use Oumi?

- 🏢 **Enterprise-Grade**: Built and validated by teams training models at scale
- 🔧 **Zero Boilerplate**: Get started in minutes with ready-to-use recipes for popular models and workflows. No need to write training loops or data pipelines.
- 🎯 **Research Ready**: Perfect for ML research with easily reproducible experiments, and flexible interfaces for customizing each component.
- 🌐 **Broad Model Support**: Works with most popular model architectures - from tiny models to the largest ones, text-only to multimodal.
- 🚀 **SOTA Performance**: Native support for distributed training techniques (FSDP, DDP) and optimized inference engines (vLLM, SGLang).
- 🤝 **Community First**: 100% open source with an active community. No vendor lock-in, no strings attached.

For a full tour of what Oumi can do, dive into the [documentation](https://oumi.ai/docs).

## Core Features

- 🚀 **Model Training**: Train and fine-tune models from 10M to 405B parameters using cutting-edge techniques like SFT, LoRA, QLoRA, and DPO
- 🤖 **Model Support**: Work with both text and multimodal models including Llama, Qwen, Phi, and more
- 🔄 **Data Pipeline**: Synthesize and curate training data using LLM-powered judges
- ⚡️ **Efficient Deployment**: Deploy models at scale with optimized inference engines like vLLM and SGLang
- 📊 **Comprehensive Evaluation**: Benchmark your models across standard industry metrics
- 🌎 **Universal Runtime**: Run anywhere - from local machines to cloud platforms (AWS, Azure, GCP, Lambda)
- 🔌 **Seamless Integration**: Connect with open models and commercial APIs (OpenAI, Anthropic, Vertex AI, Parasail)

## Design Philosophy

Oumi is built on principles that prioritize both research excellence and production reliability:

### Unified Experience
One consistent API spans the entire ML lifecycle, eliminating the need to switch between different tools and frameworks. This unified interface makes it easy to move from experimentation to production.

### Complete Workflow Coverage
Every aspect of foundation model development is supported out of the box - from data preparation to deployment. This comprehensive approach ensures you have all the tools you need in one place.

### Open Architecture
Built on and for open source, Oumi maintains transparency while offering seamless integration with commercial services when needed. This hybrid approach gives you the flexibility to choose the best tools for your needs.

### Research Excellence
Version-controlled configurations and standardized workflows ensure reproducible research. Track experiments, compare results, and share findings with confidence.

### Enterprise-Grade Performance
Optimized for performance at any scale, Oumi handles everything from research prototypes to production deployments efficiently.

Systematic, reproducible model development is more effective than trial-and-error experimentation.

   - Curate training data with built-in synthesis and filtering tools
   - Validate data quality with LLM judges
   - Set up evaluation benchmarks specific to your use case
   - Start with proven recipes for your chosen model and task


## Getting Started

With just a couple commands you can install Oumi, train, infer, and evaluate.

### Installation

```shell
# Install the package (CPU & NPU only)
pip install oumi  # For local development & testing

# OR, with GPU support (Requires Nvidia or AMD GPU)
pip install oumi[gpu]  # For GPU training

# To get the latest version, install from the source
pip install git+https://github.com/oumi-ai/oumi.git
```

### Usage

  ```shell
  # Training
  oumi train -c configs/recipes/smollm/sft/135m/quickstart_train.yaml

  # Evaluation
  oumi evaluate -c configs/recipes/smollm/evaluation/135m/quickstart_eval.yaml

  # Inference
  oumi infer -c configs/recipes/smollm/inference/135m_infer.yaml --interactive
  ```

  For more advanced options, see the [training](https://oumi.ai/docs/latest/user_guides/train/train.html), [evaluation](https://oumi.ai/docs/latest/user_guides/evaluate/evaluate.html), and [inference](https://oumi.ai/docs/latest/user_guides/infer/infer.html) guides.

### Examples &  Recipes

Explore the growing collection of ready-to-use configurations for state-of-the-art models and training workflows:

#### 🦙 Llama Family

| Model | Example Configurations |
|-------|------------------------|
| Llama 3.1 8B | [FFT](/configs/recipes/llama3_1/sft/8b_full/train.yaml) • [LoRA](/configs/recipes/llama3_1/sft/8b_lora/train.yaml) • [QLoRA](/configs/recipes/llama3_1/sft/8b_qlora/train.yaml) • [Pre-training](/configs/recipes/llama3_1/pretraining/8b/train.yaml) • [Inference](/configs/recipes/llama3_1/inference/8b_infer.yaml) • [Evaluation](/configs/recipes/llama3_1/evaluation/8b_eval.yaml) |
| Llama 3.1 70B | [FFT](/configs/recipes/llama3_1/sft/70b_full/train.yaml) • [LoRA](/configs/recipes/llama3_1/sft/70b_lora/train.yaml) • [QLoRA](/configs/recipes/llama3_1/sft/70b_qlora/train.yaml) • [Inference](/configs/recipes/llama3_1/inference/70b_infer.yaml) • [Evaluation](/configs/recipes/llama3_1/evaluation/70b_eval.yaml) |
| Llama 3.1 405B | [FFT](/configs/recipes/llama3_1/sft/405b_full/train.yaml) • [LoRA](/configs/recipes/llama3_1/sft/405b_lora/train.yaml) • [QLoRA](/configs/recipes/llama3_1/sft/405b_qlora/train.yaml) |
| Llama 3.2 1B | [FFT](/configs/recipes/llama3_2/sft/1b_full/train.yaml) • [LoRA](/configs/recipes/llama3_2/sft/1b_lora/train.yaml) • [QLoRA](/configs/recipes/llama3_2/sft/1b_qlora/train.yaml) • [Inference (vLLM)](/configs/recipes/llama3_2/inference/1b_vllm_infer.yaml) • [Inference (SGLang)](/configs/recipes/llama3_2/inference/1b_sglang_infer.yaml) • [Inference](/configs/recipes/llama3_2/inference/1b_infer.yaml) • [Evaluation](/configs/recipes/llama3_2/evaluation/1b_eval.yaml) |
| Llama 3.2 3B | [FFT](/configs/recipes/llama3_2/sft/3b_full/train.yaml) • [LoRA](/configs/recipes/llama3_2/sft/3b_lora/train.yaml) • [QLoRA](/configs/recipes/llama3_2/sft/3b_qlora/train.yaml) • [Inference (vLLM)](/configs/recipes/llama3_2/inference/3b_vllm_infer.yaml) • [Inference (SGLang)](/configs/recipes/llama3_2/inference/3b_sglang_infer.yaml) • [Inference](/configs/recipes/llama3_2/inference/3b_infer.yaml) • [Evaluation](/configs/recipes/llama3_2/evaluation/3b_eval.yaml) |
| Llama 3.2 Vision 11B | [SFT](/configs/recipes/vision/llama3_2_vision/sft/11b_full/train.yaml) • [Inference (vLLM)](/configs/recipes/vision/llama3_2_vision/inference/11b_vllm_infer.yaml) • [Inference (SGLang)](/configs/recipes/vision/llama3_2_vision/inference/11b_sglang_infer.yaml) • [Evaluation](/configs/recipes/vision/llama3_2_vision/evaluation/11b_eval.yaml) |

#### 🎨 Vision Models

| Model | Example Configurations |
|-------|------------------------|
| Llama 3.2 Vision 11B | [SFT](/configs/recipes/vision/llama3_2_vision/sft/11b_full/train.yaml) • [LoRA](/configs/recipes/vision/llama3_2_vision/sft/11b_lora/train.yaml) • [Inference (vLLM)](/configs/recipes/vision/llama3_2_vision/inference/11b_vllm_infer.yaml) • [Inference (SGLang)](/configs/recipes/vision/llama3_2_vision/inference/11b_sglang_infer.yaml) • [Evaluation](/configs/recipes/vision/llama3_2_vision/evaluation/11b_eval.yaml) |
| LLaVA 7B | [SFT](/configs/recipes/vision/llava_7b/sft/train.yaml) • [Inference (vLLM)](configs/recipes/vision/llava_7b/inference/vllm_infer.yaml) • [Inference](/configs/recipes/vision/llava_7b/inference/infer.yaml) |
| Phi3 Vision 4.2B | [SFT](/configs/recipes/vision/phi3/sft/train.yaml) • [Inference (vLLM)](configs/recipes/vision/phi3/inference/vllm_infer.yaml) |
| BLIP-2 3.6B | [SFT](/configs/recipes/vision/blip2_opt_2.7b/sft/oumi_gcp_job.yaml) |
| Qwen2-VL 2B | [SFT](/configs/recipes/vision/qwen2_vl_2b/sft/train.yaml) • [Inference (vLLM)](configs/recipes/vision/qwen2_vl_2b/inference/vllm_infer.yaml) • [Inference (SGLang)](configs/recipes/vision/qwen2_vl_2b/inference/sglang_infer.yaml) • [Inference](configs/recipes/vision/qwen2_vl_2b/inference/infer.yaml) |
| SmolVLM-Instruct 2B | [SFT](/configs/recipes/vision/smolvlm/sft/gcp_job.yaml) |


#### 🎯 Training Techniques

| Model | Example Configurations |
|-------|------------------------|
| Llama 3.1 8B | [FSDP Training](/configs/recipes/llama3_1/sft/8b_lora/fsdp_train.yaml) • [Long Context](/configs/recipes/llama3_1/sft/8b_full/longctx_train.yaml) |
| Phi-3 | [DPO](/configs/recipes/phi3/dpo/train.yaml) • [DPO+FSDP](/configs/recipes/phi3/dpo/fsdp_nvidia_24g_train.yaml) |
| FineWeb Pre-training | [DDP](/configs/examples/fineweb_ablation_pretraining/ddp/train.yaml) • [FSDP](/configs/examples/fineweb_ablation_pretraining/fsdp/train.yaml) |

#### 🚀 Inference

| Model | Example Configurations |
|-------|------------------------|
| Llama 3.1 8B | [Native](/configs/recipes/llama3_1/inference/8b_infer.yaml) |
| Llama 3.1 70B | [Native](/configs/recipes/llama3_1/inference/70b_infer.yaml) |
| Llama 3.2 Vision 11B | [Native](/configs/recipes/vision/llama3_2_vision/inference/11b_infer.yaml) • [vLLM](/configs/recipes/vision/llama3_2_vision/inference/11b_vllm_infer.yaml) • [SGLang](/configs/recipes/vision/llama3_2_vision/inference/11b_sglang_infer.yaml) |
| GPT-2 | [Native](/configs/recipes/gpt2/inference/infer.yaml) |
| Mistral | [Bulk Inference](/configs/examples/bulk_inference/mistral_small_infer.yaml) |

## Example Notebooks

Comprehensive tutorials and guides to help you master Oumi:

| Tutorial | Description |
|----------|-------------|
| [🎯 Getting Started: A Tour](/notebooks/Oumi%20-%20A%20Tour.ipynb) | Comprehensive overview of Oumi's architecture and core capabilities |
| **Model Training & Finetuning** |
| [🔧 Model Finetuning Guide](/notebooks/Oumi%20-%20Finetuning%20Tutorial.ipynb) | Step-by-step guide to efficient model finetuning techniques |
| **Deployment & Infrastructure** |
| [🔄 vLLM Inference Engine](/notebooks/Oumi%20-%20Using%20vLLM%20Engine%20for%20Inference.ipynb) | High-performance inference using vLLM |
| [☁️ Remote Training](/notebooks/Oumi%20-%20Running%20Jobs%20Remotely.ipynb) | Guide to running jobs on cloud platforms |
| [🖥️ Custom Clusters](/notebooks/Oumi%20-%20Launching%20Jobs%20on%20Custom%20Clusters.ipynb) | Setting up and using custom compute clusters |
| **Datasets & Evaluation** |
| [⚖️ Custom Judge](/notebooks/Oumi%20-%20Custom%20Judge.ipynb) | Creating custom evaluation metrics and judges |
| [📈 Oumi Judge](/notebooks/Oumi%20-%20Oumi%20Judge.ipynb) | Using Oumi's built-in evaluation framework |

## Documentation

See the [Oumi documentation](https://oumi.ai/docs) to learn more about all the platform's capabilities.

## Contributing

This is a community-first effort. Contributions are very welcome! 🚀

Please check the [`CONTRIBUTING.md`](https://github.com/oumi-ai/oumi/blob/main/CONTRIBUTING.md) for guidelines on how to contribute to the project.

If you want to contribute, but you are short of ideas, please reach out (<contact@oumi.ai>)!

## Acknowledgements

Oumi makes use of [several libraries](https://oumi.ai/docs/latest/about/acknowledgements.html) and tools from the open-source community. We would like to acknowledge and deeply thank the contributors of these projects! 🙏

## Citation

If you find Oumi useful in your research, please consider citing it:

```bibtex
@software{oumi2025,
  author = {Oumi Community},
  title = {Oumi: an Open, Collaborative Platform for Training Large Foundation Models},
  month = {January},
  year = {2025},
  url = {https://github.com/oumi-ai/oumi}
}
```

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
