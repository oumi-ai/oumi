<table border="0">
 <tr>
    <td width="150">
      <img src="docs/_static/logo/oumi_logo_dark.png" alt="Oumi Logo" width="150"/>
    </td>
    <td>
      <h1>Oumi: Open Universal Machine Intelligence</h1>
      <p>E2E Foundation Model Research Platform - Community-first & Enterprise-grade</p>
    </td>
 </tr>
</table>

[![PyPI version](https://badge.fury.io/py/oumi.svg)](https://badge.fury.io/py/oumi)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Pre-review Tests](https://github.com/oumi-ai/oumi/actions/workflows/pretest.yaml/badge.svg?branch=main)](https://github.com/oumi-ai/oumi/actions/workflows/pretest.yaml)
[![Documentation](https://img.shields.io/badge/docs-oumi-blue.svg)](https://oumi.ai/docs/latest/index.html)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

Oumi is a community-first, end-to-end platform for advanced AI research and development. It provides comprehensive support for foundation model workflows - from pretraining and post-training, to data curation, synthesis, and evaluation. Built to be flexible and enterprise-grade, Oumi serves both researchers pushing the boundaries of AI and organizations building production-ready solutions.

<p align="center">
   <b>Check out the detailed Oumi docs!</b>
   <br>
   ↓↓↓↓↓↓
   <br>
   https://oumi.ai/docs
   <br>
   ↑↑↑↑↑↑
</p>

## Features

Oumi is designed to support all common foundation model workflows. As a few examples, with Oumi you can:
-  **Tune** all the most popular and highest performing open text and multimodal models with the latest techniques for the highest quality and efficiency. From the smallest models to the largest ones. On your laptop, your cluster, or any common cloud.
- **Synthesize** data with the largest open models, curate it with LLM judges and readily use it to train and evaluate.
- Perform **inference** with most common inference engines and **evaluation** with most common benchmarks.
- Do many more things out of the box and **anything else** you desire easily thanks to Oumi’s composable and extensible design.

Some of Oumi's key features and design principles include:

- **End-to-end Unified Platform**: Support for the full ML lifecycle with one consistent interface - from pretraining to data curation, data synthesis, fine-tuning (SFT, LoRA, QLoRA, DPO), inference, and evaluation. Seamlessly work with both open models (Llama, QWEN, Phi and others) and commercial APIs (OpenAI, Anthropic, Vertex AI), with both text and multimodal models.

- **Easy To Use**: Prebuilt ready-to-use workflows and recipes for post training and other common operations.

- **Extensible Architecture**: Easily add new models, datasets, training approaches and evaluation metrics. Built with modularity and extensibility in mind.

- **Run Anywhere**: Train and evaluate models seamlessly across environments - from local machines to remote clusters and clouds (AWS, Azure, GCP, Lambda), with native support for Jupyter notebooks and VS Code debugging.

- **Built for Scale**: First-class support for distributed training with PyTorch DDP and FSDP. Efficiently handle models up to 405B parameters.

- **Research-Grade**: Version-controlled configurations via YAML files and CLI arguments ensure fully reproducible experiments across training and evaluation pipelines. Standardization of datasets, evaluation and other experimentation steps make it easy to share, collaborate and build on each other's work.

- **Enterprise-Grade**: Comprehensive test coverage, detailed documentation, and strong support make Oumi reliable for both research and production use cases.

If there's a feature that you think is missing, let us know or join us in making it a reality by sending a [feature request](https://github.com/oumi-ai/oumi/issues/new?template=feature_request.md), or [contributing directly](development/contributing)!

For a full tour of what Oumi can do, dive into the [documentation](https://oumi.ai/docs).

## Getting Started

With just a couple commands you can install Oumi, train, infer, and evaluate.

### Installation

```shell
# Clone the repository
git clone git@github.com:oumi-ai/oumi.git
cd oumi

# Install the package (CPU & NPU only)
pip install -e .  # For local development & testing

# OR, with GPU support (Requires Nvidia or AMD GPU)
pip install -e ".[gpu]"  # For GPU training
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
| Llama 3.2 3B | [FFT](/configs/recipes/llama3_2/sft/3b_full/train.yaml) • [LoRA](/configs/recipes/llama3_2/sft/3b_lora/train.yaml) • [QLoRA](/configs/recipes/llama3_2/sft/3b_qlora/train.yaml) • [Inference](/configs/recipes/llama3_2/inference/3b_infer.yaml)  • [Evaluation](/configs/recipes/llama3_2/evaluation/3b_eval.yaml)|
| Llama 3.2 Vision 11B | [SFT](/configs/recipes/vision/llama3_2_vision/sft/11b_train.yaml) • [Inference (SG-Lang)](/configs/recipes/vision/llama3_2_vision/inference/11b_sglang_infer.yaml) • [Inference (vLLM)](/configs/recipes/vision/llama3_2_vision/inference/11b_vllm_infer.yaml) • [Evaluation](/configs/recipes/vision/llama3_2_vision/evaluation/11b_eval.yaml) |

#### 🎨 Vision Models

| Model | Example Configurations |
|-------|------------------------|
| Llama 3.2 Vision 11B | [SFT](/configs/recipes/vision/llama3_2_vision/sft/11b_train.yaml) • [Inference (SG-Lang)](/configs/recipes/vision/llama3_2_vision/inference/11b_sglang_infer.yaml) • [Inference (vLLM)](/configs/recipes/vision/llama3_2_vision/inference/11b_vllm_infer.yaml) • [Evaluation](/configs/recipes/vision/llama3_2_vision/evaluation/11b_eval.yaml) |
| LLaVA 7B | [SFT](/configs/recipes/vision/llava_7b/sft/train.yaml) • [Inference (vLLM)](configs/recipes/vision/llava_7b/inference/vllm_infer.yaml) • [Inference](/configs/recipes/vision/llava_7b/inference/infer.yaml) |
| Phi3 Vision 4.2B | [SFT](/configs/recipes/vision/phi3/sft/train.yaml) |
| BLIP-2 3.6B | [SFT](/configs/recipes/vision/blip2_opt_2.7b/sft/oumi_gcp_job.yaml) |
| Qwen2-VL 2B | [SFT](/configs/recipes/vision/qwen2_vl_2b/sft/train.yaml) |
| SmolVLM-Instruct 2B | [SFT](/configs/recipes/vision/smolvlm/sft/gcp_job.yaml) |


#### 🎯 Training Techniques

| Model | Example Configurations |
|-------|------------------------|
| Llama 3.1 8B | [FSDP Training](/configs/recipes/llama3_1/sft/8b_lora/fsdp_train.yaml) • [Long Context](/configs/recipes/llama3_1/sft/8b_full/longctx_train.yaml) |
| Phi-3 | [DPO](/configs/recipes/phi3/dpo/train.yaml) • [DPO+FSDP](/configs/recipes/phi3/dpo/fsdp_nvidia_24g_train.yaml) |
| FineWeb Pretraining | [DDP](/configs/examples/fineweb_ablation_pretraining/ddp/train.yaml) • [FSDP](/configs/examples/fineweb_ablation_pretraining/fsdp/train.yaml) |

#### 🚀 Inference

| Model | Example Configurations |
|-------|------------------------|
| Llama 3.1 8B | [Native](/configs/recipes/llama3_1/inference/8b_infer.yaml) |
| Llama 3.1 70B | [Native](/configs/recipes/llama3_1/inference/70b_infer.yaml) |
| Llama 3.2 Vision 11B | [Native](/configs/recipes/vision/llama3_2_vision/inference/11b_infer.yaml) • [vLLM](/configs/recipes/vision/llama3_2_vision/inference/11b_vllm_infer.yaml) • [SG-Lang](/configs/recipes/vision/llama3_2_vision/inference/11b_sglang_infer.yaml) |
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
@software{oumi2024,
  author = {Oumi Community},
  title = {Oumi: an Open, Collaborative Platform for Training Large Foundation Models},
  month = {January},
  year = {2025},
  url = {https://github.com/oumi-ai/oumi}
}
```

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
