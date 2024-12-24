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

Oumi is a community-first, end-to-end platform for advanced AI research and development. It provides comprehensive support for foundation model workflows - from pretraining and post-training, to data curation, synthesis, and evaluation. Built with enterprise-grade quality and reliability, Oumi serves both researchers pushing the boundaries of AI and organizations building production-ready solutions.

<p align="center">
   <b>Check out our docs!</b>
   <br>
   ↓↓↓↓↓↓
   <br>
   https://oumi.ai/docs
   <br>
   <b>Password:</b> c155c7d02520
   <br>
   ↑↑↑↑↑↑
</p>

## Features

Oumi is designed to be fully flexible yet easy to use:

- **Run Anywhere**: Train and evaluate models seamlessly across environments - from local machines to remote clusters, with native support for Jupyter notebooks and VS Code debugging.

- **Comprehensive Training**: Support for the full ML lifecycle - from pretraining to fine-tuning (SFT, LoRA, QLoRA, DPO) to evaluation. Built for both research exploration and production deployment.

- **Built for Scale**: First-class support for distributed training with PyTorch DDP and FSDP. Efficiently handle models up to 405B parameters.

- **Reproducible Research**: Version-controlled configurations via YAML files and CLI arguments ensure fully reproducible experiments across training and evaluation pipelines.

- **Unified Interface**: One consistent interface for everything - data processing, training, evaluation, and inference. Seamlessly work with both open models and commercial APIs (OpenAI, Anthropic, Vertex AI).

- **Extensible Architecture**: Easily add new models, datasets, training approaches and evaluation metrics. Built with modularity in mind.

- **Production Ready**: Comprehensive test coverage, detailed documentation, and enterprise-grade support make Oumi reliable for both research and production use cases.

If there's a feature that you think is missing, let us know or join us in making it a reality by sending a [feature request](https://github.com/oumi-ai/oumi/issues/new?template=feature_request.md), or [contributing directly](development/contributing)!

For a full tour of what Oumi can do, dive into our [documentation](https://oumi.ai/docs).

## Getting Started

With just a couple commands you can install Oumi, train, infer, and evaluate. All it would take is something like the following:

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
   oumi train -c configs/recipes/smollm/sft/135m/train_quickstart.yaml

   # Evaluation
   oumi evaluate -c configs/recipes/smollm/evaluation/135m_eval_quickstart.yaml \
   --tasks "[{evaluation_platform: lm_harness, task_name: m_mmlu_en}]"

   # Inference
   oumi infer -c configs/recipes/smollm/inference/135m_infer.yaml \
   --generation.max_new_tokens 40 \
   --generation.temperature 0.7 \
   --interactive
   ```

   For more advanced training options, see the [training guide](/docs/user_guides/train/train.md) and [distributed training](docs/user_guides/train/distributed_training.md).

### Examples &  Recipes

Explore our growing collection of ready-to-use configurations for state-of-the-art models and training workflows:

#### 🦙 Llama Family

| Model | Example Configurations |
|-------|------------------------|
| Llama 3.1 8B | [LoRA](/configs/recipes/llama3_1/sft/8b_lora/train.yaml) • [Full Finetune](/configs/recipes/llama3_1/sft/8b_full/train.yaml) • [QLoRA](/configs/recipes/llama3_1/sft/8b_qlora/train.yaml) • [Pre-training](/configs/recipes/llama3_1/pretraining/8b/train.yaml) • [Inference](/configs/recipes/llama3_1/inference/8b_infer.yaml) • [Evaluation](/configs/recipes/llama3_1/evaluation/8b_eval.yaml) |
| Llama 3.1 70B | [LoRA](/configs/recipes/llama3_1/sft/70b_lora/train.yaml) • [Full Finetune](/configs/recipes/llama3_1/sft/70b_full/train.yaml) • [QLoRA](/configs/recipes/llama3_1/sft/70b_qlora/train.yaml) • [Inference](/configs/recipes/llama3_1/inference/70b_infer.yaml) • [Evaluation](/configs/recipes/llama3_1/evaluation/70b_eval.yaml) |
| Llama 3.1 405B | [LoRA](/configs/recipes/llama3_1/sft/405b_lora/train.yaml) • [QLoRA](/configs/recipes/llama3_1/sft/405b_qlora/train.yaml) • [Full Finetune](/configs/recipes/llama3_1/sft/405b_full/train.yaml) |
| Llama 3.2 3B | [Full Finetune](/configs/recipes/llama3_2/sft/3b_full/train.yaml) • [LoRA](/configs/recipes/llama3_2/sft/3b_lora/train.yaml) • [QLoRA](/configs/recipes/llama3_2/sft/3b_qlora/train.yaml) • [Evaluation](/configs/recipes/llama3_2/evaluation/3b_eval.yaml) • [Inference](/configs/recipes/llama3_2/inference/3b_infer.yaml) |
| Llama 3.2 Vision 11B | [SFT](/configs/recipes/vision/llama3_2_vision/sft/11b_sft_train.yaml) • [Inference (SG-Lang)](/configs/recipes/vision/llama3_2_vision/inference/11b_infer_sglang.yaml) • [Inference (vLLM)](/configs/recipes/vision/llama3_2_vision/inference/11b_infer_vllm.yaml) • [Evaluation](/configs/recipes/vision/llama3_2_vision/evaluation/11b_eval.yaml) |

#### 🎨 Vision Models

| Model | Example Configurations |
|-------|------------------------|
| Llama 3.2 Vision 11B | [SFT](/configs/recipes/vision/llama3_2_vision/sft/11b_sft_train.yaml) • [Inference (SG-Lang)](/configs/recipes/vision/llama3_2_vision/inference/11b_infer_sglang.yaml) • [Inference (vLLM)](/configs/recipes/vision/llama3_2_vision/inference/11b_infer_vllm.yaml) • [Evaluation](/configs/recipes/vision/llama3_2_vision/evaluation/11b_eval.yaml) |
| LLaVA 7B | [SFT](/configs/recipes/vision/llava_7b/sft/7b_sft_train.yaml) • [Inference](/configs/recipes/vision/llava_7b/inference/infer.yaml) |
| Phi3 Vision | [SFT](/configs/recipes/vision/phi3/sft/sft_train.yaml) |
| Qwen2-VL 2B | [SFT](/configs/recipes/vision/qwen2_vl_2b/sft/sft_train.yaml) |

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
| Llama 3.2 Vision 11B | [Native](/configs/recipes/vision/llama3_2_vision/inference/11b_infer.yaml) • [vLLM](/configs/recipes/vision/llama3_2_vision/inference/11b_infer_vllm.yaml) • [SG-Lang](/configs/recipes/vision/llama3_2_vision/inference/11b_infer_sglang.yaml) |
| GPT-2 | [Native](/configs/recipes/gpt2/inference/infer.yaml) |
| Mistral | [Bulk Inference](/configs/examples/bulk_inference/mistral_small_infer.yaml) |

## Example Notebooks

Comprehensive tutorials and guides to help you master Oumi:

| Tutorial | Description |
|----------|-------------|
| [🎯 Getting Started: A Tour](/notebooks/Oumi%20-%20A%20Tour.ipynb) | Comprehensive overview of Oumi's architecture and core capabilities |
| **Model Training & Finetuning** |
| [🔧 Model Finetuning Guide](/notebooks/Oumi%20-%20Finetuning%20Tutorial.ipynb) | Step-by-step guide to efficient model finetuning techniques |
| [🦙 Advanced Llama Training](/notebooks/Oumi%20-%20Tuning%20Llama.ipynb) | In-depth walkthrough of Llama model training and optimization |
| **Deployment & Infrastructure** |
| [🚀 Distributed Inference](/notebooks/Oumi%20-%20Multinode%20Inference%20on%20Polaris.ipynb) | Guide to scaling inference across multiple compute nodes |
| [🔄 vLLM Inference Engine](/notebooks/Oumi%20-%20Using%20vLLM%20Engine%20for%20Inference.ipynb) | High-performance inference using vLLM |
| [☁️ Remote Training](/notebooks/Oumi%20-%20Running%20Jobs%20Remotely.ipynb) | Guide to running jobs on cloud platforms |
| [🖥️ Custom Clusters](/notebooks/Oumi%20-%20Launching%20Jobs%20on%20Custom%20Clusters.ipynb) | Setting up and using custom compute clusters |
| **Datasets & Evaluation** |
| [📊 Dataset Management](/notebooks/Oumi%20-%20Datasets%20Tutorial.ipynb) | Best practices for dataset preparation and processing |
| [⚖️ Custom Judge](/notebooks/Oumi%20-%20Custom%20Judge.ipynb) | Creating custom evaluation metrics and judges |
| [📈 Oumi Judge](/notebooks/Oumi%20-%20Oumi%20Judge.ipynb) | Using Oumi's built-in evaluation framework |

## Documentation

See the [Oumi documentation](https://oumi.ai/docs) to learn more about all the platform's capabilities.

## Contributing

Did we mention that this is a community-first effort? All contributions are welcome!

Please check the `CONTRIBUTING.md` file for guidelines on how to contribute to the project.

If you want to contribute but you are short of ideas, please reach out (<contact@oumi.ai>)!

## Acknowledgements

Oumi makes use of [several libraries](https://oumi.ai/docs/latest/about/acknowledgements.html) and tools from the open-source community. We would like to acknowledge and deeply thank the contributors of these projects! 🚀

## Citation

If you find Oumi useful in your research, please consider citing it using the following entry:

```bibtex
@software{oumi2024,
  author = {Oumi Community},
  title = {Oumi: an Open, Collaborative Platform for Training Large Foundation Models},
  month = {November},
  year = {2024},
  url = {https://github.com/oumi-ai/oumi}
}
```

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
