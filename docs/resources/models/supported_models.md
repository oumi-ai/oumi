# Supported Models

This page lists all the language models that can be used with Oumi. Thanks to the integration with the [🤗 Transformers](https://github.com/huggingface/transformers) library, you can easily use any of these models for training, evaluation, or inference.

Models prefixed with a checkmark (✅) have been thoroughly tested and validated by the Oumi community, with ready-to-use recipes available in the {gh}`configs <configs/recipes>` directory.

## 📚 Model Categories

### Instruct Models

| Model | Size | Paper | HF Hub  | License  | Open [^1] | Recommended Parameters |
|-------|------|-------|---------|----------|------|------------------------|
| ✅ SmolLM-Instruct | 135M/360M/1.7B | [Blog](https://huggingface.co/blog/smollm) | [Hub](https://huggingface.co/HuggingFaceTB/SmolLM-135M-Instruct) | Apache 2.0 | ✅ | |
| ✅ DeepSeek R1 Family | 1.5B/8B/32B/70B/671B | [Blog](https://api-docs.deepseek.com/news/news250120) | [Hub](https://huggingface.co/deepseek-ai/DeepSeek-R1) | MIT | ❌ | |
| ✅ Llama 3.1 Instruct | 8B/70B/405B | [Paper](https://arxiv.org/abs/2407.21783) | [Hub](https://huggingface.co/meta-llama/Llama-3.1-70b-instruct) | [License](https://llama.meta.com/llama3/license/) | ❌  | |
| ✅ Llama 3.2 Instruct | 1B/3B | [Paper](https://arxiv.org/abs/2407.21783) | [Hub](https://huggingface.co/meta-llama/Llama-3.2-3b-instruct) | [License](https://llama.meta.com/llama3/license/) | ❌  | |
| ✅ Llama 3.3 Instruct | 70B | [Paper](https://arxiv.org/abs/2407.21783) | [Hub](https://huggingface.co/meta-llama/Llama-3.3-70b-instruct) | [License](https://llama.meta.com/llama3/license/) | ❌  | |
| ✅ Phi-3.5-Instruct | 4B/14B | [Paper](https://arxiv.org/abs/2404.14219) | [Hub](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) | [License](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/LICENSE) | ❌  | |
| ✅ Falcon-H1-Instruct | 0.5B/1.5B/3B/7B/34B | Paper coming soon | [Hub](https://huggingface.co/tiiuae/Falcon-H1-34B-Instruct) | [License](https://falconllm.tii.ae/falcon-terms-and-conditions.html ) | ❌ | |
| Qwen2.5-Instruct | 0.5B-70B | [Paper](https://arxiv.org/abs/2412.15115) | [Hub](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | [License](https://github.com/QwenLM/Qwen/blob/main/LICENSE) | ❌  | |
| OLMo 2 Instruct | 7B | [Paper](https://arxiv.org/abs/2501.00656) | [Hub](https://huggingface.co/allenai/OLMo-2-1124-7B) | Apache 2.0 | ✅ | |
| ✅ OLMo 3 Instruct | 7B/32B | [Paper](https://arxiv.org/abs/2402.00838) | [Hub](https://huggingface.co/allenai/OLMo-3-7B-Instruct) | Apache 2.0 | ✅ | |
| MPT-Instruct | 7B | [Blog](https://www.mosaicml.com/blog/mpt-7b) | [Hub](https://huggingface.co/mosaicml/mpt-7b-instruct) | Apache 2.0 | ✅ | |
| Command R | 35B/104B | [Blog](https://cohere.com/blog/command-r7b) | [Hub](https://huggingface.co/CohereForAI/c4ai-command-r-plus) | [License](https://cohere.com/c4ai-cc-by-nc-license) | ❌ | |
| Granite-3.1-Instruct | 2B/8B | [Paper](https://github.com/ibm-granite/granite-3.0-language-models/blob/main/paper.pdf) | [Hub](https://huggingface.co/ibm-granite/granite-3.1-8b-instruct) | Apache 2.0 | ❌ | |
| Gemma 2 Instruct | 2B/9B | [Blog](https://ai.google.dev/gemma) | [Hub](https://huggingface.co/google/gemma-2-2b-it) | [License](https://ai.google.dev/gemma/terms) | ❌ | |
| ✅ Gemma 3 Instruct | 4B/12B/27B | [Blog](https://ai.google.dev/gemma) | [Hub](https://huggingface.co/google/gemma-3-27b-it) | [License](https://ai.google.dev/gemma/terms) | ❌ | |
| DBRX-Instruct | 130B MoE | [Blog](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm) | [Hub](https://huggingface.co/databricks/dbrx-instruct) | Apache 2.0 | ❌ | |
| Falcon-Instruct | 7B/40B | [Paper](https://arxiv.org/abs/2306.01116) | [Hub](https://huggingface.co/tiiuae/falcon-7b-instruct) | Apache 2.0 | ❌  | |
| ✅ Llama 4 Scout Instruct | 17B (Activated) 109B (Total) | [Paper](https://arxiv.org/abs/2407.21783) | [Hub](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) | [License](https://llama.meta.com/llama4/license/) | ❌  | |
| ✅ Llama 4 Maverick Instruct | 17B (Activated) 400B (Total) | [Paper](https://arxiv.org/abs/2407.21783) | [Hub](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct) | [License](https://llama.meta.com/llama4/license/) | ❌  | |

### Vision-Language Models

| Model | Size | Paper | HF Hub | License | Open [^1] | Recommended Parameters |
|-------|------|-------|---------|----------|------|---------------------|
| ✅ Llama 3.2 Vision | 11B | [Paper](https://arxiv.org/abs/2407.21783) | [Hub](https://huggingface.co/meta-llama/Llama-3.2-11b-vision) | [License](https://llama.meta.com/llama3/license/) | ❌  | |
| ✅ LLaVA-1.5 | 7B | [Paper](https://arxiv.org/abs/2310.03744) | [Hub](https://huggingface.co/llava-hf/llava-1.5-7b-hf) | [License](https://ai.meta.com/llama/license) | ❌ | |
| ✅ Phi-3 Vision | 4.2B | [Paper](https://arxiv.org/abs/2404.14219) | [Hub](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) | [License](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/blob/main/LICENSE) | ❌ | |
| ✅ BLIP-2 | 3.6B | [Paper](https://arxiv.org/abs/2301.12597) | [Hub](https://huggingface.co/Salesforce/blip2-opt-2.7b) | MIT | ❌ | |
| ✅ Qwen2-VL | 2B | [Blog](https://qwenlm.github.io/blog/qwen2-vl/) | [Hub](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) | [License](https://github.com/QwenLM/Qwen/blob/main/LICENSE) | ❌  | |
| ✅ Qwen3-VL | 2B/4B/8B | [Blog](https://qwen.ai/blog?id=99f0335c4ad9ff6153e517418d48535ab6d8afef) | [Hub](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | [License](https://github.com/QwenLM/Qwen/blob/main/LICENSE) | ❌  | |
| ✅ SmolVLM-Instruct | 2B | [Blog](https://huggingface.co/blog/smolvlm) | [Hub](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct) | Apache 2.0 | ✅  | |

### Base Models

| Model | Size | Paper | HF Hub | License | Open [^1] | Recommended Parameters |
|-------|------|-------|---------|----------|------|---------------------|
| ✅ SmolLM2 | 135M/360M/1.7B | [Blog](https://huggingface.co/blog/smollm) | [Hub](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) | Apache 2.0 | ✅ | |
| ✅ Llama 3.2 | 1B/3B | [Paper](https://arxiv.org/abs/2407.21783) | [Hub](https://huggingface.co/meta-llama/Llama-3.2-3b) | [License](https://llama.meta.com/llama3/license/) | ❌  | |
| ✅ Llama 3.1 | 8B/70B/405B | [Paper](https://arxiv.org/abs/2407.21783) | [Hub](https://huggingface.co/meta-llama/Llama-3.1-70b) | [License](https://llama.meta.com/llama3/license/) | ❌  | |
| ✅ GPT-2 | 124M-1.5B | [Paper](https://arxiv.org/abs/2005.14165) | [Hub](https://huggingface.co/gpt2) | MIT | ❌ | |
| DeepSeek V2 | 7B/13B | [Paper](https://arxiv.org/abs/2405.04434) | [Hub](https://huggingface.co/deepseek-ai/deepseek-llm-7b-v2) | [License](https://github.com/deepseek-ai/DeepSeek-LLM/blob/main/LICENSE-MODEL) | ❌ | |
| Gemma2 | 2B/9B | [Blog](https://ai.google.dev/gemma) | [Hub](https://huggingface.co/google/gemma2-7b) | [License](https://ai.google.dev/gemma/terms) | ❌ | |
| GPT-J | 6B | [Blog](https://www.eleuther.ai/artifacts/gpt-j) | [Hub](https://huggingface.co/EleutherAI/gpt-j-6b) | Apache 2.0 | ✅ | |
| GPT-NeoX | 20B | [Paper](https://arxiv.org/abs/2204.06745) | [Hub](https://huggingface.co/EleutherAI/gpt-neox-20b) | Apache 2.0 | ✅ | |
| Mistral | 7B | [Paper](https://arxiv.org/abs/2310.06825) | [Hub](https://huggingface.co/mistralai/Mistral-7B-v0.1) | Apache 2.0 | ❌  | |
| Mixtral | 8x7B/8x22B | [Blog](https://mistral.ai/news/mixtral-of-experts/) | [Hub](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) | Apache 2.0 | ❌  | |
| MPT | 7B | [Blog](https://www.mosaicml.com/blog/mpt-7b) | [Hub](https://huggingface.co/mosaicml/mpt-7b) | Apache 2.0 | ✅ | |
| OLMo | 1B/7B | [Paper](https://arxiv.org/abs/2402.00838) | [Hub](https://huggingface.co/allenai/OLMo-7B-hf) | Apache 2.0 | ✅ | |
| ✅ Llama 4 Scout | 17B (Activated) 109B (Total) | [Paper](https://arxiv.org/abs/2407.21783) | [Hub](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E) | [License](https://llama.meta.com/llama4/license/) | ❌  | |

### Reasoning Models

| Model | Size | Paper | HF Hub | License | Open [^1] | Recommended Parameters |
|-------|------|-------|---------|----------|------|---------------------|
| ✅ Qwen3 | 600M/1.7B/4B/8B/14B/32B/30B-A3B/235B-A22B | [Blog](https://qwenlm.github.io/blog/qwen3/) | [Hub](https://huggingface.co/Qwen/Qwen3-235B-A22B) | Apache 2.0 | ❌ | |
| ✅ Qwen3-Next | 80B-A3B | [Blog](https://qwenlm.github.io/blog/qwen3/) | [Hub](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B) | Apache 2.0 | ❌ | |
| Qwen QwQ | 32B | [Blog](https://qwenlm.github.io/blog/qwq-32b-preview/) | [Hub](https://huggingface.co/Qwen/QwQ-32B-Preview) | Apache 2.0 | ❌ | |
| Phi-4-reasoning-plus | 14B | [Blog](https://azure.microsoft.com/en-us/blog/one-year-of-phi-small-language-models-making-big-leaps-in-ai/) | [Hub](https://huggingface.co/microsoft/Phi-4-reasoning-plus) | MIT | ❌ | |

### Code Models

| Model | Size | Paper | HF Hub | License | Open [^1] | Recommended Parameters |
|-------|------|-------|---------|----------|------|---------------------|
| ✅ Qwen2.5 Coder | 0.5B-32B | [Blog](https://qwenlm.github.io/blog/qwen2.5/) | [Hub](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct) | [License](https://github.com/QwenLM/Qwen/blob/main/LICENSE) | ❌  | |
| DeepSeek Coder | 1.3B-33B | [Paper](https://arxiv.org/abs/2401.02954) | [Hub](https://huggingface.co/deepseek-ai/deepseek-coder-7b-instruct) | [License](https://github.com/deepseek-ai/DeepSeek-LLM/blob/main/LICENSE-MODEL) | ❌  | |
| StarCoder 2 | 3B/7B/15B | [Paper](https://arxiv.org/abs/2402.19173) | [Hub](https://huggingface.co/bigcode/starcoder2-15b) | [License](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement) | ✅ | |

### Math Models

| Model | Size | Paper | HF Hub | License | Open [^1]  | Recommended Parameters |
|-------|------|-------|---------|----------|------|---------------------|
| DeepSeek Math | 7B | [Paper](https://arxiv.org/abs/2401.02954) | [Hub](https://huggingface.co/deepseek-ai/deepseek-math-7b-instruct) | [License](https://github.com/deepseek-ai/DeepSeek-LLM/blob/main/LICENSE-MODEL) | ❌  | |

## Additional Resources

- [Model Cards](https://huggingface.co/oumi-ai)

[^1]: Open models are defined as models with fully open weights, training code, and data, and a permissive license. See [Open Source Definitions](https://opensource.org/ai) for more information.
