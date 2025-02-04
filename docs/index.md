

# 🚀 Oumi: Open-Source Platform for Foundation Models

Everything you need to build state-of-the-art foundation models—end-to-end.

Oumi is a **fully open-source platform** that simplifies the entire lifecycle of foundation models: **data preparation, training, evaluation, and deployment**. Whether you're experimenting on a laptop or running large-scale training on a cluster, Oumi provides the tools and workflows you need.

---

## 🔥 Why Use Oumi?

✅ **Train & Fine-Tune** models from **10M to 405B parameters** (LoRA, QLoRA, DPO, SFT)  
🤖 **Work with Multimodal Models** (Llama, DeepSeek, Qwen, Phi, and more)  
📊 **Evaluate Models** using comprehensive benchmarks  
🔄 **Synthesize & Curate Data** with LLM-powered judges  
⚡ **Deploy Models Efficiently** with optimized inference engines (vLLM, SGLang)  
🌎 **Run Anywhere**: Laptops, Clusters, or Cloud (AWS, Azure, GCP, Lambda)  
🔌 **Integrate Seamlessly** with OpenAI, Anthropic, Vertex AI, Together, Parasail, and more  

All with **one consistent API**, production-grade reliability, and research flexibility.

---

## 📖 Getting Started

### 🚀 Quickstart

```bash
# Install Oumi (CPU & NPU only)
pip install oumi  

# OR, with GPU support (Requires Nvidia or AMD GPU)
pip install oumi[gpu]  

# Install the latest version from source
git clone https://github.com/oumi-ai/oumi.git
cd oumi
pip install .
```

### 🏃 Run Your First Training Job

```bash
oumi train -c configs/recipes/smollm/sft/135m/quickstart_train.yaml
```

### 📊 Evaluate a Model

```bash
oumi evaluate -c configs/recipes/smollm/evaluation/135m/quickstart_eval.yaml
```

### 🔍 Perform Inference

```bash
oumi infer -c configs/recipes/smollm/inference/135m_infer.yaml --interactive
```

For more details, check the [installation guide](https://github.com/oumi-ai/oumi#installation).

---

## ☁️ Running Jobs on the Cloud

Easily run your jobs on major cloud platforms:

```bash
# GCP
oumi launch up -c configs/recipes/smollm/sft/135m/quickstart_gcp_job.yaml

# AWS
oumi launch up -c configs/recipes/smollm/sft/135m/quickstart_aws_job.yaml

# Azure
oumi launch up -c configs/recipes/smollm/sft/135m/quickstart_azure_job.yaml

# Lambda
oumi launch up -c configs/recipes/smollm/sft/135m/quickstart_lambda_job.yaml
```

*Note: Oumi is in beta, and some features are still evolving.*

---

## 🛠 Features

### 🎯 Zero Boilerplate
- Ready-to-use **recipes** for popular models & workflows
- No need to write training loops or data pipelines

### 🏢 Enterprise-Grade
- Built for **scalability** and **reliability**
- Proven by teams training models at scale

### 🎓 Research Ready
- Easily **reproducible experiments**
- Flexible interfaces for customizing every component

### 🌍 Broad Model Support
- Supports **tiny to large-scale models**, including **multimodal** ones
- Compatible with **Hugging Face Transformers**

### 🚀 State-of-the-Art Performance
- Built-in support for **distributed training** (FSDP, DDP)
- Optimized inference engines (**vLLM, SGLang**)

### 🤝 Community First
- 100% **open-source** with **no vendor lock-in**
- Active **developer and research community**

---

## 📚 Example Configurations

Oumi supports a variety of foundation models with ready-to-use configurations:

### 🦙 Llama Family
| Model                | Configurations |
|----------------------|---------------|
| Llama 3.1 8B        | LoRA, QLoRA, Inference (vLLM), Evaluation |
| Llama 3.1 70B       | LoRA, QLoRA, Inference, Evaluation |
| Llama 3.2 3B        | LoRA, QLoRA, Inference (vLLM, SGLang), Evaluation |
| Llama 3.3 70B       | LoRA, QLoRA, Inference, Evaluation |

### 🎨 Vision Models
| Model                 | Configurations |
|-----------------------|---------------|
| Llama 3.2 Vision 11B  | SFT, LoRA, Inference (vLLM, SGLang), Evaluation |
| LLaVA 7B             | SFT, Inference (vLLM) |
| Phi3 Vision 4.2B     | SFT, Inference (vLLM) |

For a full list of supported models, check the [Oumi documentation](https://oumi.ai/docs/models).

---

## 🤝 Contributing

We welcome contributions! To get started:

1. Read the [CONTRIBUTING.md](https://github.com/oumi-ai/oumi/blob/main/CONTRIBUTING.md)
2. Join our **[Discord community](https://discord.gg/oumi-ai)** to discuss, ask questions, and share insights.
3. Explore open issues and start contributing 🚀

---

## 📜 License

Oumi is licensed under the **Apache License 2.0**. See the [LICENSE](https://github.com/oumi-ai/oumi/blob/main/LICENSE) file for more details.

---

## 📝 Citation

If you use Oumi in your research, please cite it:

```bibtex
@software{oumi2025,
  author = {Oumi Community},
  title = {Oumi: An Open, End-to-End Platform for Building Large Foundation Models},
  year = {2025},
  url = {https://github.com/oumi-ai/oumi}
}
```

---

## ✨ Acknowledgements

Oumi is built on the shoulders of giants! A huge thanks to the open-source community and contributors who make this possible. 🚀
