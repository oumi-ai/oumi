# DCVLR: Data Curation for Vision Language Reasoning

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue.svg)](https://neurips.cc/Conferences/2025)
[![Competition](https://img.shields.io/badge/Competition-Open-green.svg)](https://dcvlr.org)

---

<div align="center">

  <h3>
   🌐 <a href="https://dcvlr-neurips.github.io">Official webpage</a> •
   🚀 <a href="https://oumi-ai.typeform.com/to/LnYoisi5">Sign up for updates</a> •
   🎯 <a href="https://oumi-ai.typeform.com/to/OGPuRt6U">Apply for GPU credits (sponsored by Lambda Labs)</a>
   </h3>
</div>

---


DCVLR is the first open-data, open-models, open-source competition for data curation in vision-language reasoning, hosted at NeurIPS 2025.


## 🎯 Challenge

Participants can leverage any source datasets to curate high-quality instruction-tuning datasets (1K or 10K examples). Participants are encouraged to explore diverse curation strategies, from synthetic data generation to subset selection. Submissions will be evaluated by fine-tuning an undisclosed, open-source vision-language model on the curated data and measuring performance across a wide variety of benchmarks.

## 🚀 Quick Start

Get started with training in minutes:

```bash
# Install oumi
uv pip install "oumi[gpu]"

# Train with Molmo-7B-O
oumi train -c molmo-o --dataset dataset.jsonl

# Train with Qwen2.5-VL-7B-Instruct
oumi train -c qwen2.5-vl-7b-instruct --dataset dataset.jsonl
```

## 📅 Key Dates

| Date | Milestone |
|------|-----------|
| **June 11, 2025** | Release of Competition Materials |
| **July 1, 2025** | Submission Portal Opens |
| **October 1, 2025** | Final Submission Deadline |
| **November 1, 2025** | Results Announced |
| **December 2025** | NeurIPS 2025 Presentation |


## 📚 Competition Resources

| Resource | Description | Link |
|----------|-------------|------|
| 📊 **Starter Kit** | Comprehensive starter kit with example datasets, training scripts, and best practices | [Access Starter Kit](https://huggingface.co/datasets/oumi-ai/dcvlr-starter-kit) |
| 💻 **Training Scripts** | Starting scripts for fine-tuning multiple vision-language models | [View Scripts](https://github.com/oumi-ai/oumi/tree/main/configs/projects/dcvlr) |
| 🧪 **Evaluation Code** | Scripts to evaluate model outputs on diverse benchmark development sets | [Get Code](https://github.com/oumi-ai/oumi/tree/main/configs/projects/dcvlr) |
| ☁️ **Compute Resources** | GPU credits from Lambda Labs for participants | [Apply for Credits](https://oumi-ai.typeform.com/to/OGPuRt6U") |
| 📚 **Documentation** | Complete guides and tutorials | [View Documentation](https://oumi.ai/docs) |

## 🤝 Sponsors

- **Lambda Labs** - Compute Resources
- **Oumi.ai** - Competition Support

## 📞 Contact

Have questions? Get in touch with the DCVLR team:

- **Website**: [dcvlr.org](https://dcvlr.org)
- **Email**: [Contact Form](https://dcvlr.org/contact)
