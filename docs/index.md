<div class="oumi-hero-stats">
  <h1 class="oumi-hero-title">Open Universal Machine Intelligence</h1>
  <p class="oumi-hero-tagline">Everything you need to build state-of-the-art foundation models, end-to-end.</p>
  <div class="oumi-stats-bar">
    <a href="about/license.html" class="oumi-stat">
      <span class="oumi-stat-value">100%</span>
      <span class="oumi-stat-label">Open Source</span>
    </a>
    <span class="oumi-stat-divider"></span>
    <a href="resources/recipes.html" class="oumi-stat">
      <span class="oumi-stat-value">200+</span>
      <span class="oumi-stat-label">Recipes</span>
    </a>
    <span class="oumi-stat-divider"></span>
    <a href="resources/models/models.html" class="oumi-stat">
      <span class="oumi-stat-value">100+</span>
      <span class="oumi-stat-label">Models</span>
    </a>
    <span class="oumi-stat-divider"></span>
    <a href="https://github.com/oumi-ai/oumi" class="oumi-stat">
      <span class="oumi-stat-value">8.8k</span>
      <span class="oumi-stat-label">GitHub Stars</span>
    </a>
  </div>
  <a href="get_started/quickstart.html" class="oumi-hero-cta">Get Started →</a>
</div>

## What is Oumi?

Oumi is an open-source platform designed for ML engineers and researchers who want to train, fine-tune, evaluate, and deploy foundation models. Whether you're fine-tuning a small language model on a single GPU or training a 405B parameter model across a cluster, Oumi provides a unified interface that scales with your needs.

**Who is Oumi for?**

- **ML Engineers** building production AI systems who need reliable training pipelines and deployment options
- **Researchers** experimenting with new training methods, architectures, or datasets
- **Teams** who want a consistent workflow from local development to cloud-scale training

**What problems does Oumi solve?**

- **Fragmented tooling**: Instead of stitching together different libraries for training, evaluation, and deployment, Oumi provides one cohesive platform
- **Scaling complexity**: The same configuration works locally and on cloud infrastructure (AWS, GCP, Azure, Lambda Labs)
- **Reproducibility**: YAML-based configs make experiments easy to track, share, and reproduce

```{toctree}
:maxdepth: 2
:hidden:
:caption: Getting Started

Home <self>
get_started/quickstart
get_started/installation
get_started/core_concepts
get_started/tutorials
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: User Guides

user_guides/train/train
user_guides/infer/infer
user_guides/evaluate/evaluate
user_guides/analyze/analyze
user_guides/judge/judge
user_guides/launch/launch
user_guides/synth
user_guides/tune
user_guides/quantization
user_guides/customization
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Resources

resources/models/models
resources/datasets/datasets
resources/recipes

```

```{toctree}
:maxdepth: 2
:hidden:
:caption: Reference

API Reference <api/oumi>
CLI Reference <cli/commands>
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: FAQ

faq/troubleshooting
faq/oom
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Development

development/dev_setup
development/contributing
development/code_of_conduct
development/style_guide
development/docs_guide
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: About

about/acknowledgements
about/license
about/citations
```

## Quick Start

**Prerequisites:** Python 3.10+, pip. GPU recommended for larger models (CPU works for small models like SmolLM-135M).

Install Oumi and start training in minutes:

```bash
# Install with GPU support (or use `pip install oumi` for CPU-only)
pip install oumi[gpu]

# Train a model
oumi train -c configs/recipes/smollm/sft/135m/quickstart_train.yaml

# Run inference
oumi infer -c configs/recipes/smollm/inference/135m_infer.yaml --interactive
```

For detailed setup instructions including virtual environments and cloud setup, see the {doc}`installation guide <get_started/installation>`.

## What will you build?

Oumi provides a unified interface across the entire model development lifecycle. The workflows below cover training, evaluation, inference, data synthesis, hyperparameter tuning, and cloud deployment—all driven by YAML configs that work identically on your laptop or a multi-node cluster.

::::{grid} 1 2 3 3
:gutter: 2

:::{grid-item-card} Fine-tune a model on my data
:link: user_guides/train/train
:link-type: doc

Start with a pre-trained model and customize it for your task using SFT, LoRA, DPO, GRPO, and more.
:::

:::{grid-item-card} Evaluate my model's performance
:link: user_guides/evaluate/evaluate
:link-type: doc

Run benchmarks and compare against baselines using standard evaluation suites and LLM judges.
:::

:::{grid-item-card} Deploy a model for inference
:link: user_guides/infer/infer
:link-type: doc

Run inference anywhere—vLLM and llama.cpp locally, or OpenAI and Anthropic remotely—with a unified interface.
:::

:::{grid-item-card} Generate synthetic training data
:link: user_guides/synth
:link-type: doc

Create high-quality training data with LLM-powered synthesis pipelines.
:::

:::{grid-item-card} Optimize my hyperparameters
:link: user_guides/tune
:link-type: doc

Find the best learning rate, batch size, and other settings automatically using bayesian optimization.
:::

:::{grid-item-card} Run training on cloud GPUs
:link: user_guides/launch/launch
:link-type: doc

Launch jobs on AWS, GCP, Azure, or Lambda Labs with a single command.
:::

::::

## Hands-on Notebooks

Explore the most common Oumi workflows hands-on. These notebooks run in Google Colab with pre-configured environments—just click and start experimenting. Try "A Tour" for a high-level overview, or dive straight into a specific topic.

::::{grid} 1 2 3 3
:gutter: 2
:class-container: notebooks-grid

:::{grid-item-card} Getting Started: A Tour
:link: https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi%20-%20A%20Tour.ipynb
:link-type: url

Quick tour of core features: training, evaluation, inference, and job management
:::

:::{grid-item-card} Model Finetuning Guide
:link: https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi%20-%20Finetuning%20Tutorial.ipynb
:link-type: url

End-to-end guide to LoRA tuning with data prep, training, and evaluation
:::

:::{grid-item-card} Model Distillation
:link: https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi%20-%20Distill%20a%20Large%20Model.ipynb
:link-type: url

Guide to distilling large models into smaller, efficient ones
:::

:::{grid-item-card} Model Evaluation
:link: https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi%20-%20Evaluation%20with%20Oumi.ipynb
:link-type: url

Comprehensive model evaluation using Oumi's evaluation framework
:::

:::{grid-item-card} Remote Training
:link: https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi%20-%20Running%20Jobs%20Remotely.ipynb
:link-type: url

Launch and monitor training jobs on cloud platforms (AWS, Azure, GCP, Lambda)
:::

:::{grid-item-card} LLM-as-a-Judge
:link: https://colab.research.google.com/github/oumi-ai/oumi/blob/main/notebooks/Oumi%20-%20Simple%20Judge.ipynb
:link-type: url

Filter and curate training data with built-in judges
:::

::::

## Community & Support

Oumi is a community-first effort. Whether you are a developer, a researcher, or a non-technical user, all contributions are very welcome!

- Join our [Discord community](https://discord.gg/oumi) to get help, share your experiences, and chat with the team
- Check the {doc}`FAQ <faq/troubleshooting>` for common questions and troubleshooting
- Open an issue on [GitHub](https://github.com/oumi-ai/oumi/issues) for bug reports or feature requests
- Read [`CONTRIBUTING.md`](https://github.com/oumi-ai/oumi/blob/main/CONTRIBUTING.md) to send your first Pull Request
- Explore our [open collaboration](https://oumi.ai/community) page to join community research efforts

```{raw} html
<script>
document.addEventListener('DOMContentLoaded', function() {
  document.querySelectorAll('.notebooks-grid a').forEach(function(link) {
    link.setAttribute('target', '_blank');
    link.setAttribute('rel', 'noopener noreferrer');
  });
});
</script>
```
