# Training Environments

```{toctree}
:maxdepth: 2
:caption: Training Environments
:hidden:

local
vscode
notebooks
```

Training machine learning models requires different environments as you progress from initial experimentation & debugging to large-scale deployment.

Oumi supports training in various environments to suit different workflows and preferences. Moving between environments is streamlined through consistent configuration:

- The `train.yaml` config file outlines your model, dataset, and training parameters,
- The `job_config.yaml` contains your resource requirements (optional for training locally).

## Environment Overview

| Environment | Best For | Advantages | Resource Scale | Setup Complexity |
|------------|----------|--------------|----------------|------------------|
| {doc}`Local <local>` | Initial development, algorithmic testing | Provides rapid development cycles with immediate feedback loops. | CPU only, Single GPU, Multi-GPU (1-8) | 🟢 Easy:<br>Python + GPU drivers |
| {doc}`VSCode, Cursor <vscode>` | Debugging | Step-by-step debugging capabilities with seamless Git integration and remote development support which allows you to debug your code running on a remote GPU machine. | CPU only, Single GPU, Multi-GPU (1-8) | 🟡 Moderate:<br>IDE setup + extensions |
| {doc}`Notebooks <notebooks>` | Research, interactive experimentation, visualization | Enables fluid experimentation with real-time code execution and immediate feedback. | CPU only, Single GPU, Multi-GPU (1-8) | 🟢 Easy:<br>Jupyter setup |
| {doc}`Remote </user_guides/launch/launch>` | Production training, large-scale deployment, hyper-parameter tuning | Enterprise-grade deployment capabilities with automated resource allocation and cluster management. Integrates seamlessly with major cloud providers, with support for integrating with custom clusters. | Multi-node deployments (16+ GPUs)<br>Frontier-scale (1000+ GPUs) | Scales with size:<br>• 🟡 Moderate: Single node (1-8 GPUs)<br>• 🔴 Complex: Multi-node (16-64 GPUs)<br>• 🔴 Advanced: Large cluster (64+ GPUs) |

## Recommended Workflow

While Oumi supports multiple training environments, we recommend a systematic progression through development stages:

- **Start Local and Small:** Begin with local development using smaller models (like LLaMA-3.2-1B) to establish core functionality.
  - If you are on CPU, even smaller models like `SmolLM-135m` and `gpt2` are recommended for faster experimentation.
- **Debug in VSCode (or IDE of choice):** Leverage VSCode's integrated debugging tools to identify and resolve issues faster (much easier than print statements everywhere 😉).
- **Scale to Small Distributed:** Test on multi-GPU setups (e.g., 1x8 GPU configurations) to validate distributed training.
- **Deploy to Cluster:** Scale to cloud providers (GCP, AWS, Lambda Labs, etc.) or custom clusters (Polaris, Frontier, etc.) when ready for full-scale training.

## Next Steps

- Check out {doc}`configuration options </user_guides/train/configuration>`
- Set up {doc}`monitoring tools </user_guides/train/monitoring>`
