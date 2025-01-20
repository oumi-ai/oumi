# VSCode Integration

VSCode provides a powerful IDE environment with debugging, testing, and code intelligence features to quickly debug your training and inference jobs. This guide covers how to configure vscode and how to get started debugging `oumi` jobs.

This guide will also work for other IDEs based on vscode, such as [Cursor](https://www.cursor.com/).

## Environment Setup

### 1. Install VSCode

Download and install VSCode from the [official website](https://code.visualstudio.com/download).

If you're new to VSCode, check out the [VSCode Getting Started Guide](https://code.visualstudio.com/docs/getstarted/getting-started) for more information.

### 2. Install Recommended Extensions

Install these VSCode extensions for the best development experience:

| Category | Extension | Purpose | Recommended |
|----------|-----------|---------|-------------|
| Python Development | [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) | Core Python support | Required |
| Remote Development | [Remote Development](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack) | Remote development support | Required for remote hosts |
| Experimentation | [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter) | Notebook support | Recommended |
| Code Quality | [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) | Python linting and formatting | Strongly recommended |
| Code Quality | [Even Better TOML](https://marketplace.visualstudio.com/items?itemName=tamasfe.even-better-toml) | TOML file support | Recommended |
| Code Quality | [YAML by Red Hat](https://marketplace.visualstudio.com/items?itemName=redhat.vscode-yaml) | YAML file support | Recommended |
| Documentation | [markdownlint](https://marketplace.visualstudio.com/items?itemName=DavidAnson.vscode-markdownlint) | Markdown file support | Recommended |

### 3. Workspace Configuration

Create or update `.vscode/settings.json` with recommended settings:

```json
{
  "editor.defaultFormatter": "charliermarsh.ruff",
  "editor.rulers": [88],
  "python.testing.pytestArgs": ["tests"],
  "python.testing.unittestEnabled": false,
  "python.testing.pytestEnabled": true,
  "python.languageServer": "Pylance",
  "[toml]": {
    "editor.defaultFormatter": "tamasfe.even-better-toml"
  },
  "[yaml]": {
    "editor.defaultFormatter": "redhat.vscode-yaml"
  },
  "notebook.formatOnSave.enabled": true
}
```

### 4. Add Launch Configurations

Create or update `.vscode/launch.json`.

Here are some example configurations you can customize:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Train - Single GPU",
      "type": "debugpy",
      "request": "launch",
      "module": "oumi",
      "args": [
        "train",
        "-c",
        "configs/my_training_config.yaml"
      ],
      "justMyCode": true
    },
    {
      "name": "Train - Multi-GPU (FSDP)",
      "type": "debugpy",
      "request": "launch",
      "module": "torch.distributed.run",
      "console": "integratedTerminal",
      "args": [
        "--standalone",
        "--nproc-per-node",
        "4",
        "-m",
        "oumi",
        "train",
        "-c",
        "configs/my_fsdp_config.yaml"
      ]
    },
    {
      "name": "Interactive Inference",
      "type": "debugpy",
      "request": "launch",
      "module": "oumi",
      "args": [
        "infer",
        "--interactive",
        "-c",
        "configs/my_inference_config.yaml"
      ],
      "justMyCode": true
    }
  ]
}
```

For configuration details, see {doc}`/user_guides/train/configuration`.

   To customize these configurations for your needs:

   1. **Training Configuration**
      - Change the config path in `args` to point to your YAML config file
      - Adjust `justMyCode` to `false` if you need to debug into library code
      - For multi-GPU training, modify `--nproc-per-node` to match your GPU count

   2. **Inference Configuration**
      - Update the config path to your inference YAML file
      - Add additional arguments like `--image` for multimodal models
      - Set environment variables using the `env` field if needed

   3. **Common Customizations**
      ```json
      {
        "env": {
          "CUDA_VISIBLE_DEVICES": "0",  // Specify GPU devices
          "WANDB_MODE": "disabled"      // Disable wandb logging
        },
        "console": "integratedTerminal", // Use integrated terminal
        "justMyCode": false             // Debug into library code
      }
      ```

## Development Features

### 1. Debugging Support

VSCode provides powerful debugging capabilities for Oumi. For detailed information, see the [Python Debugging Documentation](https://code.visualstudio.com/docs/python/debugging).

1. **Interactive Debugging**
   - Set breakpoints in your code
   - Inspect variables and state
   - Step through execution
   - Use watch expressions

2. **Debug Console**

The debug console allows you to inspect the state of your training and inference jobs.

For example, you can access the model configuration, training state, training batch, or any other variables:

   ```python
   # Access training state
   print(trainer.model.config)  # View model configuration
   print(trainer.state.global_step)  # Current training step

   # Inspect batch data
   batch = next(iter(trainer.get_train_dataloader()))
   print(batch.keys())  # View batch structure

   # Check gradients
   for name, param in trainer.model.named_parameters():
       if param.grad is not None:
           print(f"{name}: grad_norm = {param.grad.norm().item()}")
   ```

### 2. Testing Support

VSCode's Test Explorer integrates well with Oumi's pytest-based tests. For detailed information, see the [Python Testing Documentation](https://code.visualstudio.com/docs/python/testing).

VSCode's Test Explorer provides several useful features for testing, including the ability to browse and run tests through the sidebar's Test Explorer, execute individual tests or entire test files, and debug tests using breakpoints.

### 3. Remote Development Support

VSCode's Remote Development extension pack allows you to use a remote machine, container, or Windows Subsystem for Linux (WSL) as a full development environment.

This is particularly useful for training models on remote GPU servers. For detailed information, see the [Remote Development Guide](https://code.visualstudio.com/docs/remote/remote-overview).

The `oumi launch` CLI automatically configures the remote environment for you whenever you launch a job. You can quickly open a remote workspace selecting your cluster name from Remote Explorer tab.

You can open and edit files directly on the remote machine, use the integrated terminal for remote commands, debug remote processes, and forward ports for services like TensorBoard. This creates a smooth development experience that feels just like working locally.

### 4. Jupyter Integration

VSCode provides excellent support for Jupyter notebooks, both local and remote. For detailed information, see [Working with Jupyter Notebooks](https://code.visualstudio.com/docs/datascience/jupyter-notebooks) or our {doc}`Notebooks setup guide</user_guides/train/environments/notebooks>`.

## Next Steps

- Set up {doc}`monitoring tools </user_guides/train/monitoring>` for tracking progress
- Explore {doc}`remote training </user_guides/launch/launch>` for cloud resources
