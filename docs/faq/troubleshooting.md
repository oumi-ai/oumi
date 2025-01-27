# Troubleshooting

## Getting Help

Running into a problem? Check in with us on [Discord](https://discord.gg/oumi)-we're happy to help!

Still can't find a solution? Let us know by filing a new [GitHub Issue](https://github.com/oumi-ai/oumi/issues).

## Common Issues

### Pre-commit hook errors with VS Code

- When committing changes, you may encounter an error with pre-commit hooks related to missing imports.
- To fix this, make sure to start your vscode instance after activating your conda environment.

     ```shell
     conda activate oumi
     code .  # inside the Oumi directory
     ```

### Out of Memory (OOM)

See {doc}`oom` for more information.

### Launching Remote Jobs Fail due to File Mounts

When running a remote job using a command like:

```shell
oumi launch up -c your/config/file.yaml
```

It's common to see failures with errors like:

```
ValueError: File mount source '~/.netrc' does not exist locally. To fix: check if it exists, and correct the path.
```

These errors indicate that your JobConfig contains a reference to a file that does not exist on your local machine. You can remove the offending line from your yaml file's {py:attr}`~oumi.core.configs.JobConfig.file_mounts` to resolve the error if it's unneeded. Otherwise, here's how to resolve the error for specific files often mounted by Oumi jobs:

- `~/.netrc`: This file contains your Weights and Biases (WandB) credentials, which are needed to log your run's metrics to WandB.
  - To fix, follow [these instructions](/development/dev_setup.md#optional-set-up-weights-and-biases)
  - If you don't require WandB logging, disable either TrainingParams.{py:attr}`~oumi.core.configs.TrainingParams.enable_wandb` or EvaluationConfig.{py:attr}`~oumi.core.configs.EvaluationConfig.enable_wandb`, for training and evaluation jobs respectively. This is needed in addition to removing the file mount to prevent an error.
- `~/.cache/huggingface/token`: This file contains your Huggingface credentials, which are needed to access gated datasets/models on HuggingFace Hub.
  - To fix, follow [these instructions](/development/dev_setup.md#optional-set-up-huggingface)

### Training Stability & NaN Loss

- Lower the initial learning rate
- Enable gradient clipping (or, apply further clipping if already enabled)
- Add learning rate warmup

```python
config = TrainingConfig(
    training=TrainingParams(
        max_grad_norm=0.5,
        optimizer="adamw_torch_fused",
        warmup_ratio=0.01,
        lr_scheduler_type="cosine",
        learning_rate=1e-5,
    ),
)
```

### Inference Issues

- Verify {doc}`model </resources/models/models>` and [tokenizer](/resources/models/models.md#tokenizer-integration) paths are correct
- Ensure [input data](/user_guides/infer/infer.md#input-data) is correctly formatted and preprocessed
- Validate that the {doc}`inference engine </user_guides/infer/inference_engines>` is compatible with your model type

### Quantization-Specific Issues

Decreased model performance:

- Increase `lora_r` and `lora_alpha` parameters in {py:obj}`oumi.core.configs.PeftParams`
