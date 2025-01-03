# Troubleshooting

```{attention}
Section under construction. Contributions welcome!
```

## Getting Help

For more help, check the [Oumi GitHub Issues](https://github.com/oumi-ai/oumi/issues) and [Discord](https://discord.gg/oumi).

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

### Training Stability & NaN Loss

- Lower the initial learning rate
- Enable gradient clipping
- Add learning rate warmup

```python
config = TrainingConfig(
    training=TrainingParams(
        max_grad_norm=1.0,
        optimizer="adamw_torch_fused",
        warmup_steps=100,
        lr_scheduler_type="cosine",
        learning_rate=1e-5,
    ),
)
```

### Inference Issues

- Verify model and tokenizer paths are correct
- Ensure input data is correctly formatted and preprocessed
- Validate that the inference engine is compatible with your model type

### Quantization-Specific Issues

Decreased model performance:

- Increase `lora_r` and `lora_alpha` parameters in {py:obj}`oumi.core.configs.PeftParams`
